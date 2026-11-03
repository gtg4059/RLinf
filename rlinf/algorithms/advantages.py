# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


def _distributed_mean(values: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` across ranks when distributed is initialized.

    The default process group is NCCL, so the reduction runs on the current
    CUDA device even when ``values`` were gathered on CPU. The scalar comes
    back on ``values``' device.
    """
    total = values.detach().to(dtype=torch.float32).sum()
    count = torch.tensor(
        float(values.numel()), device=total.device, dtype=torch.float32
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        reduce_device = total.device
        if reduce_device.type == "cpu" and torch.cuda.is_available():
            reduce_device = torch.device("cuda", torch.cuda.current_device())
        total = total.to(reduce_device)
        count = count.to(reduce_device)
        torch.distributed.all_reduce(total)
        torch.distributed.all_reduce(count)
    mean = total / count.clamp(min=1.0)
    return mean.to(device=values.device, dtype=values.dtype)


def _center_by_episode_return(
    returns: torch.Tensor,
    dones: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Subtract the mean episode return from every return-to-go.

    An episode starts at step 0 and again on the step after a done. The
    baseline is the mean of those start returns, so a long miss does not
    outweigh a short open just by having more chunks. The critic target
    stays the uncentered return.
    """
    steps = returns.shape[0]
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != returns.shape:
        raise ValueError(
            "dones steps "
            f"{tuple(ended.shape)} do not match returns {tuple(returns.shape)}"
        )
    start = torch.zeros_like(ended)
    start[0] = True
    if steps > 1:
        start[1:] = ended[:-1]
    if loss_mask is not None:
        start = start & loss_mask.bool()
    selected = returns[start]
    if selected.numel() == 0 and not (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ):
        return returns
    baseline = _distributed_mean(selected)
    return returns - baseline.to(device=returns.device, dtype=returns.dtype)


def _zero_unopened_prefix(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    success_chunk_reward_min: float,
    reach_chunk_reward_min: float,
) -> torch.Tensor:
    """Zero a miss's actor advantage before the door first moves.

    A chunk reward above ``success_chunk_reward_min`` marks that attempt as
    a success, and its advantages stay. On a miss, chunks before the first
    reward above ``reach_chunk_reward_min`` become 0. The move chunk and
    the chunks after it keep the centered advantage. A miss that never
    moves the door is all zeros. This does not change the critic target.

    Args:
        advantages: Centered actor advantages, shape ``[n_steps, batch]``.
        rewards: Per-chunk rewards, same shape as ``advantages``.
        dones: Done flags with one leading bootstrap row.
        success_chunk_reward_min: Chunk reward that marks an open.
        reach_chunk_reward_min: Chunk reward that marks the first door move.
            Must be below ``success_chunk_reward_min``.

    Returns:
        Advantages with the unopened miss prefix set to 0.
    """
    if float(reach_chunk_reward_min) >= float(success_chunk_reward_min):
        raise ValueError(
            "reach_chunk_reward_min must be below success_chunk_reward_min, "
            f"got {reach_chunk_reward_min} and {success_chunk_reward_min}."
        )
    steps = advantages.shape[0]
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != advantages.shape or rewards.shape != advantages.shape:
        raise ValueError(
            "advantage shape "
            f"{tuple(advantages.shape)} does not match rewards "
            f"{tuple(rewards.shape)} or dones steps {tuple(ended.shape)}."
        )
    attempt = _attempt_index(ended)
    opened = rewards > float(success_chunk_reward_min)
    moved = rewards > float(reach_chunk_reward_min)
    out = advantages
    n_attempts = int(attempt.max().item()) + 1 if steps else 0
    for attempt_id in range(n_attempts):
        in_attempt = attempt == attempt_id
        if not bool(in_attempt.any()):
            continue
        miss = in_attempt.any(dim=0) & ~(opened & in_attempt).any(dim=0)
        if not bool(miss.any()):
            continue
        seen = (moved & in_attempt).to(dtype=torch.long).cumsum(dim=0)
        prefix = in_attempt & (seen == 0) & miss.unsqueeze(0)
        out = torch.where(prefix, torch.zeros_like(out), out)
    return out


def _zero_before_reach(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    reach_chunk_reward_min: float,
    success_chunk_reward_min: Optional[float] = None,
) -> torch.Tensor:
    """Zero every attempt's actor advantage before the door first moves.

    The move chunk and the chunks after it keep their advantage. When
    ``success_chunk_reward_min`` is set, only an attempt that opens has
    its prefix zeroed, so a miss keeps the grasp reward. Otherwise a
    trajectory that never moves the door is all zeros. Returns are unchanged.

    Args:
        advantages: Actor advantages, shape ``[n_steps, batch]``.
        rewards: Per-chunk rewards, same shape as ``advantages``.
        dones: Done flags with one leading bootstrap row.
        reach_chunk_reward_min: Chunk reward that marks the first door move.
        success_chunk_reward_min: When set, only attempts that open are masked.

    Returns:
        Advantages with the pre-move prefix set to 0.
    """
    steps = advantages.shape[0]
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != advantages.shape or rewards.shape != advantages.shape:
        raise ValueError(
            "advantage shape "
            f"{tuple(advantages.shape)} does not match rewards "
            f"{tuple(rewards.shape)} or dones steps {tuple(ended.shape)}."
        )
    attempt = _attempt_index(ended)
    moved = rewards > float(reach_chunk_reward_min)
    opened = None
    if success_chunk_reward_min is not None:
        opened = rewards > float(success_chunk_reward_min)
    out = advantages
    n_attempts = int(attempt.max().item()) + 1 if steps else 0
    for attempt_id in range(n_attempts):
        in_attempt = attempt == attempt_id
        if not bool(in_attempt.any()):
            continue
        seen = (moved & in_attempt).to(dtype=torch.long).cumsum(dim=0)
        prefix = in_attempt & (seen == 0)
        if opened is not None:
            is_open = (opened & in_attempt).any(dim=0)
            prefix = prefix & is_open.unsqueeze(0)
        out = torch.where(prefix, torch.zeros_like(out), out)
    return out


def _scale_unopened_advantages(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    success_chunk_reward_min: float,
    scale: float,
) -> torch.Tensor:
    """Multiply a miss attempt's actor advantage by ``scale``.

    An attempt with any chunk above ``success_chunk_reward_min`` keeps its
    advantage. The critic target is unchanged.

    Args:
        advantages: Actor advantages, shape ``[n_steps, batch]``.
        rewards: Per-chunk rewards, same shape as ``advantages``.
        dones: Done flags with one leading bootstrap row.
        success_chunk_reward_min: Chunk reward that marks an open.
        scale: Fraction in ``[0, 1]`` applied to every miss chunk.

    Returns:
        Advantages with miss attempts scaled.
    """
    scale_f = float(scale)
    if scale_f < 0.0 or scale_f > 1.0:
        raise ValueError(
            f"failure_advantage_scale must be in [0, 1], got {scale_f}."
        )
    steps = advantages.shape[0]
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != advantages.shape or rewards.shape != advantages.shape:
        raise ValueError(
            "advantage shape "
            f"{tuple(advantages.shape)} does not match rewards "
            f"{tuple(rewards.shape)} or dones steps {tuple(ended.shape)}."
        )
    attempt = _attempt_index(ended)
    opened = rewards > float(success_chunk_reward_min)
    out = advantages
    n_attempts = int(attempt.max().item()) + 1 if steps else 0
    for attempt_id in range(n_attempts):
        in_attempt = attempt == attempt_id
        if not bool(in_attempt.any()):
            continue
        miss = in_attempt.any(dim=0) & ~(opened & in_attempt).any(dim=0)
        if not bool(miss.any()):
            continue
        mask = in_attempt & miss.unsqueeze(0)
        out = torch.where(mask, out * scale_f, out)
    return out


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    baseline = kwargs.get("advantage_baseline", None)
    if baseline not in (None, "none", "batch_mean"):
        raise ValueError(
            f"advantage_baseline must be none or batch_mean, got {baseline!r}."
        )
    # batch_mean ignores the value head for the actor. Zeros keep gamma.
    if baseline == "batch_mean":
        if values is None:
            values = torch.zeros(
                (rewards.shape[0] + 1, *rewards.shape[1:]),
                dtype=rewards.dtype,
                device=rewards.device,
            )
        else:
            values = torch.zeros_like(values)

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    # Terminations (CRI reset, open) end the backup. A timeout is only a
    # truncation: ``terminals`` stays false there so the final value remains.
    cut = kwargs.get("terminals", dones)
    if cut is None:
        cut = dones
    # A failure end (CRI reset) keeps its penalty on this chunk. The value
    # and the advantage of that chunk do not enter earlier chunks. A success
    # above ``success_chunk_reward_min`` still backs up. A timeout is not a
    # cut, so its final value still bootstraps.
    isolate_failure = bool(kwargs.get("isolate_failure_terminal", False))
    success_min = float(kwargs.get("success_chunk_reward_min", 0.5))
    block = torch.zeros(
        rewards.shape[1:], dtype=torch.bool, device=rewards.device
    )

    for step in reversed(range(T)):
        bootstrap = ~cut[step + 1]
        if isolate_failure:
            bootstrap = bootstrap & ~block
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * bootstrap
                - values[step]
            )

        gae = delta + gamma * gae_lambda * bootstrap * gae
        returns[step] = gae if critic_free else gae + values[step]
        if isolate_failure:
            ended = cut[step + 1]
            if ended.dtype != torch.bool:
                ended = ended != 0
            while ended.ndim > rewards[step].ndim and ended.shape[-1] == 1:
                ended = ended.squeeze(-1)
            failed = ended & (rewards[step] <= success_min)
            block = failed
            gae = torch.where(failed, torch.zeros_like(gae), gae)

    advantages = returns - values[:-1] if not critic_free else returns
    if baseline == "batch_mean":
        advantages = _center_by_episode_return(returns, dones, loss_mask)

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    scale = kwargs.get("failure_advantage_scale", None)
    scale_misses = scale is not None and float(scale) != 1.0
    if kwargs.get("zero_failure_prefix", False) or kwargs.get(
        "advantage_after_reach", False
    ):
        if dones is None:
            raise ValueError("reach advantage masking requires dones.")
        reach_min = float(kwargs.get("reach_chunk_reward_min", 0.1))
        if kwargs.get("advantage_after_reach", False):
            advantages = _zero_before_reach(
                advantages,
                rewards,
                dones,
                reach_min,
                success_chunk_reward_min=(
                    float(kwargs.get("success_chunk_reward_min", 0.5))
                    if scale_misses
                    else None
                ),
            )
        elif kwargs.get("zero_failure_prefix", False):
            advantages = _zero_unopened_prefix(
                advantages,
                rewards,
                dones,
                success_chunk_reward_min=float(
                    kwargs.get("success_chunk_reward_min", 0.5)
                ),
                reach_chunk_reward_min=reach_min,
            )
    if scale_misses:
        if dones is None:
            raise ValueError("failure advantage scaling requires dones.")
        advantages = _scale_unopened_advantages(
            advantages,
            rewards,
            dones,
            success_chunk_reward_min=float(
                kwargs.get("success_chunk_reward_min", 0.5)
            ),
            scale=float(scale),
        )
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("opd")
def compute_opd_advantages(
    prev_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    normalize_advantages: bool = False,
    **kwargs,
):
    """Compute OPD advantages from frozen teacher token log-probabilities."""
    assert teacher_logprobs is not None, (
        "OPD advantage computation requires post-rollout teacher_logprobs."
    )
    assert prev_logprobs is not None, (
        "OPD advantage computation requires prev_logprobs from student rollout."
    )
    assert teacher_logprobs.shape == prev_logprobs.shape, (
        f"teacher_logprobs shape {teacher_logprobs.shape} must match "
        f"prev_logprobs shape {prev_logprobs.shape}."
    )
    assert not normalize_advantages, (
        "VLA-OPD uses raw reverse-KL rewards; set normalize_advantages to False."
    )
    num_action_chunks = kwargs.get("num_action_chunks", None)
    assert num_action_chunks is not None, (
        "OPD advantage computation requires num_action_chunks."
    )
    advantages = teacher_logprobs.float() - prev_logprobs.float()
    assert advantages.shape[-1] % num_action_chunks == 0, (
        f"OPD token count {advantages.shape[-1]} must be divisible by "
        f"num_action_chunks {num_action_chunks}."
    )
    advantages = advantages.reshape(*advantages.shape[:-1], num_action_chunks, -1)
    if loss_mask is not None:
        target_steps = loss_mask.shape[0]
        assert advantages.shape[0] in {target_steps, target_steps + 1}, (
            f"OPD advantages time dimension {advantages.shape[0]} must match "
            f"loss_mask time dimension {target_steps} or include one bootstrap step."
        )
        advantages = advantages[:target_steps]

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None


@register_advantage("episode_success")
def compute_episode_success_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    success_return_threshold: float = 0.8,
    success_chunk_reward_min: float = 0.0,
    success_advantage: float = 0.5,
    approach_advantage: float = 0.0,
    approach_return_max: float = 0.2,
    speed_horizon_chunks: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Reinforce door motion on successes, and handle approach before contact.

    ``rewards`` is the per-episode return from ``calculate_scores``. When
    that return is at least ``success_return_threshold``, chunks whose own
    reward is strictly above ``success_chunk_reward_min`` get
    ``success_advantage``, scaled up when the episode is shorter than
    ``speed_horizon_chunks``. An episode below ``approach_return_max`` never
    received the door-reach bonus, so its positive chunks are the handle
    approach and get ``approach_advantage`` with no speed scale. Episodes in
    between are near misses and stay zero. There is no baseline and no
    negative.

    Args:
        rewards: Episode returns. One score per batch column, or
            ``[n_groups, group_size]``.
        loss_mask: Trainable steps, shape ``[n_steps, batch]``.
        success_return_threshold: Minimum return counted as a success.
        success_chunk_reward_min: Chunks at or below this reward are not
            reinforced. Zero keeps only positive reward chunks.
        success_advantage: Advantage written on a success's positive chunks.
        approach_advantage: Advantage written when the door was never reached.
        approach_return_max: Returns below this never earned the door bonus.
        speed_horizon_chunks: Chunk horizon used to prefer shorter successes.
            ``0`` leaves every success at ``success_advantage``.

    Returns:
        Advantages with the same shape as ``loss_mask``, and no returns.
    """
    step_rewards = kwargs.get("step_rewards")
    if loss_mask is None:
        raise ValueError("episode_success advantages require a loss_mask.")
    if step_rewards is None:
        raise ValueError("episode_success advantages require step_rewards.")
    episode_return = rewards.reshape(-1).to(dtype=torch.float32)
    if step_rewards.shape[-1] != episode_return.shape[0]:
        raise ValueError(
            "step_rewards batch "
            f"{step_rewards.shape[-1]} does not match returns {episode_return.shape[0]}"
        )
    live = loss_mask.to(dtype=torch.float32)
    positive = (step_rewards > float(success_chunk_reward_min)).to(dtype=live.dtype)
    success = episode_return >= float(success_return_threshold)
    approach = episode_return < float(approach_return_max)
    horizon = float(speed_horizon_chunks)
    if horizon > 0.0:
        length = live.sum(dim=0).clamp(min=1.0)
        speed_scale = ((horizon - length) / horizon).clamp(min=0.0, max=1.0)
    else:
        speed_scale = torch.ones(
            episode_return.shape[0], dtype=live.dtype, device=live.device
        )
    advantages = positive * live * (
        success.to(dtype=live.dtype).view(1, -1)
        * float(success_advantage)
        * speed_scale.view(1, -1)
        + approach.to(dtype=live.dtype).view(1, -1) * float(approach_advantage)
    )
    return advantages, None


def _attempt_index(ended: torch.Tensor) -> torch.Tensor:
    """Attempt id of each step. The done step still belongs to the attempt it ends."""
    seen = ended.long().cumsum(dim=0)
    return seen - ended.long()


def _return_to_go_by_attempt(
    step_rewards: torch.Tensor, ended: torch.Tensor
) -> torch.Tensor:
    """Remaining reward inside the current attempt, including this step."""
    remaining = torch.zeros_like(step_rewards)
    running = torch.zeros(
        step_rewards.shape[-1],
        dtype=step_rewards.dtype,
        device=step_rewards.device,
    )
    for step in reversed(range(step_rewards.shape[0])):
        if bool(ended[step].any()):
            running = torch.where(ended[step], torch.zeros_like(running), running)
        running = running + step_rewards[step]
        remaining[step] = running
    return remaining


def _keep_finished_after_success(
    rewards: torch.Tensor,
    ended: torch.Tensor,
    trainable: torch.Tensor,
    success_min: float,
) -> torch.Tensor:
    """Drop an attempt that follows a success and never finishes.

    A finish is this attempt's own open, or a done that is not that open
    (the timeout). The rollout ending the attempt early is neither, so
    those steps leave the loss and the mean. Their time penalty is already
    0. The open that came before stays. A miss that starts the horizon
    stays, including its penalty.
    """
    if rewards.numel() == 0:
        return trainable
    attempt = _attempt_index(ended)
    opened = rewards > float(success_min)
    n_attempts = int(attempt.max().item()) + 1
    success_rows = []
    closed_rows = []
    for attempt_id in range(n_attempts):
        in_attempt = attempt == attempt_id
        success_rows.append((opened & in_attempt).any(dim=0))
        closed_rows.append((ended & in_attempt).any(dim=0))
    success_mat = torch.stack(success_rows, dim=0)
    closed_mat = torch.stack(closed_rows, dim=0)
    prior_success = torch.zeros_like(success_mat)
    if n_attempts > 1:
        prior_success[1:] = success_mat.cumsum(dim=0)[:-1] > 0
    drop_attempt = prior_success & ~success_mat & ~closed_mat
    drop = torch.zeros_like(trainable)
    for attempt_id in range(n_attempts):
        drop = drop | (
            (attempt == attempt_id) & drop_attempt[attempt_id].view(1, -1)
        )
    return trainable & ~drop


def _center_selected(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Subtract the per-step mean of the selected batch entries."""
    selected_f = selected.to(dtype=values.dtype)
    count = selected_f.sum(dim=-1, keepdim=True)
    safe_count = count.clamp(min=1.0)
    mean = (values * selected_f).sum(dim=-1, keepdim=True) / safe_count
    return (values - mean) * selected_f


@register_advantage("grouped_return")
def compute_grouped_return_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    success_chunk_reward_min: float = 0.5,
    outcome_groups: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Center each attempt's return-to-go inside its own outcome group.

    An attempt runs until its done. A chunk reward above
    ``success_chunk_reward_min`` (the +1 open) marks that attempt as a
    success. Successes are centered only with other successes that are
    live at the same chunk, and misses only with other misses. The +1 is
    common to the success group, so the sign ranks the remaining dense
    cost: a faster, closer open stays positive. A miss that keeps the
    hand nearer the handle stays positive against a miss that moves away.
    The two groups are not centered together, so a miss is not made
    negative merely for lacking +1. A later attempt in the same horizon is
    its own episode. There is no critic return.

    Args:
        rewards: Episode scores from ``calculate_scores``. Unused; the
            per-chunk rewards arrive in ``step_rewards``.
        loss_mask: Optional trainable steps, shape ``[n_steps, batch]``.
        success_chunk_reward_min: Chunk reward that marks an open.
        outcome_groups: When False, center every live attempt together.

    Returns:
        Advantages with shape ``[n_steps, batch]``, and no returns.
    """
    del rewards
    step_rewards = kwargs.get("step_rewards")
    dones = kwargs.get("dones")
    if step_rewards is None:
        raise ValueError("grouped_return advantages require step_rewards.")
    if dones is None:
        raise ValueError("grouped_return advantages require dones.")
    rewards_f = step_rewards.to(dtype=torch.float32)
    steps, _batch = rewards_f.shape
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != rewards_f.shape:
        raise ValueError(
            "dones steps "
            f"{tuple(ended.shape)} do not match rewards {tuple(rewards_f.shape)}"
        )
    trainable = torch.ones_like(ended)
    if loss_mask is not None:
        trainable = loss_mask.bool()
        if trainable.shape != rewards_f.shape:
            raise ValueError(
                "loss_mask shape "
                f"{tuple(trainable.shape)} does not match rewards "
                f"{tuple(rewards_f.shape)}"
            )
    attempt = _attempt_index(ended)
    live_rewards = rewards_f * trainable.to(dtype=rewards_f.dtype)
    remaining = _return_to_go_by_attempt(live_rewards, ended)
    if not outcome_groups:
        return _center_selected(remaining, trainable), None
    opened_step = torch.zeros_like(ended)
    n_attempts = int(attempt.max().item()) + 1 if steps else 0
    open_mark = rewards_f > float(success_chunk_reward_min)
    for attempt_id in range(n_attempts):
        in_attempt = (attempt == attempt_id) & trainable
        opened = (open_mark & in_attempt).any(dim=0)
        opened_step = opened_step | (in_attempt & opened.view(1, -1))
    miss_step = trainable & ~opened_step
    advantages = _center_selected(remaining, opened_step) + _center_selected(
        remaining, miss_step
    )
    return advantages, None


@register_advantage("global_return")
def compute_global_return_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Center return-to-go across every live attempt.

    Opens and misses share one mean at each chunk. The sparse bonus stays
    in an open's return, so that open is positive against a miss. When the
    open rate falls, those higher returns are still above the batch and the
    policy can climb back. A batch of only misses ranks the higher return
    above the lower one. An attempt that starts after an open and then
    stops before its own open or timeout has its reward cleared to 0 and
    is left out of the mean and the loss. A miss from the start stays.
    There is no critic return.

    Args:
        rewards: Episode scores from ``calculate_scores``. Unused.
        loss_mask: Optional trainable steps, shape ``[n_steps, batch]``.

    Returns:
        Advantages with shape ``[n_steps, batch]``, and no returns.
    """
    success_min = float(kwargs.pop("success_chunk_reward_min", 0.5))
    step_rewards = kwargs.get("step_rewards")
    dones = kwargs.get("dones")
    if step_rewards is not None and dones is not None:
        rewards_f = step_rewards.to(dtype=torch.float32)
        ended = dones[1 : 1 + rewards_f.shape[0]].bool()
        if ended.shape == rewards_f.shape:
            if loss_mask is None:
                trainable = torch.ones_like(ended)
            else:
                trainable = loss_mask.bool()
            kept = _keep_finished_after_success(
                rewards_f,
                ended,
                trainable,
                success_min,
            )
            # The stored step reward of that tail is already 0. Zero the
            # copy used for return-to-go as well, then drop the steps.
            kwargs["step_rewards"] = torch.where(
                kept.to(device=step_rewards.device),
                step_rewards,
                torch.zeros_like(step_rewards),
            )
            loss_mask = kept
            step_loss_mask = kwargs.get("step_loss_mask")
            if (
                step_loss_mask is not None
                and step_loss_mask.shape[:2] == kept.shape
            ):
                extra = step_loss_mask.ndim - 2
                view = kept.view(*kept.shape, *([1] * extra))
                step_loss_mask &= view.expand_as(step_loss_mask)
    return compute_grouped_return_advantages(
        rewards,
        loss_mask=loss_mask,
        outcome_groups=False,
        success_chunk_reward_min=success_min,
        **kwargs,
    )


@register_advantage("success_first")
def compute_success_first_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    success_chunk_reward_min: float = 0.5,
    success_advantage: float = 0.5,
    miss_advantage_fraction: float = 0.5,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Reinforce every open, then prefer the shorter open.

    An attempt runs until its done. A chunk reward above
    ``success_chunk_reward_min`` marks that attempt as a success. Every
    success scores at least ``success_advantage``. Centering the attempt
    return among successes at the same attempt index keeps the time gap,
    so a shorter open scores higher, and the lowest open is lifted back to
    ``success_advantage``. CRI resets share a negative budget of
    ``miss_advantage_fraction`` times the success total, so the reward-0
    resets are pulled toward an open but cannot outweigh the opens. A
    batch with no open gives those resets zero. When ``terminals`` is set,
    a horizon timeout is not a reset: its advantage stays 0 and it does
    not take any of the miss budget. Either score is spread evenly over
    the attempt, so a longer attempt does not gain weight just by having
    more chunks. There is no critic return.

    Args:
        rewards: Episode scores from ``calculate_scores``. Unused; the
            per-chunk rewards arrive in ``step_rewards``.
        loss_mask: Optional trainable steps, shape ``[n_steps, batch]``.
        success_chunk_reward_min: Chunk reward that marks an open.
        success_advantage: Floor on a success attempt's total advantage.
        miss_advantage_fraction: Share of the success total assigned to
            CRI-reset attempts, split across those attempts. Clamped to
            ``[0, 1]``.

    Returns:
        Advantages with shape ``[n_steps, batch]``, and no returns.
    """
    del rewards
    step_rewards = kwargs.get("step_rewards")
    dones = kwargs.get("dones")
    if step_rewards is None:
        raise ValueError("success_first advantages require step_rewards.")
    if dones is None:
        raise ValueError("success_first advantages require dones.")
    rewards_f = step_rewards.to(dtype=torch.float32)
    steps, _batch = rewards_f.shape
    ended = dones[1 : 1 + steps].bool()
    if ended.shape != rewards_f.shape:
        raise ValueError(
            "dones steps "
            f"{tuple(ended.shape)} do not match rewards {tuple(rewards_f.shape)}"
        )
    # ``terminals`` marks a CRI reset or an open. A timeout is only in
    # ``dones``. Without ``terminals``, every non-open attempt stays a miss.
    terminals = kwargs.get("terminals")
    if terminals is None:
        cri_end = None
    else:
        cri_end = terminals[1 : 1 + steps].bool()
        if cri_end.shape != rewards_f.shape:
            raise ValueError(
                "terminals steps "
                f"{tuple(cri_end.shape)} do not match rewards "
                f"{tuple(rewards_f.shape)}"
            )
    trainable = torch.ones_like(ended)
    if loss_mask is not None:
        trainable = loss_mask.bool()
        if trainable.shape != rewards_f.shape:
            raise ValueError(
                "loss_mask shape "
                f"{tuple(trainable.shape)} does not match rewards "
                f"{tuple(rewards_f.shape)}"
            )
    attempt = _attempt_index(ended)
    live_rewards = rewards_f * trainable.to(dtype=rewards_f.dtype)
    open_mark = rewards_f > float(success_chunk_reward_min)
    advantages = torch.zeros_like(rewards_f)
    floor = rewards_f.new_tensor(float(success_advantage))
    fraction = min(max(float(miss_advantage_fraction), 0.0), 1.0)
    n_attempts = int(attempt.max().item()) + 1 if steps else 0
    records: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for attempt_id in range(n_attempts):
        in_span = attempt == attempt_id
        in_attempt = in_span & trainable
        present = in_attempt.any(dim=0)
        if not bool(present.any()):
            continue
        opened = (open_mark & in_attempt).any(dim=0)
        # The reset step itself can be masked out of the loss. It still
        # belongs to this attempt, so the CRI end is read on the full span.
        if cri_end is None:
            failed = present & ~opened
        else:
            failed = (
                present
                & ~opened
                & (cri_end & in_span).any(dim=0)
            )
        score = torch.zeros(
            rewards_f.shape[-1], dtype=rewards_f.dtype, device=rewards_f.device
        )
        if bool(opened.any()):
            selected = opened.to(dtype=rewards_f.dtype)
            attempt_return = (
                live_rewards * in_attempt.to(dtype=rewards_f.dtype)
            ).sum(dim=0)
            count = selected.sum().clamp(min=1.0)
            mean = (attempt_return * selected).sum() / count
            centered = (attempt_return - mean) * selected
            lifted = centered + (floor - centered[opened].min())
            score = torch.where(opened, lifted, score)
        records.append((in_attempt, present, opened, failed, score))
    total_pos = rewards_f.new_zeros(())
    n_miss = rewards_f.new_zeros(())
    for _in_attempt, _present, _opened, failed, score in records:
        total_pos = total_pos + score.sum()
        n_miss = n_miss + failed.to(dtype=rewards_f.dtype).sum()
    miss_each = rewards_f.new_zeros(())
    if float(total_pos) > 0.0 and float(n_miss) > 0.0:
        miss_each = -total_pos * fraction / n_miss
    for in_attempt, _present, _opened, failed, score in records:
        score = torch.where(failed, miss_each, score)
        length = in_attempt.to(dtype=rewards_f.dtype).sum(dim=0).clamp(min=1.0)
        advantages = advantages + in_attempt.to(dtype=rewards_f.dtype) * (
            score / length
        ).view(1, -1)
    return advantages, None


@register_advantage("min_time")
def compute_min_time_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Center the undiscounted time return on every live chunk.

    ``rewards`` is the undiscounted return up to the done. A success
    termination therefore ends the time penalty, and a timeout keeps paying
    until the horizon. Subtracting the batch mean keeps that difference:
    a shorter penalty stays positive. When ``advantage_horizon_steps`` is
    positive, that centered return is divided by this span and by the number
    of live chunks, so each
    episode has the same total weight. Set the span to the timeout return
    (penalty per step times the horizon) so a one-step gap stays ``1/horizon``.
    Masked hold steps stay zero. A span of 0 leaves the centered return
    unchanged.

    Args:
        rewards: Episode returns. One score per batch column, or
            ``[n_groups, group_size]``.
        loss_mask: Trainable steps, shape ``[n_steps, batch]``.

    Returns:
        Advantages with the same shape as ``loss_mask``, and no returns.
    """
    if loss_mask is None:
        raise ValueError("min_time advantages require a loss_mask.")
    episode_return = rewards.reshape(-1).to(dtype=torch.float32)
    live = loss_mask.to(dtype=torch.float32)
    if live.shape[-1] != episode_return.shape[0]:
        raise ValueError(
            "loss_mask batch "
            f"{live.shape[-1]} does not match returns {episode_return.shape[0]}"
        )
    centered = episode_return - episode_return.mean()
    horizon = float(kwargs.get("advantage_horizon_steps", 0.0) or 0.0)
    if horizon > 0.0:
        length = live.sum(dim=0).clamp(min=1.0)
        centered = centered / horizon / length
    advantages = live * centered.view(1, -1)
    return advantages, None
