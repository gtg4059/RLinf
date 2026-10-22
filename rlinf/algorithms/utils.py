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


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def reference_kl_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor | None,
    kl_penalty_type: str,
    action_dim: int,
) -> torch.Tensor:
    """Mean KL of the current policy against a frozen reference.

    Uses the same token layout as ``preprocess_loss_inputs`` for
    ``logprob_type: token_level``. ``kl`` / ``k1`` is ``log π - log π_ref``.

    Args:
        logprobs: Current-policy log probabilities.
        ref_logprobs: Reference-policy log probabilities, same shape.
        loss_mask: Steps that still contribute to the loss. ``None`` keeps all.
        kl_penalty_type: Estimator name accepted by ``kl_penalty``.
        action_dim: Last-dimension width before the token reshape.

    Returns:
        Scalar mean divergence on the masked tokens.
    """
    batch_size = logprobs.shape[0]
    current = logprobs.reshape(batch_size, -1, action_dim)
    reference = ref_logprobs.reshape(batch_size, -1, action_dim)
    divergence = kl_penalty(current, reference, kl_penalty_type)
    if loss_mask is None:
        return divergence.mean()
    weights = loss_mask.unsqueeze(-1).expand_as(divergence).to(divergence.dtype)
    denominator = weights.sum()
    if denominator.item() == 0.0:
        return divergence.sum() * 0.0
    return (divergence * weights).sum() / denominator


def keep_through_first_done(step_dones: torch.Tensor) -> torch.Tensor:
    """Keep steps up to and including the first done along the last dimension.

    A success inside an action chunk ends that episode. Later steps in the
    same chunk are hold padding and must not enter the return or the loss.

    Args:
        step_dones: Bool tensor, done flags for each step in a chunk.

    Returns:
        Bool tensor of the same shape. True on steps that still belong to
        the episode that the chunk started.
    """
    done = step_dones.bool()
    seen = done.cumsum(dim=-1)
    return (seen == 0) | (done & (seen == 1))


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    if kwargs["reward_type"] == "chunk_level":
        # TODO: need check
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        # Dones are stored with one leading bootstrap row. Drop rewards after
        # the first done in the chunk so the reset hold is not part of the
        # episode that just succeeded, and GAE does not pull in the next chunk.
        step_dones = dones[1:] if dones.shape[0] == rewards.shape[0] + 1 else dones
        if step_dones.shape == rewards.shape and rewards.shape[-1] > 1:
            keep = keep_through_first_done(step_dones)
            rewards = rewards * keep.to(dtype=rewards.dtype)
            kwargs["step_loss_mask"] = keep
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    # Per-chunk rewards, kept so episode_success can ignore the approach.
    # The sum stops at the done, so a success termination ends the time penalty.
    step_rewards = rewards
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
        }
    )
    if kwargs.get("adv_type") in ("episode_success", "grouped_return"):
        kwargs["step_rewards"] = step_rewards

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "grpo_dynamic":
        grouped_rewards = (
            rewards.reshape(-1, kwargs["num_sequence"]).transpose(0, 1).contiguous()
        )
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    elif kwargs["adv_type"] == "raw":
        kwargs.update({"rewards": rewards})

    else:
        assert False, f"Unsupported adv_type {kwargs['adv_type']}"

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    # remember to call contiguous() to ensure correctness when being
    # transmitted through channels
    advantages = advantages.transpose(0, 1).contiguous()  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1).contiguous()  # [bsz, seq_len]

    return advantages, returns


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    versions: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    if reward_type == "chunk_level":
        # A mask shaped like the action chunk (one flag per step) must stay
        # that shape so steps after a mid-chunk success are dropped. A mask
        # with one flag per chunk flattens with the advantage.
        advantage_count = advantages.numel()
        advantages = advantages.flatten()
        if loss_mask is not None and loss_mask.numel() == advantage_count:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None and loss_mask_sum.numel() == advantage_count:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    proximal_logprobs = kwargs.get("proximal_logprobs", None)
    if logprob_type == "token_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)
        if kwargs.get("loss_type") == "opd":
            assert advantages.shape == logprobs.shape, (
                f"OPD advantages shape {advantages.shape} must match "
                f"logprobs shape {logprobs.shape}."
            )
        else:
            advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=-1)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[..., 0]

    elif logprob_type == "chunk_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=[1, 2])
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[:, 0, 0]

    target_shape = logprobs.shape
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)
    versions = expand_to_target_dim(versions, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "proximal_logprobs": proximal_logprobs,
            "versions": versions,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def resolve_update_epoch(
    update_epoch: int,
    critic_warmup_update_epoch: int,
    optimizer_steps: int,
    critic_warmup_steps: int,
) -> int:
    """How many passes to take over the rollout just collected.

    While the actor is frozen, the value head can train for several passes on
    that same rollout. After warmup this returns ``update_epoch`` so the
    policy is not updated more than once per rollout.

    Args:
        update_epoch: Passes used once the actor is training.
        critic_warmup_update_epoch: Passes used on a warmup rollout.
        optimizer_steps: Value/actor updates finished so far.
        critic_warmup_steps: Updates that must finish before the actor unfreezes.

    Returns:
        The epoch count for this rollout.
    """
    update_epoch = int(update_epoch)
    if int(critic_warmup_steps) > 0 and int(optimizer_steps) < int(critic_warmup_steps):
        return max(update_epoch, int(critic_warmup_update_epoch))
    return update_epoch


def expand_to_target_dim(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
