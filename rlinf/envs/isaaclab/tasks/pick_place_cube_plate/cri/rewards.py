# Copyright 2026 The RLinf Authors.
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

"""IsaacLab ``CRI_OVF_exp`` barrier used as a dense cube-plate reward term."""

from __future__ import annotations

import torch

from .constants import CRI_FILTER_LIMIT
from .constants import CRI_OVF_SIGMA
from .constants import CRI_OVF_TERM_PENALTY
from .constants import CRI_OVF_THRESHOLD
from .constants import TIME_PENALTY_WEIGHT


def resolve_cri_exceed_mode(mode: str | None) -> str | None:
    """Return ``soft``, ``hard``, ``reset``, or ``None`` when unset.

    ``soft`` slows the arm near the CRI limit and adds the barrier penalty.
    ``hard`` rewinds to the last scene-reset pose and drops that penalty.
    ``reset`` ends the attempt, restores the scene, and leaves the reward at 0.
    """
    if mode is None:
        return None
    name = str(mode).strip().lower()
    if name not in ("soft", "hard", "reset"):
        raise ValueError(
            f"cri_exceed_mode must be 'soft', 'hard', or 'reset', got {mode!r}"
        )
    return name


def cri_exceed_settings(
    mode: str | None,
    *,
    boundary_rewind: bool,
    termination: bool,
    penalty_weight: float,
) -> tuple[str | None, bool, bool, float]:
    """Resolve the exceedance trigger into rewind, termination, and penalty.

    An unset trigger leaves the three flags as configured. ``soft`` turns
    rewind off and keeps the penalty. Termination follows the flag, so a
    failure can reset. ``hard`` rewinds, turns termination off, and drops
    the dense penalty. ``reset`` turns rewind off, ends the attempt, and
    drops the dense penalty so the trajectory reward stays 0.
    """
    resolved = resolve_cri_exceed_mode(mode)
    if resolved == "soft":
        return resolved, False, bool(termination), float(penalty_weight)
    if resolved == "hard":
        return resolved, True, False, 0.0
    if resolved == "reset":
        return resolved, False, True, 0.0
    return None, bool(boundary_rewind), bool(termination), float(penalty_weight)


def cri_ovf_exp(
    cri: torch.Tensor,
    *,
    limit: float | torch.Tensor = CRI_FILTER_LIMIT,
    sigma: float = CRI_OVF_SIGMA,
    ovf_threshold: float = CRI_OVF_THRESHOLD,
) -> torch.Tensor:
    """Soft CRI overflow penalty. Matches IsaacLab ``mdp.CRI_OVF_exp``.

    ``cri`` is ``(B, K)`` or ``(B,)``. ``limit`` is a scalar or one value per
    row. Returns a non-negative ``(B,)`` term: 0 at CRI=0, ~1 at ``limit``,
    then linear up to 1 at ``ovf_threshold``. Apply a negative reward weight.
    A row limit above ``ovf_threshold`` keeps the same linear span as the
    scalar 0.96 limit.
    """
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = torch.as_tensor(limit, dtype=values.dtype, device=values.device)
    if limit_t.ndim == 0:
        limit_t = limit_t.expand_as(values)
    else:
        limit_t = limit_t.reshape(values.shape)
    headroom = (limit_t - values).clamp(min=0.0)
    exp_pen = torch.exp(-float(sigma) * headroom) - torch.exp(
        -float(sigma) * limit_t
    )
    excess = (values - limit_t).clamp(min=0.0)
    default_span = max(float(ovf_threshold) - float(CRI_FILTER_LIMIT), 1e-6)
    span = (float(ovf_threshold) - limit_t).clamp(min=1e-6)
    span = torch.where(
        limit_t >= float(ovf_threshold),
        limit_t.new_full((), default_span).expand_as(span),
        span,
    )
    lin_pen = (excess / span).clamp(max=1.0)
    return exp_pen + lin_pen


def cri_exceed_linear(
    cri: torch.Tensor,
    *,
    limit: float | torch.Tensor = CRI_FILTER_LIMIT,
) -> torch.Tensor:
    """Linear overflow term ``y = 1 + x`` once CRI is on or past the limit.

    ``x`` is ``cri - limit``. Below the limit the term is 0. On the limit it
    is 1, and each further unit of CRI adds 1. ``cri`` is ``(B, K)`` or
    ``(B,)``. ``limit`` is a scalar or one value per row.
    """
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = torch.as_tensor(limit, dtype=values.dtype, device=values.device)
    if limit_t.ndim == 0:
        limit_t = limit_t.expand_as(values)
    else:
        limit_t = limit_t.reshape(values.shape)
    excess = (values - limit_t).clamp(min=0.0)
    return torch.where(values >= limit_t, excess + 1.0, torch.zeros_like(excess))


def cri_soft_approach_scale(
    cri: torch.Tensor,
    *,
    limit: float | torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Arm-command scale in ``[0, 1]``.

    The scale is 1 far below ``limit`` and falls to 0 as CRI reaches it, on
    the same exponential as ``cri_ovf_exp``. At and above the limit it is 1
    again, so the arm can move off the barrier. The penalty covers that side.
    """
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = torch.as_tensor(limit, dtype=values.dtype, device=values.device)
    if limit_t.ndim == 0:
        limit_t = limit_t.expand_as(values)
    else:
        limit_t = limit_t.reshape(values.shape)
    headroom = (limit_t - values).clamp(min=0.0)
    exp_pen = torch.exp(-float(sigma) * headroom) - torch.exp(
        -float(sigma) * limit_t
    )
    approaching = (1.0 - exp_pen).clamp(0.0, 1.0)
    return torch.where(values >= limit_t, torch.ones_like(approaching), approaching)


def apply_cri_soft_arm_limit(
    actions: torch.Tensor,
    current_q: torch.Tensor,
    cri: torch.Tensor,
    *,
    limit: float | torch.Tensor,
    sigma: float,
    arm_dims: int = 7,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pull absolute arm targets toward the current joints as CRI nears the limit.

    The gripper channel stays put. Rows with ``row_mask`` false are unchanged,
    so a chunk hold is not undone. Non-tensor actions are returned as given.
    """
    if not torch.is_tensor(actions) or actions.shape[-1] < arm_dims:
        return actions
    scale = cri_soft_approach_scale(cri, limit=limit, sigma=sigma)
    scale = scale.to(device=actions.device, dtype=actions.dtype).reshape(-1)
    joints = current_q.to(device=actions.device, dtype=actions.dtype)
    if joints.shape[-1] < arm_dims or joints.shape[0] != actions.shape[0]:
        return actions
    cmd = actions.clone()
    arm = cmd[..., :arm_dims]
    pose = joints[..., :arm_dims]
    width = scale.reshape(-1, *([1] * (arm.ndim - 1)))
    blended = pose + width * (arm - pose)
    if row_mask is not None:
        keep = row_mask.to(device=cmd.device).reshape(-1).bool()
        keep = keep.reshape(-1, *([1] * (arm.ndim - 1)))
        blended = torch.where(keep, blended, arm)
    cmd[..., :arm_dims] = blended
    return cmd


def cri_ovf_violated(
    cri: torch.Tensor,
    *,
    threshold: float | torch.Tensor = CRI_FILTER_LIMIT,
) -> torch.Tensor:
    """Return ``(B,)`` bool mask: raw ``cri_max >= threshold`` (Isaac ``CRI_OVF`` term).

    ``threshold`` may be a scalar or a per-row tensor of shape ``(B,)``.
    """
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    if limit_t.ndim == 0:
        return values >= limit_t
    return values >= limit_t.reshape(values.shape)


def apply_cri_chunk_penalty(
    rewards: torch.Tensor,
    exceeded: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Add ``penalty`` once to each chunk that crossed the CRI limit.

    ``rewards`` is ``[num_envs, chunk_steps]``. The charge is written on the
    last step, so a chunk-level sum counts it one time even when several
    steps in the chunk are over the limit.
    """
    charge = float(penalty)
    if charge == 0.0:
        return rewards
    hit = exceeded.reshape(-1).bool()
    if hit.shape[0] != rewards.shape[0] or not bool(hit.any()):
        return rewards
    out = rewards.clone()
    out[hit, -1] = out[hit, -1] + charge
    return out


def cri_ovf_termination_step_penalty() -> float:
    """One-shot hard termination penalty: success step reward (dt * weight 1.0) / 3."""
    return float(CRI_OVF_TERM_PENALTY)


def time_step_penalty(weight: float | None = None) -> float:
    """Per-step living cost. Default is ``-ISAACLAB_STEP_DT / 100``."""
    if weight is None:
        return float(TIME_PENALTY_WEIGHT)
    return float(weight)


def time_penalty_live_mask(
    num_envs: int,
    *,
    hold_after_done: bool = False,
    was_frozen: torch.Tensor | None = None,
    success_once: torch.Tensor | None = None,
    until_horizon: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Rows that still pay the living cost this step.

    Charge until this attempt's success. The success step still pays
    (``success_once`` is updated after the reward). A later attempt pays
    again once that latch is cleared. Frozen hold-after-done rows do not
    pay. ``until_horizon`` keeps the charge on every row through the
    timeout, including after the door is open.
    """
    if device is None:
        if success_once is not None:
            device = success_once.device
        elif was_frozen is not None:
            device = was_frozen.device
        else:
            device = "cpu"
    mask = torch.ones(int(num_envs), dtype=torch.bool, device=device)
    if until_horizon:
        return mask
    if hold_after_done and was_frozen is not None:
        mask = mask & ~was_frozen.to(device=mask.device)
    if success_once is not None:
        mask = mask & ~success_once.to(device=mask.device)
    return mask


# Isaac Lab extras key: ``log["Episode_Reward/<term>"]`` is the episode sum
# of ``weight * term``. TensorBoard shows it under ``env/Episode_Reward/``.
CRI_OVF_REWARD_KEY: str = "Episode_Reward/cri_ovf"
CRI_REWIND_REWARD_KEY: str = "Episode_Reward/cri_rewind"
TIME_PENALTY_REWARD_KEY: str = "Episode_Reward/time_penalty"
ACTION_RATE_REWARD_KEY: str = "Episode_Reward/action_rate"


def cri_rewind_penalty_reward(
    rewind: torch.Tensor, penalty: float
) -> torch.Tensor:
    """One charge of ``penalty`` on each row that rewound this step.

    ``penalty`` is the step value, negative when a rewind should cost.
    Rows that did not rewind pay 0.

    Args:
        rewind: ``(N,)`` bool, true on rows rewound this step.
        penalty: Reward added once for each rewind.

    Returns:
        Step reward, shape ``(N,)``.
    """
    hit = rewind.reshape(-1).to(dtype=torch.float32)
    return float(penalty) * hit


def cri_ovf_reward(ovf: torch.Tensor, weight: float) -> torch.Tensor:
    """Weighted CRI term added to the step reward (negative ``weight``)."""
    return float(weight) * torch.as_tensor(ovf, dtype=torch.float32)


def accumulate_cri_episode_reward(
    ovf_sum: torch.Tensor,
    penalty_sum: torch.Tensor,
    ovf: torch.Tensor,
    weight: float,
) -> None:
    """Add this step's barrier and weighted reward into Isaac-style episode sums."""
    term = torch.as_tensor(ovf, dtype=ovf_sum.dtype, device=ovf_sum.device).reshape(-1)
    n = int(ovf_sum.shape[0])
    ovf_sum += term[:n]
    penalty_sum += float(weight) * term[:n]


def cri_episode_reward_logs(penalty_sum: torch.Tensor) -> dict[str, torch.Tensor]:
    """Episode sum of CRI reward-function penalty (soft + hard)."""
    return {CRI_OVF_REWARD_KEY: penalty_sum.clone()}


def accumulate_time_episode_reward(
    penalty_sum: torch.Tensor,
    weight: float,
    live_mask: torch.Tensor | None = None,
) -> None:
    """Add this step's living cost into the episode sum (live rows only)."""
    step = penalty_sum.new_full(penalty_sum.shape, float(weight))
    if live_mask is not None:
        step = step * live_mask.to(device=step.device, dtype=step.dtype).reshape_as(step)
    penalty_sum += step


def time_episode_reward_logs(penalty_sum: torch.Tensor) -> dict[str, torch.Tensor]:
    """Isaac Lab-style episode time-penalty curve."""
    return {TIME_PENALTY_REWARD_KEY: penalty_sum.clone()}


def action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Squared L2 change of the command, matching IsaacLab ``action_rate_l2``.

    Args:
        action: Command this step, shape ``(N, A)``.
        prev_action: Command last step, same shape.

    Returns:
        Non-negative rate, shape ``(N,)``.
    """
    delta = (action - prev_action).reshape(action.shape[0], -1)
    return torch.sum(torch.square(delta), dim=-1)
