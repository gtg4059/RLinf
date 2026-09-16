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


def cri_ovf_exp(
    cri: torch.Tensor,
    *,
    limit: float = CRI_FILTER_LIMIT,
    sigma: float = CRI_OVF_SIGMA,
    ovf_threshold: float = CRI_OVF_THRESHOLD,
) -> torch.Tensor:
    """Soft CRI overflow penalty. Matches IsaacLab ``mdp.CRI_OVF_exp``.

    ``cri`` is ``(B, K)`` or ``(B,)``. Returns a non-negative ``(B,)`` term:
    0 at CRI=0, ~1 at ``limit``, then linear up to 1 at ``ovf_threshold``.
    Apply a negative reward weight.
    """
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = values.new_tensor(float(limit))
    headroom = (limit_t - values).clamp(min=0.0)
    exp_pen = torch.exp(-float(sigma) * headroom) - torch.exp(
        values.new_tensor(-float(sigma) * float(limit))
    )
    excess = (values - limit_t).clamp(min=0.0)
    span = max(float(ovf_threshold) - float(limit), 1e-6)
    lin_pen = (excess / span).clamp(max=1.0)
    return exp_pen + lin_pen


def cri_ovf_violated(
    cri: torch.Tensor,
    *,
    threshold: float = CRI_FILTER_LIMIT,
) -> torch.Tensor:
    """Return ``(B,)`` bool mask: raw ``cri_max >= threshold`` (Isaac ``CRI_OVF`` term)."""
    values = torch.as_tensor(cri, dtype=torch.float32)
    if values.ndim == 0:
        values = values.unsqueeze(0)
    if values.ndim > 1:
        values = values.amax(dim=-1)
    limit_t = values.new_tensor(float(threshold))
    return values >= limit_t


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
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Rows that still pay the living cost this step.

    Charge until the first RLinf-cycle success. The success step still
    pays (``success_once`` is updated after the reward). Leftover Isaac
    episodes after that success do not. Frozen hold-after-done rows
    also do not pay.
    """
    if device is None:
        if success_once is not None:
            device = success_once.device
        elif was_frozen is not None:
            device = was_frozen.device
        else:
            device = "cpu"
    mask = torch.ones(int(num_envs), dtype=torch.bool, device=device)
    if hold_after_done and was_frozen is not None:
        mask = mask & ~was_frozen.to(device=mask.device)
    if success_once is not None:
        mask = mask & ~success_once.to(device=mask.device)
    return mask


# Isaac Lab extras key: ``log["Episode_Reward/<term>"]`` is the episode sum
# of ``weight * term``. TensorBoard shows it under ``env/Episode_Reward/``.
CRI_OVF_REWARD_KEY: str = "Episode_Reward/cri_ovf"
TIME_PENALTY_REWARD_KEY: str = "Episode_Reward/time_penalty"


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
