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

"""Hold last actions/obs for done envs when auto_reset is off."""

from __future__ import annotations

from typing import Any

import torch


def clone_obs(obs: Any) -> Any:
    """Deep-clone a tensor or nested observation mapping."""
    if isinstance(obs, dict):
        return {key: clone_obs(value) for key, value in obs.items()}
    if torch.is_tensor(obs):
        return obs.clone()
    return obs


def _row_mask(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """Boolean row mask on ``tensor``'s device (policy actions may be CPU)."""
    if mask.device != tensor.device:
        return mask.to(device=tensor.device)
    return mask


def _match_rows(src: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    """Align ``src`` to ``dest`` for index assignment (Isaac vs policy dtypes)."""
    return src.to(device=dest.device, dtype=dest.dtype)


def apply_hold_actions(
    actions: torch.Tensor | None,
    hold_actions: torch.Tensor | None,
    frozen: torch.Tensor,
) -> torch.Tensor | None:
    """Replace frozen rows with the last commanded action (abs-joint hold)."""
    if actions is None or hold_actions is None or not bool(frozen.any()):
        return actions
    mask = _row_mask(frozen, actions)
    held = actions.clone()
    held[mask] = _match_rows(hold_actions, held)[mask]
    return held


def overwrite_frozen_obs(obs: Any, hold_obs: Any, frozen: torch.Tensor) -> Any:
    """Restore the last observation for frozen env rows."""
    if hold_obs is None or not bool(frozen.any()):
        return obs
    if isinstance(obs, dict):
        return {
            key: overwrite_frozen_obs(obs[key], hold_obs[key], frozen)
            for key in obs
        }
    if torch.is_tensor(obs) and obs.shape[:1] == frozen.shape:
        mask = _row_mask(frozen, obs)
        out = obs.clone()
        out[mask] = _match_rows(hold_obs, out)[mask]
        return out
    return obs


def update_hold_actions(
    hold_actions: torch.Tensor | None,
    actions: torch.Tensor,
    frozen: torch.Tensor,
) -> torch.Tensor:
    """Keep prior holds on frozen rows; refresh the rest."""
    if hold_actions is None or hold_actions.shape != actions.shape:
        return actions.clone()
    mask = _row_mask(frozen, actions)
    updated = _match_rows(hold_actions, actions).clone()
    live = ~mask
    if bool(live.any()):
        updated[live] = _match_rows(actions, updated)[live]
    return updated
