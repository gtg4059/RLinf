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

"""Helpers for Isaac Lab in-step reset while an RLinf cycle continues."""

from __future__ import annotations

import torch


def isaac_reset_row_ids(
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> torch.Tensor:
    """Row ids that ``ManagerBasedRLEnv.step`` resets before returning.

    Isaac Lab resets terminated/truncated envs inside ``step``. RLinf can keep
    its 450-step cycle (metrics, elapsed) and treat those rows as a new Isaac
    episode.
    """
    done = terminations.reshape(-1).bool() | truncations.reshape(-1).bool()
    return torch.nonzero(done, as_tuple=False).reshape(-1)


def zero_rows(tensor: torch.Tensor, row_ids: torch.Tensor) -> None:
    """In-place zero selected leading-dimension rows."""
    if tensor is None or row_ids is None or int(row_ids.numel()) == 0:
        return
    ids = row_ids.to(device=tensor.device).reshape(-1)
    tensor.index_fill_(0, ids, 0)


def set_chunk_hold(
    hold_mask: torch.Tensor,
    hold_actions: torch.Tensor | None,
    states: torch.Tensor,
    row_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold abs-joint at post-reset ``states`` until the next policy chunk.

    Leftover chunk actions were planned for the pre-reset pose. Applying them
    after Isaac teleports the arm slams the desk (stiffness 400).
    """
    ids = torch.as_tensor(row_ids, device=hold_mask.device).reshape(-1)
    hold_mask = hold_mask.clone()
    hold_mask.index_fill_(0, ids, True)
    src = states
    row_ids_on_src = ids.to(device=src.device)
    if hold_actions is None or hold_actions.shape != src.shape:
        held = src.clone()
    else:
        held = hold_actions.to(device=src.device, dtype=src.dtype).clone()
        held[row_ids_on_src] = src[row_ids_on_src]
    return hold_mask, held
