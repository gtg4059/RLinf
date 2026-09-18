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

"""Pure fridge-door openness math (no Isaac Lab import)."""

from __future__ import annotations

import torch


def compute_door_openness(
    joint_pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Return hinge travel as a fraction of ``[lower, upper]``.

    Args:
        joint_pos: Door joint position, shape ``(N,)``.
        lower: Soft lower limit, shape ``(N,)``.
        upper: Soft upper limit, shape ``(N,)``.

    Returns:
        Openness in ``[0, 1]`` (clamped), shape ``(N,)``.
    """
    span = (upper - lower).clamp(min=1e-6)
    return ((joint_pos - lower) / span).clamp(0.0, 1.0)
