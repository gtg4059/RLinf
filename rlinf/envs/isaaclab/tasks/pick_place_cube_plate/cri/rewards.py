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
from .constants import CRI_OVF_THRESHOLD


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
