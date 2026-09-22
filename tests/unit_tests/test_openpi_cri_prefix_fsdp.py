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

"""CRI-prefix tensors must not mix float32 into the FSDP root FlatParameter."""

import pytest
import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi import cast_cri_prefix_params_to_bfloat16


class _CriPrefixStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.cri_in_proj = nn.Linear(9, 8)
        self.cri_out_proj = nn.Linear(8, 16)
        self.cri_pos = nn.Parameter(torch.zeros(9, 4))
        self.other = nn.Linear(4, 4)


def test_cast_cri_prefix_params_to_bfloat16_leaves_other_float32():
    model = _CriPrefixStub()
    cast_cri_prefix_params_to_bfloat16(model)
    assert model.cri_pos.dtype == torch.bfloat16
    # OpenPI embed_prefix force-casts CRI to float32; linears must stay fp32.
    assert model.cri_in_proj.weight.dtype == torch.float32
    assert model.cri_out_proj.weight.dtype == torch.float32
    assert model.other.weight.dtype == torch.float32


def test_cast_cri_to_proj_dtype_matches_weight():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0ForRLActionPrediction,
    )

    class _Host:
        cri_in_proj = nn.Linear(9, 4)

    _Host.cri_in_proj.weight.data = _Host.cri_in_proj.weight.data.to(torch.bfloat16)
    _Host.cri_in_proj.bias.data = _Host.cri_in_proj.bias.data.to(torch.bfloat16)
    cri = torch.ones(2, 9)
    out = OpenPi0ForRLActionPrediction._cast_cri_to_proj_dtype(_Host(), cri)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 9)


def test_cri_prefix_linears_are_fsdp_no_split_names():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0ForRLActionPrediction,
    )

    names = OpenPi0ForRLActionPrediction._no_split_names.fget(_CriPrefixStub())
    assert "cri_in_proj" in names
    assert "cri_out_proj" in names
    assert "cri_pos" in names
    assert "action_in_proj" in names
