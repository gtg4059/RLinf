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

"""CRI prefix encoder matches the 19999 adapter checkpoint layout."""

from __future__ import annotations

from pathlib import Path

import torch

from rlinf.models.embodiment.openpi.cri_prefix import (
    CRIPositionalEmbedding,
    encode_cri_prefix_tokens,
    normalize_cri_prefix_state_dict,
    vlm_value_prefix_mask,
    wrap_bare_cri_pos_parameter,
)

ADAPTER_CKPT = (
    Path(__file__).resolve().parents[2]
    / "checkpoint"
    / "pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999"
)


def test_normalize_cri_prefix_state_dict_remaps_bare_pos() -> None:
    remapped = normalize_cri_prefix_state_dict({"cri_pos": torch.zeros(9, 2048)})
    assert "cri_pos" not in remapped
    assert remapped["cri_pos.weight"].shape == (9, 2048)


def test_wrap_bare_cri_pos_parameter_replaces_root_param() -> None:
    host = torch.nn.Module()
    host.cri_pos = torch.nn.Parameter(torch.ones(9, 2048))
    wrap_bare_cri_pos_parameter(host)
    assert isinstance(host.cri_pos, CRIPositionalEmbedding)
    assert tuple(host.cri_pos.weight.shape) == (9, 2048)
    assert host.cri_pos._fsdp_wrap_name == "cri_pos"
    assert "cri_pos.weight" in host.state_dict()
    assert "cri_pos" not in host.state_dict()


def test_cri_pos_module_is_uniform_dtype() -> None:
    module = CRIPositionalEmbedding(9, 2048)
    dtypes = {param.dtype for param in module.parameters()}
    assert dtypes == {torch.float32}


def test_encode_cri_prefix_tokens_clamps_range() -> None:
    cri_in = torch.nn.Linear(9, 256)
    cri_out = torch.nn.Linear(256, 9 * 2048)
    cri_pos = torch.zeros(9, 2048)
    low = encode_cri_prefix_tokens(torch.full((1, 9), -5.0), cri_in, cri_out, cri_pos)
    zero = encode_cri_prefix_tokens(torch.zeros(1, 9), cri_in, cri_out, cri_pos)
    assert torch.allclose(low, zero)


def test_encode_cri_prefix_tokens_shape() -> None:
    cri_in = torch.nn.Linear(9, 256)
    cri_out = torch.nn.Linear(256, 9 * 2048)
    cri_pos = torch.zeros(9, 2048)
    tokens = encode_cri_prefix_tokens(torch.zeros(4, 9), cri_in, cri_out, cri_pos)
    assert tokens.shape == (4, 9, 2048)


def test_vlm_mask_includes_adapter_cri_tokens() -> None:
    seq_len = 256 * 3 + 200 + 9
    mask = vlm_value_prefix_mask(seq_len, num_images_in_input=2, value_vlm_mode="mean_token")
    assert len(mask) == 977
    assert sum(mask) == 256 * 2 + 200 + 9


def test_adapter_checkpoint_cri_keys_encode() -> None:
    weights = ADAPTER_CKPT / "model.safetensors"
    if not weights.is_file():
        import pytest

        pytest.skip("PolarIS CRI-prefix adapter checkpoint is not present")

    from safetensors.torch import load_file

    state = load_file(str(weights), device="cpu")
    cri_in = torch.nn.Linear(9, 256)
    cri_out = torch.nn.Linear(256, 9 * 2048)
    cri_in.weight.data.copy_(state["cri_in_proj.weight"])
    cri_in.bias.data.copy_(state["cri_in_proj.bias"])
    cri_out.weight.data.copy_(state["cri_out_proj.weight"])
    cri_out.bias.data.copy_(state["cri_out_proj.bias"])
    cri_pos = state["cri_pos"].clone()
    tokens = encode_cri_prefix_tokens(torch.zeros(2, 9), cri_in, cri_out, cri_pos)
    assert tokens.shape == (2, 9, 2048)
    assert torch.isfinite(tokens).all()
