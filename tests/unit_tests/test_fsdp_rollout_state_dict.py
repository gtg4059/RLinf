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

"""Helpers that rebuild unwrapped FSDP1 tensors for patch weight sync."""

import pytest
import torch

from rlinf.hybrid_engines.fsdp.state_dict_utils import (
    all_gather_local_shard_to_shape,
    clean_fsdp_state_dict_key,
    collect_cleaned_named_tensors,
    fill_missing_named_tensors,
    lookup_module_tensor,
    reconstruct_gathered_flat,
    unwrap_fsdp_module,
)


def test_clean_fsdp_state_dict_key_strips_wrapper_segments():
    assert (
        clean_fsdp_state_dict_key(
            "_fsdp_wrapped_module.paligemma._checkpoint_wrapped_module."
            "model.language_model.embed_tokens.weight"
        )
        == "paligemma.model.language_model.embed_tokens.weight"
    )


def test_all_gather_reshapes_when_local_numel_already_matches():
    tensor = torch.arange(6, dtype=torch.float32)
    out = all_gather_local_shard_to_shape(tensor, (2, 3))
    assert tuple(out.shape) == (2, 3)
    assert torch.equal(out, tensor.reshape(2, 3))


def test_reconstruct_gathered_flat_concat_and_padding():
    expected = (8, 4)
    full = torch.arange(32, dtype=torch.int32)
    out = reconstruct_gathered_flat(full, expected, world_size=2)
    assert tuple(out.shape) == expected
    assert torch.equal(out.reshape(-1), full)

    padded = torch.arange(10, dtype=torch.int32)
    out = reconstruct_gathered_flat(padded, (3, 3), world_size=2)
    assert tuple(out.shape) == (3, 3)
    assert torch.equal(out.reshape(-1), padded[:9])


def test_reconstruct_gathered_flat_rejects_multi_param_blob():
    with pytest.raises(RuntimeError, match="Cannot reconstruct"):
        reconstruct_gathered_flat(torch.arange(20), (3, 3), world_size=2)


def test_embed_tokens_local_shard_is_exactly_half():
    assert 263323648 * 2 == 257152 * 2048


class _TinyCriModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.cri_pos = torch.nn.Parameter(torch.ones(3, 4))


def test_unwrap_and_lookup_recover_cri_pos():
    model = _TinyCriModule()
    inner = torch.nn.Module()
    inner.cri_pos = model.cri_pos
    inner.linear = model.linear
    wrapper = torch.nn.Module()
    wrapper._fsdp_wrapped_module = inner

    assert unwrap_fsdp_module(wrapper) is inner
    assert torch.equal(lookup_module_tensor(wrapper, "cri_pos"), model.cri_pos)
    assert torch.equal(
        lookup_module_tensor(wrapper, "linear.weight"), model.linear.weight
    )


def test_collect_and_fill_recover_omitted_cri_pos():
    model = _TinyCriModule()
    collected = collect_cleaned_named_tensors(model)
    assert "cri_pos" in collected
    assert "linear.weight" in collected

    partial = {"linear.weight": model.linear.weight.detach()}
    fill_missing_named_tensors(model, partial, ["cri_pos", "linear.weight"])
    assert "cri_pos" in partial
    assert torch.equal(partial["cri_pos"], model.cri_pos)
