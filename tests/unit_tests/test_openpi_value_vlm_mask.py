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

"""VLM value prefix mask must match OpenPI prefix length (incl. PolarIS CRI 220)."""

import pytest
import torch

pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.openpi_action_model import vlm_value_prefix_mask


def test_mean_token_mask_matches_polaris_cri_prefix():
    # 2 cameras + unused 3rd slot + max_token_len=220
    seq_len = 256 * 3 + 220
    mask = vlm_value_prefix_mask(seq_len, num_images_in_input=2, value_vlm_mode="mean_token")
    assert len(mask) == seq_len
    dummy = torch.zeros(16, seq_len, 8)
    selected = dummy[:, mask, :]
    assert selected.shape == (16, 256 * 2 + 220, 8)


def test_mean_token_mask_keeps_pi05_default_200():
    seq_len = 256 * 3 + 200
    mask = vlm_value_prefix_mask(seq_len, num_images_in_input=2, value_vlm_mode="mean_token")
    assert len(mask) == 968
    assert sum(mask) == 256 * 2 + 200


def test_last_and_first_token_use_actual_seq_len():
    seq_len = 988
    last = vlm_value_prefix_mask(seq_len, 2, "last_token")
    first = vlm_value_prefix_mask(seq_len, 2, "first_token")
    assert last[-1] is True and sum(last) == 1
    assert first[0] is True and sum(first) == 1
    assert len(last) == seq_len == len(first)
