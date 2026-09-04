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

"""Episode-stat gathering for eval/train env_info (scalar vs per-env)."""

from __future__ import annotations

import numpy as np
import torch

from rlinf.workers.env.env_worker import gather_episode_stat


def test_gather_indexes_per_env_vector():
    values = torch.tensor([0.1, 0.2, 0.3, 0.4])
    mask = torch.tensor([False, True, False, True])
    got = gather_episode_stat(values, mask)
    assert torch.equal(got, torch.tensor([0.2, 0.4]))


def test_gather_expands_zero_dim_scalar():
    mask = torch.tensor([True, False, True])
    got = gather_episode_stat(torch.tensor(1.5), mask)
    assert got.shape == (2,)
    assert torch.allclose(got, torch.tensor([1.5, 1.5]))


def test_gather_expands_python_and_numpy_scalars():
    mask = torch.tensor([True, True, False])
    assert torch.equal(gather_episode_stat(3, mask), torch.tensor([3, 3]))
    got = gather_episode_stat(np.float32(0.25), mask)
    assert got.shape == (2,)
    assert torch.allclose(got, torch.tensor([0.25, 0.25]))


def test_gather_empty_mask_from_scalar():
    mask = torch.zeros(4, dtype=torch.bool)
    got = gather_episode_stat(torch.tensor(2.0), mask)
    assert got.shape == (0,)
