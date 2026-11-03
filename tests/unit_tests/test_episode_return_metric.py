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

import pytest
import torch

from rlinf.utils.metric_utils import finished_episode_returns


def test_episode_return_adds_success_and_time_and_drops_the_tail():
    """A short open sums to more than a long one. The unfinished tail is out."""
    short = [-0.002, 0.998]
    long = [-0.002, -0.002, -0.002, 0.998]
    tail = [-0.002, -0.002]
    rewards = torch.tensor(
        [
            short + [0.0, 0.0] + tail,
            long + tail,
        ],
        dtype=torch.float32,
    ).T
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    dones[1, 0] = True
    dones[3, 1] = True

    returns = finished_episode_returns(rewards, dones)

    assert returns.tolist() == pytest.approx([0.996, 0.992])


def test_timeout_keeps_its_time_penalty():
    """A miss that reaches its done keeps every time step and no success."""
    rewards = torch.tensor([[-0.002, -0.002, -0.002]], dtype=torch.float32).T
    dones = torch.tensor([[False, False, True]]).T
    returns = finished_episode_returns(rewards, dones)
    assert returns.item() == pytest.approx(-0.006)


def test_chunk_layout_matches_the_flat_sum():
    """``[n_chunks, batch, chunk]`` sums the same steps as a flat trajectory."""
    rewards = torch.tensor(
        [
            [[-0.002, -0.002, 0.998], [-0.002, -0.002, -0.002]],
        ],
        dtype=torch.float32,
    )
    dones = torch.tensor(
        [
            [[False, False, True], [False, False, True]],
        ]
    )
    returns = finished_episode_returns(rewards, dones)
    assert sorted(returns.tolist()) == pytest.approx([-0.006, 0.994])
