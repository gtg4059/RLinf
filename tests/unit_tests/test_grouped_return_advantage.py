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

"""Grouped return advantages center opens and misses separately."""

import pytest
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns


def _advantages(rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grouped_return",
        rewards=rewards.unsqueeze(-1),
        dones=dones.unsqueeze(-1),
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=None,
        group_size=1,
        success_chunk_reward_min=0.5,
    )
    assert result.get("returns") is None
    return result["advantages"][:, :, 0]


def test_fast_open_beats_slow_open_and_a_close_miss_stays_positive():
    """Dense cost ranks each group. Missing +1 does not negate a close miss."""
    rewards = torch.tensor(
        [
            [-0.01, -0.05, -0.01, -0.08],
            [1.00, -0.05, -0.01, -0.08],
            [-0.20, -0.05, -0.01, -0.08],
            [-0.20, 1.00, -0.01, -0.08],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(5, 4, dtype=torch.bool)
    dones[2, 0] = True
    dones[4, :] = True

    advantages = _advantages(rewards, dones)

    assert advantages[0, 0] > 0
    assert advantages[0, 1] < 0
    assert advantages[0, 0] == pytest.approx(-advantages[0, 1])
    assert advantages[0, 2] > 0
    assert advantages[0, 3] < 0
    # The fast open's later miss is ranked with misses, not with the +1.
    assert advantages[2, 0] < advantages[2, 2]


def test_a_lone_outcome_has_no_direction():
    """One open and one miss have nothing inside their group to beat."""
    rewards = torch.tensor([[1.0, -0.2]], dtype=torch.float32)
    dones = torch.zeros(2, 2, dtype=torch.bool)
    dones[1, :] = True

    advantages = _advantages(rewards, dones)

    assert advantages.shape == (1, 2)
    assert torch.all(advantages == 0)
