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

import rlinf.algorithms.advantages  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.utils import clear_unfinished_tail_rewards


def test_clear_zeros_only_the_unfinished_tail():
    """A later open keeps its penalty. A miss from the start keeps its own."""
    penalty = -0.002
    opened = 1.0 + penalty
    rewards = torch.tensor(
        [
            [[penalty, opened], [penalty, penalty], [penalty, opened]],
            [[penalty, penalty], [penalty, penalty], [penalty, opened]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(3, 3, 2, dtype=torch.bool)
    dones[1, 0, 1] = True
    dones[1, 2, 1] = True
    dones[2, 1, 1] = True
    dones[2, 2, 1] = True
    before = rewards.clone()

    clear_unfinished_tail_rewards(rewards, dones)

    torch.testing.assert_close(rewards[0], before[0])
    torch.testing.assert_close(rewards[1, 0], torch.zeros(2))
    torch.testing.assert_close(rewards[1, 1], before[1, 1])
    torch.testing.assert_close(rewards[1, 2], before[1, 2])


def test_tail_leaves_the_update_and_a_first_miss_stays():
    """The post-success tail is masked. The horizon that never opens is not."""
    penalty = -0.002
    opened = 1.0 + penalty
    rewards = torch.tensor(
        [
            [[penalty, opened], [penalty, penalty]],
            [[penalty, penalty], [penalty, penalty]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(3, 2, 2, dtype=torch.bool)
    dones[1, 0, 1] = True
    dones[2, 1, 1] = True

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="global_return",
        rewards=rewards,
        dones=dones,
        reward_type="chunk_level",
        loss_mask=torch.ones_like(rewards, dtype=torch.bool),
        group_size=1,
        success_chunk_reward_min=0.5,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(rewards[1, 0], torch.zeros(2))
    torch.testing.assert_close(rewards[0, 1], torch.full((2,), penalty))
    mask = result["step_loss_mask"]
    assert mask[0].tolist() == [[True, True], [True, True]]
    assert mask[1, 0].tolist() == [False, False]
    assert mask[1, 1].tolist() == [True, True]
    advantages = result["advantages"]
    assert advantages[1, 0].item() == pytest.approx(0.0)
    assert advantages[0, 0].item() > 0.0
    assert advantages[0, 1].item() < 0.0
