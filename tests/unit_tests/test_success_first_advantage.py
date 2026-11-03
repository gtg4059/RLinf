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

from rlinf.algorithms.advantages import (
    compute_gae_advantages_and_returns,
    compute_success_first_advantages,
)
from rlinf.algorithms.registry import calculate_adv_and_returns


def test_success_first_prefers_the_shorter_open_and_pulls_the_miss():
    """Opens stay positive. The shorter one totals more. A miss is negative."""
    rewards = torch.tensor(
        [
            [-0.1, -0.1, -0.1],
            [1.0, -0.1, -0.1],
            [-0.1, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(5, 3, dtype=torch.bool)
    dones[2, 0] = True
    dones[4, 1] = True

    advantages, returns = compute_success_first_advantages(
        rewards=torch.zeros(3),
        loss_mask=None,
        success_chunk_reward_min=0.5,
        success_advantage=0.5,
        step_rewards=rewards,
        dones=dones,
    )

    assert returns is None
    fast = advantages[:, 0]
    slow = advantages[:, 1]
    miss = advantages[:, 2]
    assert torch.all(fast[:2] > 0)
    assert torch.all(slow > 0)
    assert torch.all(fast[2:] < 0)
    assert torch.all(miss < 0)
    # Returns 0.9 and 0.7. The floor is 0.5 and the 0.2 gap is kept.
    # Two misses share half of the 1.2 success total, so each totals -0.3.
    assert torch.isclose(fast[:2].sum(), torch.tensor(0.7))
    assert torch.isclose(fast[2:].sum(), torch.tensor(-0.3))
    assert torch.isclose(slow.sum(), torch.tensor(0.5))
    assert torch.isclose(miss.sum(), torch.tensor(-0.3))
    assert fast[:2].sum() + slow.sum() > -(fast[2:].sum() + miss.sum())
    assert slow.sum() > miss.sum()


def test_success_first_ignores_a_masked_chunk_tail():
    """A masked step adds no return and gets no advantage."""
    rewards = torch.tensor(
        [[-0.1, -0.1], [1.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    dones = torch.zeros(4, 2, dtype=torch.bool)
    dones[2, :] = True
    loss_mask = torch.tensor(
        [[True, True], [True, True], [False, True]],
        dtype=torch.bool,
    )

    advantages, _ = compute_success_first_advantages(
        rewards=torch.zeros(2),
        loss_mask=loss_mask,
        success_chunk_reward_min=0.5,
        success_advantage=0.5,
        step_rewards=rewards,
        dones=dones,
    )

    assert advantages[2, 0] == 0
    # The unmasked step after the open is a new miss, so it is pulled down.
    assert advantages[2, 1] < 0
    # Both opens end on step 1. The masked tail is not part of either return,
    # so the open totals match and stay positive.
    assert torch.isclose(advantages[:2, 0].sum(), advantages[:2, 1].sum())
    assert advantages[:2, 0].sum() > 0


def test_success_first_does_not_punish_a_batch_with_no_open():
    """Failures alone must not move the policy. There is no open to pull toward."""
    rewards = torch.full((3, 4), -0.1)
    dones = torch.zeros(4, 4, dtype=torch.bool)
    dones[-1, :] = True

    advantages, _ = compute_success_first_advantages(
        rewards=torch.zeros(4),
        loss_mask=None,
        success_chunk_reward_min=0.5,
        success_advantage=0.5,
        miss_advantage_fraction=0.5,
        step_rewards=rewards,
        dones=dones,
    )

    assert torch.all(advantages == 0)


def test_success_first_timeout_is_not_a_cri_reset():
    """The open stays positive. A CRI reset is the capped 0. A timeout is 0."""
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(4, 3, dtype=torch.bool)
    dones[2, :] = True
    terminals = torch.zeros(4, 3, dtype=torch.bool)
    terminals[2, 0] = True
    terminals[2, 1] = True

    advantages, returns = compute_success_first_advantages(
        rewards=torch.zeros(3),
        loss_mask=None,
        success_chunk_reward_min=0.5,
        success_advantage=0.5,
        miss_advantage_fraction=0.5,
        step_rewards=rewards,
        dones=dones,
        terminals=terminals,
    )

    assert returns is None
    opened = advantages[:, 0]
    reset = advantages[:, 1]
    timeout = advantages[:, 2]
    assert torch.all(opened[:2] > 0)
    assert torch.all(reset[:2] < 0)
    assert torch.all(timeout == 0)
    assert torch.all(opened[2] == 0)
    assert torch.all(reset[2] == 0)
    assert torch.isclose(opened.sum(), torch.tensor(0.5))
    assert torch.isclose(reset.sum(), torch.tensor(-0.25))


def test_success_first_counts_a_masked_cri_reset_step():
    """Masking the reset step still marks that attempt as the reward-0 miss."""
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[2, :] = True
    terminals = dones.clone()
    loss_mask = torch.tensor([[True, True], [False, True]], dtype=torch.bool)

    advantages, _ = compute_success_first_advantages(
        rewards=torch.zeros(2),
        loss_mask=loss_mask,
        success_chunk_reward_min=0.5,
        success_advantage=0.5,
        miss_advantage_fraction=0.5,
        step_rewards=rewards,
        dones=dones,
        terminals=terminals,
    )

    assert advantages[1, 0] == 0
    assert advantages[0, 0] < 0
    assert torch.all(advantages[:, 1] > 0)
    assert torch.isclose(advantages[:, 1].sum(), torch.tensor(0.5))
    assert torch.isclose(advantages[:, 0].sum(), torch.tensor(-0.25))


def test_batch_mean_centers_episode_returns_and_ignores_the_value():
    """A value of 10 does not enter the advantage. Returns stay uncentered."""
    rewards = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[-1, :] = True
    values = torch.full((3, 2), 10.0)

    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
    )

    assert returns[:, 0].tolist() == pytest.approx([1.0, 1.0])
    assert returns[:, 1].tolist() == pytest.approx([0.0, 0.0])
    assert advantages[:, 0].tolist() == pytest.approx([0.5, 0.5])
    assert advantages[:, 1].tolist() == pytest.approx([-0.5, -0.5])


def test_batch_mean_restarts_after_a_done():
    """The open and the later miss are separate episodes."""
    rewards = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    dones = torch.zeros(5, 1, dtype=torch.bool)
    dones[2, 0] = True
    dones[4, 0] = True

    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=torch.zeros(5, 1),
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
        advantage_baseline="batch_mean",
    )

    assert returns[:, 0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert advantages[:, 0].tolist() == pytest.approx([0.5, 0.5, -0.5, -0.5])


def test_embodied_batch_mean_survives_chunk_preprocess():
    """Chunk-level GAE receives the baseline and ignores prev_values."""
    rewards = torch.tensor([[[0.0], [0.0]], [[1.0], [0.0]]])
    dones = torch.zeros(3, 2, 1, dtype=torch.bool)
    dones[-1, :, :] = True
    values = torch.full((3, 2, 1), 10.0)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae",
        rewards=rewards,
        dones=dones,
        values=values,
        gamma=1.0,
        gae_lambda=1.0,
        reward_type="chunk_level",
        normalize_advantages=False,
        advantage_baseline="batch_mean",
    )

    assert result["returns"][:, 0, 0].tolist() == pytest.approx([1.0, 1.0])
    assert result["returns"][:, 1, 0].tolist() == pytest.approx([0.0, 0.0])
    assert result["advantages"][:, 0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert result["advantages"][:, 1, 0].tolist() == pytest.approx([-0.5, -0.5])
