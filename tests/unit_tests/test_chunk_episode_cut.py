# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunk rewards stop at the first success inside the chunk."""

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    keep_through_first_done,
    loss_mask_through_first_done,
    preprocess_embodied_advantages_inputs,
)
from rlinf.utils.metric_utils import align_loss_mask, values_aligned_to_returns


def test_keep_through_first_done_drops_the_hold_tail():
    dones = torch.tensor([[False, False, True, False, False]])
    keep = keep_through_first_done(dones)
    assert keep.tolist() == [[True, True, True, False, False]]


def test_failure_reset_step_leaves_the_loss_and_success_stays():
    dones = torch.tensor(
        [
            [False, True, False],
            [False, False, True],
        ]
    )
    rewards = torch.tensor(
        [
            [0.0, -0.02, 1.0],
            [-0.01, 0.0, 1.0],
        ]
    )
    mask = loss_mask_through_first_done(dones, rewards, success_min=0.5)
    assert mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]


def test_chunk_reward_excludes_steps_after_success_and_the_next_chunk():
    # Two chunks, three steps. Success on step 1 of chunk 0.
    # Rewards after that step, including the whole next chunk, must not be
    # added into chunk 0. Chunk 1 stays its own episode.
    rewards = torch.tensor(
        [
            [[1.0, 10.0, 100.0]],
            [[7.0, 8.0, 9.0]],
        ]
    )
    dones = torch.zeros(3, 1, 3, dtype=torch.bool)
    dones[1, 0, 1] = True
    out = preprocess_embodied_advantages_inputs(
        rewards,
        dones,
        values=torch.zeros(3, 1, 1),
        reward_type="chunk_level",
        adv_type="gae",
    )
    # Chunk 0 keeps 1+10. The 100 after success is dropped, not merged
    # into chunk 1. Chunk 1 is still 7+8+9.
    assert out["rewards"].tolist() == [[11.0], [24.0]]
    assert out["dones"][1].item() is True
    assert out["dones"][2].item() is False
    assert out["step_loss_mask"][0, 0].tolist() == [True, True, False]
    assert out["step_loss_mask"][1, 0].tolist() == [True, True, True]


def test_logged_values_drop_the_bootstrap_row():
    """The extra GAE value is not part of the logged value function."""
    prev_values = torch.tensor([0.1, 0.2, 9.0]).reshape(3, 1, 1)
    returns = torch.zeros(2, 1, 1)
    aligned = values_aligned_to_returns(prev_values, returns)
    assert torch.allclose(aligned.reshape(-1), torch.tensor([0.1, 0.2]))

    same_length = torch.tensor([0.1, 0.2]).reshape(2, 1, 1)
    assert values_aligned_to_returns(same_length, returns).shape[0] == 2


def test_chunk_advantage_metric_accepts_the_step_mask():
    """Advantages are one value per chunk. The step mask stays 15 wide."""
    advantages = torch.zeros(60, 32, 1)
    step_mask = torch.ones(60, 32, 15, dtype=torch.bool)
    step_mask[0, 0, 3:] = False
    aligned = align_loss_mask(step_mask, advantages)
    assert aligned.shape == advantages.shape
    assert bool(aligned[0, 0, 0])
    assert int(aligned.sum()) == advantages.numel()

    # Critic returns are [micro, 1, 1] after the token unsqueeze.
    # The step mask is [micro, chunk, 1].
    returns = torch.zeros(8, 1, 1)
    critic_mask = torch.ones(8, 15, 1, dtype=torch.bool)
    critic_mask[0, 3:, 0] = False
    aligned = align_loss_mask(critic_mask, returns)
    assert aligned.shape == returns.shape
    assert bool(aligned.all())


def test_timeout_bootstraps_and_cri_termination_does_not():
    """A horizon timeout keeps the final value. A CRI end stays at reward 0.

    Both rows end, so ``dones`` is true. Only the CRI row is a termination.
    With bootstrap on, the timeout return is the final value and the CRI
    return is 0. The timeout step stays in the loss. The CRI step does not.
    """
    rewards = torch.zeros(1, 2, 3)
    dones = torch.zeros(2, 2, 3, dtype=torch.bool)
    dones[1, :, 2] = True
    terminations = torch.zeros_like(dones)
    terminations[1, 1, 2] = True
    # Values are one per chunk, plus the bootstrap row. 0.5 is V at the end.
    values = torch.zeros(2, 2, 1)
    values[1, :, 0] = 0.5

    masked = preprocess_embodied_advantages_inputs(
        rewards.clone(),
        dones,
        values=values,
        terminations=terminations,
        reward_type="chunk_level",
        adv_type="gae",
        bootstrap_truncation=True,
    )
    assert masked["step_loss_mask"][0, 0].tolist() == [True, True, True]
    assert masked["step_loss_mask"][0, 1].tolist() == [True, True, False]

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae",
        rewards=rewards,
        dones=dones,
        values=values,
        terminations=terminations,
        gamma=1.0,
        gae_lambda=1.0,
        reward_type="chunk_level",
        normalize_advantages=False,
        bootstrap_truncation=True,
    )
    # One chunk, so the return is the bootstrapped value or 0.
    assert abs(result["returns"][0, 0].item() - 0.5) < 1e-6
    assert abs(result["returns"][0, 1].item()) < 1e-6


def test_chunk_level_logprob_collapses_per_step_mask():
    """A 15-step mask must not be compared with a batch of 16 ratios."""
    batch, steps, action_dim = 16, 15, 8
    logprobs = torch.zeros(batch, steps, action_dim)
    advantages = torch.ones(batch, 1)
    loss_mask = torch.ones(batch, steps, dtype=torch.bool)
    loss_mask[:, 5:] = False
    loss_mask[0] = False
    loss_mask_sum = torch.arange(steps, dtype=torch.float32).expand(batch, steps)
    loss, metrics = policy_loss(
        loss_type="actor",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=action_dim,
        logprobs=logprobs,
        old_logprobs=logprobs.clone(),
        advantages=advantages,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum.clone(),
        max_episode_steps=450,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert loss.ndim == 0
    assert metrics["actor/clip_fraction"] == 0.0
