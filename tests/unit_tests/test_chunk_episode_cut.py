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

from rlinf.algorithms.utils import (
    keep_through_first_done,
    preprocess_embodied_advantages_inputs,
)
from rlinf.utils.metric_utils import align_loss_mask


def test_keep_through_first_done_drops_the_hold_tail():
    dones = torch.tensor([[False, False, True, False, False]])
    keep = keep_through_first_done(dones)
    assert keep.tolist() == [[True, True, True, False, False]]


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
