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

"""Chunk-level embodied rollout size vs actor global_batch_size."""

from __future__ import annotations

import pytest

from rlinf.config import embodied_chunk_rollout_size_per_actor_rank


def test_two_gpu_cri_openpi_sizes_divide():
    rollout = embodied_chunk_rollout_size_per_actor_rank(
        total_num_envs=128,
        rollout_epoch=2,
        max_steps_per_rollout_epoch=450,
        num_action_chunks=15,
        actor_world_size=2,
    )
    assert rollout == 3840
    global_batch_size = 2560
    batch_per_rank = global_batch_size // 2
    assert rollout % batch_per_rank == 0
    assert global_batch_size % (128 * 2) == 0


def test_stale_eight_gpu_gbs_fails_on_two_gpus():
    rollout = embodied_chunk_rollout_size_per_actor_rank(
        total_num_envs=128,
        rollout_epoch=2,
        max_steps_per_rollout_epoch=450,
        num_action_chunks=15,
        actor_world_size=2,
    )
    batch_per_rank = 10240 // 2
    assert rollout % batch_per_rank != 0


def test_rejects_non_positive_world_size():
    with pytest.raises(ValueError):
        embodied_chunk_rollout_size_per_actor_rank(128, 2, 450, 15, 0)
