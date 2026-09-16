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

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.filter import (
    abs_joint_to_qd_nom,
    shift_cri_filter_obs,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
    CRI_OVF_TERM_PENALTY,
    ISAACLAB_STEP_DT,
    TIME_PENALTY_WEIGHT,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
    CRI_OVF_REWARD_KEY,
    TIME_PENALTY_REWARD_KEY,
    accumulate_cri_episode_reward,
    accumulate_time_episode_reward,
    cri_episode_reward_logs,
    cri_ovf_exp,
    cri_ovf_reward,
    cri_ovf_termination_step_penalty,
    cri_ovf_violated,
    time_episode_reward_logs,
    time_penalty_live_mask,
    time_step_penalty,
)


def test_cri_ovf_exp_zero_at_rest():
    cri = torch.zeros(2, 9)
    pen = cri_ovf_exp(cri, limit=0.96, sigma=20.0, ovf_threshold=2.0)
    assert pen.shape == (2,)
    assert torch.allclose(pen, torch.zeros(2), atol=1e-6)


def test_cri_ovf_exp_one_at_limit():
    cri = torch.zeros(1, 9)
    cri[0, -1] = 0.96
    pen = cri_ovf_exp(cri, limit=0.96, sigma=20.0, ovf_threshold=2.0)
    assert float(pen[0]) == pytest.approx(1.0, abs=1e-5)


def test_cri_ovf_exp_grows_past_limit():
    at_limit = torch.full((1, 9), 0.96)
    above = torch.full((1, 9), 1.48)
    p_lim = cri_ovf_exp(at_limit, limit=0.96, sigma=20.0, ovf_threshold=2.0)
    p_hi = cri_ovf_exp(above, limit=0.96, sigma=20.0, ovf_threshold=2.0)
    assert float(p_hi[0]) > float(p_lim[0])
    assert float(p_hi[0]) == pytest.approx(1.5, abs=1e-5)


def test_cri_ovf_exp_weight_scale():
    cri = torch.zeros(1, 9)
    cri[0, 0] = 0.96
    pen = cri_ovf_exp(cri)
    reward = -0.02 * pen
    assert float(reward[0]) == pytest.approx(-0.02, abs=1e-5)


def test_cri_episode_reward_logs_match_isaaclab_episode_sum():
    ovf_sum = torch.zeros(2)
    penalty_sum = torch.zeros(2)
    accumulate_cri_episode_reward(ovf_sum, penalty_sum, torch.tensor([0.0, 1.0]), -0.002)
    accumulate_cri_episode_reward(ovf_sum, penalty_sum, torch.tensor([0.5, 1.0]), -0.002)
    logs = cri_episode_reward_logs(penalty_sum)
    assert torch.allclose(logs[CRI_OVF_REWARD_KEY], torch.tensor([-0.001, -0.004]))
    assert torch.allclose(
        cri_ovf_reward(torch.tensor([0.5, 1.0]), -0.002),
        torch.tensor([-0.001, -0.002]),
    )


def test_abs_joint_to_qd_nom_matches_isaaclab():
    q = torch.zeros(2, 7)
    q_tgt = torch.zeros(2, 7)
    q_tgt[:, 0] = 0.04
    qd = abs_joint_to_qd_nom(q_tgt, q, dt=0.02)
    assert torch.allclose(qd[:, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(qd[:, 1:], torch.zeros(2, 6))


def test_cri_ovf_violated_at_and_below_threshold():
    below = torch.full((2, 9), 0.95)
    at = torch.full((2, 9), 0.96)
    above = torch.full((2, 9), 1.1)
    assert not bool(cri_ovf_violated(below, threshold=0.96).any())
    assert bool(cri_ovf_violated(at, threshold=0.96).all())
    assert bool(cri_ovf_violated(above, threshold=0.96).all())


def test_cri_ovf_termination_penalty_is_success_reward_third():
    expected = -ISAACLAB_STEP_DT / 3.0
    assert CRI_OVF_TERM_PENALTY == pytest.approx(expected)
    assert cri_ovf_termination_step_penalty() == pytest.approx(expected)
    assert cri_ovf_termination_step_penalty() == pytest.approx(-0.022222, abs=1e-4)


def test_time_step_penalty_is_dt_over_100():
    expected = -ISAACLAB_STEP_DT / 100.0
    assert TIME_PENALTY_WEIGHT == pytest.approx(expected)
    assert time_step_penalty() == pytest.approx(expected)
    assert time_step_penalty() == pytest.approx(-0.0006666667, abs=1e-9)


def test_time_step_penalty_30_steps_is_success_third():
    saved = 30.0 * abs(time_step_penalty())
    assert saved == pytest.approx(0.3 * ISAACLAB_STEP_DT, abs=1e-9)
    assert saved == pytest.approx(ISAACLAB_STEP_DT / 3.0, abs=3e-3)


def test_time_episode_reward_logs_skip_frozen():
    penalty_sum = torch.zeros(2)
    live = torch.tensor([True, False])
    accumulate_time_episode_reward(penalty_sum, -0.001, live)
    accumulate_time_episode_reward(penalty_sum, -0.001, live)
    logs = time_episode_reward_logs(penalty_sum)
    assert torch.allclose(logs[TIME_PENALTY_REWARD_KEY], torch.tensor([-0.002, 0.0]))


def test_time_penalty_live_mask_stops_after_first_success():
    success_once = torch.tensor([False, True, False])
    mask = time_penalty_live_mask(3, success_once=success_once)
    assert torch.equal(mask, torch.tensor([True, False, True]))


def test_time_penalty_live_mask_success_step_still_pays():
    """``success_once`` flips after the reward, so the hit step is charged."""
    success_once = torch.zeros(2, dtype=torch.bool)
    before = time_penalty_live_mask(2, success_once=success_once)
    success_once[0] = True
    after = time_penalty_live_mask(2, success_once=success_once)
    assert torch.equal(before, torch.tensor([True, True]))
    assert torch.equal(after, torch.tensor([False, True]))


def test_time_penalty_live_mask_hold_after_done_still_skips_frozen():
    was_frozen = torch.tensor([True, False])
    success_once = torch.tensor([False, False])
    mask = time_penalty_live_mask(
        2,
        hold_after_done=True,
        was_frozen=was_frozen,
        success_once=success_once,
    )
    assert torch.equal(mask, torch.tensor([False, True]))


def test_time_penalty_sum_only_until_first_success():
    penalty_sum = torch.zeros(2)
    success_once = torch.zeros(2, dtype=torch.bool)
    for step in range(5):
        live = time_penalty_live_mask(2, success_once=success_once)
        accumulate_time_episode_reward(penalty_sum, -0.001, live)
        if step == 1:
            success_once[0] = True
    logs = time_episode_reward_logs(penalty_sum)
    # env0: steps 0-1 (success on step 1), env1: all 5 steps
    assert torch.allclose(logs[TIME_PENALTY_REWARD_KEY], torch.tensor([-0.002, -0.005]))


def test_shift_cri_filter_obs_first_row_zero():
    cri = torch.arange(18, dtype=torch.float32).reshape(2, 9).numpy()
    cri_obs, _ = shift_cri_filter_obs(cri)
    assert cri_obs[0].sum() == 0.0
    assert (cri_obs[1] == cri[0]).all()
