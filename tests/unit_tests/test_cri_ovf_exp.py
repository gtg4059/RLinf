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
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import cri_ovf_exp


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


def test_abs_joint_to_qd_nom_matches_isaaclab():
    q = torch.zeros(2, 7)
    q_tgt = torch.zeros(2, 7)
    q_tgt[:, 0] = 0.04
    qd = abs_joint_to_qd_nom(q_tgt, q, dt=0.02)
    assert torch.allclose(qd[:, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(qd[:, 1:], torch.zeros(2, 6))


def test_shift_cri_filter_obs_first_row_zero():
    cri = torch.arange(18, dtype=torch.float32).reshape(2, 9).numpy()
    cri_obs, _ = shift_cri_filter_obs(cri)
    assert cri_obs[0].sum() == 0.0
    assert (cri_obs[1] == cri[0]).all()
