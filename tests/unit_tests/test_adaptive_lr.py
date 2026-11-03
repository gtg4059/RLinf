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

"""rsl_rl adaptive learning-rate schedule."""

from pathlib import Path

import pytest

from rlinf.algorithms.adaptive_lr import adapt_learning_rate, kl_for_adaptive_lr


def test_adaptive_lr_shrinks_when_kl_is_above_twice_the_target():
    assert adapt_learning_rate(5.0e-4, kl_mean=0.04, desired_kl=0.016) == pytest.approx(
        5.0e-4 / 1.5
    )


def test_adaptive_lr_grows_when_kl_is_below_half_the_target():
    assert adapt_learning_rate(5.0e-4, kl_mean=0.004, desired_kl=0.016) == pytest.approx(
        5.0e-4 * 1.5
    )


def test_rollout_eval_mode_restores_training():
    from torch import nn

    from rlinf.models.embodiment.openpi.rollout_eval import use_rollout_eval_mode

    layer = nn.Linear(2, 2)
    layer.train()
    with use_rollout_eval_mode(layer):
        assert layer.training is False
    assert layer.training is True
    layer.eval()
    with use_rollout_eval_mode(layer):
        assert layer.training is False
    assert layer.training is False


def test_adaptive_lr_uses_update_kl_when_the_reference_penalty_is_off():
    metrics = {"actor/kl_loss": 0.0, "actor/update_kl": 0.2, "actor/approx_kl": 4.0}
    assert kl_for_adaptive_lr(metrics, kl_beta=0.0) == pytest.approx(0.2)
    assert kl_for_adaptive_lr(metrics, kl_beta=0.05) == pytest.approx(0.0)


def test_fridge_yaml_starts_adaptive_lr_at_1e6_for_desired_kl_005():
    text = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config"
        / "isaaclab_kitchen_open_fridge_ppo_openpi_pi05.yaml"
    ).read_text()
    assert "replay_old_logprob: False" in text
    assert "lr: 1.0e-6" in text
    assert "lr_scheduler: adaptive" in text
    assert "desired_kl: 0.05" in text
    assert "adaptive_lr_min: 1.0e-7" in text


def test_adaptive_lr_holds_inside_the_band_and_at_the_floor():
    assert adapt_learning_rate(5.0e-4, kl_mean=0.016, desired_kl=0.016) == pytest.approx(
        5.0e-4
    )
    assert adapt_learning_rate(1.2e-5, kl_mean=1.0, desired_kl=0.016) == pytest.approx(
        1.0e-5
    )
    assert adapt_learning_rate(9.0e-3, kl_mean=0.001, desired_kl=0.016) == pytest.approx(
        1.0e-2
    )
