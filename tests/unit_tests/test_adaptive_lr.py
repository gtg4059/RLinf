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

import pytest

from rlinf.algorithms.adaptive_lr import adapt_learning_rate


def test_adaptive_lr_shrinks_when_kl_is_above_twice_the_target():
    assert adapt_learning_rate(5.0e-4, kl_mean=0.04, desired_kl=0.016) == pytest.approx(
        5.0e-4 / 1.5
    )


def test_adaptive_lr_grows_when_kl_is_below_half_the_target():
    assert adapt_learning_rate(5.0e-4, kl_mean=0.004, desired_kl=0.016) == pytest.approx(
        5.0e-4 * 1.5
    )


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
