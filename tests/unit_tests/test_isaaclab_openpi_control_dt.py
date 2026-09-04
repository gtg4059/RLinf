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

"""IsaacLab cube-plate control dt must match OpenPI DROID (15 Hz)."""

from __future__ import annotations

import pytest

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
    DROID_CONTROL_DT,
    DROID_CONTROL_FREQUENCY_HZ,
    ISAACLAB_DECIMATION,
    ISAACLAB_SIM_DT,
    ISAACLAB_STEP_DT,
)


def test_isaaclab_step_dt_matches_openpi_droid():
    assert DROID_CONTROL_FREQUENCY_HZ == 15.0
    assert ISAACLAB_DECIMATION == 4
    assert ISAACLAB_STEP_DT == pytest.approx(DROID_CONTROL_DT)
    assert ISAACLAB_SIM_DT * ISAACLAB_DECIMATION == pytest.approx(DROID_CONTROL_DT)
    assert ISAACLAB_STEP_DT == pytest.approx(1.0 / 15.0)
    assert ISAACLAB_SIM_DT == pytest.approx(1.0 / 60.0)
