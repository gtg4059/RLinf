# Copyright 2025 The RLinf Authors.
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

"""CRI constants matching IsaacLab Safetics / Panda analysis bundled under ``openpi.cri``."""

from __future__ import annotations

import os
from pathlib import Path

# Package root: contains lib/, Engine/, ST_AnalysisInfo.json, Robot_Model/.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Safetics CUDA CRI output width for the bundled Panda analysis:
# 8 RobotColliPoint channels + 1 aggregate channel (engine returns shape (B, 9)).
# Per-point channels may be ~0 without scene obstacles; the last channel still carries signal.
NUM_CRI_POINTS: int = 9

# Training observation width: cri_pre only (IsaacLab CRI-F policy has no time-scale s).
NUM_CRI_OBS_DIM: int = NUM_CRI_POINTS

# IsaacLab JointVelocityCriFilterActionCfg / enable_cri_filter_mode defaults.
CRI_FILTER_LIMIT: float = 0.96
CBF_ALPHA: float = 0.02

# IsaacLab CRI_OVF_exp (reach mdp.rewards) + cube-plate sparse-success scale.
CRI_OVF_SIGMA: float = 20.0
CRI_OVF_THRESHOLD: float = 2.0
CRI_PENALTY_WEIGHT: float = -0.02

# cube-plate ManagerBasedRLEnv: decimation 4 * sim.dt 1/200.
ISAACLAB_STEP_DT: float = 0.02

# IsaacLab articulation_data._store_cri_output_buffers
CRI_CLAMP_MIN: float = 0.0
CRI_CLAMP_MAX: float = 2.0

# IsaacLab SFD_CRI_ZERO_VEL_EPS (articulation_data._apply_cri_zero_vel_filter)
DEFAULT_ZERO_VEL_EPS: float = float(os.environ.get("SFD_CRI_ZERO_VEL_EPS", "1e-6"))

# Franka Panda arm DOF used by DROID + bundled Panda ST_AnalysisInfo.json.
DEFAULT_NUM_JOINTS: int = 7

# DROID teleop / control rate (examples/droid/main.py).
DROID_CONTROL_FREQUENCY_HZ: float = 15.0
DROID_CONTROL_DT: float = 1.0 / DROID_CONTROL_FREQUENCY_HZ
