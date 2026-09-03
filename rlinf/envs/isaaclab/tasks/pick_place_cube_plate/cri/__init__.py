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

"""CRI (Collision Risk Index) computation for OpenPI safety modality.

Computes ``CRI(q, qd)`` via the Safetics CUDA solver (IsaacLab ``sfd_coreservice``)
before multimodal fusion. Near-zero joint velocity forces CRI to 0 (first step / at rest).

Offline RLDS annotation (DROID → CRI-augmented dataset) lives in
``rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.annotate_rlds``.
"""

from .compute import compute_cri
from .constants import (
    CBF_ALPHA,
    CRI_CLAMP_MAX,
    CRI_CLAMP_MIN,
    CRI_FILTER_LIMIT,
    DEFAULT_NUM_JOINTS,
    DEFAULT_ZERO_VEL_EPS,
    DROID_CONTROL_DT,
    NUM_CRI_OBS_DIM,
    NUM_CRI_POINTS,
    PACKAGE_DIR,
)
from .filter import compute_episode_cri_f
from .postprocess import apply_cri_zero_vel_filter, clamp_cri
from .solver import CriSolver, resolve_analysis_dir
from .velocity import joint_velocity_from_positions

__all__ = [
    "CBF_ALPHA",
    "CRI_CLAMP_MAX",
    "CRI_CLAMP_MIN",
    "CRI_FILTER_LIMIT",
    "DEFAULT_NUM_JOINTS",
    "DEFAULT_ZERO_VEL_EPS",
    "DROID_CONTROL_DT",
    "NUM_CRI_OBS_DIM",
    "NUM_CRI_POINTS",
    "PACKAGE_DIR",
    "CriSolver",
    "apply_cri_zero_vel_filter",
    "clamp_cri",
    "compute_cri",
    "compute_episode_cri_f",
    "joint_velocity_from_positions",
    "resolve_analysis_dir",
]
