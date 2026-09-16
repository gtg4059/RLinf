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
    CRI_OVF_SIGMA,
    CRI_OVF_TERM_PENALTY,
    CRI_OVF_THRESHOLD,
    CRI_PENALTY_WEIGHT,
    DEFAULT_NUM_JOINTS,
    DEFAULT_ZERO_VEL_EPS,
    DROID_CONTROL_DT,
    ISAACLAB_DECIMATION,
    ISAACLAB_SIM_DT,
    ISAACLAB_STEP_DT,
    NUM_CRI_OBS_DIM,
    NUM_CRI_POINTS,
    PACKAGE_DIR,
    TIME_PENALTY_WEIGHT,
)
from .filter import abs_joint_to_qd_nom
from .filter import compute_episode_cri_f
from .postprocess import apply_cri_zero_vel_filter, clamp_cri
from .rewards import (
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
from .solver import CriSolver, resolve_analysis_dir
from .velocity import joint_velocity_from_positions

__all__ = [
    "CBF_ALPHA",
    "CRI_CLAMP_MAX",
    "CRI_CLAMP_MIN",
    "CRI_FILTER_LIMIT",
    "CRI_OVF_SIGMA",
    "CRI_OVF_TERM_PENALTY",
    "CRI_OVF_THRESHOLD",
    "CRI_PENALTY_WEIGHT",
    "DEFAULT_NUM_JOINTS",
    "DEFAULT_ZERO_VEL_EPS",
    "DROID_CONTROL_DT",
    "ISAACLAB_DECIMATION",
    "ISAACLAB_SIM_DT",
    "ISAACLAB_STEP_DT",
    "NUM_CRI_OBS_DIM",
    "NUM_CRI_POINTS",
    "PACKAGE_DIR",
    "TIME_PENALTY_WEIGHT",
    "TIME_PENALTY_REWARD_KEY",
    "CriSolver",
    "abs_joint_to_qd_nom",
    "apply_cri_zero_vel_filter",
    "clamp_cri",
    "compute_cri",
    "compute_episode_cri_f",
    "CRI_OVF_REWARD_KEY",
    "accumulate_cri_episode_reward",
    "accumulate_time_episode_reward",
    "cri_episode_reward_logs",
    "cri_ovf_exp",
    "cri_ovf_reward",
    "cri_ovf_termination_step_penalty",
    "cri_ovf_violated",
    "joint_velocity_from_positions",
    "time_episode_reward_logs",
    "time_penalty_live_mask",
    "time_step_penalty",
    "resolve_analysis_dir",
]
