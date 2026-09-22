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

"""MDP terms for DROID abs-joint-pos kitchen fridge open-door.

Reuses pick_place DROID proprio / gripper actions. Success matches Arena
``OpenDoorTask``: joint travel past ``openness_threshold`` of the hinge range.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.managers import SceneEntityCfg

from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door import (
    compute_door_openness,
    door_reached_from_rest,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.mdp import (  # noqa: F401
    BinaryJointPositionZeroToOneAction,
    BinaryJointPositionZeroToOneActionCfg,
    arm_joint_pos,
    arm_joint_vel,
    gripper_pos,
)

_FRIDGE_DOOR_JOINT = "fridge_door_joint"


def _fridge_door_index(fridge, joint_name: str) -> int:
    names = list(fridge.data.joint_names)
    if joint_name in names:
        return names.index(joint_name)
    for i, name in enumerate(names):
        if "door" in name.lower() and "fridge" in name.lower():
            return i
    for i, name in enumerate(names):
        if "door" in name.lower():
            return i
    raise KeyError(
        f"Fridge door joint {joint_name!r} not in {names}. "
        "Expected Arena ``fridge_door_joint``."
    )


def fridge_door_openness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_name: str = _FRIDGE_DOOR_JOINT,
) -> torch.Tensor:
    """Fraction of the fridge door hinge range (0 = closed)."""
    fridge = env.scene[asset_cfg.name]
    idx = _fridge_door_index(fridge, joint_name)
    pos = fridge.data.joint_pos[:, idx]
    lower = fridge.data.soft_joint_pos_limits[:, idx, 0]
    upper = fridge.data.soft_joint_pos_limits[:, idx, 1]
    return compute_door_openness(pos, lower, upper)


def fridge_door_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_name: str = _FRIDGE_DOOR_JOINT,
    rest_openness: float = 0.0,
    min_openness_change: float = 0.05,
) -> torch.Tensor:
    """Arena reach: fridge door moved from rest (``min_openness_change``)."""
    return door_reached_from_rest(
        fridge_door_openness(env, asset_cfg, joint_name),
        rest_openness=rest_openness,
        min_openness_change=min_openness_change,
    )


def fridge_door_is_open(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_name: str = _FRIDGE_DOOR_JOINT,
    openness_threshold: float = 0.2,
) -> torch.Tensor:
    """Arena ``OpenDoorTask`` success: openness > ``openness_threshold``."""
    return fridge_door_openness(env, asset_cfg, joint_name) > openness_threshold


def reset_fridge_door(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fridge"),
    joint_name: str = _FRIDGE_DOOR_JOINT,
    reset_openness: float = 0.0,
) -> None:
    """Close (or partially open) the fridge door on reset."""
    fridge = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=fridge.data.joint_pos.device, dtype=torch.long)
    if int(ids.numel()) == 0:
        return
    idx = _fridge_door_index(fridge, joint_name)
    lower = fridge.data.soft_joint_pos_limits[ids, idx, 0]
    upper = fridge.data.soft_joint_pos_limits[ids, idx, 1]
    target = lower + float(reset_openness) * (upper - lower)
    joint_pos = fridge.data.joint_pos[ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, idx] = target
    fridge.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
