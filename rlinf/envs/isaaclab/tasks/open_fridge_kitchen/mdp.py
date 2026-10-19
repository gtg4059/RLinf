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
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door_reward import (
    LEFT_FINGERTIP_OFFSET_M,
    RIGHT_FINGERTIP_OFFSET_M,
    hinge_handle_segment,
    quat_apply_wxyz,
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


def _body_index(asset, needles: tuple[str, ...]) -> int | None:
    names = [str(name).lower() for name in asset.body_names]
    for needle in needles:
        for index, name in enumerate(names):
            if needle in name:
                return index
    return None


def _body_point(robot, index: int, offset: tuple[float, float, float]) -> torch.Tensor:
    """World position of a point fixed in one body frame."""
    positions = robot.data.body_pos_w[:, index, :]
    quats = getattr(robot.data, "body_quat_w", None)
    local = positions.new_tensor(offset).expand(positions.shape[0], 3)
    if quats is None:
        return positions + local
    return positions + quat_apply_wxyz(quats[:, index, :], local)


def _gripper_position(robot) -> torch.Tensor:
    """World position of the grasp, at the middle of the two fingertip ends.

    Each offset is the outer face of that finger's pad. Averaging the two
    puts the query between the fingertips, so a pad on the handle bar is
    within a few centimeters of the bar centerline.
    """
    left = _body_index(robot, ("left_inner_finger", "left_finger"))
    right = _body_index(robot, ("right_inner_finger", "right_finger"))
    if left is not None and right is not None:
        return 0.5 * (
            _body_point(robot, left, LEFT_FINGERTIP_OFFSET_M)
            + _body_point(robot, right, RIGHT_FINGERTIP_OFFSET_M)
        )
    positions = robot.data.body_pos_w
    index = _body_index(
        robot, ("panda_hand", "panda_link8", "gripper", "tool0", "wrist")
    )
    if index is None:
        index = 0
    return positions[:, index, :]


def _fridge_door_pose(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """World pose of the fridge-door link, or None when the body is missing."""
    fridge = env.scene["fridge"]
    index = _body_index(fridge, ("refrigerator032_fridge_door",))
    if index is None:
        return None
    pos = fridge.data.body_pos_w[:, index, :]
    quat = getattr(fridge.data, "body_quat_w", None)
    if quat is None:
        return None
    return pos, quat[:, index, :]


def fridge_handle_reach(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gripper position and the hinged handle bar, shape ``(N, 9)``.

    Columns are gripper xyz, bar start xyz, bar end xyz. The bar is
    ``fridge_door_handle_main`` expressed in the door link and moved by the
    live hinge pose. If that link is missing, the segment copies the gripper
    so the distance stays zero and no approach reward is invented.
    """
    del asset_cfg
    robot = env.scene["robot"]
    gripper = _gripper_position(robot)
    pose = _fridge_door_pose(env)
    if pose is None:
        return torch.cat([gripper, gripper, gripper], dim=-1)
    start, end = hinge_handle_segment(*pose)
    return torch.cat([gripper, start, end], dim=-1)


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
    ids = torch.as_tensor(
        env_ids, device=fridge.data.joint_pos.device, dtype=torch.long
    )
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
