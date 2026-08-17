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

"""MDP terms for DROID abs-joint-pos cube→bowl pick-and-place.

Observation / gripper action semantics mirror Isaac Lab Arena's DROID
embodiment (``joint_pos``, ``gripper_pos`` in ``[0, 1]``, abs joint targets).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import *  # noqa: F401,F403
from isaaclab.envs.mdp.actions import binary_joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass

_PANDA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)


def _panda_arm_joint_indices(robot) -> list[int]:
    return [i for i, name in enumerate(robot.data.joint_names) if name in _PANDA_ARM_JOINT_NAMES]


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Panda 7-DoF arm joint positions (Arena ``joint_pos``)."""
    robot = env.scene[asset_cfg.name]
    joint_indices = _panda_arm_joint_indices(robot)
    return robot.data.joint_pos[:, joint_indices]


def arm_joint_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Panda 7-DoF arm joint velocities (rad/s) for online CRI."""
    robot = env.scene[asset_cfg.name]
    joint_indices = _panda_arm_joint_indices(robot)
    return robot.data.joint_vel[:, joint_indices]


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gripper open→close mapped to ``[0, 1]`` (Arena ``gripper_pos``)."""
    robot = env.scene[asset_cfg.name]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name == "finger_joint"
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos / (torch.pi / 4)


class BinaryJointPositionZeroToOneAction(binary_joint_actions.BinaryJointPositionAction):
    """Binary gripper with OpenPI/DROID convention: ``>0.5`` closes."""

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            binary_mask = actions == 1
        else:
            binary_mask = actions > 0.5
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Config for :class:`BinaryJointPositionZeroToOneAction`."""

    class_type: type[ActionTerm] = BinaryJointPositionZeroToOneAction


def object_placed_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    destination_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.10,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Success when the cube rests in/near the destination (Arena bowl) XY."""
    cube = env.scene[object_cfg.name]
    destination = env.scene[destination_cfg.name]
    cube_pos = cube.data.root_pos_w
    dest_pos = destination.data.root_pos_w
    xy_dist = torch.norm(cube_pos[:, :2] - dest_pos[:, :2], dim=-1)
    height_ok = (cube_pos[:, 2] - dest_pos[:, 2]).abs() < height_threshold
    vel_ok = torch.norm(cube.data.root_lin_vel_w, dim=-1) < velocity_threshold
    return (xy_dist < xy_threshold) & height_ok & vel_ok


# Back-compat alias for older configs / callers.
object_placed_on_plate = object_placed_on_destination


def reset_object_pose_uniform(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Reset a rigid object's root pose with small uniform noise (single layout)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = asset.data.root_pos_w.device

    base_pos = asset.data.default_root_state[env_ids, 0:3].clone()
    base_pos[:, 0] += env.scene.env_origins[env_ids, 0]
    base_pos[:, 1] += env.scene.env_origins[env_ids, 1]
    base_pos[:, 2] += env.scene.env_origins[env_ids, 2]

    def _sample(key: str) -> torch.Tensor:
        low, high = pose_range.get(key, (0.0, 0.0))
        return (high - low) * torch.rand(n, device=device) + low

    base_pos[:, 0] += _sample("x")
    base_pos[:, 1] += _sample("y")
    base_pos[:, 2] += _sample("z")

    quat = asset.data.default_root_state[env_ids, 3:7].clone()
    velocities = torch.zeros(n, 6, device=device)
    asset.write_root_pose_to_sim(torch.cat([base_pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
