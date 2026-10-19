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

"""Rewind an arm to the pose stored at the last scene reset.

Matches Isaac Lab ``cri_reach_env_cfg``: crossing the CRI limit does not end
the episode. The arm returns to the spawn captured when the scene was last
reset, and the horizon reset still runs that original reset.
"""

from __future__ import annotations

import torch


def cri_boundary_rewind_mask(
    cri_hit: torch.Tensor,
    was_frozen: torch.Tensor,
    success_once: torch.Tensor,
    openness: torch.Tensor | None = None,
    openness_threshold: float = 0.2,
) -> torch.Tensor:
    """Rows that should rewind: CRI crossed, the attempt is still live.

    A door already past ``openness_threshold`` is a reach. Those rows are
    left where they are so the open is not undone. Frozen hold rows and
    rows that already succeeded are left as well.

    Args:
        cri_hit: ``(N,)`` bool, true where CRI is at or above the limit.
        was_frozen: ``(N,)`` bool, rows held after a previous done.
        success_once: ``(N,)`` bool, rows that already reached.
        openness: Optional door openness, ``(N,)``.
        openness_threshold: Strict reach level. Rows above it are not rewound.

    Returns:
        ``(N,)`` bool mask.
    """
    rewind = (
        cri_hit.reshape(-1).bool()
        & ~was_frozen.reshape(-1).bool()
        & ~success_once.reshape(-1).bool()
    )
    if openness is not None:
        rewind = rewind & (openness.reshape(-1) <= float(openness_threshold))
    return rewind


def cri_boundary_episode_metrics(
    success_once: torch.Tensor,
    rewind_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Episode stats for CRI rewinds and opens that never rewound.

    ``success_without_cri`` is 1 only when the door crossed 0.2 and the arm
    was never sent back to the spawn. The mean over episodes is that rate.
    ``cri_rewind_count`` is how many times the arm rewound. ``cri_rewind_once``
    is 1 when that count is positive.

    Args:
        success_once: ``(N,)`` bool, door opened past 0.2.
        rewind_count: ``(N,)`` number of CRI rewinds in the episode.

    Returns:
        Float tensors ``cri_rewind_count``, ``cri_rewind_once``, and
        ``success_without_cri``.
    """
    count = rewind_count.reshape(-1).to(dtype=torch.float32)
    opened = success_once.reshape(-1).bool()
    rewound = count > 0
    return {
        "cri_rewind_count": count,
        "cri_rewind_once": rewound.to(dtype=torch.float32),
        "success_without_cri": (opened & ~rewound).to(dtype=torch.float32),
    }


def capture_robot_spawn(isaac_env, env_ids: torch.Tensor | None = None) -> None:
    """Store the robot joint state currently in the sim.

    Called after a scene reset. A later CRI rewind restores this pose and
    does not replace it, so the horizon reset remains the source of the
    original start.
    """
    robot = isaac_env.scene["robot"]
    joint_pos = robot.data.joint_pos.detach()
    joint_vel = robot.data.joint_vel.detach()
    if not hasattr(isaac_env, "_rlinf_spawn_q"):
        isaac_env._rlinf_spawn_q = joint_pos.clone()
        isaac_env._rlinf_spawn_qd = joint_vel.clone()
    if env_ids is None:
        isaac_env._rlinf_spawn_q.copy_(joint_pos)
        isaac_env._rlinf_spawn_qd.copy_(joint_vel)
        return
    ids = torch.as_tensor(env_ids, device=joint_pos.device).reshape(-1).long()
    isaac_env._rlinf_spawn_q[ids] = joint_pos[ids]
    isaac_env._rlinf_spawn_qd[ids] = joint_vel[ids]


def rewind_robot_to_spawn(isaac_env, env_ids: torch.Tensor):
    """Put ``env_ids`` back on the stored spawn and return fresh observations.

    The fridge door and the episode clock are not reset. Cameras are
    rendered when the sim has RTX sensors so the image matches the arm.
    """
    robot = isaac_env.scene["robot"]
    if not hasattr(isaac_env, "_rlinf_spawn_q"):
        capture_robot_spawn(isaac_env)
    ids = torch.as_tensor(env_ids, device=robot.data.joint_pos.device)
    ids = ids.reshape(-1).long()
    position = isaac_env._rlinf_spawn_q[ids]
    velocity = isaac_env._rlinf_spawn_qd[ids]
    robot.write_joint_state_to_sim(position, velocity, env_ids=ids)
    robot.set_joint_position_target(position, env_ids=ids)
    if hasattr(robot, "set_joint_velocity_target"):
        robot.set_joint_velocity_target(velocity, env_ids=ids)
    isaac_env.sim.forward()
    if isaac_env.sim.has_rtx_sensors():
        renders = max(int(getattr(isaac_env.cfg, "num_rerenders_on_reset", 1)), 1)
        for _ in range(renders):
            isaac_env.sim.render()
    isaac_env.obs_buf = isaac_env.observation_manager.compute(update_history=False)
    return isaac_env.obs_buf
