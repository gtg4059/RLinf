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

"""RLinf wrapper for the DROID kitchen fridge open-door Isaac Lab task."""

from __future__ import annotations

import gymnasium as gym

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env import (
    IsaaclabPickPlaceCubePlateEnv,
)

GYM_ID = "Isaac-OpenFridge-Kitchen-Droid-AbsJointPos-v0"
_STAND_HEIGHT_M = 0.8


def register_open_fridge_kitchen_env() -> str:
    """Register the Isaac Lab gym id if it is not already present."""
    if GYM_ID not in gym.envs.registry:
        gym.register(
            id=GYM_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg:"
                    "OpenFridgeKitchenEnvCfg"
                ),
            },
        )
    return GYM_ID


class IsaaclabOpenFridgeKitchenEnv(IsaaclabPickPlaceCubePlateEnv):
    """Open the kitchen fridge door with DROID abs joint-pos actions.

    Reuses the pick-place CRI-F / time-penalty / hold step path. Only the
    Isaac Lab scene construction differs (kitchen USD + DROID stand).
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        init_params = cfg.init_params
        self._stand_height_m = float(
            init_params.get("stand_height_m", _STAND_HEIGHT_M)
        )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        num_envs = int(self.cfg.init_params.num_envs)
        seed = int(self.seed)
        env_id = self.isaaclab_env_id
        table_cam = getattr(self.cfg.init_params, "table_cam", None)
        wrist_cam = getattr(self.cfg.init_params, "wrist_cam", None)
        eval_cam = getattr(self.cfg.init_params, "eval_cam", None)
        episode_length_s = getattr(self.cfg.init_params, "episode_length_s", None)
        sim_dt = getattr(self.cfg.init_params, "sim_dt", None)
        decimation = getattr(self.cfg.init_params, "decimation", None)
        stand_height_m = self._stand_height_m

        def make_env_isaaclab():
            import os

            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg import (
                OpenFridgeKitchenEnvCfg,
                _resolve_arena_assets_root,
            )
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.compose_on_stand import (
                ARENA_DROID_STAND_FOOTPRINT_XY_M,
                ARENA_DROID_STAND_TRANSLATE_XYZ,
                compose_droid_on_stand,
            )

            if env_id not in gym.envs.registry:
                gym.register(
                    id=env_id,
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    disable_env_checker=True,
                    kwargs={
                        "env_cfg_entry_point": (
                            "rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg:"
                            "OpenFridgeKitchenEnvCfg"
                        ),
                    },
                )
            isaac_env_cfg = OpenFridgeKitchenEnvCfg()
            isaac_env_cfg.seed = seed
            isaac_env_cfg.scene.num_envs = num_envs
            arena_root = _resolve_arena_assets_root()
            composed = compose_droid_on_stand(
                arena_root,
                stand_height_m=stand_height_m,
                footprint_translate_xyz=ARENA_DROID_STAND_TRANSLATE_XYZ,
                footprint_xy_m=ARENA_DROID_STAND_FOOTPRINT_XY_M,
            )
            isaac_env_cfg.scene.robot.spawn.usd_path = composed

            if episode_length_s is not None:
                isaac_env_cfg.episode_length_s = float(episode_length_s)
            if sim_dt is not None:
                isaac_env_cfg.sim.dt = float(sim_dt)
            if decimation is not None:
                isaac_env_cfg.decimation = int(decimation)
                isaac_env_cfg.sim.render_interval = isaac_env_cfg.decimation
            if table_cam is not None:
                h = int(table_cam.height)
                w = int(table_cam.width)
                isaac_env_cfg.scene.external_camera.height = h
                isaac_env_cfg.scene.external_camera.width = w
                isaac_env_cfg.scene.external_camera_2.height = h
                isaac_env_cfg.scene.external_camera_2.width = w
            if wrist_cam is not None:
                isaac_env_cfg.scene.wrist_camera.height = int(wrist_cam.height)
                isaac_env_cfg.scene.wrist_camera.width = int(wrist_cam.width)
            if eval_cam is not None:
                isaac_env_cfg.scene.eval_camera.height = int(eval_cam.height)
                isaac_env_cfg.scene.eval_camera.width = int(eval_cam.width)
                if getattr(eval_cam, "pos", None) is not None:
                    isaac_env_cfg.scene.eval_camera.offset.pos = tuple(eval_cam.pos)
                if getattr(eval_cam, "rot", None) is not None:
                    isaac_env_cfg.scene.eval_camera.offset.rot = tuple(eval_cam.rot)
                if getattr(eval_cam, "eye", None) is not None:
                    isaac_env_cfg.viewer.eye = tuple(eval_cam.eye)
                if getattr(eval_cam, "lookat", None) is not None:
                    isaac_env_cfg.viewer.lookat = tuple(eval_cam.lookat)

            env = gym.make(
                env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab
