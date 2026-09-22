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

import torch

from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door import door_reached_from_rest
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
        self.reach_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_door_openness = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _wrap_obs(self, obs):
        policy = obs.get("policy", obs) if isinstance(obs, dict) else {}
        openness = policy.get("door_openness") if isinstance(policy, dict) else None
        if openness is not None:
            self._last_door_openness = torch.as_tensor(
                openness, device=self.device, dtype=torch.float32
            ).reshape(-1)
        return super()._wrap_obs(obs)

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if getattr(self, "reach_once", None) is None:
            return
        if env_idx is not None:
            ids = torch.as_tensor(env_idx, device=self.device).reshape(-1)
            self.reach_once.index_fill_(0, ids, False)
            return
        self.reach_once.zero_()

    def _record_metrics(self, step_reward, terminations, infos):
        infos = super()._record_metrics(step_reward, terminations, infos)
        openness = getattr(self, "_last_door_openness", None)
        if openness is None:
            return infos
        self.reach_once = self.reach_once | door_reached_from_rest(openness)
        episode = infos.setdefault("episode", {})
        episode["reach_once"] = self.reach_once.clone()
        episode["door_openness"] = openness.clone()
        return infos

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
            # DLSS / post-AA each allocate RTX ParameterBlocks per camera.
            # Kitchen CameraCfg views already sit near the descriptor-set limit.
            try:
                import carb

                settings = carb.settings.get_settings()
                settings.set("/rtx/post/dlss/enabled", False)
                settings.set("/rtx/post/aa/algo", 0)
            except Exception:
                pass

            from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg import (
                OpenFridgeKitchenEnvCfg,
                _resolve_arena_assets_root,
            )
            from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.compose_on_stand import (
                compose_droid_on_stand_arena,
            )
            from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.materials import (
                patch_fridge_material_terminals,
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
            composed = compose_droid_on_stand_arena(
                arena_root, stand_height_m=stand_height_m
            )
            isaac_env_cfg.scene.robot.spawn.usd_path = composed
            robot_pos = getattr(self.cfg.init_params, "robot_pos", None)
            robot_rot = getattr(self.cfg.init_params, "robot_rot", None)
            if robot_pos is not None:
                isaac_env_cfg.scene.robot.init_state.pos = tuple(robot_pos)
            if robot_rot is not None:
                isaac_env_cfg.scene.robot.init_state.rot = tuple(robot_rot)

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
            albedo = (
                f"{arena_root}/background_library/"
                "lightwheel_kitchen_one_wall_coastal/textures/"
                "T_Refrigerator032_BC001_0.png"
            )
            patched = 0
            for env_i in range(num_envs):
                patched += patch_fridge_material_terminals(
                    env.sim.stage,
                    f"/World/envs/env_{env_i}/Kitchen",
                    albedo,
                )
            print(
                f"[open_fridge] kitchen_usd={isaac_env_cfg.scene.kitchen.spawn.usd_path} "
                f"fridge_meshes_rebound={patched}",
                flush=True,
            )
            return env, sim_app

        return make_env_isaaclab
