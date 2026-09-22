#!/usr/bin/env python3
"""Dump one reset frame of fridge DROID cameras (no policy)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.pop("DISPLAY", None)

from isaaclab.app import AppLauncher

sim_app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import numpy as np
from PIL import Image

from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.compose_on_stand import (
    compose_droid_on_stand_arena,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg import (
    OpenFridgeKitchenEnvCfg,
    _resolve_arena_assets_root,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.materials import (
    patch_fridge_material_terminals,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.placement import (
    arena_open_fridge_robot_pos,
)


def _save_rgb(path: Path, img) -> None:
    if hasattr(img, "detach"):
        img = img.detach().cpu()
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    out = Path("/workspace/RLinf/logs/fridge_cam_dump")
    out.mkdir(parents=True, exist_ok=True)

    cfg = OpenFridgeKitchenEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0
    cfg.scene.robot.spawn.usd_path = compose_droid_on_stand_arena(
        _resolve_arena_assets_root(), stand_height_m=0.8
    )
    cfg.scene.robot.init_state.pos = arena_open_fridge_robot_pos()
    print("robot_pos", cfg.scene.robot.init_state.pos)
    print("robot_rot", cfg.scene.robot.init_state.rot)
    print("ext1", cfg.scene.external_camera.offset.rot)
    print("ext2", cfg.scene.external_camera_2.offset.rot)

    env_id = "Isaac-OpenFridge-Kitchen-Dump-v0"
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": OpenFridgeKitchenEnvCfg},
        )
    env = gym.make(env_id, cfg=cfg, render_mode="rgb_array").unwrapped
    arena_root = _resolve_arena_assets_root()
    patch_fridge_material_terminals(
        env.sim.stage,
        "/World/envs/env_0/Kitchen",
        f"{arena_root}/background_library/lightwheel_kitchen_one_wall_coastal/"
        "textures/T_Refrigerator032_BC001_0.png",
    )
    obs, _ = env.reset()
    policy = obs["policy"] if "policy" in obs else obs
    for key in ("external_camera", "external_camera_2", "wrist_camera", "eval_camera"):
        _save_rgb(out / f"{key}.png", policy[key])
        print("wrote", out / f"{key}.png")
    env.close()


if __name__ == "__main__":
    main()
