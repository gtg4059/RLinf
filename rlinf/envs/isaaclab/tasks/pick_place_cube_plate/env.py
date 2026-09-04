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

"""RLinf wrapper for the DROID cube→bowl Isaac Lab task (Arena maple table)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

GYM_ID = "Isaac-PickPlace-Cube-Plate-Droid-AbsJointPos-v0"

# DROID RLDS observation cameras:
#   exterior_image_1_left, exterior_image_2_left, wrist_image_left
# See https://droid-dataset.github.io/droid/the-droid-dataset
_DEFAULT_MAIN_IMAGE_KEYS = (
    "external_camera",
    "exterior_image_1_left",
    "table_cam",
    "external_cam",
    "main_cam",
)
_DEFAULT_EXTERIOR2_IMAGE_KEYS = (
    "external_camera_2",
    "exterior_image_2_left",
    "external_cam_2",
)
_DEFAULT_WRIST_IMAGE_KEYS = (
    "wrist_camera",
    "wrist_cam",
    "wrist_image_left",
    "wrist_cam_left",
)
_DEFAULT_JOINT_KEYS = (
    "joint_pos",
    "arm_joint_pos",
    "joint_position",
)
_DEFAULT_JOINT_VEL_KEYS = (
    "joint_vel",
    "arm_joint_vel",
    "joint_velocity",
)
_DEFAULT_GRIPPER_KEYS = (
    "gripper_pos",
    "gripper_position",
)
_DEFAULT_EVAL_IMAGE_KEYS = (
    "eval_camera",
)


def _as_batched_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value.copy())
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device)


def _lookup_first(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _lookup_image(
    image_src: dict[str, Any], keys: tuple[str, ...]
) -> Any:
    """Lookup an image key, including Arena ``*_rgb`` suffixes."""
    image = _lookup_first(image_src, keys)
    if image is None:
        image = _lookup_first(image_src, tuple(f"{k}_rgb" for k in keys))
    return image


def _select_policy_main_image(
    main_t: torch.Tensor,
    exterior2_t: torch.Tensor | None,
    use_exterior2: torch.Tensor | None,
) -> torch.Tensor:
    """Pick the OpenPI base image, matching DROID RLDS 50:50 exterior sampling.

    ``use_exterior2`` is a per-env bool mask. Physical ``exterior2_t`` is left
    unchanged for logging; only the policy ``main_images`` slot is swapped.
    """
    if use_exterior2 is None or exterior2_t is None:
        return main_t
    mask = use_exterior2.reshape(-1).bool()
    if main_t.shape != exterior2_t.shape:
        return main_t
    if main_t.dim() == 3:
        return exterior2_t if bool(mask[0]) else main_t
    view_shape = (main_t.shape[0],) + (1,) * (main_t.dim() - 1)
    return torch.where(mask.view(view_shape), exterior2_t, main_t)


def wrap_droid_obs(
    obs: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    task_description: str,
    main_image_keys: tuple[str, ...] = _DEFAULT_MAIN_IMAGE_KEYS,
    exterior2_image_keys: tuple[str, ...] = _DEFAULT_EXTERIOR2_IMAGE_KEYS,
    wrist_image_keys: tuple[str, ...] = _DEFAULT_WRIST_IMAGE_KEYS,
    joint_keys: tuple[str, ...] = _DEFAULT_JOINT_KEYS,
    joint_vel_keys: tuple[str, ...] = _DEFAULT_JOINT_VEL_KEYS,
    gripper_keys: tuple[str, ...] = _DEFAULT_GRIPPER_KEYS,
    eval_image_keys: tuple[str, ...] = _DEFAULT_EVAL_IMAGE_KEYS,
    use_exterior2: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Map DROID / Arena-style Isaac Lab obs to RLinf canonical keys.

    Emits the three DROID RLDS cameras:
    ``main_images`` (policy base: exterior_1, or exterior_2 when sampled),
    ``extra_view_images`` (physical exterior_2 as ``[N, 1, H, W, C]``), and
    ``wrist_images``.
    """
    policy = obs.get("policy", obs)
    camera_obs = obs.get("camera_obs", {})
    image_src = {**camera_obs, **policy}

    main_image = _lookup_image(image_src, main_image_keys)
    exterior2_image = _lookup_image(image_src, exterior2_image_keys)
    wrist_image = _lookup_image(image_src, wrist_image_keys)
    eval_image = _lookup_image(image_src, eval_image_keys)
    if main_image is None:
        raise KeyError(
            "Missing DROID exterior_image_1_left / main camera in Isaac Lab obs. "
            f"Tried keys={main_image_keys}. Available={list(image_src.keys())}"
        )

    state = policy.get("state")
    if state is not None:
        states = _as_batched_tensor(state, device).float()
    else:
        joint = _lookup_first(policy, joint_keys)
        gripper = _lookup_first(policy, gripper_keys)
        if joint is None or gripper is None:
            raise KeyError(
                "Missing DROID proprioception. Provide `state` (8,) or "
                f"joint keys={joint_keys} and gripper keys={gripper_keys}. "
                f"Available={list(policy.keys())}"
            )
        joint_t = _as_batched_tensor(joint, device).float()
        gripper_t = _as_batched_tensor(gripper, device).float()
        if gripper_t.dim() == 1:
            gripper_t = gripper_t.unsqueeze(-1)
        if gripper_t.shape[-1] != 1:
            gripper_t = gripper_t.reshape(gripper_t.shape[0], -1)[:, :1]
        states = torch.cat([joint_t, gripper_t], dim=-1)

    if states.shape[-1] != 8:
        raise ValueError(
            f"DROID state must be 8-D (joint7+gripper1), got shape {tuple(states.shape)}"
        )

    main_t = _as_batched_tensor(main_image, device)
    exterior2_t = (
        _as_batched_tensor(exterior2_image, device)
        if exterior2_image is not None
        else None
    )
    env_obs: dict[str, Any] = {
        "main_images": _select_policy_main_image(main_t, exterior2_t, use_exterior2),
        "states": states,
        "task_descriptions": [task_description] * num_envs,
        "extra_view_images": None,
        "exterior2_images": None,
        "wrist_images": None,
        "joint_vel": None,
        "cri": None,
    }
    joint_vel = _lookup_first(policy, joint_vel_keys)
    if joint_vel is not None:
        env_obs["joint_vel"] = _as_batched_tensor(joint_vel, device).float()
    if wrist_image is not None:
        env_obs["wrist_images"] = _as_batched_tensor(wrist_image, device)
    if exterior2_t is not None:
        # Flat 4D for still/video logging; 5D for multi-view pipelines.
        env_obs["exterior2_images"] = exterior2_t
        if exterior2_t.dim() == 4:
            env_obs["extra_view_images"] = exterior2_t.unsqueeze(1)
        else:
            env_obs["extra_view_images"] = exterior2_t
    # Optional third-person viewer (not a DROID dataset camera).
    if eval_image is not None:
        env_obs["eval_images"] = _as_batched_tensor(eval_image, device)
    return env_obs


def register_pick_place_cube_plate_env() -> str:
    """Register the Isaac Lab gym id if it is not already present."""
    if GYM_ID not in gym.envs.registry:
        gym.register(
            id=GYM_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env_cfg:"
                    "PickPlaceCubePlateEnvCfg"
                ),
            },
        )
    return GYM_ID


class IsaaclabPickPlaceCubePlateEnv(IsaaclabBaseEnv):
    """Cube-on-plate pick-and-place with DROID abs joint-pos actions."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        init_params = cfg.init_params
        self._main_image_keys = tuple(
            init_params.get("main_image_keys", _DEFAULT_MAIN_IMAGE_KEYS)
        )
        self._exterior2_image_keys = tuple(
            init_params.get("exterior2_image_keys", _DEFAULT_EXTERIOR2_IMAGE_KEYS)
        )
        self._wrist_image_keys = tuple(
            init_params.get("wrist_image_keys", _DEFAULT_WRIST_IMAGE_KEYS)
        )
        self._joint_keys = tuple(init_params.get("joint_keys", _DEFAULT_JOINT_KEYS))
        self._joint_vel_keys = tuple(
            init_params.get("joint_vel_keys", _DEFAULT_JOINT_VEL_KEYS)
        )
        self._gripper_keys = tuple(
            init_params.get("gripper_keys", _DEFAULT_GRIPPER_KEYS)
        )
        # IsaacLab CRI-F: one run_cri_filter(q, qd_nom) per env.step.
        # Policy CRI is the previous-tick cri_pre cache (reset/first obs = 0).
        self._compute_cri = bool(init_params.get("compute_cri", True))
        self._cri_filter = bool(init_params.get("cri_filter", True))
        self._cri_limit = float(init_params.get("cri_limit", 0.96))
        self._cbf_alpha = float(init_params.get("cbf_alpha", 0.02))
        self._cri_filter_enabled = bool(init_params.get("cri_filter_enabled", False))
        self._cri_penalty_weight = float(init_params.get("cri_penalty_weight", -0.02))
        self._cri_penalty_limit = float(init_params.get("cri_penalty_limit", 0.96))
        self._cri_penalty_sigma = float(init_params.get("cri_penalty_sigma", 20.0))
        self._cri_ovf_threshold = float(init_params.get("cri_ovf_threshold", 2.0))
        self._cri_step_dt = float(init_params.get("cri_step_dt", 0.02))
        self._cri_solver = None
        self._cri_obs_cache = None
        self._last_q = None
        self._cri_solve_count = 0
        self._cri_step_count = 0
        self._last_task_success = None
        self._last_cri_max = None
        self._last_cri_ovf = None
        # Match openpi DROID RLDS: sample one exterior view per episode
        # (tf.random.uniform() > 0.5 in droid_rlds_dataset.restructure).
        self._sample_exterior_camera = bool(
            init_params.get("sample_exterior_camera", False)
        )
        self._sample_exterior_camera_prob = float(
            init_params.get("sample_exterior_camera_prob", 0.5)
        )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self._exterior_rng = torch.Generator()
        self._exterior_rng.manual_seed(int(self.seed))
        self._use_exterior2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._init_cri_buffers()

    def _make_env_function(self):
        num_envs = int(self.cfg.init_params.num_envs)
        seed = int(self.seed)
        env_id = self.isaaclab_env_id
        table_cam = getattr(self.cfg.init_params, "table_cam", None)
        wrist_cam = getattr(self.cfg.init_params, "wrist_cam", None)
        eval_cam = getattr(self.cfg.init_params, "eval_cam", None)
        episode_length_s = getattr(self.cfg.init_params, "episode_length_s", None)

        def make_env_isaaclab():
            import os

            # Force headless to avoid GLX errors in worker subprocesses.
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # Local import: env_cfg pulls Isaac Lab and requires AppLauncher first.
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env_cfg import (
                PickPlaceCubePlateEnvCfg,
            )

            if env_id not in gym.envs.registry:
                gym.register(
                    id=env_id,
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    disable_env_checker=True,
                    kwargs={
                        "env_cfg_entry_point": (
                            "rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env_cfg:"
                            "PickPlaceCubePlateEnvCfg"
                        ),
                    },
                )
            isaac_env_cfg = PickPlaceCubePlateEnvCfg()
            isaac_env_cfg.seed = seed
            isaac_env_cfg.scene.num_envs = num_envs

            if episode_length_s is not None:
                isaac_env_cfg.episode_length_s = float(episode_length_s)
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

    def _resample_exterior_camera(self, env_ids: torch.Tensor | None = None) -> None:
        """Bernoulli-sample which exterior camera is the policy base image."""
        if not self._sample_exterior_camera:
            return
        if env_ids is None:
            n = self.num_envs
            samples = torch.rand(n, generator=self._exterior_rng)
            self._use_exterior2 = (samples < self._sample_exterior_camera_prob).to(
                device=self.device
            )
            return
        ids = torch.as_tensor(env_ids, device="cpu").reshape(-1)
        samples = torch.rand(int(ids.numel()), generator=self._exterior_rng)
        self._use_exterior2[ids.to(device=self.device)] = (
            samples < self._sample_exterior_camera_prob
        ).to(device=self.device)

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        self._reset_cri_filter_rows(env_ids)
        self._resample_exterior_camera(env_ids)
        obs, infos = super().reset(seed=seed, env_ids=env_ids)
        self._remember_q(obs)
        return obs, infos

    def _init_cri_buffers(self) -> None:
        from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
            DEFAULT_NUM_JOINTS,
            NUM_CRI_POINTS,
        )

        self._cri_obs_cache = torch.zeros(
            self.num_envs, NUM_CRI_POINTS, device=self.device, dtype=torch.float32
        )
        self._last_q = torch.zeros(
            self.num_envs, DEFAULT_NUM_JOINTS, device=self.device, dtype=torch.float32
        )
        self._cri_solve_count = 0
        self._cri_step_count = 0

    def _reset_cri_filter_rows(self, env_ids: torch.Tensor | None = None) -> None:
        """IsaacLab CRI-F reset: CRI obs=0, no solver."""
        if self._cri_obs_cache is None:
            return
        self._last_task_success = None
        self._last_cri_max = None
        self._last_cri_ovf = None
        if env_ids is None:
            self._cri_obs_cache.zero_()
            self._last_q.zero_()
            self._cri_solve_count = 0
            self._cri_step_count = 0
            return
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        self._cri_obs_cache.index_fill_(0, ids, 0.0)
        self._last_q.index_fill_(0, ids, 0.0)

    def _remember_q(self, env_obs: dict[str, Any]) -> None:
        if self._last_q is None or env_obs.get("states") is None:
            return
        self._last_q.copy_(env_obs["states"][:, : self._last_q.shape[-1]])

    def _wrap_obs(self, obs):
        use_exterior2 = self._use_exterior2 if self._sample_exterior_camera else None
        env_obs = wrap_droid_obs(
            obs,
            num_envs=self.num_envs,
            device=self.device,
            task_description=self.task_description,
            main_image_keys=self._main_image_keys,
            exterior2_image_keys=self._exterior2_image_keys,
            wrist_image_keys=self._wrist_image_keys,
            joint_keys=self._joint_keys,
            joint_vel_keys=self._joint_vel_keys,
            gripper_keys=self._gripper_keys,
            use_exterior2=use_exterior2,
        )
        return self._attach_cri(env_obs)

    def _get_cri_solver(self):
        if self._cri_solver is None:
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri import CriSolver

            self._cri_solver = CriSolver(
                batch_size=max(int(self.num_envs), 1),
                device=self.device,
                cri_filter=True,
                cri_limit=self._cri_limit,
                cbf_alpha=self._cbf_alpha,
                filter_enabled=self._cri_filter_enabled,
            )
        return self._cri_solver

    def _attach_cri(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Attach previous-tick ``cri_filter_pre``. Does not call the solver."""
        if not self._compute_cri:
            return env_obs
        if self._cri_obs_cache is None:
            self._init_cri_buffers()
        env_obs["cri"] = self._cri_obs_cache.to(
            device=self.device, dtype=torch.float32
        )
        return env_obs

    def _store_cri_pre(self, cri_pre: torch.Tensor) -> None:
        if self._cri_obs_cache is None:
            self._init_cri_buffers()
        cri = cri_pre.to(device=self.device, dtype=torch.float32)
        if cri.dim() == 1:
            cri = cri.unsqueeze(0)
        self._cri_obs_cache.copy_(cri[: self.num_envs])

    def _solve_cri_for_action(self, actions: Any) -> torch.Tensor:
        """One ``run_cri_filter(q, qd_nom)`` for this env.step."""
        from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.filter import (
            abs_joint_to_qd_nom,
        )

        act = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        q_tgt = act[..., :7]
        qd_nom = abs_joint_to_qd_nom(q_tgt, self._last_q, self._cri_step_dt)
        self._cri_step_count += 1
        self._cri_solve_count += 1
        result = self._get_cri_solver().run_cri_filter(self._last_q, qd_nom)
        cri = torch.as_tensor(
            result["cri_pre"], device=self.device, dtype=torch.float32
        )
        if cri.dim() == 1:
            cri = cri.unsqueeze(0)
        return cri[: self.num_envs]

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        if self._last_task_success is None:
            task_hit = step_reward > 0
        else:
            task_hit = self._last_task_success
        self.success_once = self.success_once | task_hit
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        if self._last_cri_max is not None:
            episode_info["cri_max"] = self._last_cri_max
        if self._last_cri_ovf is not None:
            episode_info["cri_ovf"] = self._last_cri_ovf
        if self._cri_step_count > 0:
            ratio = float(self._cri_solve_count) / float(self._cri_step_count)
            if self._cri_obs_cache is not None:
                episode_info["cri_solves_per_step"] = self._cri_obs_cache.new_full(
                    (self.num_envs,), ratio
                )
            else:
                episode_info["cri_solves_per_step"] = torch.full(
                    (self.num_envs,),
                    ratio,
                    device=self.device,
                    dtype=torch.float32,
                )
        infos["episode"] = episode_info
        return infos

    def step(self, actions=None, auto_reset=True):
        from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
            cri_ovf_exp,
        )

        cri_pre = None
        if actions is not None and self._compute_cri:
            cri_pre = self._solve_cri_for_action(actions)

        obs, step_reward, terminations, truncations, infos = self.env.step(actions)

        terminations = terminations.clone()
        truncations = truncations.clone()
        r_task = step_reward.clone()
        step_reward = r_task.clone()

        if isinstance(infos, dict):
            success = infos.get("success", infos.get("is_success"))
            if success is not None:
                success_t = torch.as_tensor(success, device=terminations.device).bool()
                if success_t.ndim == 0:
                    success_t = success_t.expand_as(terminations)
                terminations = terminations | success_t.reshape_as(terminations)

        ovf = None
        if cri_pre is not None and self._cri_penalty_weight != 0.0:
            ovf = cri_ovf_exp(
                cri_pre,
                limit=self._cri_penalty_limit,
                sigma=self._cri_penalty_sigma,
                ovf_threshold=self._cri_ovf_threshold,
            )
            step_reward = r_task + self._cri_penalty_weight * ovf.to(
                device=r_task.device, dtype=r_task.dtype
            )

        self._last_task_success = r_task > 0
        self._last_cri_max = None if cri_pre is None else cri_pre.amax(dim=-1)
        self._last_cri_ovf = ovf

        obs = self._wrap_obs(obs)
        self._remember_q(obs)
        if cri_pre is not None:
            self._store_cri_pre(cri_pre)
        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations

        infos = self._record_metrics(step_reward, terminations, {})
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos
