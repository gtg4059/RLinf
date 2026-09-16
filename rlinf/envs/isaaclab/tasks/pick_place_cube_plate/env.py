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
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.hold import (
    apply_hold_actions,
    clone_obs,
    overwrite_frozen_obs,
    update_hold_actions,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.isaac_reset import (
    isaac_reset_row_ids,
    set_chunk_hold,
    zero_rows,
)

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


def measured_arm_q_qd(
    obs: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IsaacLab measured arm ``(q, qd)`` for CRI.

    ``q`` is observed joint positions (``states[..., :7]``).
    ``qd`` is Isaac ``robot.data.joint_vel`` exposed as policy ``joint_vel``.
    """
    states = obs.get("states")
    if states is None:
        raise RuntimeError("CRI needs observed joint positions in obs['states']")
    q = torch.as_tensor(states, device=device, dtype=torch.float32)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = q[:num_envs, :7]
    if q.shape[-1] != 7:
        raise RuntimeError(f"CRI needs 7 arm joints in states, got shape {tuple(q.shape)}")
    qd = obs.get("joint_vel")
    if qd is None:
        raise RuntimeError(
            "CRI needs IsaacLab measured joint_vel in obs. "
            "Policy ObservationsCfg must include arm_joint_vel."
        )
    qd_t = torch.as_tensor(qd, device=device, dtype=torch.float32)
    if qd_t.dim() == 1:
        qd_t = qd_t.unsqueeze(0)
    qd_t = qd_t[:num_envs, :7]
    if qd_t.shape != q.shape:
        raise RuntimeError(
            f"CRI q/qd shape mismatch: q={tuple(q.shape)} qd={tuple(qd_t.shape)}"
        )
    return q, qd_t


def build_traj_info(
    *,
    states: torch.Tensor | None,
    actions: Any = None,
    cri: torch.Tensor | None = None,
    cri_ovf: torch.Tensor | None = None,
    joint_vel: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-step robot / CRI arrays aligned with a RecordVideo frame.

    ``cri`` is this-tick ``cri_pre`` (the motion that produced the frame),
    not the delayed policy observation.
    """
    traj: dict[str, torch.Tensor] = {}
    if states is not None:
        state_t = torch.as_tensor(states, dtype=torch.float32)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        traj["states"] = state_t
        traj["q"] = state_t[..., :7]
        if state_t.shape[-1] > 7:
            traj["gripper"] = state_t[..., 7:8]
    if actions is not None:
        action_t = torch.as_tensor(actions, dtype=torch.float32)
        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)
        traj["action"] = action_t
    if cri is not None:
        cri_t = torch.as_tensor(cri, dtype=torch.float32)
        if cri_t.dim() == 1:
            cri_t = cri_t.unsqueeze(0)
        traj["cri"] = cri_t
    if joint_vel is not None:
        qd_t = torch.as_tensor(joint_vel, dtype=torch.float32)
        if qd_t.dim() == 1:
            qd_t = qd_t.unsqueeze(0)
        traj["qd"] = qd_t
    return traj


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
        # IsaacLab CRI-F: one run_cri_filter(q, qd) per env.step from
        # measured observations (joint_pos, joint_vel). Policy CRI is the
        # previous-tick cri_pre cache (reset/first obs = 0).
        self._compute_cri = bool(init_params.get("compute_cri", True))
        self._cri_filter = bool(init_params.get("cri_filter", True))
        self._cri_limit = float(init_params.get("cri_limit", 0.96))
        self._cbf_alpha = float(init_params.get("cbf_alpha", 0.02))
        self._cri_filter_enabled = bool(init_params.get("cri_filter_enabled", False))
        self._cri_penalty_weight = float(init_params.get("cri_penalty_weight", -0.02))
        self._cri_penalty_limit = float(init_params.get("cri_penalty_limit", 0.96))
        self._cri_penalty_sigma = float(init_params.get("cri_penalty_sigma", 20.0))
        self._cri_ovf_threshold = float(init_params.get("cri_ovf_threshold", 2.0))
        self._cri_ovf_termination_enabled = bool(
            init_params.get("cri_ovf_termination_enabled", False)
        )
        term_threshold = init_params.get("cri_ovf_term_threshold", None)
        self._cri_ovf_term_threshold = float(
            self._cri_penalty_limit
            if term_threshold is None
            else term_threshold
        )
        term_penalty = init_params.get("cri_ovf_term_penalty", None)
        if term_penalty is None:
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
                cri_ovf_termination_step_penalty,
            )

            self._cri_ovf_term_penalty = cri_ovf_termination_step_penalty()
        else:
            self._cri_ovf_term_penalty = float(term_penalty)
        self._time_penalty_weight = float(init_params.get("time_penalty_weight", 0.0))
        from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
            DROID_CONTROL_DT,
        )

        self._cri_step_dt = float(init_params.get("cri_step_dt", DROID_CONTROL_DT))
        self._cri_solver = None
        self._cri_obs_cache = None
        self._last_q = None
        self._cri_solve_count = 0
        self._cri_step_count = 0
        self._last_task_success = None
        self._last_cri_ovf_term = None
        self._last_time_penalty = None
        # Match openpi DROID RLDS: sample one exterior view per episode
        # (tf.random.uniform() > 0.5 in droid_rlds_dataset.restructure).
        self._sample_exterior_camera = bool(
            init_params.get("sample_exterior_camera", False)
        )
        self._sample_exterior_camera_prob = float(
            init_params.get("sample_exterior_camera_prob", 0.5)
        )
        # False: Isaac Lab resets done envs inside step(); RLinf keeps the
        # 450-step cycle so pick-place can succeed more than once.
        # True: freeze leftover steps (no Isaac reset) until bootstrap reset.
        self._hold_after_done = bool(init_params.get("hold_after_done", False))
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
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_actions = None
        self._hold_obs = None
        self._success_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._cri_ovf_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._cri_penalty_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._cri_ovf_term_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._time_penalty_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._chunk_hold = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._chunk_hold_actions = None
        self._init_cri_buffers()

    def chunk_step(self, chunk_actions):
        """Clear intra-chunk reset-hold so the next policy chunk can act."""
        self._chunk_hold.zero_()
        self._chunk_hold_actions = None
        return super().chunk_step(chunk_actions)

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if getattr(self, "_success_count", None) is None:
            return
        if env_idx is not None:
            ids = torch.as_tensor(env_idx, device=self.device).reshape(-1)
            self._success_count.index_fill_(0, ids, 0.0)
            if getattr(self, "_cri_ovf_sum", None) is not None:
                self._cri_ovf_sum.index_fill_(0, ids, 0.0)
                self._cri_penalty_sum.index_fill_(0, ids, 0.0)
            if getattr(self, "_cri_ovf_term_count", None) is not None:
                self._cri_ovf_term_count.index_fill_(0, ids, 0.0)
            if getattr(self, "_time_penalty_sum", None) is not None:
                self._time_penalty_sum.index_fill_(0, ids, 0.0)
            return
        self._success_count.zero_()
        if getattr(self, "_cri_ovf_sum", None) is not None:
            self._cri_ovf_sum.zero_()
            self._cri_penalty_sum.zero_()
        if getattr(self, "_cri_ovf_term_count", None) is not None:
            self._cri_ovf_term_count.zero_()
        if getattr(self, "_time_penalty_sum", None) is not None:
            self._time_penalty_sum.zero_()

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

        def make_env_isaaclab():
            import os

            # Force headless to avoid GLX errors in worker subprocesses.
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.scene_guard import (
                harden_isaac_scene,
                prepare_isaac_render,
            )

            maple_usd = prepare_isaac_render()

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
            if maple_usd.is_file():
                isaac_env_cfg.scene.table.spawn.usd_path = str(maple_usd)

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
            harden_isaac_scene(env)
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
        self._clear_hold(env_ids)
        if getattr(self, "_chunk_hold", None) is not None:
            self._chunk_hold.zero_()
            self._chunk_hold_actions = None
        self._remember_q(obs)
        self._hold_obs = clone_obs(obs)
        return obs, self._attach_traj_info(infos, obs)

    def _sync_after_isaac_reset(self, env_ids: torch.Tensor) -> None:
        """Refresh CRI/camera after Isaac's in-step reset. Keep RLinf metrics."""
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if int(ids.numel()) == 0:
            return
        if self._cri_obs_cache is not None:
            zero_rows(self._cri_obs_cache, ids)
        if self._last_q is not None:
            zero_rows(self._last_q, ids)
        self._resample_exterior_camera(ids)

    def _clear_hold(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear freeze-after-done state (full reset or selected env rows)."""
        if env_ids is None:
            self._frozen.zero_()
            self._hold_actions = None
            self._hold_obs = None
            return
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        self._frozen.index_fill_(0, ids, False)

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
        self._last_cri_ovf_term = None
        self._last_time_penalty = None
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

    def _attach_traj_info(
        self,
        infos: Any,
        obs: dict[str, Any] | None,
        actions: Any = None,
        cri_pre: torch.Tensor | None = None,
        ovf: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Attach per-step ``traj`` arrays for RecordVideo dumps."""
        payload = infos if isinstance(infos, dict) else {}
        states = None
        joint_vel = None
        if isinstance(obs, dict):
            states = obs.get("states")
            joint_vel = obs.get("joint_vel")
        cri = cri_pre
        if cri is None and self._cri_obs_cache is not None:
            cri = self._cri_obs_cache
        payload["traj"] = build_traj_info(
            states=states,
            actions=actions,
            cri=cri,
            cri_ovf=ovf,
            joint_vel=joint_vel,
        )
        return payload

    def _store_cri_pre(self, cri_pre: torch.Tensor) -> None:
        if self._cri_obs_cache is None:
            self._init_cri_buffers()
        cri = cri_pre.to(device=self.device, dtype=torch.float32)
        if cri.dim() == 1:
            cri = cri.unsqueeze(0)
        self._cri_obs_cache.copy_(cri[: self.num_envs])

    def _solve_cri_from_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        """One ``run_cri_filter(q, qd)`` from this-tick Isaac observations."""
        q, qd = measured_arm_q_qd(obs, num_envs=self.num_envs, device=self.device)
        self._cri_step_count += 1
        self._cri_solve_count += 1
        result = self._get_cri_solver().run_cri_filter(q, qd)
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
        self._success_count = self._success_count + task_hit.to(
            dtype=self._success_count.dtype
        )
        episode_info["success_once"] = self.success_once.clone()
        episode_info["success_count"] = self._success_count.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        if self._last_cri_ovf_term is not None:
            episode_info["cri_ovf_term"] = self._last_cri_ovf_term
            episode_info["cri_ovf_term_count"] = self._cri_ovf_term_count.clone()
        if getattr(self, "_cri_penalty_sum", None) is not None:
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
                cri_episode_reward_logs,
            )

            episode_info.update(cri_episode_reward_logs(self._cri_penalty_sum))
        if self._last_time_penalty is not None:
            episode_info["time_penalty"] = self._last_time_penalty
        if getattr(self, "_time_penalty_sum", None) is not None:
            from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
                time_episode_reward_logs,
            )

            episode_info.update(time_episode_reward_logs(self._time_penalty_sum))
        infos["episode"] = episode_info
        return infos

    def step(self, actions=None, auto_reset=True):
        from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import (
            accumulate_cri_episode_reward,
            accumulate_time_episode_reward,
            cri_ovf_exp,
            cri_ovf_violated,
            time_penalty_live_mask,
        )

        was_frozen = self._frozen.clone()
        hold_after_done = self._hold_after_done
        isaac_reset_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        cri_reset_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self._last_cri_ovf_term = None
        if hold_after_done and actions is not None:
            actions = apply_hold_actions(actions, self._hold_actions, was_frozen)
        elif (
            (not hold_after_done)
            and actions is not None
            and bool(self._chunk_hold.any())
        ):
            actions = apply_hold_actions(
                actions, self._chunk_hold_actions, self._chunk_hold
            )

        if hold_after_done and bool(was_frozen.all()) and self._hold_obs is not None:
            step_reward = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            )
            terminations = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self._elapsed_steps += 1
            truncations = self.elapsed_steps >= self.cfg.max_episode_steps
            infos = self._record_metrics(step_reward, terminations, {})
            infos = self._attach_traj_info(
                infos, self._hold_obs, actions=actions
            )
            return self._hold_obs, step_reward, terminations, truncations, infos

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

        # Isaac already reset these rows inside env.step. Do not reset RLinf
        # elapsed/return/success_once so the 450-step cycle can succeed again.
        isaac_reset_ids = isaac_reset_row_ids(terminations, truncations)
        if (not hold_after_done) and int(isaac_reset_ids.numel()) > 0:
            self._sync_after_isaac_reset(isaac_reset_ids)

        obs = self._wrap_obs(obs)
        if hold_after_done and bool(was_frozen.any()):
            obs = overwrite_frozen_obs(obs, self._hold_obs, was_frozen)

        cri_pre = None
        ovf = None
        if self._compute_cri:
            cri_pre = self._solve_cri_from_obs(obs)
            ovf = cri_ovf_exp(
                cri_pre,
                limit=self._cri_penalty_limit,
                sigma=self._cri_penalty_sigma,
                ovf_threshold=self._cri_ovf_threshold,
            )
            if self._cri_penalty_weight != 0.0:
                step_reward = r_task + self._cri_penalty_weight * ovf.to(
                    device=r_task.device, dtype=r_task.dtype
                )
            cri_hit = cri_ovf_violated(
                cri_pre, threshold=self._cri_ovf_term_threshold
            ).reshape(-1)
            self._last_cri_ovf_term = cri_hit.clone()
            if self._cri_ovf_termination_enabled and bool(cri_hit.any()):
                terminations = terminations | cri_hit.reshape_as(terminations)
                step_reward = step_reward + self._cri_ovf_term_penalty * cri_hit.to(
                    device=step_reward.device, dtype=step_reward.dtype
                )
                if not hold_after_done:
                    already_reset = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                    if int(isaac_reset_ids.numel()) > 0:
                        already_reset[isaac_reset_ids] = True
                    need_reset = cri_hit & ~already_reset
                    if bool(need_reset.any()):
                        cri_reset_ids = torch.nonzero(
                            need_reset, as_tuple=False
                        ).reshape(-1)
                        reset_obs_raw, _ = self.env.reset(env_ids=cri_reset_ids)
                        reset_obs = self._wrap_obs(reset_obs_raw)
                        obs = overwrite_frozen_obs(obs, reset_obs, need_reset)
                        self._sync_after_isaac_reset(cri_reset_ids)
                        cri_pre = cri_pre.clone()
                        cri_pre[cri_reset_ids] = 0.0

        if hold_after_done and bool(was_frozen.any()):
            r_task = r_task.clone()
            step_reward = step_reward.clone()
            frozen_r = was_frozen.to(device=r_task.device)
            r_task[frozen_r] = 0
            step_reward[frozen_r] = 0
            terminations[was_frozen.to(device=terminations.device)] = True
            if ovf is not None:
                ovf = ovf.clone()
                ovf[was_frozen.to(device=ovf.device)] = 0
            if self._last_cri_ovf_term is not None:
                self._last_cri_ovf_term = self._last_cri_ovf_term.clone()
                self._last_cri_ovf_term[was_frozen.to(device=self.device)] = False

        live_mask = time_penalty_live_mask(
            self.num_envs,
            hold_after_done=hold_after_done,
            was_frozen=was_frozen,
            success_once=getattr(self, "success_once", None),
            device=self.device,
        )
        time_pen = None
        if self._time_penalty_weight != 0.0:
            time_pen = step_reward.new_zeros(self.num_envs)
            time_pen[live_mask] = self._time_penalty_weight
            step_reward = step_reward + time_pen
            if getattr(self, "_time_penalty_sum", None) is not None:
                accumulate_time_episode_reward(
                    self._time_penalty_sum,
                    self._time_penalty_weight,
                    live_mask,
                )
        self._last_time_penalty = time_pen

        self._last_task_success = r_task > 0
        if self._last_cri_ovf_term is not None:
            self._cri_ovf_term_count = self._cri_ovf_term_count + (
                self._last_cri_ovf_term.to(dtype=self._cri_ovf_term_count.dtype)
            )
        if ovf is not None and getattr(self, "_cri_penalty_sum", None) is not None:
            accumulate_cri_episode_reward(
                self._cri_ovf_sum,
                self._cri_penalty_sum,
                ovf,
                self._cri_penalty_weight,
            )
            if (
                self._cri_ovf_termination_enabled
                and self._last_cri_ovf_term is not None
            ):
                self._cri_penalty_sum = self._cri_penalty_sum + (
                    self._cri_ovf_term_penalty
                    * self._last_cri_ovf_term.to(
                        device=self._cri_penalty_sum.device,
                        dtype=self._cri_penalty_sum.dtype,
                    )
                )

        self._remember_q(obs)
        post_reset_ids = isaac_reset_ids
        if int(cri_reset_ids.numel()) > 0:
            if int(post_reset_ids.numel()) == 0:
                post_reset_ids = cri_reset_ids
            else:
                post_reset_ids = torch.cat(
                    [post_reset_ids, cri_reset_ids]
                ).unique()
        if (
            (not hold_after_done)
            and int(post_reset_ids.numel()) > 0
            and obs.get("states") is not None
        ):
            self._chunk_hold, self._chunk_hold_actions = set_chunk_hold(
                self._chunk_hold,
                self._chunk_hold_actions,
                obs["states"],
                post_reset_ids,
            )
        if cri_pre is not None:
            self._store_cri_pre(cri_pre)
            if (not hold_after_done) and int(post_reset_ids.numel()) > 0:
                zero_rows(self._cri_obs_cache, post_reset_ids)
        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations

        infos = self._record_metrics(step_reward, terminations, {})
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        infos = self._attach_traj_info(
            infos, obs, actions=actions, cri_pre=cri_pre, ovf=ovf
        )

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
            self._clear_hold(torch.arange(self.num_envs, device=self.device)[dones])

        if hold_after_done:
            if actions is not None:
                self._hold_actions = update_hold_actions(
                    self._hold_actions, actions, was_frozen
                )
            self._hold_obs = clone_obs(obs)
            self._frozen = was_frozen | dones.to(device=was_frozen.device)

        return obs, step_reward, terminations, truncations, infos
