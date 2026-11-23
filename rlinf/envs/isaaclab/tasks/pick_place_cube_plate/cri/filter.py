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

"""CRI-F helpers matching IsaacLab ``articulation_data`` CBF-QP filter mode.

- One ``run_cri_filter(q, qd_RL)`` per tick; policy CRI is ``cri_pre``.
- Command is library ``qd_cmd``. A point over the force/pressure limit adds
  one row ``g = J^T dir``, and the QP moves only along that row.
- A joint the QP leaves on the request keeps its absolute target.
- Time-scale ``s`` is not used. ``delta = ||qd_cmd - qd_RL||``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .constants import CBF_ALPHA
from .constants import CRI_FILTER_LIMIT
from .constants import DEFAULT_NUM_JOINTS
from .constants import DROID_CONTROL_DT
from .constants import NUM_CRI_POINTS
from .velocity import joint_velocity_from_positions


def abs_joint_to_qd_nom(
    q_tgt: np.ndarray | torch.Tensor,
    q: np.ndarray | torch.Tensor,
    dt: float,
) -> np.ndarray | torch.Tensor:
    """Convert absolute joint targets to IsaacLab ``qd_nom = (q_tgt - q) / dt``."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if isinstance(q_tgt, torch.Tensor) or isinstance(q, torch.Tensor):
        tgt = torch.as_tensor(q_tgt)
        cur = torch.as_tensor(q, device=tgt.device, dtype=tgt.dtype)
        return (tgt - cur) / dt
    return (np.asarray(q_tgt) - np.asarray(q)) / dt


# Franka Panda implicit actuators in the fridge task: joints 1–4 at
# 2.175 rad/s, joints 5–7 at 2.61 rad/s.
DEFAULT_ARM_QD_LIMIT = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)


def clip_arm_velocity(
    qd: torch.Tensor,
    limit: torch.Tensor | tuple[float, ...] = DEFAULT_ARM_QD_LIMIT,
) -> torch.Tensor:
    """Clamp each arm joint velocity to the actuator limit.

    The position error ``(q_tgt - q) / dt`` can ask for a speed the Panda
    drive will never track. The CRI filter should see the fastest command
    the actuator can actually execute.
    """
    lim = torch.as_tensor(limit, device=qd.device, dtype=qd.dtype).reshape(-1)
    if int(lim.numel()) == 1:
        lim = lim.expand(qd.shape[-1])
    lim = lim[: qd.shape[-1]]
    return qd.clamp(-lim, lim)


def joint_command_changed(
    qd_req: torch.Tensor,
    qd_cmd: torch.Tensor,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> torch.Tensor:
    """True for joints the linear-velocity projection moved.

    The library projects ``qd_req`` along ``g = J^T dir`` so the reduced
    quantity is the point's approach speed ``dir · v_lin``. A joint the
    projection leaves on the request has a zero entry in every active row.

    Args:
        qd_req: Clipped requested arm velocity, ``(N, 7)``.
        qd_cmd: Library command, same shape.
        rel_tol: Relative change that counts as a QP move.
        abs_tol: Absolute change that counts as a QP move.

    Returns:
        A boolean mask, ``(N, 7)``.
    """
    req = torch.as_tensor(qd_req)
    cmd = torch.as_tensor(qd_cmd, device=req.device, dtype=req.dtype)
    delta = (cmd - req).abs()
    scale = req.abs().clamp_min(1e-6)
    tol = torch.maximum(rel_tol * scale, torch.full_like(scale, abs_tol))
    return delta > tol


def apply_filtered_arm_targets(
    actions: torch.Tensor,
    q: torch.Tensor,
    qd_cmd: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Replace the 7 arm targets with ``q + qd_cmd * dt``.

    ``actions`` is ``(N, >=7)`` absolute joint positions. A gripper channel
    after the arm is left as written. When ``qd_cmd`` is
    ``(q_tgt - q) / dt``, the arm targets come back unchanged.

    Args:
        actions: Policy or hold targets, ``(N, >=7)``.
        q: Current arm joint positions, ``(N, 7)``.
        qd_cmd: Filtered joint velocity, ``(N, 7)``.
        dt: Control period in seconds.

    Returns:
        A copy of ``actions`` whose arm columns track ``qd_cmd``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    out = actions.clone()
    cmd = torch.as_tensor(qd_cmd, device=out.device, dtype=out.dtype)
    cur = torch.as_tensor(q, device=out.device, dtype=out.dtype)
    if cmd.ndim == 1:
        cmd = cmd.unsqueeze(0)
    if cur.ndim == 1:
        cur = cur.unsqueeze(0)
    n = out.shape[0]
    out[:, :DEFAULT_NUM_JOINTS] = cur[:n, :DEFAULT_NUM_JOINTS] + (
        cmd[:n, :DEFAULT_NUM_JOINTS] * dt
    )
    return out


def shift_cri_filter_obs(
    cri_pre: np.ndarray,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Policy-obs delay: t=0 is CRI=0 (and extra=0); later ticks read the previous solve."""
    cri = np.asarray(cri_pre, dtype=np.float32)
    if cri.ndim == 1:
        cri = cri[None, :]
    t_len = cri.shape[0]
    cri_obs = np.zeros_like(cri, dtype=np.float32)
    extra_obs = None
    if extra is not None:
        extra_arr = np.asarray(extra, dtype=np.float32).reshape(t_len, -1)
        extra_obs = np.zeros_like(extra_arr, dtype=np.float32)
    if t_len > 1:
        cri_obs[1:] = cri[:-1]
        if extra_obs is not None:
            extra_obs[1:] = extra_arr[:-1]
    return cri_obs, extra_obs


def command_delta(qd_cmd: np.ndarray | None, qd_rl: np.ndarray) -> np.ndarray:
    """Per-row ``||qd_cmd - qd_RL||`` matching IsaacLab ``apply_cri_filter``."""
    qd = np.asarray(qd_rl, dtype=np.float32)
    if qd.ndim == 1:
        qd = qd[None, :]
    if qd_cmd is None:
        return np.zeros((qd.shape[0],), dtype=np.float32)
    cmd = np.asarray(qd_cmd, dtype=np.float32)
    if cmd.ndim == 1:
        cmd = cmd[None, :]
    if cmd.shape != qd.shape:
        cmd = np.broadcast_to(cmd, qd.shape)
    return np.linalg.norm(cmd - qd, axis=-1).astype(np.float32, copy=False)


def compute_episode_cri_f(
    joint_positions: np.ndarray,
    *,
    solver: Any,
    dt: float = DROID_CONTROL_DT,
    cri_limit: float = CRI_FILTER_LIMIT,
    cbf_alpha: float = CBF_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(qd_rl, cri_obs, delta_obs, qd_cmd)`` via ``run_cri_filter``.

    ``cri_obs`` / ``delta_obs`` are 1-step delayed (IsaacLab CRI-F policy CRI).
    ``qd_cmd`` is the instantaneous CBF-QP command (same length as ``qd_rl``).
    """
    del cri_limit, cbf_alpha
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 2 or q.shape[-1] < DEFAULT_NUM_JOINTS:
        raise ValueError(f"expected joint_positions (T, >=7), got {q.shape}")
    q = q[:, :DEFAULT_NUM_JOINTS]
    qd = joint_velocity_from_positions(q, dt=dt)
    if not hasattr(solver, "run_cri_filter"):
        raise RuntimeError("solver.run_cri_filter is missing")

    t_len = q.shape[0]
    batch = int(getattr(solver, "batch_size", t_len) or t_len)
    cri_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    cmd_chunks: list[np.ndarray] = []
    for start in range(0, t_len, batch):
        end = min(start + batch, t_len)
        result = solver.run_cri_filter(q[start:end], qd[start:end])
        cri_pre = np.asarray(result["cri_pre"], dtype=np.float32)
        if cri_pre.ndim == 1:
            cri_pre = cri_pre[None, :]
        if cri_pre.shape[-1] != NUM_CRI_POINTS:
            raise ValueError(f"expected CRI width {NUM_CRI_POINTS}, got {cri_pre.shape}")
        qd_rl = qd[start:end].astype(np.float32, copy=False)
        qd_cmd = result.get("qd_cmd")
        if qd_cmd is None:
            qd_cmd = qd_rl
        else:
            qd_cmd = np.asarray(qd_cmd, dtype=np.float32)
            if qd_cmd.ndim == 1:
                qd_cmd = qd_cmd[None, :]
        delta = result.get("delta")
        if delta is None:
            delta = command_delta(qd_cmd, qd_rl)
        else:
            delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        cri_chunks.append(cri_pre)
        delta_chunks.append(delta)
        cmd_chunks.append(qd_cmd)

    cri_inst = np.concatenate(cri_chunks, axis=0)
    delta_inst = np.concatenate(delta_chunks, axis=0)
    qd_cmd = np.concatenate(cmd_chunks, axis=0)
    cri_obs, delta_obs = shift_cri_filter_obs(cri_inst, delta_inst)
    assert delta_obs is not None
    return qd, cri_obs, delta_obs.reshape(-1), qd_cmd
