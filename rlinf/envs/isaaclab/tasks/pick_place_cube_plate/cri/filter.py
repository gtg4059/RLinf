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
- The call also returns ``impact_jacobian`` ``{robot, points, 6, joints}``
  and ``impact_dir`` ``{robot, points, 3}``.
- The point with the largest CRI supplies ``J`` and ``dir``. Its approach
  ``dir · v_lin`` is removed and the tangential velocity at that point stays.
- The uniform speed scale is off unless ``apply_speed_scale`` is set.
  At or above the point limit the direction-change vector is applied with
  weight ``min(1, direction_gain * peak / point_limit)``, so the configured
  gain is the floor and a higher peak uses more of the cut.
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


def subtract_approach_velocity(
    qd: torch.Tensor,
    jacobian: torch.Tensor | np.ndarray | None,
    direction: torch.Tensor | np.ndarray | None,
    cri: torch.Tensor | np.ndarray | None,
    *,
    hard_limit: float = 0.96,
    point_limit: float = 0.8,
    strong_cri: float = 0.92,
    strong_target: float = 0.0095,
    apply_speed_scale: bool = False,
    direction_gain: float = 1.0,
) -> torch.Tensor:
    """Remove approach speed. Scale joint speed only when asked.

    That point's linear Jacobian and impact direction give::

        v_lin = impact_jacobian[..., :3, :] @ qd
        approach = (impact_dir * v_lin).sum(-1)
        v_lin_tangential = v_lin - approach * impact_dir

    With the speed scale off, a 9-channel peak at or above ``point_limit``
    applies ``min(1, direction_gain * peak / point_limit)`` of the joint
    change that removes the highest point's incoming approach. At the limit
    that weight equals ``direction_gain``; a higher peak uses more, up to the
    full cut. Tangential speed stays, including a peak already over
    ``hard_limit``. With the speed scale on, the full cut stays under
    ``hard_limit`` and, at or above ``strong_cri``, is followed by
    ``min(1, strong_target / peak)`` on every joint. Motion away from the
    point keeps its sign. A row under ``point_limit`` keeps ``qd``.

    Args:
        qd: Requested joint velocity, ``(N, 7)``.
        jacobian: Per-point spatial Jacobian, ``(N, P, 6, joints)``.
        direction: Impact direction, ``(N, P, 3)``.
        cri: Per-point CRI, ``(N, C)``. The row peak uses every channel.
        hard_limit: Mild-path ceiling. Rows at or above it skip the mild cut.
        point_limit: The peak point loses its approach at or above this.
        strong_cri: 9-channel peak where the speed scale also applies.
        strong_target: CRI the speed scale aims at once ``strong_cri`` is hit.
        apply_speed_scale: When false, only the approach component is removed.
        direction_gain: Floor weight of the approach-removal vector at
            ``point_limit``. A weight of 1 applies the full cut.

    Returns:
        Joint velocity with that point's incoming approach removed. The
        uniform speed scale runs only when ``apply_speed_scale`` is set.
    """
    out = torch.as_tensor(qd, dtype=torch.float32).clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if jacobian is None or direction is None or cri is None:
        return out
    jac = torch.as_tensor(jacobian, dtype=torch.float32, device=out.device)
    dire = torch.as_tensor(direction, dtype=torch.float32, device=out.device)
    cri_t = torch.as_tensor(cri, dtype=torch.float32, device=out.device)
    if jac.ndim == 3:
        jac = jac.unsqueeze(0)
    if dire.ndim == 2:
        dire = dire.unsqueeze(0)
    if cri_t.ndim == 1:
        cri_t = cri_t.unsqueeze(0)
    if jac.ndim != 4 or jac.shape[-2] < 3 or dire.shape[-1] < 3:
        return out
    n = out.shape[0]
    if int(cri_t.shape[0]) == 1 and n > 1:
        cri_t = cri_t.expand(n, -1)
    if int(jac.shape[0]) == 1 and n > 1:
        jac = jac.expand(n, -1, -1, -1)
    elif int(jac.shape[0]) < n:
        n = int(jac.shape[0])
    if int(dire.shape[0]) == 1 and n > 1:
        dire = dire.expand(n, -1, -1)
    elif int(dire.shape[0]) < n:
        n = int(dire.shape[0])
    if int(cri_t.shape[0]) < n:
        n = int(cri_t.shape[0])
    points = min(int(jac.shape[1]), int(dire.shape[1]), int(cri_t.shape[-1]))
    joints = min(int(out.shape[-1]), int(jac.shape[-1]))
    if n == 0 or points == 0 or joints == 0:
        return out
    peak = cri_t[:n].amax(dim=-1)
    speed_on = bool(apply_speed_scale)
    if speed_on:
        strong = peak >= float(strong_cri)
        mild = (peak < float(hard_limit)) & ~strong
        if not bool((strong | mild).any()):
            return out
    else:
        strong = torch.zeros(n, dtype=torch.bool, device=out.device)
        mild = torch.ones(n, dtype=torch.bool, device=out.device)
    point_cri = cri_t[:n, :points]
    peak_idx = point_cri.argmax(dim=-1)
    batch = torch.arange(n, device=out.device)
    j_lin = jac[:n, :points, :3, :joints]
    impact_dir = dire[:n, :points, :3][batch, peak_idx]
    impact_dir = impact_dir / impact_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qd_j = out[:n, :joints]
    # v_lin = impact_jacobian[..., :3, :] @ qd
    # approach = (impact_dir * v_lin).sum(-1)
    # v_lin_tangential = v_lin - approach * impact_dir
    v_lin = torch.matmul(j_lin[batch, peak_idx], qd_j.unsqueeze(-1))
    approach = (impact_dir * v_lin.squeeze(-1)).sum(-1, keepdim=True)
    v_lin_tangential = v_lin.squeeze(-1) - approach * impact_dir
    entering = approach.squeeze(-1) > 0.0
    finite = torch.isfinite(v_lin_tangential).all(dim=-1)
    point_hot = point_cri[batch, peak_idx] >= float(point_limit)
    if not speed_on:
        # The 9-channel peak is the direction gate. Speed scaling stays off.
        point_hot = peak >= float(point_limit)
    drop_approach = (strong | (mild & point_hot)) & entering & finite
    delta_v = (v_lin_tangential - v_lin.squeeze(-1)).unsqueeze(-1)
    delta_v = torch.where(
        drop_approach.view(n, 1, 1), delta_v, torch.zeros_like(delta_v)
    )
    dq = torch.matmul(torch.linalg.pinv(j_lin[batch, peak_idx]), delta_v).squeeze(-1)
    if speed_on:
        qd_new = qd_j + dq
    else:
        # At point_limit the weight is direction_gain. Above it the weight rises to 1.
        gain = (
            float(direction_gain) * peak / max(float(point_limit), 1e-6)
        ).clamp(max=1.0)
        qd_new = qd_j + dq * gain.unsqueeze(-1)
    if speed_on:
        speed_scale = (float(strong_target) / peak.clamp_min(1e-6)).clamp(max=1.0)
        qd_new = torch.where(
            strong.unsqueeze(-1), qd_new * speed_scale.unsqueeze(-1), qd_new
        )
    lim = torch.as_tensor(DEFAULT_ARM_QD_LIMIT, device=out.device, dtype=out.dtype)
    qd_new = qd_new.clamp(-lim[:joints], lim[:joints])
    use = drop_approach | strong
    out[:n, :joints] = torch.where(use.unsqueeze(-1), qd_new, qd_j)
    return out


def joint_command_changed(
    qd_req: torch.Tensor,
    qd_cmd: torch.Tensor,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-4,
) -> torch.Tensor:
    """True for joints the approach removal moved.

    The correction realizes ``v_lin - (dir · v_lin) dir`` at the highest-CRI
    point, so the reduced quantity is that point's approach speed. A joint
    the correction leaves on the request stays at the absolute target when
    no joint on the row moves.

    Args:
        qd_req: Clipped requested arm velocity, ``(N, 7)``.
        qd_cmd: Joint velocity after the approach removal, same shape.
        rel_tol: Relative change that counts as a moved joint.
        abs_tol: Absolute change that counts as a moved joint.

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
