#!/usr/bin/env python3
"""Compare CRI(q, qd_nom) vs CRI(q, measured qd) on recorded eval trajs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.constants import (
    DROID_CONTROL_DT,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.rewards import cri_ovf_exp
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.cri.solver import CriSolver


def _row_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def _load_frames(root: Path, dt: float) -> dict[str, np.ndarray]:
    q_pre, q_post, qd_nom, qd_meas, cri_log = [], [], [], [], []
    hold, lift = [], []
    for npz in sorted(root.glob("seed_*/step_*/*_traj.npz")):
        z = np.load(npz)
        q = z["q"].reshape(-1, 7).astype(np.float64)
        qd = z["qd"].reshape(-1, 7).astype(np.float64)
        act = z["action"].reshape(-1, 8).astype(np.float64)
        cri = z["cri"].reshape(-1, 9).astype(np.float32)
        g = z["gripper"].reshape(-1)
        t_len = q.shape[0]
        if t_len < 2:
            continue
        pre = q[:-1]
        post = q[1:]
        nom = (act[1:, :7] - pre) / dt
        meas = qd[1:]
        logged = cri[1:]
        g1 = g[1:]
        cmd_g = act[1:, 7]
        dq1 = post[:, 1] - pre[:, 1]
        held = (g1 > 0.15) & (cmd_g > 0.5)
        q_pre.append(pre)
        q_post.append(post)
        qd_nom.append(nom)
        qd_meas.append(meas)
        cri_log.append(logged)
        hold.append(held)
        lift.append(held & (dq1 < -0.003))
    return {
        "q_pre": np.concatenate(q_pre, axis=0),
        "q_post": np.concatenate(q_post, axis=0),
        "qd_nom": np.concatenate(qd_nom, axis=0),
        "qd_meas": np.concatenate(qd_meas, axis=0),
        "cri_log": np.concatenate(cri_log, axis=0),
        "hold": np.concatenate(hold, axis=0),
        "lift": np.concatenate(lift, axis=0),
    }


def _valid_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    nom_n = _row_norm(data["qd_nom"])
    meas_n = _row_norm(data["qd_meas"])
    jump = _row_norm(data["q_post"] - data["q_pre"])
    return (
        np.isfinite(nom_n)
        & np.isfinite(meas_n)
        & (nom_n < 3.0)
        & (meas_n < 3.0)
        & (jump < 0.5)
    )


def _solve_cri(solver: CriSolver, q: np.ndarray, qd: np.ndarray, batch: int) -> np.ndarray:
    out = []
    for start in range(0, q.shape[0], batch):
        end = min(start + batch, q.shape[0])
        cri = solver.run_cri_filter(q[start:end], qd[start:end])["cri_pre"]
        if cri.ndim == 1:
            cri = cri[None, :]
        out.append(np.asarray(cri, dtype=np.float32))
    return np.concatenate(out, axis=0)


def _amax(cri: np.ndarray) -> np.ndarray:
    return np.nanmax(cri, axis=-1)


def _ovf(cri: np.ndarray) -> np.ndarray:
    return cri_ovf_exp(torch.from_numpy(cri)).numpy()


def _stats(name: str, old: np.ndarray, new: np.ndarray) -> dict[str, float]:
    delta = new - old
    return {
        "n": float(old.size),
        "old_mean": float(old.mean()),
        "old_med": float(np.median(old)),
        "old_p90": float(np.quantile(old, 0.9)),
        "old_ge096": float((old >= 0.96).mean()),
        "new_mean": float(new.mean()),
        "new_med": float(np.median(new)),
        "new_p90": float(np.quantile(new, 0.9)),
        "new_ge096": float((new >= 0.96).mean()),
        "delta_mean": float(delta.mean()),
        "delta_med": float(np.median(delta)),
        "abs_delta_mean": float(np.abs(delta).mean()),
        "abs_delta_med": float(np.median(np.abs(delta))),
        "abs_delta_p90": float(np.quantile(np.abs(delta), 0.9)),
    }


def _print_stats(title: str, s: dict[str, float]) -> None:
    print(f"\n== {title}  n={int(s['n'])} ==")
    print(
        f"  old  mean={s['old_mean']:.3f} med={s['old_med']:.3f} "
        f"p90={s['old_p90']:.3f} >=0.96={s['old_ge096']*100:.1f}%"
    )
    print(
        f"  new  mean={s['new_mean']:.3f} med={s['new_med']:.3f} "
        f"p90={s['new_p90']:.3f} >=0.96={s['new_ge096']*100:.1f}%"
    )
    print(
        f"  new-old  mean={s['delta_mean']:+.3f} med={s['delta_med']:+.3f}  "
        f"|d| mean={s['abs_delta_mean']:.3f} med={s['abs_delta_med']:.3f} "
        f"p90={s['abs_delta_p90']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=Path(
            "logs/20260913-04:08:26-isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri/video/eval"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dt = DROID_CONTROL_DT
    data = _load_frames(args.traj_root, dt)
    ok = _valid_mask(data)
    print(f"frames={ok.size} valid={int(ok.sum())} skipped={int((~ok).sum())}")
    print(
        f"|qd_nom| med={np.median(_row_norm(data['qd_nom'][ok])):.3f}  "
        f"|qd_meas| med={np.median(_row_norm(data['qd_meas'][ok])):.3f}  "
        f"ratio med={np.median(_row_norm(data['qd_nom'][ok]) / np.maximum(_row_norm(data['qd_meas'][ok]), 1e-6)):.2f}"
    )

    solver = CriSolver(
        batch_size=args.batch_size,
        device=args.device,
        cri_filter=True,
        filter_enabled=False,
        warmup_rounds=2,
    )

    q_pre = data["q_pre"][ok]
    q_post = data["q_post"][ok]
    qd_nom = data["qd_nom"][ok]
    qd_meas = data["qd_meas"][ok]
    cri_log = data["cri_log"][ok]
    hold = data["hold"][ok]
    lift = data["lift"][ok]

    cri_old = _solve_cri(solver, q_pre, qd_nom, args.batch_size)
    cri_new = _solve_cri(solver, q_post, qd_meas, args.batch_size)
    cri_sameq_nom = _solve_cri(solver, q_post, qd_nom, args.batch_size)

    old_max = _amax(cri_old)
    new_max = _amax(cri_new)
    sameq_nom_max = _amax(cri_sameq_nom)
    meas_max = _amax(cri_new)
    log_max = _amax(cri_log)

    recon_err = np.abs(old_max - log_max)
    print(
        f"\nrecompute old vs logged cri_max: "
        f"mae={recon_err.mean():.3f} med={np.median(recon_err):.3f} "
        f"p90={np.quantile(recon_err, 0.9):.3f}"
    )

    _print_stats("pipeline  CRI(q_pre, qd_nom) vs CRI(q_post, qd_meas)", _stats("p", old_max, new_max))
    _print_stats("velocity-only at q_post  qd_nom vs qd_meas", _stats("v", sameq_nom_max, meas_max))
    _print_stats("pipeline ovf", _stats("po", _ovf(cri_old), _ovf(cri_new)))
    _print_stats("velocity-only ovf", _stats("vo", _ovf(cri_sameq_nom), _ovf(cri_new)))

    if hold.any():
        _print_stats(
            "HOLD pipeline cri_max",
            _stats("h", old_max[hold], new_max[hold]),
        )
    if lift.any():
        _print_stats(
            "LIFT (held, q1 down) pipeline cri_max",
            _stats("l", old_max[lift], new_max[lift]),
        )
        _print_stats(
            "LIFT velocity-only cri_max",
            _stats("lv", sameq_nom_max[lift], meas_max[lift]),
        )


if __name__ == "__main__":
    main()
