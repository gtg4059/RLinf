#!/usr/bin/env python3
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

"""Standalone CRI worker process (Isaac Sim torch 2.7, not OpenPI torch).

Talks a length-prefixed pickle protocol on stdin plus a dedicated write fd
(``CRI_IPC_WRITE_FD``). Import path is the ``cri`` package itself so
``rlinf.envs.isaaclab`` (Isaac env wrappers) is never loaded.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any
from typing import BinaryIO

_ipc_in: BinaryIO | None = None
_ipc_out: BinaryIO | None = None
_ipc_mod: Any = None


def _scrub_sys_path() -> None:
    kept: list[str] = []
    for entry in sys.path:
        norm = entry.replace("\\", "/")
        # Keep Isaac's ml_archive so SafetyCore can resolve torch 2.7 c10.
        if "isaac_sim" in norm and "omni.isaac.ml_archive" not in norm:
            continue
        if "/kit/python/" in norm or "/python_packages" in norm:
            continue
        kept.append(entry)
    isaac_site = (os.environ.get("CRI_ISAAC_TORCH_SITE") or "").strip()
    if isaac_site:
        kept = [p for p in kept if p != isaac_site]
        kept.insert(0, isaac_site)
    cri_parent = str(Path(__file__).resolve().parent.parent)
    if cri_parent not in kept:
        kept.insert(0, cri_parent)
    sys.path[:] = kept


def _load_ipc():
    """Load ``ipc.py`` by path so ``cri/__init__.py`` (torch) is not imported yet."""
    global _ipc_mod
    if _ipc_mod is not None:
        return _ipc_mod
    import importlib.util

    path = Path(__file__).resolve().parent / "ipc.py"
    spec = importlib.util.spec_from_file_location("_cri_ipc_standalone", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ipc_mod = module
    return module


def _bind_ipc() -> None:
    """Pickle replies on the dedicated fd; print/C++ stdout go to stderr."""
    global _ipc_in, _ipc_out
    ipc = _load_ipc()

    _ipc_in = sys.stdin.buffer
    raw_fd = os.environ.get(ipc.CRI_IPC_WRITE_FD_ENV, "").strip()
    if raw_fd:
        _ipc_out = os.fdopen(int(raw_fd), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
        return
    _ipc_out = sys.stdout.buffer


def _send(obj: object) -> None:
    assert _ipc_out is not None
    _load_ipc().pickle_send(_ipc_out, obj)


def _recv() -> object:
    assert _ipc_in is not None
    return _load_ipc().pickle_recv(_ipc_in)


def main() -> int:
    _scrub_sys_path()
    _bind_ipc()
    try:
        import numpy as np
        import torch
        from cri.solver import CriSolver

        kwargs = _recv()
        if not isinstance(kwargs, dict):
            raise TypeError(f"expected init dict, got {type(kwargs)}")
        solver = CriSolver(inprocess=True, **kwargs)
        _send({"ok": True, "torch": torch.__file__, "cuda": torch.version.cuda})
    except Exception as exc:
        try:
            _send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, OSError, ValueError):
            traceback.print_exc()
        return 1

    while True:
        try:
            msg = _recv()
        except EOFError:
            return 0
        if not msg or msg.get("op") == "stop":
            return 0
        op = msg.get("op")
        if op not in ("compute", "run_cri_filter"):
            _send({"error": f"unknown op {op}"})
            continue
        try:
            if op == "run_cri_filter":
                result = solver.run_cri_filter(msg["q"], msg["qd"])
                _send(
                    {
                        "cri_pre": np.asarray(result["cri_pre"], dtype=np.float32),
                        "qd_cmd": np.asarray(result["qd_cmd"], dtype=np.float32),
                        "delta": np.asarray(result["delta"], dtype=np.float32),
                        "cri_limit": float(result["cri_limit"]),
                        "cbf_alpha": float(result["cbf_alpha"]),
                        "approach_limit": float(result["approach_limit"]),
                        "enabled": bool(result["enabled"]),
                    }
                )
                continue
            cri = solver.compute(msg["q"], msg["qd"], return_torch=False)
            _send({"cri": np.asarray(cri, dtype=np.float32)})
        except Exception as exc:
            _send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


if __name__ == "__main__":
    raise SystemExit(main())
