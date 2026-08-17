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

"""Standalone CRI worker process (venv torch, not Isaac Sim torch).

Talks a length-prefixed pickle protocol on stdin/stdout. Import path is the
``cri`` package itself so ``rlinf.envs.isaaclab`` (Isaac env wrappers) is never
loaded.
"""

from __future__ import annotations

import pickle
import struct
import sys
import traceback
from pathlib import Path


def _scrub_sys_path() -> None:
    kept: list[str] = []
    for entry in sys.path:
        norm = entry.replace("\\", "/")
        if "isaac_sim" in norm or "omni.isaac.ml_archive" in norm:
            continue
        kept.append(entry)
    cri_parent = str(Path(__file__).resolve().parent.parent)
    if cri_parent not in kept:
        kept.insert(0, cri_parent)
    sys.path[:] = kept


def _send(obj: object) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack("<Q", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _recv() -> object:
    header = sys.stdin.buffer.read(8)
    if len(header) < 8:
        raise EOFError("CRI worker stdin closed")
    (n,) = struct.unpack("<Q", header)
    data = sys.stdin.buffer.read(n)
    if len(data) < n:
        raise EOFError("CRI worker stdin truncated")
    return pickle.loads(data)


def main() -> int:
    _scrub_sys_path()
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
        _send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 1

    while True:
        try:
            msg = _recv()
        except EOFError:
            return 0
        if not msg or msg.get("op") == "stop":
            return 0
        if msg.get("op") != "compute":
            _send({"error": f"unknown op {msg.get('op')}"})
            continue
        try:
            cri = solver.compute(msg["q"], msg["qd"], return_torch=False)
            _send({"cri": np.asarray(cri, dtype=np.float32)})
        except Exception as exc:
            _send({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


if __name__ == "__main__":
    raise SystemExit(main())
