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

"""Keep Isaac Sim's bundled torch off actor/rollout ``PYTHONPATH``.

``setup_python_env.sh`` appends ``omni.isaac.ml_archive/pip_prebundle``. That
directory shadows the OpenPI venv torch. Isaac's torch is not Blackwell-safe
and FSDP/NCCL then fails with illegal memory access.

Env workers still need the rest of the Isaac PYTHONPATH (kit, simulation_app,
core_archive). Only the ML archive is removed for training/inference workers.
"""

from __future__ import annotations

import os
from typing import Mapping

ISAAC_ML_ARCHIVE_MARKERS = ("omni.isaac.ml_archive",)
_ENV_WORKER_MODULE_PREFIXES = ("rlinf.workers.env",)
_SCRUB_ENV_DISABLE = "RLINF_KEEP_ISAAC_ML_ARCHIVE"
_PATH_KEYS = ("PYTHONPATH",)


def entry_is_isaac_ml_archive(entry: str) -> bool:
    """Return True if a path entry is Isaac Sim's ML pip prebundle."""
    norm = (entry or "").replace("\\", "/")
    return any(marker in norm for marker in ISAAC_ML_ARCHIVE_MARKERS)


def scrub_isaac_ml_archive_path(value: str, pathsep: str = os.pathsep) -> str:
    """Drop ``omni.isaac.ml_archive`` segments from a path-like env value."""
    kept = [
        entry
        for entry in value.split(pathsep)
        if entry and not entry_is_isaac_ml_archive(entry)
    ]
    return pathsep.join(kept)


def scrub_isaac_ml_archive_env(
    env_vars: Mapping[str, str],
) -> dict[str, str]:
    """Copy ``env_vars`` with Isaac ML-archive entries removed from path vars."""
    out = dict(env_vars)
    for key in _PATH_KEYS:
        if key not in out:
            continue
        cleaned = scrub_isaac_ml_archive_path(out[key])
        if cleaned:
            out[key] = cleaned
        else:
            del out[key]
    return out


def worker_keeps_isaac_ml_archive(worker_cls: type) -> bool:
    """Env workers keep Isaac's torch; actor/rollout must not."""
    module = getattr(worker_cls, "__module__", "") or ""
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in _ENV_WORKER_MODULE_PREFIXES
    )


def should_scrub_isaac_ml_archive(worker_cls: type) -> bool:
    """Whether Cluster.allocate should strip the ML archive for this worker."""
    raw = (os.environ.get(_SCRUB_ENV_DISABLE) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return False
    return not worker_keeps_isaac_ml_archive(worker_cls)


def torch_is_isaac_build(torch_file: str | None = None) -> bool:
    """True if the imported torch came from Isaac Sim's ML archive."""
    if torch_file is None:
        try:
            import torch
        except ImportError:
            return False
        torch_file = getattr(torch, "__file__", "") or ""
    return entry_is_isaac_ml_archive(torch_file) or "isaac_sim" in torch_file.replace(
        "\\", "/"
    )
