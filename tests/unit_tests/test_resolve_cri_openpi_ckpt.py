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

"""Tests for checkpoint/ vs checkpoints/ OpenPI CRI path resolution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOLVE_SCRIPT = REPO_ROOT / "examples/embodiment/scripts/resolve_cri_openpi_ckpt.sh"


def _resolve(repo: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    cmd = f"source '{RESOLVE_SCRIPT}' && resolve_cri_openpi_ckpt '{repo}'"
    completed = subprocess.run(
        ["bash", "-c", cmd],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CRI_OPENPI_CKPT": "", **(env or {})},
    )
    return completed.returncode, completed.stdout.strip()


def _touch_weights(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"stub")


def test_prefers_checkpoint_over_checkpoints(tmp_path: Path) -> None:
    preferred = tmp_path / "checkpoint" / "pi05_droid_cri_rlinf_49999"
    legacy = tmp_path / "checkpoints" / "pi05_droid_cri_rlinf_49999"
    _touch_weights(preferred)
    _touch_weights(legacy)

    code, path = _resolve(tmp_path)
    assert code == 0
    assert path == str(preferred)


def test_falls_back_to_checkpoints(tmp_path: Path) -> None:
    legacy = tmp_path / "checkpoints" / "pi05_droid_cri_rlinf_49999"
    _touch_weights(legacy)

    code, path = _resolve(tmp_path)
    assert code == 0
    assert path == str(legacy)


def test_honors_explicit_cri_openpi_ckpt(tmp_path: Path) -> None:
    preferred = tmp_path / "checkpoint" / "pi05_droid_cri_rlinf_49999"
    override = tmp_path / "custom_ckpt"
    _touch_weights(preferred)
    override.mkdir()

    code, path = _resolve(tmp_path, env={"CRI_OPENPI_CKPT": str(override)})
    assert code == 0
    assert path == str(override)


def test_missing_checkpoint_returns_nonzero(tmp_path: Path) -> None:
    code, path = _resolve(tmp_path)
    assert code != 0
    assert path == ""
