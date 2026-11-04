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

"""Tests for the fridge PolarIS install check."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPO_ROOT / "examples/embodiment/scripts/prepare_fridge_open_polaris.sh"
MOUNTS = REPO_ROOT / "docker/runtime_mounts.sh"

_RELATIVE_FILES = (
    "checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000/model.safetensors",
    "checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000/config.json",
    ".assets/isaaclab_arena/background_library/lightwheel_kitchen_one_wall_coastal/scene.usd",
    ".assets/isaaclab_arena/lightwheel_sdk/floorplan/robocasa-robocasakitchen-1-1/scene.usd",
    ".assets/isaaclab_arena/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
    ".assets/isaaclab_arena/object_library/srl_robolab_assets/robots/franka_stand_grey.usda",
    "rlinf/envs/isaaclab/tasks/pick_place_cube_plate/cri/lib/libSFD_CoreService.so",
    "rlinf/envs/isaaclab/tasks/pick_place_cube_plate/cri/Engine/Chest_Shape5_Cover_off_Force/model.onnx",
    "rlinf/envs/isaaclab/tasks/pick_place_cube_plate/cri/Engine/Chest_Shape5_Cover_off_Pressure/model.onnx",
)


def _fake_repo(tmp_path: Path, *, drop: str | None = None) -> Path:
    root = tmp_path / "repo"
    (root / "docker").mkdir(parents=True)
    (root / "docker" / "runtime_mounts.sh").symlink_to(MOUNTS)
    (root / "setup_conda_env.sh").write_text("#!/bin/bash\n")
    (root / "VERSION").write_text("5.1.0\n")
    for relative in _RELATIVE_FILES:
        if relative == drop:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return root


def _docker_shim(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/bash\nexit 1\n")
    docker.chmod(stat.S_IRWXU)
    return bin_dir


def _run(root: Path, bin_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RLINF_PREPARE_REPO_ROOT"] = str(root)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env.pop("ISAAC_SIM_PATH", None)
    env.pop("ISAAC_PATH", None)
    env.pop("IMAGE_TAG", None)
    return subprocess.run(
        ["bash", str(PREPARE), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_check_only_accepts_local_tree_without_image(tmp_path: Path) -> None:
    completed = _run(_fake_repo(tmp_path), _docker_shim(tmp_path), "--check-only")
    assert completed.returncode == 0, completed.stderr
    assert "local tree ready; image still missing" in completed.stdout
    assert "MISSING image=rlinf:embodied-isaaclab-u24" in completed.stderr


def test_check_only_rejects_missing_checkpoint(tmp_path: Path) -> None:
    root = _fake_repo(
        tmp_path,
        drop="checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000/model.safetensors",
    )
    completed = _run(root, _docker_shim(tmp_path), "--check-only")
    assert completed.returncode == 1
    assert "MISSING checkpoint/" in completed.stderr
