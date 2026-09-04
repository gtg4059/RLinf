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

"""Tests for CRI PPO train launcher and Isaac Lab Docker image resolution."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "examples/embodiment/scripts/train_cri_openpi_ckpt.sh"
MOUNTS_SCRIPT = REPO_ROOT / "docker/runtime_mounts.sh"
CONFIG_NAME = "isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri"


def _run_bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _write_docker_shim(directory: Path, present_tags: set[str]) -> Path:
    bin_dir = directory / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    cases = "\n".join(f'    "{tag}") exit 0 ;;' for tag in sorted(present_tags))
    docker.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then\n'
        "  case \"$3\" in\n"
        f"{cases}\n"
        "    *) exit 1 ;;\n"
        "  esac\n"
        "fi\n"
        "exit 1\n"
    )
    docker.chmod(stat.S_IRWXU)
    return bin_dir


def _resolve_isaaclab_image(tmp_path: Path, present_tags: set[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    bin_dir = _write_docker_shim(tmp_path, present_tags)
    path = f"{bin_dir}:{os.environ.get('PATH', '')}"
    completed = _run_bash(
        f"source '{MOUNTS_SCRIPT}' && rlinf_resolve_isaaclab_image",
        env={"PATH": path, "IMAGE_TAG": "", **(env or {})},
    )
    return completed.returncode, completed.stdout.strip()


def test_prefers_u24_image_when_present(tmp_path: Path) -> None:
    code, tag = _resolve_isaaclab_image(
        tmp_path,
        {"rlinf:embodied-isaaclab-u24", "rlinf:embodied-isaaclab-blackwell"},
    )
    assert code == 0
    assert tag == "rlinf:embodied-isaaclab-u24"


def test_falls_back_to_blackwell_image(tmp_path: Path) -> None:
    code, tag = _resolve_isaaclab_image(tmp_path, {"rlinf:embodied-isaaclab-blackwell"})
    assert code == 0
    assert tag == "rlinf:embodied-isaaclab-blackwell"


def test_image_tag_env_wins(tmp_path: Path) -> None:
    code, tag = _resolve_isaaclab_image(
        tmp_path,
        {"rlinf:embodied-isaaclab-u24"},
        env={"IMAGE_TAG": "rlinf:custom-isaaclab"},
    )
    assert code == 0
    assert tag == "rlinf:custom-isaaclab"


def test_train_script_help_exits_zero() -> None:
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "train_cri_openpi_ckpt.sh" in completed.stdout
    assert CONFIG_NAME in completed.stdout


def test_train_script_dry_run_uses_checkpoint_and_config(tmp_path: Path) -> None:
    ckpt = REPO_ROOT / "checkpoint" / "pi05_droid_cri_rlinf_49999"
    if not (ckpt / "model.safetensors").is_file():
        import pytest

        pytest.skip("converted OpenPI CRI checkpoint is not present")

    bin_dir = _write_docker_shim(tmp_path, {"rlinf:embodied-isaaclab-u24"})
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "runner.max_epochs=2"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "RLINF_TRAIN_CRI_DRY_RUN": "1",
            "IMAGE_TAG": "",
        },
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    assert f"CONFIG_NAME={CONFIG_NAME}" in stdout
    assert f"CRI_OPENPI_CKPT={ckpt}" in stdout
    assert "IMAGE_TAG=rlinf:embodied-isaaclab-u24" in stdout
    assert "LAUNCH_MODE=docker" in stdout
    assert "run_embodied_isaaclab_blackwell.sh" in stdout
    assert "run_embodiment.sh" in stdout
    assert CONFIG_NAME in stdout
    assert "runner.max_epochs=2" in stdout


def _resolve_isaac_sim(repo_root: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    completed = _run_bash(
        f"source '{MOUNTS_SCRIPT}' && rlinf_resolve_isaac_sim '{repo_root}'",
        env=env,
    )
    return completed.returncode, completed.stdout.strip()


def test_resolve_isaac_sim_skips_broken_symlink_and_uses_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "setup_conda_env.sh").write_text("#!/bin/bash\n")
    (repo / "VERSION").write_text("5.1.0\n")
    (repo / "isaac_sim").symlink_to("/mnt/E/isaac_sim")

    code, resolved = _resolve_isaac_sim(repo, env={"ISAAC_SIM_PATH": "", "ISAAC_PATH": ""})
    assert code == 0
    assert resolved == str(repo.resolve())


def test_resolve_isaac_sim_prefers_explicit_env(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other_sim"
    repo.mkdir()
    other.mkdir()
    (repo / "setup_conda_env.sh").write_text("#!/bin/bash\n")
    (repo / "VERSION").write_text("5.1.0\n")
    (other / "setup_conda_env.sh").write_text("#!/bin/bash\n")
    (other / "VERSION").write_text("5.1.0\n")

    code, resolved = _resolve_isaac_sim(
        repo, env={"ISAAC_SIM_PATH": str(other), "ISAAC_PATH": ""}
    )
    assert code == 0
    assert resolved == str(other.resolve())


def test_resolve_isaac_sim_finds_this_checkout() -> None:
    if not ((REPO_ROOT / "setup_conda_env.sh").is_file() and (REPO_ROOT / "VERSION").is_file()):
        import pytest

        pytest.skip("Isaac Sim tree is not extracted into this checkout")
    code, resolved = _resolve_isaac_sim(
        REPO_ROOT, env={"ISAAC_SIM_PATH": "", "ISAAC_PATH": ""}
    )
    assert code == 0
    assert Path(resolved).resolve() == REPO_ROOT.resolve()


def test_train_script_honors_explicit_ckpt_override(tmp_path: Path) -> None:
    override = tmp_path / "custom_ckpt"
    override.mkdir()
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_CRI_DRY_RUN": "1",
            "CRI_OPENPI_CKPT": str(override),
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert f"CRI_OPENPI_CKPT={override}" in completed.stdout
