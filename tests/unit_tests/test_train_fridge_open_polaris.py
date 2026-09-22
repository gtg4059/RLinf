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

"""Tests for kitchen fridge PolarIS PPO train launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "examples/embodiment/scripts/train_fridge_open_polaris.sh"
CONFIG_NAME = "isaaclab_kitchen_open_fridge_ppo_openpi_pi05"
ADAPTER_CKPT = REPO_ROOT / "checkpoint" / "pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000"
ADAPTER_CKPT_19999 = (
    REPO_ROOT / "checkpoint" / "pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999"
)


def test_train_script_help_exits_zero() -> None:
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "train_fridge_open_polaris.sh" in completed.stdout
    assert CONFIG_NAME in completed.stdout


def test_train_script_dry_run_uses_adapter_checkpoint() -> None:
    if not (ADAPTER_CKPT / "model.safetensors").is_file() and not (
        ADAPTER_CKPT_19999 / "model.safetensors"
    ).is_file():
        import pytest

        pytest.skip("PolarIS CRI-prefix adapter checkpoint is not present")

    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "runner.max_epochs=2"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    assert f"CONFIG_NAME={CONFIG_NAME}" in stdout
    assert "POLARIS_OPENPI_CKPT=" in stdout
    assert "polaris_cri_adapter_rlinf_" in stdout
    assert "run_embodiment.sh" in stdout
    assert CONFIG_NAME in stdout
    assert "runner.max_epochs=2" in stdout


def test_train_script_honors_explicit_ckpt_override(tmp_path: Path) -> None:
    override = tmp_path / "custom_adapter"
    override.mkdir()
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
            "POLARIS_OPENPI_CKPT": str(override),
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert f"POLARIS_OPENPI_CKPT={override}" in completed.stdout


def test_train_script_accepts_config_name_then_overrides() -> None:
    """Same argv style as run_embodiment.sh: config name, then Hydra."""
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), CONFIG_NAME, "runner.max_epochs=2"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert f"CONFIG_NAME={CONFIG_NAME}" in completed.stdout
    assert "runner.max_epochs=2" in completed.stdout


def test_ppo_yaml_uses_all_gpus_at_32_train_envs_per_gpu() -> None:
    """Yaml default is this 2-GPU host: 32 train / 4 eval envs per GPU."""
    text = (REPO_ROOT / "examples/embodiment/config" / f"{CONFIG_NAME}.yaml").read_text()
    train_block, eval_block = text.split("\n  eval:\n", 1)
    assert "actor,env,rollout: all" in text
    assert "total_num_envs: 64" in train_block
    assert "rollout_epoch: 2" in train_block
    assert "max_episode_steps: 600" in train_block
    assert "episode_length_s: 40.0" in train_block
    assert "total_num_envs: 8" in eval_block
    assert "max_episode_steps: 600" in eval_block
    assert "episode_length_s: 40.0" in eval_block
    assert "global_batch_size: 5120" in text
    assert "sharding_strategy: \"no_shard\"" in text
    assert "gradient_checkpointing: True" in text
    assert "micro_batch_size: 8" in text


def test_train_script_scales_envs_to_visible_gpus() -> None:
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), CONFIG_NAME],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
            "RLINF_NUM_GPUS": "8",
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "VISIBLE_GPUS=8" in completed.stdout
    assert "env.train.total_num_envs=256" in completed.stdout
    assert "env.eval.total_num_envs=32" in completed.stdout
    assert "actor.global_batch_size=20480" in completed.stdout


def test_train_script_honors_explicit_env_count() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(TRAIN_SCRIPT),
            CONFIG_NAME,
            "env.train.total_num_envs=16",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
            "RLINF_NUM_GPUS": "2",
            "IMAGE_TAG": "rlinf:embodied-isaaclab-u24",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "env.train.total_num_envs=16" in completed.stdout
    assert "env.train.total_num_envs=64" not in completed.stdout
    assert "env.eval.total_num_envs=8" in completed.stdout


def test_train_script_rejects_unknown_config() -> None:
    completed = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "not_a_real_fridge_config"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RLINF_TRAIN_FRIDGE_DRY_RUN": "1",
            "RLINF_NO_DOCKER": "1",
        },
    )
    assert completed.returncode != 0
    assert "Hydra config not found" in completed.stderr
