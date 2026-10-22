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

from rlinf.algorithms.utils import resolve_update_epoch

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


def test_ppo_yaml_uses_all_gpus_at_64_train_envs_per_gpu() -> None:
    """Yaml default is this 2-GPU host: 64 train / 4 eval envs per GPU."""
    text = (REPO_ROOT / "examples/embodiment/config" / f"{CONFIG_NAME}.yaml").read_text()
    train_block, eval_block = text.split("\n  eval:\n", 1)
    assert "actor,env,rollout: all" in text
    assert "total_num_envs: 128" in train_block
    assert "rollout_epoch: 1" in train_block
    assert "max_episode_steps: 900" in train_block
    assert "episode_length_s: 60.0" in train_block
    assert "hold_after_done: False" in train_block
    assert "isaac_success_termination: True" in train_block
    assert "repeat_open_after_success: true" in train_block
    assert "success_bonus: 1.0" in train_block
    assert "door_progress_weight: 1.0" in train_block
    assert "reach_bonus: 0.2" in train_block
    assert "reach_bonus:" not in eval_block
    assert "door_progress_cap: 0.4" in train_block
    assert "hold_after_done: False" in eval_block
    assert "success_bonus:" not in eval_block
    assert "total_num_envs: 8" in eval_block
    assert "max_episode_steps: 900" in eval_block
    assert "episode_length_s: 60.0" in eval_block
    assert "global_batch_size: 7680" in text
    assert "val_check_interval: 50" in text
    assert "save_interval: 50" in text
    assert "adv_type: episode_success" in text
    assert "success_return_threshold: 0.8" in text
    assert "success_chunk_reward_min: 0.0" in text
    assert "success_advantage: 0.5" in text
    assert "approach_advantage: 0.25" in text
    assert "speed_horizon_chunks: 60" in text
    assert "handle_approach_weight: 0.2" in text
    assert "handle_approach_cap: 0.1" in text
    assert "group_size: 1" in text
    assert "success_floor:" not in text
    assert "success_floor_window:" not in text
    assert "loss_type: actor\n" in text
    assert "lr: 1.0e-5" in text
    assert "lr_scheduler: fixed" in text
    assert "kl_penalty: low_var_kl" in text
    assert "kl_beta: 0.05" in text
    assert "checkpoints/global_step_150" in text
    assert "critic_warmup_steps: 0" in text
    assert "critic_warmup_update_epoch: 5" in text
    assert "update_epoch: 2" in text
    assert "sharding_strategy: \"no_shard\"" in text
    assert "gradient_checkpointing: True" in text
    assert "micro_batch_size: 16" in text


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
    assert "env.train.total_num_envs=512" in completed.stdout
    assert "env.eval.total_num_envs=32" in completed.stdout
    assert "HORIZON_STEPS=600" in completed.stdout
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
    assert "env.train.total_num_envs=128" not in completed.stdout
    assert "env.eval.total_num_envs=8" in completed.stdout
    # 16 envs / 2 GPUs * (600/15) chunks. Batch follows the env override.
    assert "actor.global_batch_size=640" in completed.stdout


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


def _run_warmup_rollouts(
    *,
    base_epoch: int,
    warmup_epoch: int,
    warmup_steps: int,
) -> list[str]:
    """Mirror the actor loop: extra warmup passes stop when the actor unfreezes."""
    optimizer_steps = 0
    remaining = warmup_steps
    passes: list[str] = []
    for _rollout in range(4):
        update_epoch = resolve_update_epoch(
            base_epoch, warmup_epoch, optimizer_steps, remaining
        )
        started_in_warmup = update_epoch > base_epoch
        stop_warmup_epochs = False
        for epoch_idx in range(update_epoch):
            kind = "critic" if remaining > 0 and optimizer_steps < warmup_steps else "actor"
            passes.append(kind)
            optimizer_steps += 1
            if optimizer_steps >= warmup_steps:
                remaining = 0
            if (
                started_in_warmup
                and remaining <= 0
                and epoch_idx + 1 >= base_epoch
            ):
                stop_warmup_epochs = True
                break
        if stop_warmup_epochs:
            continue
    return passes


def test_warmup_reuses_one_rollout_then_actor_epoch_stays_one() -> None:
    assert resolve_update_epoch(1, 5, optimizer_steps=0, critic_warmup_steps=5) == 5
    assert resolve_update_epoch(1, 5, optimizer_steps=5, critic_warmup_steps=0) == 1
    passes = _run_warmup_rollouts(base_epoch=1, warmup_epoch=5, warmup_steps=5)
    assert passes[:5] == ["critic"] * 5
    assert passes[5:] == ["actor"] * 3


def test_warmup_epoch_does_not_keep_training_the_actor_on_the_same_batch() -> None:
    passes = _run_warmup_rollouts(base_epoch=1, warmup_epoch=5, warmup_steps=3)
    assert passes[:3] == ["critic"] * 3
    assert passes[3] == "actor"
