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

"""FSDP checkpoint state_dict without a training run.

``save_checkpoint`` fails only at ``runner.save_interval`` (step 50). This
test wraps the PolarIS checkpoint the same way as the actor and calls
``get_state_dict`` immediately.

Run inside the training image, on one free GPU::

    python -m pytest tests/unit_tests/test_openpi_fsdp_checkpoint_smoke.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "checkpoint" / "pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000"
VISION_KEY = (
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    "encoder.layers.0.layer_norm1.weight"
)


def _skip_unless_smoke_ready() -> None:
    pytest.importorskip("openpi")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to match actor FSDP checkpoint save")
    if not (CKPT / "model.safetensors").is_file():
        pytest.skip(f"PolarIS checkpoint is not at {CKPT}")


def test_fsdp_state_dict_includes_vision_encoder_and_cri_pos() -> None:
    """Actor save path: DCP ``get_state_dict`` on a NO_SHARD FSDP model."""
    _skip_unless_smoke_ready()
    from omegaconf import OmegaConf
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.embodiment.openpi import get_model

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    cfg = OmegaConf.create(
        {
            "model_path": str(CKPT),
            "precision": None,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi05_droid_jointpos_polaris_cri_adapter",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "joint_logprob": False,
                "num_steps": 10,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
            },
        }
    )
    model = get_model(cfg)
    model.gradient_checkpointing_enable()
    model.cuda()
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config={"strategy": "fsdp"},
        is_lora=False,
        model_type="openpi",
    )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        device_id=0,
    )
    trainable = [param for param in fsdp_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-6)
    state_dict, _ = get_state_dict(
        fsdp_model,
        optimizers=optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    assert VISION_KEY in state_dict, f"missing {VISION_KEY}"
    assert "cri_pos.weight" in state_dict
    post_ln = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "post_layernorm.weight"
    )
    assert post_ln in state_dict, f"missing {post_ln}"
    projector = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    )
    assert projector in state_dict, f"missing {projector}"
    def _checkpoint_name(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    missing = sorted(
        cleaned
        for name, _ in fsdp_model.named_parameters()
        if "flat_param" not in name
        and (cleaned := _checkpoint_name(name)) not in state_dict
    )
    assert not missing, "FSDP state_dict is missing:\n" + "\n".join(missing)
