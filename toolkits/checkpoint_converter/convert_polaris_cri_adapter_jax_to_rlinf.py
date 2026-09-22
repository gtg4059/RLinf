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

"""Convert official PolaRiS + TacVLA CRI-prefix JAX 30000 into RLinf weights.

No LoRA merge (adapter is encoder-only). Pipeline:

  1. Restore Orbax ``params/``
  2. JAX -> new PyTorch layout (SigLIP / LLM / action projs / CRI prefix)
  3. New -> legacy ``paligemma_with_expert.*`` layout
  4. Copy ``norm_stats.json`` under ``assets/droid`` (and aliases)
  5. Write ``rlinf_load_hint.json``

Example::

    python toolkits/checkpoint_converter/convert_polaris_cri_adapter_jax_to_rlinf.py \\
        --input-dir /mnt/E/openpi/checkpoints/pi05_droid_jointpos_polaris_cri_adapter/polaris_cri_adapter_20k/30000 \\
        --output-dir checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def _unwrap(tree):
    """Unwrap Orbax ``{'value': array}`` leaves to plain float32 numpy arrays."""
    if isinstance(tree, dict):
        if set(tree.keys()) == {"value"}:
            return np.asarray(tree["value"], dtype=np.float32)
        return {k: _unwrap(v) for k, v in tree.items()}
    return np.asarray(tree, dtype=np.float32)


def _copy_norm_stats(input_dir: Path, output_dir: Path) -> Path:
    candidates = [
        input_dir / "assets" / "droid" / "norm_stats.json",
        input_dir / "assets" / "droid_cri" / "norm_stats.json",
        input_dir / "droid" / "norm_stats.json",
        input_dir / "norm_stats.json",
    ]
    src_norm = next((p for p in candidates if p.is_file()), None)
    if src_norm is None:
        found = list(input_dir.rglob("norm_stats.json"))
        if not found:
            raise FileNotFoundError(f"norm_stats.json not found under {input_dir}")
        src_norm = found[0]
    for asset_id in ("assets/droid", "droid", "assets/droid_cri", "droid_cri"):
        dst = output_dir / Path(asset_id)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_norm, dst / "norm_stats.json")
    shutil.copy2(src_norm, output_dir / "norm_stats.json")
    return src_norm


def convert(input_dir: Path, output_dir: Path, action_horizon: int = 15) -> Path:
    import orbax.checkpoint as ocp

    from rlinf.utils.ckpt_convertor.openpi._core import (
        save_safetensors,
        write_config_json,
    )
    from rlinf.utils.ckpt_convertor.openpi.jax2new import (
        convert_cri_prefix,
        convert_llm,
        convert_projections,
        convert_siglip,
    )
    from rlinf.utils.ckpt_convertor.openpi.new2old import (
        ACTION_EXPERT_LM_HEAD,
        new_to_old_state_dict,
    )

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Restoring JAX params from {input_dir / 'params'}")
    restored = ocp.PyTreeCheckpointer().restore(str(input_dir / "params"))
    params = (
        restored["params"]
        if "params" in restored and "PaliGemma" not in restored
        else restored
    )
    params = _unwrap(params)

    print("[2/4] Converting JAX -> new PyTorch layout (no LoRA)")
    new_sd: dict[str, torch.Tensor] = {}
    for part in (
        convert_siglip(params),
        convert_llm(params, pi05=True),
        convert_projections(params, pi05=True),
        convert_cri_prefix(params),
    ):
        for key, value in part.items():
            new_sd[key] = value.contiguous().float()
    cri_keys = [key for key in new_sd if key.startswith("cri_")]
    if not cri_keys:
        raise RuntimeError(
            "JAX checkpoint has no CRI prefix tensors "
            "(expected cri_in_proj / cri_out_proj / cri_pos). "
            "This convertor is only for polaris_cri_adapter."
        )

    print("[3/4] Converting new -> legacy OpenPI layout")
    old_sd = new_to_old_state_dict(new_sd)
    del new_sd
    if ACTION_EXPERT_LM_HEAD not in old_sd:
        vocab = int(params["PaliGemma"]["llm"]["embedder"]["input_embedding"].shape[0])
        old_sd[ACTION_EXPERT_LM_HEAD] = torch.zeros(vocab, 1024, dtype=torch.float32)
    for key, value in list(old_sd.items()):
        old_sd[key] = value.float().contiguous()

    out_weights = output_dir / "model.safetensors"
    print(f"[4/4] Writing {out_weights} ({len(old_sd)} tensors)")
    save_safetensors(old_sd, out_weights)
    del old_sd

    write_config_json(
        {
            "action_dim": 32,
            "action_horizon": action_horizon,
            "max_token_len": 200,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": True,
            "use_cri_prefix": True,
            "num_cri_tokens": 9,
            "dtype": "float32",
        },
        output_dir,
    )
    _copy_norm_stats(input_dir, output_dir)
    hint = {
        "rlinf_model_path": str(output_dir),
        "openpi_config_name": "pi05_droid_jointpos_polaris_cri_adapter",
        "source_jax_checkpoint": str(input_dir),
        "action_dim": 32,
        "action_horizon": action_horizon,
        "max_token_len": 200,
        "use_cri_prefix": True,
        "lora_merged": False,
        "notes": [
            "Official polaris backbone + TacVLA cri_encoder (no LoRA).",
            "Set POLARIS_OPENPI_CKPT to this directory.",
            "Eval: bash examples/embodiment/scripts/eval_fridge_open_polaris.sh",
        ],
    }
    (output_dir / "rlinf_load_hint.json").write_text(
        json.dumps(hint, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done. export POLARIS_OPENPI_CKPT={output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="JAX checkpoint dir with params/ (OpenPI 30000 step)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000"),
        help="RLinf-loadable output directory",
    )
    parser.add_argument("--action-horizon", type=int, default=15)
    args = parser.parse_args(argv)
    convert(args.input_dir, args.output_dir, action_horizon=args.action_horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
