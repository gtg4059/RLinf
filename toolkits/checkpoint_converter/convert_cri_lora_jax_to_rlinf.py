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

"""Convert a pi05 DROID CRI LoRA JAX checkpoint into RLinf-loadable OpenPI weights.

Pipeline:
  1. Restore Orbax ``params/``
  2. Merge LoRA adapters into base weights (openpi issue #958 semantics)
  3. Convert JAX -> new PyTorch layout (``jax2new``)
  4. Convert new -> legacy ``paligemma_with_expert.*`` layout (``new2old``)
  5. Copy ``assets/droid_cri/norm_stats.json`` into the output

Example::

    python toolkits/checkpoint_converter/convert_cri_lora_jax_to_rlinf.py \\
        --input-dir /workspace/RLinf/49999 \\
        --output-dir /workspace/RLinf/checkpoint/pi05_droid_cri_rlinf_49999
"""

from __future__ import annotations

import argparse
import math
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


def _lora_scale(rank: int, alpha: float, rslora: bool = False) -> float:
    return alpha / math.sqrt(rank) if rslora else alpha / rank


def _merge_einsum_lora(
    base: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    *,
    einsum_expr: str,
    scale: float,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    lora_a = np.asarray(lora_a, dtype=np.float32)
    lora_b = np.asarray(lora_b, dtype=np.float32)
    delta = np.einsum(einsum_expr, lora_a, lora_b, optimize=True)
    return base + delta * scale


def _merge_attn_vec_lora(
    base: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """attn_vec_einsum merge with sum-over-N correction (openpi #958)."""
    base = np.asarray(base, dtype=np.float32)
    lora_a = np.asarray(lora_a, dtype=np.float32)
    lora_b = np.asarray(lora_b, dtype=np.float32)
    # shapes: base (L, N, H, D), lora_a (L, N, H, rank), lora_b (L, N, rank, D)
    lora_b_sum_n = np.sum(lora_b, axis=1)  # (L, rank, D)
    delta = np.einsum("lnhr,lrd->lnhd", lora_a, lora_b_sum_n, optimize=True)
    return base + delta * scale


def merge_lora_into_params(params: dict) -> dict:
    """Merge LoRA adapters for gemma_2b_lora + gemma_300m_lora pi05 experts."""
    attn = params["PaliGemma"]["llm"]["layers"]["attn"]
    mlp = params["PaliGemma"]["llm"]["layers"]["mlp"]
    mlp_1 = params["PaliGemma"]["llm"]["layers"]["mlp_1"]

    # gemma_2b_lora: rank=16, alpha=16 -> scale=1.0
    attn_scale_pg = _lora_scale(16, 16.0)
    # gemma_300m_lora: rank=32, alpha=32 -> scale=1.0
    attn_scale_act = _lora_scale(32, 32.0)
    # MLP FeedForward._dot does NOT apply alpha/rank scaling.
    mlp_scale = 1.0

    # Expert 0 (PaliGemma)
    attn["q_einsum"]["w"] = _merge_einsum_lora(
        attn["q_einsum"]["w"],
        attn["q_einsum"].pop("lora_a"),
        attn["q_einsum"].pop("lora_b"),
        einsum_expr="lndr,lnrh->lndh",
        scale=attn_scale_pg,
    )
    attn["kv_einsum"]["w"] = _merge_einsum_lora(
        attn["kv_einsum"]["w"],
        attn["kv_einsum"].pop("lora_a"),
        attn["kv_einsum"].pop("lora_b"),
        einsum_expr="labdr,labrh->labdh",
        scale=attn_scale_pg,
    )
    attn["attn_vec_einsum"]["w"] = _merge_attn_vec_lora(
        attn["attn_vec_einsum"]["w"],
        attn["attn_vec_einsum"].pop("lora_a"),
        attn["attn_vec_einsum"].pop("lora_b"),
        scale=attn_scale_pg,
    )
    mlp["gating_einsum"] = _merge_einsum_lora(
        mlp["gating_einsum"],
        mlp.pop("gating_einsum_lora_a"),
        mlp.pop("gating_einsum_lora_b"),
        einsum_expr="lafr,larh->lafh",
        scale=mlp_scale,
    )
    mlp["linear"] = _merge_einsum_lora(
        mlp["linear"],
        mlp.pop("linear_lora_a"),
        mlp.pop("linear_lora_b"),
        einsum_expr="lhr,lrf->lhf",
        scale=mlp_scale,
    )

    # Expert 1 (action expert)
    attn["q_einsum_1"]["w"] = _merge_einsum_lora(
        attn["q_einsum_1"]["w"],
        attn["q_einsum_1"].pop("lora_a"),
        attn["q_einsum_1"].pop("lora_b"),
        einsum_expr="lndr,lnrh->lndh",
        scale=attn_scale_act,
    )
    attn["kv_einsum_1"]["w"] = _merge_einsum_lora(
        attn["kv_einsum_1"]["w"],
        attn["kv_einsum_1"].pop("lora_a"),
        attn["kv_einsum_1"].pop("lora_b"),
        einsum_expr="labdr,labrh->labdh",
        scale=attn_scale_act,
    )
    attn["attn_vec_einsum_1"]["w"] = _merge_attn_vec_lora(
        attn["attn_vec_einsum_1"]["w"],
        attn["attn_vec_einsum_1"].pop("lora_a"),
        attn["attn_vec_einsum_1"].pop("lora_b"),
        scale=attn_scale_act,
    )
    mlp_1["gating_einsum"] = _merge_einsum_lora(
        mlp_1["gating_einsum"],
        mlp_1.pop("gating_einsum_lora_a"),
        mlp_1.pop("gating_einsum_lora_b"),
        einsum_expr="lafr,larh->lafh",
        scale=mlp_scale,
    )
    mlp_1["linear"] = _merge_einsum_lora(
        mlp_1["linear"],
        mlp_1.pop("linear_lora_a"),
        mlp_1.pop("linear_lora_b"),
        einsum_expr="lhr,lrf->lhf",
        scale=mlp_scale,
    )

    # Sanity: no LoRA leaves remain.
    flat_left = []

    def _walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}/{k}" if prefix else k
                if "lora" in k.lower():
                    flat_left.append(p)
                _walk(v, p)

    _walk(params)
    if flat_left:
        raise RuntimeError(f"Unmerged LoRA keys remain: {flat_left[:20]}")
    return params


def convert(input_dir: Path, output_dir: Path, action_horizon: int = 15) -> Path:
    import orbax.checkpoint as ocp

    from rlinf.utils.ckpt_convertor.openpi._core import (
        save_safetensors,
        write_config_json,
    )
    from rlinf.utils.ckpt_convertor.openpi.jax2new import (
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

    print(f"[1/5] Restoring JAX params from {input_dir / 'params'}")
    restored = ocp.PyTreeCheckpointer().restore(str(input_dir / "params"))
    params = restored["params"] if "params" in restored and "PaliGemma" not in restored else restored
    params = _unwrap(params)

    print("[2/5] Merging LoRA adapters into base weights")
    params = merge_lora_into_params(params)

    print("[3/5] Converting JAX -> new PyTorch layout")
    new_sd: dict[str, torch.Tensor] = {}
    for part in (
        convert_siglip(params),
        convert_llm(params, pi05=True),
        convert_projections(params, pi05=True),
    ):
        for k, v in part.items():
            new_sd[k] = v.contiguous().float()

    print("[4/5] Converting new -> legacy OpenPI layout")
    old_sd = new_to_old_state_dict(new_sd)
    del new_sd

    # Action-expert lm_head is unused/tied in deploy; provide a correctly shaped
    # placeholder so loaders that expect the key do not fail.
    if ACTION_EXPERT_LM_HEAD not in old_sd:
        vocab = int(params["PaliGemma"]["llm"]["embedder"]["input_embedding"].shape[0])
        old_sd[ACTION_EXPERT_LM_HEAD] = torch.zeros(vocab, 1024, dtype=torch.float32)

    # Keep float32 for LoRA-merged weights (bf16 storage drifts; see openpi #958).
    for k, v in list(old_sd.items()):
        if v.dtype != torch.float32:
            old_sd[k] = v.float().contiguous()
        else:
            old_sd[k] = v.contiguous()

    out_weights = output_dir / "model.safetensors"
    print(f"[5/5] Writing {out_weights} ({len(old_sd)} tensors)")
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
            "dtype": "float32",
        },
        output_dir,
    )

    # Preserve CRI norm stats under the asset_id used by pi05_droid_cri.
    src_norm = input_dir / "assets" / "droid_cri" / "norm_stats.json"
    if not src_norm.is_file():
        candidates = list(input_dir.rglob("norm_stats.json"))
        if not candidates:
            raise FileNotFoundError(f"norm_stats.json not found under {input_dir}")
        src_norm = candidates[0]

    for asset_id in ("droid_cri", "droid", "assets/droid_cri", "assets/droid"):
        dst = output_dir / asset_id
        if asset_id.startswith("assets/"):
            dst = output_dir / Path(asset_id)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_norm, dst / "norm_stats.json")

    # Also keep a copy next to weights for convenience.
    shutil.copy2(src_norm, output_dir / "norm_stats.json")

    print(f"Done. RLinf model_path -> {output_dir}")
    print("  export CRI_OPENPI_CKPT=" + str(output_dir))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/RLinf/49999"),
        help="JAX checkpoint dir containing params/ and assets/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/RLinf/checkpoint/pi05_droid_cri_rlinf_49999"),
        help="Output directory for RLinf-loadable safetensors",
    )
    parser.add_argument("--action-horizon", type=int, default=15)
    args = parser.parse_args(argv)
    convert(args.input_dir, args.output_dir, action_horizon=args.action_horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
