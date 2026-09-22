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

"""TacVLA CRI prefix encoder used by PolarIS adapter checkpoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CRIPositionalEmbedding(torch.nn.Module):
    """FSDP-wrappable CRI positional table.

    The adapter checkpoint stores a bare ``cri_pos`` tensor. Loaders remap
    that key to ``cri_pos.weight`` so this module is a uniform-dtype FSDP
    unit instead of a leftover root Parameter mixed with paligemma bf16.
    """

    def __init__(
        self,
        num_tokens: int,
        width: int,
        data: torch.Tensor | None = None,
    ):
        super().__init__()
        if data is None:
            data = torch.zeros(num_tokens, width)
        else:
            data = data.detach().clone()
        self.weight = torch.nn.Parameter(data)


def wrap_bare_cri_pos_parameter(model: torch.nn.Module) -> None:
    """Replace a root ``cri_pos`` Parameter with ``CRIPositionalEmbedding``.

    FSDP's full-state-dict hook looks up every managed name in
    ``module.state_dict()``. A leftover root Parameter named ``cri_pos`` is
    tracked by FSDP but omitted from that dict, which raises
    ``FSDP assumes cri_pos is in the state_dict``.
    """
    param = getattr(model, "cri_pos", None)
    if isinstance(param, CRIPositionalEmbedding):
        return
    if not isinstance(param, torch.nn.Parameter):
        return
    data = param.detach()
    assert data.ndim == 2, f"cri_pos must be 2-D, got {tuple(data.shape)}"
    module = CRIPositionalEmbedding(data.shape[0], data.shape[1], data=data)
    module.weight.requires_grad_(param.requires_grad)
    module._fsdp_wrap_name = "cri_pos"
    del model._parameters["cri_pos"]
    model.cri_pos = module


def cri_pos_weight(cri_pos: torch.Tensor | torch.nn.Module) -> torch.Tensor:
    """Return the ``(num_cri_tokens, width)`` table from a tensor or module."""
    if isinstance(cri_pos, torch.nn.Module):
        return cri_pos.weight
    return cri_pos


def normalize_cri_prefix_state_dict(state_dict: dict) -> dict:
    """Map checkpoint ``cri_pos`` onto ``cri_pos.weight`` when needed."""
    remapped = dict(state_dict)
    if "cri_pos" in remapped and "cri_pos.weight" not in remapped:
        remapped["cri_pos.weight"] = remapped.pop("cri_pos")
    return remapped


def encode_cri_prefix_tokens(
    cri: torch.Tensor,
    cri_in_proj: torch.nn.Linear,
    cri_out_proj: torch.nn.Linear,
    cri_pos: torch.Tensor,
) -> torch.Tensor:
    """Encode float CRI as prefix tokens (Linear → SiLU → Linear + pos).

    The adapter checkpoint stores ``cri_in_proj`` as ``(256, 9)``,
    ``cri_out_proj`` as ``(18432, 256)``, and ``cri_pos`` as ``(9, 2048)``.
    """
    if not torch.is_tensor(cri):
        cri = torch.as_tensor(cri)
    if cri.ndim == 1:
        cri = cri.unsqueeze(0)
    cri = torch.clamp(cri.to(dtype=torch.float32), 0.0, 2.0)
    cri = cri.to(dtype=cri_in_proj.weight.dtype, device=cri_in_proj.weight.device)
    expected = cri_in_proj.in_features
    if cri.shape[-1] < expected:
        cri = F.pad(cri, (0, expected - cri.shape[-1]))
    elif cri.shape[-1] > expected:
        cri = cri[..., :expected]
    hidden = F.silu(cri_in_proj(cri))
    pos = cri_pos_weight(cri_pos)
    num_tokens, width = pos.shape
    tokens = cri_out_proj(hidden).reshape(cri.shape[0], num_tokens, width)
    return tokens + pos.to(dtype=tokens.dtype, device=tokens.device)


def vlm_value_prefix_mask(
    seq_len: int,
    num_images_in_input: int,
    value_vlm_mode: str = "mean_token",
) -> list[bool]:
    """Build a VLM value mask that follows the actual prefix length.

    Image slots are always ``256 * 3``. Remaining tokens are language, and
    for the CRI-prefix adapter the extra 9 CRI tokens are included there.
    """
    image_slots = 3
    image_tokens = 256
    reserved = image_tokens * image_slots
    lang_token_len = max(seq_len - reserved, 0)
    if value_vlm_mode == "mean_token":
        return (
            [True] * image_tokens * num_images_in_input
            + [False] * image_tokens * (image_slots - num_images_in_input)
            + [True] * lang_token_len
        )
    if value_vlm_mode == "last_token":
        return [False] * (seq_len - 1) + [True]
    if value_vlm_mode == "first_token":
        return [True] + [False] * (seq_len - 1)
    raise ValueError(f"Unknown value_vlm_mode: {value_vlm_mode}")
