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

"""Rebuild unwrapped FSDP1 tensors for rollout weight sync."""

from __future__ import annotations

import math

import torch

_FSDP_KEY_SEGMENTS = frozenset(
    {
        "_fsdp_wrapped_module",
        "_checkpoint_wrapped_module",
        "_orig_mod",
    }
)


def clean_fsdp_state_dict_key(name: str) -> str:
    """Strip FSDP/checkpoint wrapper segments from a parameter name."""
    return ".".join(part for part in name.split(".") if part not in _FSDP_KEY_SEGMENTS)


def unwrap_fsdp_module(module: torch.nn.Module) -> torch.nn.Module:
    """Peel FSDP/checkpoint wrapper attributes until the inner module."""
    changed = True
    while changed:
        changed = False
        for attr in ("_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"):
            inner = getattr(module, attr, None)
            if inner is not None and inner is not module:
                module = inner
                changed = True
                break
    return module


def lookup_module_tensor(module: torch.nn.Module, cleaned_name: str) -> torch.Tensor | None:
    """Resolve a cleaned parameter/buffer name on a (possibly FSDP) module."""
    if not cleaned_name:
        return None
    obj: object = unwrap_fsdp_module(module)
    parts = cleaned_name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
        if isinstance(obj, torch.nn.Module):
            obj = unwrap_fsdp_module(obj)
    leaf = getattr(obj, parts[-1], None)
    if isinstance(leaf, torch.nn.Parameter) or torch.is_tensor(leaf):
        return leaf
    if isinstance(leaf, torch.nn.Module) and hasattr(leaf, "weight"):
        weight = leaf.weight
        if isinstance(weight, torch.nn.Parameter) or torch.is_tensor(weight):
            return weight
    return None


def collect_cleaned_named_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect parameters/buffers under cleaned names.

    After the first FSDP backward, ``named_parameters()`` on a rank can omit
    frozen VLM weights (and sometimes a root ``cri_pos``). Merge several
    sources so rollout sync still sees trainable tensors.
    """
    out: dict[str, torch.Tensor] = {}

    def _add(name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or not torch.is_tensor(tensor):
            return
        key = clean_fsdp_state_dict_key(name)
        if key and key not in out:
            out[key] = tensor.detach()

    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        _add(name, tensor)

    try:
        for name, tensor in model.state_dict().items():
            _add(name, tensor)
    except Exception:
        pass

    unwrapped = unwrap_fsdp_module(model)
    cri = getattr(unwrapped, "cri_pos", None)
    if cri is not None:
        if isinstance(cri, torch.nn.Parameter) or torch.is_tensor(cri):
            _add("cri_pos", cri)
        elif isinstance(cri, torch.nn.Module):
            weight = getattr(cri, "weight", None)
            if weight is not None:
                _add("cri_pos", weight)
    return out


def fill_missing_named_tensors(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    required_names: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Fill required keys that ``named_parameters`` omitted after FSDP train."""
    for name in required_names:
        if name in state_dict:
            continue
        tensor = lookup_module_tensor(model, name)
        if tensor is not None:
            state_dict[name] = tensor.detach()


def reconstruct_gathered_flat(
    gathered: torch.Tensor,
    expected_shape: tuple[int, ...],
    world_size: int,
) -> torch.Tensor:
    """Reshape concatenated FSDP local shards, dropping at most world_size-1 pad."""
    expected_numel = math.prod(expected_shape)
    padding = gathered.numel() - expected_numel
    if padding < 0 or padding >= world_size:
        raise RuntimeError(
            f"Cannot reconstruct shape {tuple(expected_shape)} from gathered "
            f"numel={gathered.numel()} (world_size={world_size})"
        )
    return gathered[:expected_numel].reshape(expected_shape)


def all_gather_local_shard_to_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    group=None,
) -> torch.Tensor:
    """Rebuild one full parameter from an FSDP1 1-D local shard."""
    expected_shape = tuple(expected_shape)
    if tuple(tensor.shape) == expected_shape:
        return tensor
    expected_numel = math.prod(expected_shape)
    if tensor.numel() == expected_numel:
        return tensor.reshape(expected_shape)

    world_size = torch.distributed.get_world_size(group)
    flat = tensor.detach().reshape(-1).contiguous()
    pieces = [torch.empty_like(flat) for _ in range(world_size)]
    torch.distributed.all_gather(pieces, flat, group=group)
    return reconstruct_gathered_flat(
        torch.cat(pieces, dim=0), expected_shape, world_size
    )
