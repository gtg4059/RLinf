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

"""Helpers for optional flash-attn that may be ABI-mismatched with torch."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

_LOGGER = logging.getLogger(__name__)
_DISABLED = False


def _purge_flash_attn_modules() -> None:
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def _force_flash_attn_2_unavailable() -> None:
    """Make transformers treat FlashAttention-2 as unavailable.

    ``is_flash_attn_2_available()`` only checks package metadata / version, so a
    wheel whose CUDA extension fails to load (torch ABI mismatch) still looks
    available and then crashes on ``from flash_attn import flash_attn_func``.
    Patch the check before ``modeling_flash_attention_utils`` is imported.
    """

    def _unavailable() -> bool:
        return False

    try:
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return

    import_utils.is_flash_attn_2_available = _unavailable

    try:
        import transformers.utils as utils

        utils.is_flash_attn_2_available = _unavailable
    except ImportError:
        pass


def disable_broken_flash_attn() -> bool:
    """Disable flash-attn when the package is present but unloadable.

    Returns:
        True if flash-attn was neutralized so callers can fall back to SDPA;
        False if flash-attn is absent or imports cleanly.
    """
    global _DISABLED
    if _DISABLED:
        return True

    if importlib.util.find_spec("flash_attn") is None:
        return False

    try:
        importlib.import_module("flash_attn")
        return False
    except Exception as exc:  # noqa: BLE001 - any load failure must not abort training
        _purge_flash_attn_modules()
        _force_flash_attn_2_unavailable()
        _DISABLED = True
        _LOGGER.warning(
            "flash-attn is installed but failed to import (%s); "
            "disabling FlashAttention-2 so transformers falls back to PyTorch SDPA.",
            exc,
        )
        return True
