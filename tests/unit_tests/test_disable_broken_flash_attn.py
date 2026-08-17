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

"""Tests for neutralizing a torch-ABI-mismatched flash-attn install."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from rlinf.utils import flash_attn as flash_attn_utils


@pytest.fixture(autouse=True)
def _reset_flash_attn_guard_state():
    flash_attn_utils._DISABLED = False
    yield
    flash_attn_utils._DISABLED = False
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


def test_disable_broken_flash_attn_noop_when_missing(monkeypatch):
    monkeypatch.setattr(
        flash_attn_utils.importlib.util, "find_spec", lambda name: None
    )
    assert flash_attn_utils.disable_broken_flash_attn() is False


def test_disable_broken_flash_attn_noop_when_importable(monkeypatch):
    fake_mod = types.ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", fake_mod)
    monkeypatch.setattr(
        flash_attn_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )
    monkeypatch.setattr(
        flash_attn_utils.importlib, "import_module", lambda name: fake_mod
    )
    assert flash_attn_utils.disable_broken_flash_attn() is False


def test_disable_broken_flash_attn_patches_transformers(monkeypatch):
    import_utils = types.ModuleType("transformers.utils.import_utils")
    utils = types.ModuleType("transformers.utils")
    transformers = types.ModuleType("transformers")

    def _available() -> bool:
        return True

    import_utils.is_flash_attn_2_available = _available
    utils.is_flash_attn_2_available = _available
    transformers.utils = utils

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)
    monkeypatch.setattr(
        flash_attn_utils.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )

    def _boom(name: str):
        raise ImportError(
            "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb"
        )

    monkeypatch.setattr(flash_attn_utils.importlib, "import_module", _boom)

    assert flash_attn_utils.disable_broken_flash_attn() is True
    assert import_utils.is_flash_attn_2_available() is False
    assert utils.is_flash_attn_2_available() is False
    # Idempotent.
    assert flash_attn_utils.disable_broken_flash_attn() is True


def test_module_is_importable():
    # Smoke check used by CI path discovery.
    assert importlib.import_module("rlinf.utils.flash_attn") is flash_attn_utils
