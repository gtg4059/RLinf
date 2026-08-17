"""Unit tests for CRI native-lib discovery (no CUDA solver required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from .constants import PACKAGE_DIR
from .solver import _clean_worker_env, _find_soname, discover_native_lib_dirs


def _require_bundled_lib_dir() -> Path:
    lib_dir = PACKAGE_DIR / "lib"
    if not (lib_dir / "libsfd_compat.so").is_file():
        pytest.skip("CRI native libs are not checked in")
    return lib_dir


def test_discover_includes_bundled_lib_and_tensorrt():
    lib_dir = _require_bundled_lib_dir()
    dirs = discover_native_lib_dirs(lib_dir)
    assert lib_dir.resolve() in dirs
    nvinfer = _find_soname("libnvinfer.so.10", dirs)
    assert nvinfer is not None, f"libnvinfer.so.10 not found in {dirs}"
    assert nvinfer.is_file()


def test_find_bundled_compat_and_cudart13():
    lib_dir = _require_bundled_lib_dir()
    dirs = discover_native_lib_dirs(lib_dir)
    assert _find_soname("libsfd_compat.so", dirs) == lib_dir / "libsfd_compat.so"
    assert _find_soname("libcudart.so.13", dirs) is not None


def test_clean_worker_env_strips_isaac_paths(monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH",
        "/workspace/RLinf:/workspace/isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle:/opt/venv/openpi/lib/python3.11/site-packages",
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/workspace/isaac_sim/kit/lib:/usr/local/cuda/lib64",
    )
    env = _clean_worker_env()
    assert "isaac_sim" not in env["PYTHONPATH"]
    assert "omni.isaac.ml_archive" not in env["PYTHONPATH"]
    assert "isaac_sim" not in env["LD_LIBRARY_PATH"]
    assert env["PYTHONNOUSERSITE"] == "1"
