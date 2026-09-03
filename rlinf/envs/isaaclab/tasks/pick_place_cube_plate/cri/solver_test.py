"""Unit tests for CRI native-lib discovery (no CUDA solver required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from .constants import PACKAGE_DIR
from .solver import _clean_worker_env, _find_soname, _resolve_bundled_lib_dir, discover_native_lib_dirs


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
    if nvinfer is None:
        pytest.skip("TensorRT 10 is not installed in this environment")
    assert nvinfer.is_file()


def test_discover_honors_cri_extra_lib_dirs(tmp_path, monkeypatch):
    lib_dir = _require_bundled_lib_dir()
    nested = tmp_path / "tensorrt_libs"
    nested.mkdir()
    (nested / "libnvinfer.so.10").write_bytes(b"stub")
    monkeypatch.setenv("CRI_EXTRA_LIB_DIRS", str(tmp_path))
    dirs = discover_native_lib_dirs(lib_dir)
    assert _find_soname("libnvinfer.so.10", dirs) == nested / "libnvinfer.so.10"


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


def test_clean_worker_env_strips_docker_isaacsim_paths(monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH",
        "/workspace/RLinf:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/kit/lib:/usr/local/cuda/lib64")
    env = _clean_worker_env()
    assert "isaac-sim" not in env["PYTHONPATH"]
    assert "isaac-sim" not in env["LD_LIBRARY_PATH"]


def test_resolve_bundled_lib_prefers_versioned_tree():
    lib_dir = _require_bundled_lib_dir()
    resolved = _resolve_bundled_lib_dir(PACKAGE_DIR)
    assert resolved.is_dir()
    assert (resolved / "libSFD_CoreService.so").is_file()
    versioned = lib_dir / "3.11-cu128"
    if versioned.is_dir() and (versioned / "libSFD_CoreService.so").is_file():
        import sys

        if sys.version_info[:2] == (3, 11):
            assert resolved == versioned.resolve() or resolved == (lib_dir / "3.11").resolve()


def test_discover_includes_versioned_lib_trees():
    lib_dir = _require_bundled_lib_dir()
    dirs = discover_native_lib_dirs(lib_dir)
    versioned = lib_dir / "3.11"
    if versioned.is_dir():
        assert versioned.resolve() in dirs
