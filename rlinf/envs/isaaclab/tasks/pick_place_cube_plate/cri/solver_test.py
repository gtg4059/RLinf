"""Unit tests for CRI native-lib discovery (no CUDA solver required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from .constants import PACKAGE_DIR
from .ipc import decode_payload_size
from .ipc import pickle_recv
from .ipc import pickle_send
from . import solver as solver_mod
from .solver import _REQUIRED_GLIBC
from .solver import _REQUIRED_GLIBCXX
from .solver import _clean_worker_env
from .solver import _find_soname
from .solver import _resolve_bundled_lib_dir
from .solver import _safety_core_needs_isaac_c10
from .solver import build_cri_worker_cmd
from .solver import discover_glibc_loader
from .solver import discover_isaac_torch_site
from .solver import discover_libstdcxx_dir
from .solver import discover_native_lib_dirs


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


def test_clean_worker_env_strips_isaac_kit_paths(monkeypatch):
    monkeypatch.setattr(solver_mod, "discover_isaac_torch_site", lambda: None)
    monkeypatch.setenv(
        "PYTHONPATH",
        "/workspace/RLinf:/workspace/isaac_sim/kit/python/lib/python3.11:"
        "/opt/venv/openpi/lib/python3.11/site-packages",
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/workspace/isaac_sim/kit/lib:/usr/local/cuda/lib64",
    )
    env = _clean_worker_env()
    assert "isaac_sim" not in env["PYTHONPATH"]
    assert "isaac_sim" not in env["LD_LIBRARY_PATH"]
    assert env["PYTHONNOUSERSITE"] == "1"


def test_clean_worker_env_strips_docker_isaacsim_paths(monkeypatch):
    monkeypatch.setattr(solver_mod, "discover_isaac_torch_site", lambda: None)
    monkeypatch.setenv(
        "PYTHONPATH",
        "/workspace/RLinf:/isaac-sim/kit/python/lib/python3.11",
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/isaac-sim/kit/lib:/usr/local/cuda/lib64")
    env = _clean_worker_env()
    assert "isaac-sim" not in env["PYTHONPATH"]
    assert "isaac-sim" not in env["LD_LIBRARY_PATH"]


def test_clean_worker_env_prepends_isaac_torch(tmp_path, monkeypatch):
    import os

    site = tmp_path / "omni.isaac.ml_archive" / "pip_prebundle"
    (site / "torch" / "lib").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("")
    monkeypatch.setattr(solver_mod, "discover_isaac_torch_site", lambda: site)
    monkeypatch.setattr(solver_mod, "_safety_core_needs_isaac_c10", lambda analysis_dir=None: True)
    monkeypatch.setenv("PYTHONPATH", "/opt/venv/openpi/lib/python3.11/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    env = _clean_worker_env()
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(site)
    assert env["CRI_ISAAC_TORCH_SITE"] == str(site)
    assert str(site / "torch" / "lib") in env["LD_LIBRARY_PATH"].split(os.pathsep)


def test_discover_isaac_torch_site_honors_env(tmp_path, monkeypatch):
    site = tmp_path / "pip_prebundle"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("")
    monkeypatch.setenv("CRI_ISAAC_TORCH_SITE", str(site))
    assert discover_isaac_torch_site() == site.resolve()


def test_safety_core_needs_isaac_c10_if_bundled():
    so = PACKAGE_DIR / "lib" / "libSafetyCore.so.1.7.1"
    if not so.is_file():
        pytest.skip("CRI native libs are not checked in")
    assert _safety_core_needs_isaac_c10(PACKAGE_DIR) is True


def test_discover_libstdcxx_dir_honors_env(tmp_path, monkeypatch):
    so = tmp_path / "libstdc++.so.6"
    so.write_bytes(b"stub " + _REQUIRED_GLIBCXX.encode() + b" end")
    monkeypatch.setenv("CRI_LIBSTDCXX_DIR", str(tmp_path))
    assert discover_libstdcxx_dir() == tmp_path.resolve()


def test_discover_libstdcxx_dir_rejects_old_so(tmp_path, monkeypatch):
    so = tmp_path / "libstdc++.so.6"
    so.write_bytes(b"GLIBCXX_3.4.30 only")
    monkeypatch.delenv("CRI_LIBSTDCXX", raising=False)
    monkeypatch.delenv("CRI_LIBSTDCXX_DIR", raising=False)
    assert discover_libstdcxx_dir(tmp_path) is None


def test_bundled_cxx_libstdcxx_if_present():
    cxx = PACKAGE_DIR / "lib" / "cxx" / "libstdc++.so.6"
    if not cxx.is_file():
        pytest.skip("cri/lib/cxx is not populated on this machine")
    found = discover_libstdcxx_dir()
    assert found is not None
    assert (found / "libstdc++.so.6").is_file()


def test_clean_worker_env_prepends_libstdcxx(tmp_path, monkeypatch):
    import os

    so = tmp_path / "libstdc++.so.6"
    so.write_bytes(b"stub " + _REQUIRED_GLIBCXX.encode())
    monkeypatch.setenv("CRI_LIBSTDCXX_DIR", str(tmp_path))
    monkeypatch.setattr(solver_mod, "discover_glibc_loader", lambda analysis_dir=None: None)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    env = _clean_worker_env()
    first = env["LD_LIBRARY_PATH"].split(os.pathsep)[0]
    assert Path(first).resolve() == tmp_path.resolve()


def test_discover_glibc_loader_honors_env(tmp_path, monkeypatch):
    loader = tmp_path / "ld-linux-x86-64.so.2"
    loader.write_bytes(b"stub-loader")
    (tmp_path / "libc.so.6").write_bytes(b"stub " + _REQUIRED_GLIBC.encode())
    monkeypatch.setenv("CRI_GLIBC_LOADER", str(loader))
    assert discover_glibc_loader() == loader.resolve()


def test_discover_glibc_loader_rejects_old_libc(tmp_path, monkeypatch):
    loader = tmp_path / "ld-linux-x86-64.so.2"
    loader.write_bytes(b"stub-loader")
    (tmp_path / "libc.so.6").write_bytes(b"GLIBC_2.35 only")
    monkeypatch.delenv("CRI_GLIBC_LOADER", raising=False)
    monkeypatch.delenv("CRI_GLIBC_DIR", raising=False)
    assert discover_glibc_loader(tmp_path) is None


def test_bundled_cxx_glibc_loader_if_present():
    loader = PACKAGE_DIR / "lib" / "cxx" / "ld-linux-x86-64.so.2"
    libc = PACKAGE_DIR / "lib" / "cxx" / "libc.so.6"
    if not loader.is_file() or not libc.is_file():
        pytest.skip("cri/lib/cxx glibc is not populated on this machine")
    found = discover_glibc_loader()
    assert found is not None
    assert found.resolve() == loader.resolve()


def test_build_cri_worker_cmd_prefixes_loader(tmp_path, monkeypatch):
    loader = tmp_path / "ld-linux-x86-64.so.2"
    loader.write_bytes(b"stub-loader")
    (tmp_path / "libc.so.6").write_bytes(b"stub " + _REQUIRED_GLIBC.encode())
    monkeypatch.setenv("CRI_GLIBC_LOADER", str(loader))
    worker = tmp_path / "sfd_worker.py"
    worker.write_text("# stub\n")
    env = {"LD_LIBRARY_PATH": str(tmp_path)}
    cmd = build_cri_worker_cmd("/usr/bin/python", worker, env)
    assert cmd[:3] == [str(loader.resolve()), "--library-path", str(tmp_path)]
    assert cmd[-2:] == ["-u", str(worker)]


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


def test_pickle_roundtrip():
    import io

    buf = io.BytesIO()
    pickle_send(buf, {"ok": True, "n": 3})
    buf.seek(0)
    assert pickle_recv(buf) == {"ok": True, "n": 3}


def test_pickle_recv_rejects_log_header():
    import io

    buf = io.BytesIO(b"[spike] mlockall applied\n")
    with pytest.raises(ValueError, match="invalid"):
        pickle_recv(buf)


def test_decode_payload_size_rejects_zero():
    import struct

    with pytest.raises(ValueError, match="invalid"):
        decode_payload_size(struct.pack("<Q", 0))


def test_dedicated_ipc_ignores_stdout_logs():
    import os
    import pickle
    import struct
    import subprocess
    import sys
    import textwrap

    from .ipc import CRI_IPC_WRITE_FD_ENV

    script = textwrap.dedent(
        """
        import os
        import pickle
        import struct
        import sys

        fd = int(os.environ["CRI_IPC_WRITE_FD"])
        out = os.fdopen(fd, "wb", buffering=0)
        print("[spike] mlockall applied", flush=True)
        header = sys.stdin.buffer.read(8)
        n = int.from_bytes(header, "little")
        obj = pickle.loads(sys.stdin.buffer.read(n))
        payload = pickle.dumps({"ok": True, "echo": obj}, protocol=pickle.HIGHEST_PROTOCOL)
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)
        out.flush()
        """
    )
    ipc_r, ipc_w = os.pipe()
    os.set_inheritable(ipc_w, True)
    env = os.environ.copy()
    env[CRI_IPC_WRITE_FD_ENV] = str(ipc_w)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            pass_fds=(ipc_w,),
        )
    finally:
        os.close(ipc_w)
    ipc_in = os.fdopen(ipc_r, "rb", buffering=0)
    try:
        assert proc.stdin is not None
        pickle_send(proc.stdin, {"hello": 1})
        reply = pickle_recv(ipc_in, timeout_s=10)
        assert reply == {"ok": True, "echo": {"hello": 1}}
        stdout, _ = proc.communicate(timeout=10)
        assert b"[spike]" in stdout
    finally:
        ipc_in.close()
        if proc.poll() is None:
            proc.kill()


def test_discover_includes_versioned_lib_trees():
    lib_dir = _require_bundled_lib_dir()
    dirs = discover_native_lib_dirs(lib_dir)
    versioned = lib_dir / "3.11"
    if versioned.is_dir():
        assert versioned.resolve() in dirs
