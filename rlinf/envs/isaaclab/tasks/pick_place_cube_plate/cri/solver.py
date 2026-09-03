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

"""Standalone Safetics CRI solver bootstrap (no Isaac Sim / Omniverse)."""

from __future__ import annotations

import ctypes
import logging
import os
import site
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import CBF_ALPHA
from .constants import CRI_FILTER_LIMIT
from .constants import DEFAULT_NUM_JOINTS
from .constants import DEFAULT_ZERO_VEL_EPS
from .constants import NUM_CRI_POINTS
from .cri_realtime_monitor import configure_cri_filter
from .filter import command_delta
from .postprocess import apply_cri_zero_vel_filter, clamp_cri

logger = logging.getLogger(__name__)

# Native deps that ``import`` cannot see unless they are already mapped.
# Changing ``LD_LIBRARY_PATH`` after process start does not help on Linux.
_PRELOAD_SONAMES = (
    "libsfd_compat.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libnvinfer.so.10",
    "libnvonnxparser.so.10",
    "libtensorflowlite_c.so",
    "libjsoncpp.so.25",
    "libcrypto++.so.8",
    "libSafeticsFoundation.so.1.7.1",
    "libSafetyCore.so.1.7.1",
    "libSFD_CoreService.so",
)

# Must live under the analysis ``lib/`` tree. These are not in git (see
# ``cri/lib/.gitignore``); copy them locally or point ``OPENPI_CRI_ANALYSIS_DIR``.
_REQUIRED_BUNDLED_SONAMES = (
    "libsfd_compat.so",
    "libSFD_CoreService.so",
    "libSafeticsFoundation.so.1.7.1",
    "libSafetyCore.so.1.7.1",
)

# Bundled under this package (lib/, Engine/, ST_AnalysisInfo.json, Robot_Model/).
_PACKAGE_ANALYSIS_DIR = Path(__file__).resolve().parent
# Host checkout first, then the path Isaac Lab uses inside the RLinf container.
_FALLBACK_ISAACLAB_ARTICULATION_CANDIDATES = (
    Path("/home/safetics/IsaacLab/source/isaaclab/isaaclab/assets/articulation"),
    Path("/opt/envs/isaaclab/source/isaaclab/isaaclab/assets/articulation"),
)


def resolve_analysis_dir(analysis_dir: str | Path | None = None) -> Path:
    """Resolve the CUDACRI analysis directory containing ``lib/`` and ``ST_AnalysisInfo.json``.

    Order: explicit arg → ``OPENPI_CRI_ANALYSIS_DIR`` → ``SFD_CRI_ANALYSIS_DIR`` →
    ``openpi/cri`` package dir → IsaacLab articulation path (if present).
    """
    if analysis_dir is not None:
        root = Path(analysis_dir).expanduser().resolve()
    else:
        env = os.environ.get("OPENPI_CRI_ANALYSIS_DIR") or os.environ.get("SFD_CRI_ANALYSIS_DIR")
        if env:
            root = Path(env).expanduser().resolve()
        elif (_PACKAGE_ANALYSIS_DIR / "lib").is_dir() and (_PACKAGE_ANALYSIS_DIR / "ST_AnalysisInfo.json").is_file():
            root = _PACKAGE_ANALYSIS_DIR
        else:
            fallback = next(
                (cand for cand in _FALLBACK_ISAACLAB_ARTICULATION_CANDIDATES if cand.is_dir()),
                None,
            )
            if fallback is None:
                raise FileNotFoundError(
                    "CRI analysis dir not found. Expected lib/ + ST_AnalysisInfo.json under "
                    f"{_PACKAGE_ANALYSIS_DIR}, or set OPENPI_CRI_ANALYSIS_DIR."
                )
            root = fallback.resolve()
    lib_dir = _resolve_bundled_lib_dir(root)
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"CUDACRI lib not found: {root / 'lib'}")
    if not (root / "ST_AnalysisInfo.json").is_file():
        raise FileNotFoundError(f"ST_AnalysisInfo.json not found under {root}")
    missing = [name for name in _REQUIRED_BUNDLED_SONAMES if not (lib_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "CRI native libraries are not stored in git. Copy them into "
            f"{lib_dir} or set OPENPI_CRI_ANALYSIS_DIR to a tree that contains "
            f"them. Missing: {', '.join(missing)}"
        )
    return root


def _resolve_bundled_lib_dir(analysis_dir: Path) -> Path:
    """Prefer ``lib/<py>[-cu128]/`` (docker 3.11 + Isaac), else flat ``lib/``.

    ``rlinf/rlinf:agentic-rlinf0.4-isaaclab`` is Python 3.11. Isaac torch
    makes ``sfd_setup`` ask for ``lib/3.11-cu128``; the deploy script mirrors
    ``lib/3.11`` there. Flat ``lib/`` still holds sonames for the worker preload.
    """
    root = analysis_dir / "lib"
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    for name in (f"{py}-cu128", py):
        candidate = root / name
        if candidate.is_dir() and (candidate / "libSFD_CoreService.so").is_file():
            return candidate
    return root


def _site_package_dirs() -> list[Path]:
    """Return candidate site-packages roots for the current interpreter."""
    dirs: list[Path] = []
    try:
        raw_sites = list(site.getsitepackages())
    except Exception:
        raw_sites = []
    try:
        raw_sites.append(site.getusersitepackages())
    except Exception:
        pass
    for raw in raw_sites:
        if raw:
            dirs.append(Path(raw))
    for entry in sys.path:
        if entry and entry.endswith("site-packages"):
            dirs.append(Path(entry))
    # EnvWorker often inherits Isaac ``PYTHONPATH``; still look at the venv that
    # shipped TensorRT next to this interpreter.
    prefix = Path(sys.prefix) / "lib"
    if prefix.is_dir():
        dirs.extend(prefix.glob("python*/site-packages"))
    extra = Path("/opt/venv/openpi/lib/python3.11/site-packages")
    if extra.is_dir():
        dirs.append(extra)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _dirs_containing_soname(root: Path, soname: str) -> list[Path]:
    """Return directories under ``root`` that contain ``soname`` (inclusive)."""
    if not root.is_dir():
        return []
    if (root / soname).is_file():
        return [root]
    found: list[Path] = []
    for match in root.rglob(soname):
        if match.is_file():
            found.append(match.parent)
    return found


def discover_native_lib_dirs(lib_dir: Path) -> list[Path]:
    """Collect directories that may contain TensorRT / CUDA / torch natives."""
    dirs: list[Path] = [lib_dir]
    if lib_dir.is_dir():
        for child in sorted(lib_dir.iterdir()):
            if child.is_dir() and any(child.glob("*.so*")):
                dirs.append(child)
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    dirs.append(torch_lib)
    nvidia_root = torch_lib.parent / "nvidia"
    if nvidia_root.is_dir():
        dirs.extend(p for p in nvidia_root.glob("*/lib") if p.is_dir())
        dirs.extend(p for p in nvidia_root.glob("*/lib64") if p.is_dir())

    extra_roots: list[Path] = []
    env_dirs = os.environ.get("CRI_EXTRA_LIB_DIRS") or os.environ.get("TENSORRT_LIB")
    if env_dirs:
        extra_roots.extend(Path(p).expanduser() for p in env_dirs.split(":") if p)
    repo_root = os.environ.get("REPO_PATH") or os.environ.get("REPO_ROOT")
    if repo_root:
        extra_roots.append(Path(repo_root) / ".assets" / "tensorrt")
    extra_roots.append(Path("/workspace/RLinf/.assets/tensorrt"))
    for root in extra_roots:
        dirs.append(root)
        dirs.extend(_dirs_containing_soname(root, "libnvinfer.so.10"))

    for site_dir in _site_package_dirs():
        for rel in (
            "tensorrt_libs",
            "tensorrt_cu13_libs",
            "nvidia/cu13/lib",
            "nvidia/cuda_runtime/lib",
            "nvidia/cuda_runtime/lib64",
        ):
            candidate = site_dir / rel
            if candidate.is_dir():
                dirs.append(candidate)
                dirs.extend(_dirs_containing_soname(candidate, "libnvinfer.so.10"))

    # Leftover CUDA 13 wheel extract used when pip package is not installed.
    cuda13_extract = Path("/tmp/cuda13rt/extract/nvidia/cu13/lib")
    if cuda13_extract.is_dir():
        dirs.append(cuda13_extract)

    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_soname(soname: str, lib_dirs: list[Path]) -> Path | None:
    for lib_dir in lib_dirs:
        candidate = lib_dir / soname
        if candidate.is_file():
            return candidate
    return None


def _preload_native_libs(lib_dirs: list[Path]) -> None:
    """``dlopen`` required sonames with ``RTLD_GLOBAL`` so the extension can link.

    ``os.environ['LD_LIBRARY_PATH']`` is ignored for libraries loaded after
    process start; EnvWorker is already running when CRI first imports.
    """
    mode = os.RTLD_GLOBAL | os.RTLD_NOW
    missing_required = []
    for soname in _PRELOAD_SONAMES:
        path = _find_soname(soname, lib_dirs)
        if path is None:
            if soname in ("libnvinfer.so.10", "libnvonnxparser.so.10"):
                missing_required.append(soname)
            continue
        try:
            ctypes.CDLL(str(path), mode=mode)
        except OSError as exc:
            logger.warning("CRI preload skipped %s: %s", path, exc)
    if missing_required:
        searched = ", ".join(str(p) for p in lib_dirs)
        raise ImportError(
            "CRI native libs missing: "
            + ", ".join(missing_required)
            + ". Install TensorRT 10 (pip package tensorrt-libs) or set "
            "CRI_EXTRA_LIB_DIRS / TENSORRT_LIB. Searched: "
            + searched
        )


def _bootstrap_sfd_coreservice(analysis_dir: Path):
    """Import ``sfd_coreservice`` without pulling IsaacLab / Omniverse packages."""
    lib_dir = _resolve_bundled_lib_dir(analysis_dir)
    for path in (analysis_dir, lib_dir, analysis_dir / "lib"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    lib_dirs = discover_native_lib_dirs(analysis_dir / "lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [str(p) for p in lib_dirs] + [os.environ.get("LD_LIBRARY_PATH", "")]
    )
    _preload_native_libs(lib_dirs)

    # Prefer lightweight path bootstrap; fall back to IsaacLab sfd_setup when present.
    sfd_setup = analysis_dir / "sfd_setup.py"
    if sfd_setup.is_file():
        try:
            from sfd_setup import configure_cudacri  # type: ignore[import-not-found]

            configure_cudacri(analysis_dir)
        except Exception as exc:
            logger.debug("sfd_setup.configure_cudacri skipped: %s", exc)

    try:
        import sfd_coreservice  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            f"Failed to import sfd_coreservice from {lib_dir} "
            f"(python {sys.version_info.major}.{sys.version_info.minor}). {exc}"
        ) from exc

    return sfd_coreservice


def _torch_is_isaac_build() -> bool:
    """Isaac Sim's torch 2.7 cannot satisfy the CRI ``libc10`` symbols."""
    path = Path(getattr(torch, "__file__", "") or "").as_posix()
    return (
        "isaac_sim" in path
        or "isaac-sim" in path
        or "isaacsim" in path
        or "omni.isaac.ml_archive" in path
    )


def _pickle_send(buf, obj: object) -> None:
    import pickle
    import struct

    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    buf.write(struct.pack("<Q", len(payload)))
    buf.write(payload)
    buf.flush()


def _pickle_recv(buf, *, timeout_s: float | None = None) -> object:
    import pickle
    import select
    import struct

    if timeout_s is not None:
        ready, _, _ = select.select([buf], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"CRI worker did not respond within {timeout_s}s")
    header = buf.read(8)
    if len(header) < 8:
        raise EOFError("CRI worker exited before sending a reply")
    (n,) = struct.unpack("<Q", header)
    data = buf.read(n)
    if len(data) < n:
        raise EOFError("CRI worker reply truncated")
    return pickle.loads(data)


def _clean_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        raw = env.get(key, "")
        env[key] = os.pathsep.join(
            p
            for p in raw.split(os.pathsep)
            if p
            and "isaac_sim" not in p
            and "isaac-sim" not in p
            and "isaacsim" not in p
            and "omni.isaac.ml_archive" not in p
        )
    env["PYTHONNOUSERSITE"] = "1"
    return env


class _CriSubprocessClient:
    """Persistent CRI worker so EnvWorker can keep Isaac's torch loaded."""

    def __init__(self, solver_kwargs: dict) -> None:
        import subprocess

        worker = Path(__file__).resolve().parent / "sfd_worker.py"
        log_path = Path(os.environ.get("CRI_WORKER_LOG", "/tmp/rlinf_cri_worker.log"))
        self._log_file = log_path.open("ab")
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            env=_clean_worker_env(),
            cwd=str(worker.parent),
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            _pickle_send(self._proc.stdin, solver_kwargs)
            reply = _pickle_recv(self._proc.stdout, timeout_s=180)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"CRI worker failed to start (see {log_path}): {exc}"
            ) from exc
        if not reply.get("ok"):
            self.close()
            raise RuntimeError(
                f"CRI worker init failed: {reply.get('error')}\n{reply.get('traceback', '')}"
            )
        logger.info(
            "CRI solver worker pid=%s torch=%s",
            self._proc.pid,
            reply.get("torch"),
        )

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("CRI worker is not running")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        _pickle_send(self._proc.stdin, payload)
        reply = _pickle_recv(self._proc.stdout, timeout_s=60)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def compute(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        reply = self._call({"op": "compute", "q": q, "qd": qd})
        return np.asarray(reply["cri"], dtype=np.float32)

    def run_cri_filter(self, q: np.ndarray, qd: np.ndarray) -> dict[str, Any]:
        reply = self._call({"op": "run_cri_filter", "q": q, "qd": qd})
        return {
            "cri_pre": np.asarray(reply["cri_pre"], dtype=np.float32),
            "qd_cmd": np.asarray(reply["qd_cmd"], dtype=np.float32),
            "delta": np.asarray(reply["delta"], dtype=np.float32),
            "cri_limit": float(reply.get("cri_limit", CRI_FILTER_LIMIT)),
            "cbf_alpha": float(reply.get("cbf_alpha", CBF_ALPHA)),
            "approach_limit": float(reply.get("approach_limit", CRI_FILTER_LIMIT * (1.0 - CBF_ALPHA))),
            "enabled": bool(reply.get("enabled", True)),
        }

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                if proc.stdin is not None:
                    _pickle_send(proc.stdin, {"op": "stop"})
                    proc.stdin.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            self._proc = None
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
            self._log_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _as_batch_f64(x: np.ndarray | torch.Tensor, *, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.asarray(x, dtype=np.float64))
    elif isinstance(x, torch.Tensor):
        t = x.detach().to(dtype=torch.float64)
    else:
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")
    if t.ndim == 1:
        t = t[None, :]
    if t.ndim != 2:
        raise ValueError(f"{name} must have shape (B, J) or (J,), got {tuple(t.shape)}")
    return t.to(device=device, dtype=torch.float64).contiguous()


class CriSolver:
    """Thin wrapper around ``sfd_coreservice.CoreService`` for RLinf / OpenPI.

    Isaac Sim torch (docker ``rlinf:agentic-rlinf0.4-isaaclab``) cannot load
    the CRI extension in-process; a subprocess worker is used instead.

    Usage::

        solver = CriSolver(cri_filter=True)  # CBF-QP; compute() returns cri_pre
        cri = solver.compute(q, qd)  # (B, NUM_CRI_POINTS) float32
    """

    def __init__(
        self,
        analysis_dir: str | Path | None = None,
        *,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        num_joints: int = DEFAULT_NUM_JOINTS,
        zero_vel_eps: float = DEFAULT_ZERO_VEL_EPS,
        warmup_rounds: int | None = None,
        inprocess: bool | None = None,
        cri_filter: bool = False,
        cri_limit: float = CRI_FILTER_LIMIT,
        cbf_alpha: float = CBF_ALPHA,
        filter_enabled: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CriSolver requires CUDA (sfd_coreservice GPU path).")
        self.analysis_dir = resolve_analysis_dir(analysis_dir)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.device = torch.device(device or "cuda:0")
        self.num_joints = int(num_joints)
        self.zero_vel_eps = float(zero_vel_eps)
        self.num_cri_points = NUM_CRI_POINTS
        self._cri_filter = bool(cri_filter)
        self._cri_limit = float(cri_limit) if cri_limit > 0.0 else CRI_FILTER_LIMIT
        self._cbf_alpha = float(cbf_alpha) if cbf_alpha > 0.0 else CBF_ALPHA
        self._filter_enabled = bool(filter_enabled)
        self._approach_limit = self._cri_limit * (1.0 - self._cbf_alpha)
        self._remote: _CriSubprocessClient | None = None
        self._solver = None

        use_inprocess = (not _torch_is_isaac_build()) if inprocess is None else bool(inprocess)
        if not use_inprocess:
            self._remote = _CriSubprocessClient(
                {
                    "analysis_dir": str(self.analysis_dir),
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                    "num_joints": self.num_joints,
                    "zero_vel_eps": self.zero_vel_eps,
                    "warmup_rounds": warmup_rounds,
                    "cri_filter": self._cri_filter,
                    "cri_limit": self._cri_limit,
                    "cbf_alpha": self._cbf_alpha,
                    "filter_enabled": self._filter_enabled,
                }
            )
            return

        sfd = _bootstrap_sfd_coreservice(self.analysis_dir)
        self._solver = sfd.CoreService(str(self.analysis_dir), self.batch_size)
        self._solver.RunSolver_CUDA_LoadAnalysisForCRI(str(self.analysis_dir))

        if self._cri_filter:
            self._require_cri_filter_api()
            cfg = configure_cri_filter(
                self._solver,
                cri_limit=self._cri_limit,
                alpha=self._cbf_alpha,
                enabled=self._filter_enabled,
            )
            self._cri_limit = float(cfg["cri_limit"])
            self._cbf_alpha = float(cfg["cbf_alpha"])
            self._approach_limit = float(cfg["approach_limit"])
            self._filter_enabled = bool(cfg["enabled"])

        rounds = (
            int(os.environ.get("SFD_ALLOC_WARMUP_ROUNDS", "15"))
            if warmup_rounds is None
            else int(warmup_rounds)
        )
        if rounds > 0:
            q0 = torch.zeros(self.batch_size, self.num_joints, dtype=torch.float64, device=self.device)
            qd0 = torch.zeros_like(q0)
            for _ in range(rounds):
                if self._cri_filter:
                    self._solver.run_cri_filter(q0.contiguous(), qd0.contiguous())
                else:
                    self._solver.RunSolver_CUDA_CRI_AtMotionState(q0.contiguous(), qd0.contiguous())
            torch.cuda.synchronize(self.device)
            logger.info(
                "CRI solver warm-up complete: %d rounds filter=%s (%s)",
                rounds,
                self._cri_filter,
                self.analysis_dir,
            )

    def _require_cri_filter_api(self) -> None:
        if self._remote is not None:
            return
        if self._solver is None or not hasattr(self._solver, "run_cri_filter"):
            raise RuntimeError(
                "sfd_coreservice.CoreService.run_cri_filter is missing. "
                "Deploy a CoreService build that exports the CRI filter API."
            )

    def set_cri_filter_limit(self, cri_limit: float) -> None:
        self._require_cri_filter_api()
        self._cri_limit = float(cri_limit) if cri_limit > 0.0 else CRI_FILTER_LIMIT
        if self._solver is not None and hasattr(self._solver, "set_cri_filter_limit"):
            self._solver.set_cri_filter_limit(self._cri_limit)

    def set_cbf_alpha(self, alpha: float) -> None:
        self._require_cri_filter_api()
        self._cbf_alpha = float(alpha) if alpha > 0.0 else CBF_ALPHA
        if self._solver is None:
            return
        if hasattr(self._solver, "set_cbf_alpha"):
            self._solver.set_cbf_alpha(self._cbf_alpha)
        elif hasattr(self._solver, "RunSolver_CUDA_SetCbfAlpha"):
            self._solver.RunSolver_CUDA_SetCbfAlpha(self._cbf_alpha)

    def set_cri_filter_enabled(self, enabled: bool) -> None:
        self._require_cri_filter_api()
        self._filter_enabled = bool(enabled)
        if self._solver is not None and hasattr(self._solver, "set_cri_filter_enabled"):
            self._solver.set_cri_filter_enabled(self._filter_enabled)

    def _pad_batch(
        self, q: np.ndarray | torch.Tensor, qd: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        q_t = _as_batch_f64(q, name="q", device=self.device)
        qd_t = _as_batch_f64(qd, name="qd", device=self.device)
        if q_t.shape != qd_t.shape:
            raise ValueError(f"q and qd shape mismatch: {tuple(q_t.shape)} vs {tuple(qd_t.shape)}")
        batch, joints = q_t.shape
        if batch > self.batch_size:
            raise ValueError(
                f"batch {batch} exceeds solver batch_size={self.batch_size}; "
                "construct CriSolver(batch_size=...) with a larger batch."
            )
        if joints < self.num_joints:
            raise ValueError(f"expected at least {self.num_joints} joints, got {joints}")
        if batch < self.batch_size:
            q_in = torch.zeros(self.batch_size, joints, dtype=torch.float64, device=self.device)
            qd_in = torch.zeros_like(q_in)
            q_in[:batch] = q_t
            qd_in[:batch] = qd_t
        else:
            q_in = q_t
            qd_in = qd_t
        return q_in, qd_in, batch

    def _fit_cri_width(self, cri: torch.Tensor, batch: int) -> torch.Tensor:
        cri = cri[:batch]
        if cri.shape[-1] == self.num_cri_points:
            return cri
        if cri.shape[-1] > self.num_cri_points:
            head = cri[..., : self.num_cri_points]
            if torch.max(torch.abs(head)) < 1e-12:
                return cri[..., -self.num_cri_points :]
            return head
        pad = torch.zeros(batch, self.num_cri_points - cri.shape[-1], device=cri.device, dtype=cri.dtype)
        return torch.cat([cri, pad], dim=-1)

    def _as_cpu_f32(self, value: Any, batch: int) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except (TypeError, ValueError):
                return None
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return arr
        return arr[:batch]

    def compute(
        self,
        q: np.ndarray | torch.Tensor,
        qd: np.ndarray | torch.Tensor,
        *,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Compute ``CRI(q, qd)``. Filter-on returns ``cri_pre`` from CBF-QP."""
        if self._cri_filter:
            cri_np = self.run_cri_filter(q, qd)["cri_pre"]
            if return_torch:
                return torch.as_tensor(cri_np, device=self.device, dtype=torch.float32)
            return cri_np

        if self._remote is not None:
            q_np = np.asarray(
                q.detach().cpu() if isinstance(q, torch.Tensor) else q, dtype=np.float64
            )
            qd_np = np.asarray(
                qd.detach().cpu() if isinstance(qd, torch.Tensor) else qd, dtype=np.float64
            )
            cri_np = self._remote.compute(q_np, qd_np)
            if return_torch:
                return torch.as_tensor(cri_np, device=self.device, dtype=torch.float32)
            return cri_np

        q_in, qd_in, batch = self._pad_batch(q, qd)
        qd_t = qd_in[:batch]
        cri_gpu = self._solver.RunSolver_CUDA_CRI_AtMotionState(q_in.contiguous(), qd_in.contiguous())
        torch.cuda.synchronize(self.device)
        if cri_gpu is None:
            raise RuntimeError("CRI solver returned None")

        cri = self._fit_cri_width(cri_gpu, batch)
        cri = clamp_cri(cri.float())
        cri = apply_cri_zero_vel_filter(cri, qd_t, eps=self.zero_vel_eps)

        if return_torch:
            return cri
        return cri.detach().cpu().numpy().astype(np.float32, copy=False)

    def run_cri_filter(
        self,
        q: np.ndarray | torch.Tensor,
        qd: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """One IsaacLab ``run_cri_filter(q, qd_RL)`` call for a batch."""
        self._require_cri_filter_api()
        if self._remote is not None:
            q_np = np.asarray(
                q.detach().cpu() if isinstance(q, torch.Tensor) else q, dtype=np.float64
            )
            qd_np = np.asarray(
                qd.detach().cpu() if isinstance(qd, torch.Tensor) else qd, dtype=np.float64
            )
            return self._remote.run_cri_filter(q_np, qd_np)

        q_in, qd_in, batch = self._pad_batch(q, qd)
        result = self._solver.run_cri_filter(q_in.contiguous(), qd_in.contiguous())
        torch.cuda.synchronize(self.device)
        if result is None:
            raise RuntimeError("run_cri_filter returned None")
        cri_pre = result.get("cri_pre") if isinstance(result, dict) else None
        if cri_pre is None:
            raise RuntimeError("run_cri_filter returned empty cri_pre")
        if isinstance(cri_pre, np.ndarray):
            cri_t = torch.from_numpy(np.asarray(cri_pre)).to(device=self.device)
        else:
            cri_t = cri_pre
        cri_t = self._fit_cri_width(cri_t, batch)
        cri_np = clamp_cri(cri_t.float()).detach().cpu().numpy().astype(np.float32, copy=False)

        def _as_float(key: str, default: float) -> float:
            value = result.get(key, default)
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().reshape(-1)[0].item())
            return float(value)

        enabled = result.get("enabled", self._filter_enabled)
        if isinstance(enabled, torch.Tensor):
            enabled = bool(enabled.detach().cpu().reshape(-1)[0].item())
        else:
            enabled = bool(enabled)

        qd_rl = qd_in[:batch].detach().cpu().numpy().astype(np.float32, copy=False)
        qd_cmd = self._as_cpu_f32(result.get("qd_cmd"), batch)
        if qd_cmd is None or np.asarray(qd_cmd).size == 0:
            qd_cmd = qd_rl
        else:
            qd_cmd = np.asarray(qd_cmd, dtype=np.float32)
            if qd_cmd.ndim == 1:
                qd_cmd = qd_cmd[None, :]
            qd_cmd = qd_cmd[:batch]
        delta = command_delta(qd_cmd, qd_rl)

        return {
            "cri_pre": cri_np,
            "qd_cmd": qd_cmd,
            "delta": delta,
            "cri_limit": _as_float("cri_limit", self._cri_limit),
            "cbf_alpha": _as_float("cbf_alpha", self._cbf_alpha),
            "approach_limit": _as_float("approach_limit", self._approach_limit),
            "enabled": enabled,
        }
