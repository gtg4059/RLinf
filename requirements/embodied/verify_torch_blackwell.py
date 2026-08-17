#!/usr/bin/env python3
"""Verify a venv's torch is a Blackwell-capable CUDA wheel.

Docker builds usually have no GPU, so ``torch.cuda.get_arch_list()`` is empty
even when the wheel contains ``sm_120``. At build time we check the wheel's
CUDA local tag / ``torch.version.cuda`` (>= 12.8 / cu128). When a GPU is
visible we also require ``sm_120`` in the arch list.
"""

from __future__ import annotations

import re
import sys

import torch


def _cuda_tag_num_from_version(version: str) -> int | None:
    match = re.search(r"\+cu(\d+)", version)
    if not match:
        return None
    return int(match.group(1))


def _cuda_tag_num_from_cuda_str(cuda_str: str) -> int:
    major_s, minor_s = cuda_str.split(".")[:2]
    return int(major_s) * 10 + int(minor_s)


def _torch_base_tuple(version: str) -> tuple[int, ...]:
    base = version.split("+", 1)[0]
    parts: list[int] = []
    for piece in base.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def main() -> int:
    version = torch.__version__
    cuda = torch.version.cuda
    available = torch.cuda.is_available()
    archs = torch.cuda.get_arch_list() if available else []

    print(
        f"[verify_torch_blackwell] torch={version} cuda={cuda} "
        f"available={available} archs={archs}"
    )

    if not cuda:
        print(
            "[verify_torch_blackwell] ERROR: torch is CPU-only; "
            "expected a CUDA wheel (UV_TORCH_BACKEND=cu128+).",
            file=sys.stderr,
        )
        return 1

    if _torch_base_tuple(version) < (2, 7):
        print(
            f"[verify_torch_blackwell] ERROR: torch {version} is too old; "
            "need >=2.7 for Blackwell sm_120.",
            file=sys.stderr,
        )
        return 1

    tag_num = _cuda_tag_num_from_version(version)
    if tag_num is None:
        tag_num = _cuda_tag_num_from_cuda_str(cuda)
    if tag_num < 128:
        print(
            f"[verify_torch_blackwell] ERROR: CUDA tag/version too old "
            f"(got torch={version}, cuda={cuda}); need cu128+ / CUDA 12.8+.",
            file=sys.stderr,
        )
        return 1

    if available:
        if not any(a.startswith("sm_120") for a in archs):
            print(
                "[verify_torch_blackwell] ERROR: GPU is visible but arch list "
                f"has no sm_120* (archs={archs}).",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "[verify_torch_blackwell] GPU not visible (typical in docker build); "
            "skipped sm_120 runtime check. Wheel tag/CUDA version OK."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
