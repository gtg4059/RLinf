#!/usr/bin/env bash
# Make TensorRT 10 (libnvinfer.so.10) visible to the Safetics CRI worker.
#
# The CRI subprocess cannot use Isaac Sim's torch path, so it looks for
# TensorRT next to the OpenPI venv or under .assets/tensorrt. Pin stays on
# 10.x: TensorRT 11 ships libnvinfer.so.11, which sfd_coreservice cannot load.
#
# Source this file, then:
#   export_cri_tensorrt [repo_root]   # export CRI_EXTRA_LIB_DIRS when found/installed

CRI_TENSORRT_PKG="${CRI_TENSORRT_PKG:-tensorrt-cu13-libs==10.16.1.11}"

_cri_nvinfer_in_dir() {
  local root="$1"
  [ -d "$root" ] || return 1
  if [ -f "${root}/libnvinfer.so.10" ]; then
    printf '%s\n' "$root"
    return 0
  fi
  local found
  found="$(find "$root" -name 'libnvinfer.so.10' -type f 2>/dev/null | head -n 1 || true)"
  if [ -n "$found" ]; then
    printf '%s\n' "$(dirname "$found")"
    return 0
  fi
  return 1
}

_cri_resolve_nvinfer_dir() {
  local repo_root="${1:-${REPO_PATH:-${REPO_ROOT:-}}}"
  local candidate

  if [ -n "${CRI_EXTRA_LIB_DIRS:-}" ]; then
    IFS=':' read -r -a _cri_extra <<< "${CRI_EXTRA_LIB_DIRS}"
    for candidate in "${_cri_extra[@]}"; do
      if _cri_nvinfer_in_dir "$candidate"; then
        return 0
      fi
    done
  fi
  if [ -n "${TENSORRT_LIB:-}" ] && _cri_nvinfer_in_dir "${TENSORRT_LIB}"; then
    return 0
  fi

  local py="${PYTHON:-}"
  if [ -z "$py" ] && command -v python >/dev/null 2>&1; then
    py="$(command -v python)"
  fi
  if [ -n "$py" ]; then
    local site
    site="$("$py" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)"
    if [ -n "$site" ]; then
      for candidate in \
          "${site}/tensorrt_libs" \
          "${site}/tensorrt_cu13_libs" \
          "${site}/nvidia/cu13/lib"; do
        if _cri_nvinfer_in_dir "$candidate"; then
          return 0
        fi
      done
    fi
  fi

  if [ -n "$repo_root" ] && _cri_nvinfer_in_dir "${repo_root}/.assets/tensorrt"; then
    return 0
  fi
  if _cri_nvinfer_in_dir "/workspace/RLinf/.assets/tensorrt"; then
    return 0
  fi
  return 1
}

export_cri_tensorrt() {
  local repo_root="${1:-${REPO_PATH:-${REPO_ROOT:-}}}"
  local nvinfer_dir

  if nvinfer_dir="$(_cri_resolve_nvinfer_dir "$repo_root")"; then
    export CRI_EXTRA_LIB_DIRS="${nvinfer_dir}${CRI_EXTRA_LIB_DIRS:+:${CRI_EXTRA_LIB_DIRS}}"
    return 0
  fi

  if [ -z "$repo_root" ]; then
    echo "export_cri_tensorrt: repo root is not set" >&2
    return 1
  fi

  local dest="${repo_root}/.assets/tensorrt"
  local py="${PYTHON:-}"
  if [ -z "$py" ] && command -v python >/dev/null 2>&1; then
    py="$(command -v python)"
  fi
  if [ -z "$py" ]; then
    echo "export_cri_tensorrt: no Python to install ${CRI_TENSORRT_PKG}" >&2
    return 1
  fi

  echo "Installing ${CRI_TENSORRT_PKG} into ${dest} (CRI needs libnvinfer.so.10)"
  mkdir -p "$dest"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$py" --target "$dest" "$CRI_TENSORRT_PKG"
  else
    "$py" -m pip install --target "$dest" "$CRI_TENSORRT_PKG"
  fi

  if nvinfer_dir="$(_cri_nvinfer_in_dir "$dest")"; then
    export CRI_EXTRA_LIB_DIRS="${nvinfer_dir}${CRI_EXTRA_LIB_DIRS:+:${CRI_EXTRA_LIB_DIRS}}"
    return 0
  fi
  echo "export_cri_tensorrt: ${CRI_TENSORRT_PKG} installed but libnvinfer.so.10 was not found under ${dest}" >&2
  return 1
}
