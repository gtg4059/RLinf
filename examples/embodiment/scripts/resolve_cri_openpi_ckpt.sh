#!/usr/bin/env bash
# Resolve the converted OpenPI CRI checkpoint used by Franka/DROID cube pick-place.
#
# Search order:
#   1. $CRI_OPENPI_CKPT if it is an existing directory
#   2. <repo>/checkpoint/pi05_droid_cri_rlinf_49999
#   3. <repo>/checkpoints/pi05_droid_cri_rlinf_49999
#
# Source this file, then:
#   resolve_cri_openpi_ckpt [repo_root]   # prints the path, return 0 if found
#   export_cri_openpi_ckpt [repo_root]    # export CRI_OPENPI_CKPT when found

_cri_ckpt_has_weights() {
  local d="$1"
  [ -d "$d" ] || return 1
  [ -f "${d}/model.safetensors" ] && return 0
  # Allow sharded safetensors dumps (model-00001-of-0000N.safetensors).
  local shard
  for shard in "${d}"/*.safetensors; do
    [ -f "${shard}" ] && return 0
  done
  return 1
}

resolve_cri_openpi_ckpt() {
  local repo_root="${1:-${REPO_PATH:-${REPO_ROOT:-}}}"
  if [ -z "${repo_root}" ]; then
    echo "resolve_cri_openpi_ckpt: repo root is not set" >&2
    return 1
  fi

  if [ -n "${CRI_OPENPI_CKPT:-}" ] && [ -d "${CRI_OPENPI_CKPT}" ]; then
    printf '%s\n' "${CRI_OPENPI_CKPT}"
    return 0
  fi

  local candidate
  for candidate in \
      "${repo_root}/checkpoint/pi05_droid_cri_rlinf_49999" \
      "${repo_root}/checkpoints/pi05_droid_cri_rlinf_49999"; do
    if _cri_ckpt_has_weights "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

export_cri_openpi_ckpt() {
  local resolved
  if [ -n "${CRI_OPENPI_CKPT:-}" ] && [ -d "${CRI_OPENPI_CKPT}" ]; then
    return 0
  fi
  resolved="$(resolve_cri_openpi_ckpt "${1:-}")" || return 1
  export CRI_OPENPI_CKPT="${resolved}"
}
