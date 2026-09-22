#!/usr/bin/env bash
# Resolve the RLinf-loadable PolaRiS OpenPI dir used by kitchen fridge open-door.
#
# Search order:
#   1. $POLARIS_OPENPI_CKPT if it is an existing directory
#   2. <repo>/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000
#   3. <repo>/checkpoints/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000
#   4. <repo>/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999
#   5. <repo>/checkpoints/pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999
#   6. <repo>/checkpoint/pi05_droid_jointpos_polaris_cri_rlinf_10000
#   7. <repo>/checkpoints/pi05_droid_jointpos_polaris_cri_rlinf_10000
#   8. <repo>/checkpoint/pi05_droid_jointpos_polaris
#   9. <repo>/checkpoints/torch/pi05_droid_polaris
#
# Source this file, then:
#   resolve_polaris_openpi_ckpt [repo_root]   # prints the path, return 0 if found
#   export_polaris_openpi_ckpt [repo_root]    # export POLARIS_OPENPI_CKPT when found

_polaris_ckpt_has_weights() {
  local d="$1"
  [ -d "$d" ] || return 1
  [ -f "${d}/model.safetensors" ] && return 0
  local shard
  for shard in "${d}"/*.safetensors; do
    [ -f "${shard}" ] && return 0
  done
  return 1
}

resolve_polaris_openpi_ckpt() {
  local repo_root="${1:-${REPO_PATH:-${REPO_ROOT:-}}}"
  if [ -z "${repo_root}" ]; then
    echo "resolve_polaris_openpi_ckpt: repo root is not set" >&2
    return 1
  fi

  if [ -n "${POLARIS_OPENPI_CKPT:-}" ] && [ -d "${POLARIS_OPENPI_CKPT}" ]; then
    printf '%s\n' "${POLARIS_OPENPI_CKPT}"
    return 0
  fi

  local candidate
  for candidate in \
      "${repo_root}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000" \
      "${repo_root}/checkpoints/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000" \
      "${repo_root}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999" \
      "${repo_root}/checkpoints/pi05_droid_jointpos_polaris_cri_adapter_rlinf_19999" \
      "${repo_root}/checkpoint/pi05_droid_jointpos_polaris_cri_rlinf_10000" \
      "${repo_root}/checkpoints/pi05_droid_jointpos_polaris_cri_rlinf_10000" \
      "${repo_root}/checkpoint/pi05_droid_jointpos_polaris" \
      "${repo_root}/checkpoints/torch/pi05_droid_polaris"; do
    if _polaris_ckpt_has_weights "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

export_polaris_openpi_ckpt() {
  local resolved
  if [ -n "${POLARIS_OPENPI_CKPT:-}" ] && [ -d "${POLARIS_OPENPI_CKPT}" ]; then
    return 0
  fi
  resolved="$(resolve_polaris_openpi_ckpt "${1:-}")" || return 1
  export POLARIS_OPENPI_CKPT="${resolved}"
}
