# Shared host-side mount helpers for docker/run_*.sh
# shellcheck shell=bash

# Locate a separately installed Isaac Sim 5.x standalone tree (not in the image).
# Order: ISAAC_SIM_PATH, <repo>/isaac_sim, <repo>/../isaac_sim, /mnt/E/isaac_sim.
rlinf_resolve_isaac_sim() {
  local repo_root="$1"
  local candidate
  if [ -n "${ISAAC_SIM_PATH:-}" ]; then
    candidate="${ISAAC_SIM_PATH}"
    if [ -f "${candidate}/setup_conda_env.sh" ] && [ -f "${candidate}/VERSION" ]; then
      (cd "${candidate}" && pwd)
      return 0
    fi
  fi
  for candidate in \
      "${repo_root}/isaac_sim" \
      "${repo_root}/../isaac_sim" \
      "/mnt/E/isaac_sim"; do
    if [ -f "${candidate}/setup_conda_env.sh" ] && [ -f "${candidate}/VERSION" ]; then
      (cd "${candidate}" && pwd)
      return 0
    fi
  done
  return 1
}

# Host -> container path for the converted OpenPI CRI checkpoint.
# Prefers checkpoint/pi05_droid_cri_rlinf_49999, then checkpoints/.
_RLINF_RUNTIME_MOUNTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RLINF_RESOLVE_CRI_SCRIPT="$(cd "${_RLINF_RUNTIME_MOUNTS_DIR}/.." && pwd)/examples/embodiment/scripts/resolve_cri_openpi_ckpt.sh"

rlinf_container_cri_ckpt() {
  local repo_root="$1"
  if [ -f "${_RLINF_RESOLVE_CRI_SCRIPT}" ]; then
    # shellcheck disable=SC1090
    source "${_RLINF_RESOLVE_CRI_SCRIPT}"
  fi

  local host_path=""
  if [ -n "${CRI_OPENPI_CKPT:-}" ] && [ -d "${CRI_OPENPI_CKPT}" ]; then
    host_path="${CRI_OPENPI_CKPT}"
  elif command -v resolve_cri_openpi_ckpt >/dev/null 2>&1; then
    host_path="$(resolve_cri_openpi_ckpt "${repo_root}")" || host_path=""
  fi
  if [ -z "${host_path}" ]; then
    return 1
  fi

  local repo_abs
  repo_abs="$(cd "${repo_root}" && pwd)"
  case "${host_path}" in
    /workspace/RLinf/*)
      printf '%s\n' "${host_path}"
      ;;
    "${repo_abs}"/*)
      printf '%s\n' "/workspace/RLinf/${host_path#"${repo_abs}"/}"
      ;;
    *)
      printf '%s\n' "${host_path}"
      ;;
  esac
}

# Prefer a locally built tag, then the published image the live `rlinf` container uses.
rlinf_resolve_image() {
  local preferred="${1:-}"
  local fallback="${2:-}"
  if [ -n "${IMAGE_TAG:-}" ]; then
    printf '%s\n' "${IMAGE_TAG}"
    return 0
  fi
  local tag
  for tag in ${preferred} ${fallback}; do
    [ -z "${tag}" ] && continue
    if docker image inspect "${tag}" >/dev/null 2>&1; then
      printf '%s\n' "${tag}"
      return 0
    fi
  done
  printf '%s\n' "${preferred:-${fallback}}"
}
