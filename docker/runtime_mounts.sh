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

# Container path for the converted OpenPI CRI checkpoint living in this checkout.
rlinf_container_cri_ckpt() {
  local repo_root="$1"
  if [ -n "${CRI_OPENPI_CKPT:-}" ]; then
    printf '%s\n' "${CRI_OPENPI_CKPT}"
    return 0
  fi
  if [ -d "${repo_root}/checkpoints/pi05_droid_cri_rlinf_49999" ]; then
    printf '%s\n' "/workspace/RLinf/checkpoints/pi05_droid_cri_rlinf_49999"
    return 0
  fi
  return 1
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
