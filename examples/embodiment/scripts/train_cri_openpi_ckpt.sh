#!/usr/bin/env bash
# Train OpenPI π₀.₅ CRI PPO from ``checkpoint/pi05_droid_cri_rlinf_49999``.
#
# Host entry: bind-mounts this checkout into the locally built embodied-isaaclab
# image (``rlinf:embodied-isaaclab-u24``, then ``rlinf:embodied-isaaclab-blackwell``)
# and launches
# ``isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri``.
#
# Usage:
#   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
#   CRI_OPENPI_CKPT=/path/to/pi05_droid_cri_rlinf_49999 \
#     bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
#   IMAGE_TAG=rlinf:embodied-isaaclab-u24 \
#     bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
#   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh runner.max_epochs=10
#
# Dry-run (print resolved paths and the launch command, then exit):
#   RLINF_TRAIN_CRI_DRY_RUN=1 bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_NAME="isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri"
OVERRIDES=("$@")

_usage() {
  sed -n '2,20p' "$0"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  _usage
  exit 0
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/source_isaaclab_local_env.sh"
source_isaaclab_local_env "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/resolve_cri_openpi_ckpt.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/docker/runtime_mounts.sh"

if ! export_cri_openpi_ckpt "${REPO_ROOT}"; then
  echo "ERROR: converted OpenPI CRI checkpoint not found." >&2
  echo "Expected: ${REPO_ROOT}/checkpoint/pi05_droid_cri_rlinf_49999" >&2
  echo "Or set CRI_OPENPI_CKPT=/path/to/pi05_droid_cri_rlinf_49999" >&2
  echo "Or convert JAX 49999 with:" >&2
  echo "  bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh" >&2
  exit 1
fi

IMAGE_TAG="$(rlinf_resolve_isaaclab_image)"
export IMAGE_TAG
export CRI_OPENPI_CKPT

LAUNCH_MODE="local"
CMD=(
  bash "${REPO_ROOT}/examples/embodiment/run_embodiment.sh"
  "${CONFIG_NAME}"
)
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
  CMD+=("${OVERRIDES[@]}")
fi

_in_docker() {
  [ -f /.dockerenv ]
}

_image_present() {
  command -v docker >/dev/null 2>&1 \
    && docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1
}

if ! _in_docker && [ "${RLINF_NO_DOCKER:-0}" != "1" ] && _image_present; then
  LAUNCH_MODE="docker"
  CMD=(
    bash "${REPO_ROOT}/docker/run_embodied_isaaclab_blackwell.sh"
    --
    bash examples/embodiment/run_embodiment.sh
    "${CONFIG_NAME}"
  )
  if [ "${#OVERRIDES[@]}" -gt 0 ]; then
    CMD+=("${OVERRIDES[@]}")
  fi
fi

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "CRI_OPENPI_CKPT=${CRI_OPENPI_CKPT}"
echo "IMAGE_TAG=${IMAGE_TAG}"
echo "LAUNCH_MODE=${LAUNCH_MODE}"
printf 'CMD='
printf '%q ' "${CMD[@]}"
echo

if [ "${RLINF_TRAIN_CRI_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

if [ "${LAUNCH_MODE}" = "docker" ]; then
  echo "Launching CRI PPO inside ${IMAGE_TAG} from ${CRI_OPENPI_CKPT}"
elif ! _in_docker && [ "${RLINF_NO_DOCKER:-0}" != "1" ]; then
  echo "WARNING: Docker image ${IMAGE_TAG} not found; training on the host." >&2
  echo "  Build with: bash docker/build_embodied_isaaclab_u24.sh --tag ${IMAGE_TAG}" >&2
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"
