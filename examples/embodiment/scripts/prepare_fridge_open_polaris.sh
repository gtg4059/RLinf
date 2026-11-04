#!/usr/bin/env bash
# Install what train_fridge_open_polaris.sh needs on this checkout.
#
# The launcher does not use the repo .venv and does not use the published
# rlinf/rlinf:agentic-rlinf0.4-isaaclab image. It enters a locally built
# image (rlinf:embodied-isaaclab-u24, else rlinf:embodied-isaaclab-blackwell)
# whose OpenPI env is /opt/venv/openpi.
#
# Place the external zip at the repo root before running this (checkpoint/,
# .assets/isaaclab_arena/, and cri/lib plus cri/Engine). Isaac Sim 5.1.0 is
# not in that zip: this host keeps the standalone tree extracted into the
# repo root. A sibling isaac_sim/ or ISAAC_SIM_PATH also works.
#
# Usage (from anywhere):
#   bash examples/embodiment/scripts/prepare_fridge_open_polaris.sh
#   bash examples/embodiment/scripts/prepare_fridge_open_polaris.sh --check-only
#
# --check-only verifies the local tree and prints whether the image exists.
# It does not build. Exit 0 when the tree is complete. Exit 1 when a local
# file is missing. The image is reported, not required, in --check-only.
#
# Without --check-only, a missing image runs docker/build_embodied_isaaclab_u24.sh.
# That build needs tens of GB free and a NVIDIA Docker host.

set -euo pipefail

REPO_ROOT="${RLINF_PREPARE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHECK_ONLY=0

if [ "${1:-}" = "--check-only" ]; then
  CHECK_ONLY=1
elif [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,24p' "$0"
  exit 0
elif [ "$#" -gt 0 ]; then
  echo "ERROR: unknown argument: $1" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${REPO_ROOT}/docker/runtime_mounts.sh"

CKPT="${REPO_ROOT}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000"
CRI_DIR="${REPO_ROOT}/rlinf/envs/isaaclab/tasks/pick_place_cube_plate/cri"
ARENA="${REPO_ROOT}/.assets/isaaclab_arena"

REQUIRED=(
  "${CKPT}/model.safetensors"
  "${CKPT}/config.json"
  "${ARENA}/background_library/lightwheel_kitchen_one_wall_coastal/scene.usd"
  "${ARENA}/lightwheel_sdk/floorplan/robocasa-robocasakitchen-1-1/scene.usd"
  "${ARENA}/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
  "${ARENA}/object_library/srl_robolab_assets/robots/franka_stand_grey.usda"
  "${CRI_DIR}/lib/libSFD_CoreService.so"
  "${CRI_DIR}/Engine/Chest_Shape5_Cover_off_Force/model.onnx"
  "${CRI_DIR}/Engine/Chest_Shape5_Cover_off_Pressure/model.onnx"
)

missing=0
for path in "${REQUIRED[@]}"; do
  if [ -f "${path}" ]; then
    echo "ok ${path#"${REPO_ROOT}/"}"
  else
    echo "MISSING ${path#"${REPO_ROOT}/"}" >&2
    missing=1
  fi
done

if rlinf_resolve_isaac_sim "${REPO_ROOT}" >/dev/null; then
  echo "ok isaac_sim=$(rlinf_resolve_isaac_sim "${REPO_ROOT}")"
else
  echo "MISSING Isaac Sim 5.1.0 (setup_conda_env.sh + VERSION)." >&2
  echo "Extract the standalone tree into the repo root, or into isaac_sim/, or set ISAAC_SIM_PATH." >&2
  echo "  mkdir -p isaac_sim && cd isaac_sim" >&2
  echo "  wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip" >&2
  echo "  unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip" >&2
  missing=1
fi

IMAGE_TAG="$(rlinf_resolve_isaaclab_image)"
image_present=0
if command -v docker >/dev/null 2>&1 && docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  image_present=1
  echo "ok image=${IMAGE_TAG}"
else
  echo "MISSING image=${IMAGE_TAG}" >&2
fi

if [ "${missing}" -ne 0 ]; then
  echo "Place rlinf-external-assets.zip contents at ${REPO_ROOT} first." >&2
  exit 1
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
  if [ "${image_present}" -eq 1 ]; then
    echo "ready: bash examples/embodiment/scripts/train_fridge_open_polaris.sh"
  else
    echo "local tree ready; image still missing"
  fi
  exit 0
fi

if [ "${image_present}" -eq 0 ]; then
  echo "Building ${IMAGE_TAG} (venv /opt/venv/openpi, Isaac Lab 2.3.0, torch cu128)."
  bash "${REPO_ROOT}/docker/build_embodied_isaaclab_u24.sh"
fi

echo "ready: bash examples/embodiment/scripts/train_fridge_open_polaris.sh"
