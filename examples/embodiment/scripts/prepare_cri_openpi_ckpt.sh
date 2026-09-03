#!/usr/bin/env bash
# Convert pi05_droid_cri LoRA JAX checkpoint -> RLinf-loadable OpenPI weights.
#
# Preferred (handles LoRA merge):
#   bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh \
#     /workspace/RLinf/49999 \
#     /workspace/RLinf/checkpoint/pi05_droid_cri_rlinf_49999
#
# Then:
#   bash examples/embodiment/run_embodiment.sh isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_ROOT="${1:-${REPO_ROOT}/49999}"
OUT_DIR="${2:-${REPO_ROOT}/checkpoint/pi05_droid_cri_rlinf_49999}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "ERROR: source checkpoint dir not found: ${SRC_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python "${REPO_ROOT}/toolkits/checkpoint_converter/convert_cri_lora_jax_to_rlinf.py" \
  --input-dir "${SRC_ROOT}" \
  --output-dir "${OUT_DIR}"

echo
echo "Done."
echo "  export CRI_OPENPI_CKPT=${OUT_DIR}"
echo "  bash examples/embodiment/run_embodiment.sh isaaclab_pick_place_cube_plate_ppo_openpi_pi05_cri"
