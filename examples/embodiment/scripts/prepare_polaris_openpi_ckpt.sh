#!/usr/bin/env bash
# Rebuild ``checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000``
# from the official OpenPI JAX 30000 (polaris + TacVLA CRI prefix, no LoRA).
#
# This machine already has the converted tree. Re-run only when that directory
# is missing or you have a newer JAX step.
#
# Usage:
#   bash examples/embodiment/scripts/prepare_polaris_openpi_ckpt.sh \
#     /mnt/E/openpi/checkpoints/pi05_droid_jointpos_polaris_cri_adapter/polaris_cri_adapter_20k/30000 \
#     checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000
#
# Then:
#   bash examples/embodiment/scripts/eval_fridge_open_polaris.sh
#
# Needs jax + orbax (embodied-isaaclab OpenPI venv, or a host JAX env).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_JAX="${POLARIS_OPENPI_JAX:-/mnt/E/openpi/checkpoints/pi05_droid_jointpos_polaris_cri_adapter/polaris_cri_adapter_20k/30000}"
SRC_ROOT="${1:-${DEFAULT_JAX}}"
OUT_DIR="${2:-${REPO_ROOT}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000}"

if [[ ! -d "${SRC_ROOT}/params" ]]; then
  echo "ERROR: JAX checkpoint not found (need params/): ${SRC_ROOT}" >&2
  echo "This workstation does not mount /mnt/E. Copy the OpenPI 30000 dir here, then:" >&2
  echo "  bash examples/embodiment/scripts/prepare_polaris_openpi_ckpt.sh \\" >&2
  echo "    /path/to/polaris_cri_adapter_20k/30000 \\" >&2
  echo "    ${OUT_DIR}" >&2
  echo "If ${OUT_DIR}/model.safetensors already exists, skip this and eval:" >&2
  echo "  bash examples/embodiment/scripts/eval_fridge_open_polaris.sh" >&2
  exit 1
fi

export PYTHONPATH="${REPO_PATH:-${REPO_ROOT}}:${PYTHONPATH:-}"
python "${REPO_ROOT}/toolkits/checkpoint_converter/convert_polaris_cri_adapter_jax_to_rlinf.py" \
  --input-dir "${SRC_ROOT}" \
  --output-dir "${OUT_DIR}"

echo
echo "Done."
echo "  export POLARIS_OPENPI_CKPT=${OUT_DIR}"
echo "  bash examples/embodiment/scripts/eval_fridge_open_polaris.sh"
