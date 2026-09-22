#!/usr/bin/env bash
# Evaluate converted PolaRiS OpenPI CRI-prefix weights
# (``checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000``)
# on kitchen fridge open-door. Sampling matches OpenPI serve (flow_ode).
#
# Usage:
#   bash examples/embodiment/scripts/eval_fridge_open_polaris.sh
#   POLARIS_OPENPI_CKPT=/path/to/pytorch/dir \
#     bash examples/embodiment/scripts/eval_fridge_open_polaris.sh
#   bash examples/embodiment/scripts/eval_fridge_open_polaris.sh \
#     'env.eval.total_num_envs=8'
#
# Host conda (env_isaaclab) has no ``openpi``. ``run_eval.sh`` re-enters the
# embodied-isaaclab image when the package is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_NAME="isaaclab_kitchen_open_fridge_openpi_pi05_eval"

# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/resolve_polaris_openpi_ckpt.sh"
if ! export_polaris_openpi_ckpt "${REPO_ROOT}"; then
  echo "ERROR: PolaRiS OpenPI checkpoint dir not found." >&2
  echo "Expected: ${REPO_ROOT}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000" >&2
  echo "Or set POLARIS_OPENPI_CKPT=/path/to/pytorch/dir" >&2
  echo "To rebuild from OpenPI JAX 30000 (needs jax/orbax + the params/ tree):" >&2
  echo "  bash examples/embodiment/scripts/prepare_polaris_openpi_ckpt.sh \\" >&2
  echo "    /path/to/polaris_cri_adapter_20k/30000" >&2
  exit 1
fi

echo "Evaluating PolaRiS OpenPI checkpoint dir: ${POLARIS_OPENPI_CKPT}"
cd "${REPO_ROOT}"
bash "${REPO_ROOT}/evaluations/run_eval.sh" isaaclab "${CONFIG_NAME}" "$@"
