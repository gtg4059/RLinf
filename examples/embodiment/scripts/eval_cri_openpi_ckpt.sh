#!/usr/bin/env bash
# Evaluate converted OpenPI CRI weights under ``checkpoint/pi05_droid_cri_rlinf_49999``
# (or ``$CRI_OPENPI_CKPT``) on the Isaac Lab pick-place cube→bowl task.
#
# Usage:
#   # Score the default checkpoint/ tree (no RL full_weights):
#   bash examples/embodiment/scripts/eval_cri_openpi_ckpt.sh
#
#   # Override the OpenPI directory:
#   CRI_OPENPI_CKPT=/path/to/pi05_droid_cri_rlinf \
#     bash examples/embodiment/scripts/eval_cri_openpi_ckpt.sh
#
#   # Score an RL actor full_weights.pt (adds value head load):
#   CKPT_PATH=/path/to/global_step_N/actor/model_state_dict/full_weights.pt \
#     bash examples/embodiment/scripts/eval_cri_openpi_ckpt.sh
#
# Extra Hydra overrides are forwarded, e.g.:
#   bash examples/embodiment/scripts/eval_cri_openpi_ckpt.sh \
#     'env.eval.total_num_envs=1' 'env.eval.rollout_epoch=1'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_NAME="isaaclab_pick_place_cube_plate_openpi_pi05_cri_ckpt_eval"

# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/resolve_cri_openpi_ckpt.sh"
if ! export_cri_openpi_ckpt "${REPO_ROOT}"; then
  echo "ERROR: OpenPI checkpoint dir not found." >&2
  echo "Expected: ${REPO_ROOT}/checkpoint/pi05_droid_cri_rlinf_49999" >&2
  echo "Prepare with:" >&2
  echo "  bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh \\" >&2
  echo "    ${REPO_ROOT}/49999 \\" >&2
  echo "    ${REPO_ROOT}/checkpoint/pi05_droid_cri_rlinf_49999" >&2
  exit 1
fi

EXTRA_ARGS=("$@")
if [ -n "${CKPT_PATH:-}" ]; then
  if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: CKPT_PATH is not a file: ${CKPT_PATH}" >&2
    exit 1
  fi
  EXTRA_ARGS+=(
    "runner.ckpt_path=${CKPT_PATH}"
    "rollout.model.add_value_head=True"
    "rollout.model.openpi.value_after_vlm=True"
    "rollout.model.openpi.value_vlm_mode=mean_token"
    "rollout.model.openpi.detach_critic_input=True"
  )
  echo "Evaluating RL full_weights: ${CKPT_PATH}"
else
  echo "Evaluating OpenPI CRI checkpoint dir: ${CRI_OPENPI_CKPT}"
fi

cd "${REPO_ROOT}"
bash "${REPO_ROOT}/evaluations/run_eval.sh" isaaclab "${CONFIG_NAME}" "${EXTRA_ARGS[@]}"
