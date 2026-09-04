#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
_RLINF_ORIG_ARGS=("$@")

# Optional machine-local ISAAC_* / CRI_OPENPI_CKPT / IMAGE_TAG.
# shellcheck disable=SC1091
source "${REPO_PATH}/examples/embodiment/scripts/source_isaaclab_local_env.sh"
source_isaaclab_local_env "${REPO_PATH}"

# Preferred OpenPI venv shipped in the embodied-isaaclab image (u24 / blackwell).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /opt/venv/openpi/bin/activate ]; then
    # shellcheck disable=SC1091
    source /opt/venv/openpi/bin/activate
fi

_rlinf_find_python() {
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    return 1
}

_rlinf_python_has_torch() {
    local py="$1"
    "${py}" -c "import torch" >/dev/null 2>&1
}

# Host checkout has no `python` and no OpenPI torch. Re-enter the image.
# shellcheck disable=SC1091
source "${REPO_PATH}/docker/runtime_mounts.sh"
_RLINF_IMAGE_TAG="$(rlinf_resolve_isaaclab_image)"
if ! PYTHON="$(_rlinf_find_python)" || ! _rlinf_python_has_torch "${PYTHON}"; then
    if [ ! -f /.dockerenv ] && [ "${RLINF_NO_DOCKER:-0}" != "1" ] \
        && command -v docker >/dev/null 2>&1 \
        && docker image inspect "${_RLINF_IMAGE_TAG}" >/dev/null 2>&1; then
        echo "Host Python is missing or has no torch. Re-launching inside ${_RLINF_IMAGE_TAG}" >&2
        IMAGE_TAG="${_RLINF_IMAGE_TAG}" exec bash "${REPO_PATH}/docker/run_embodied_isaaclab_blackwell.sh" -- \
            bash examples/embodiment/run_embodiment.sh "${_RLINF_ORIG_ARGS[@]}"
    fi
    echo "ERROR: no usable Python with torch found." >&2
    echo "OpenPI + IsaacLab training needs the embodied-isaaclab image:" >&2
    echo "  bash docker/run_embodied_isaaclab_blackwell.sh" >&2
    echo "  source switch_env openpi" >&2
    echo "  bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh" >&2
    exit 1
fi
export PYTHON

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Isaac Sim is a separate install (not in the Docker image). Prefer ISAAC_PATH;
# otherwise probe ./isaac_sim, this checkout (Sim extracted into the repo),
# a sibling tree, /mnt/E/isaac_sim, /workspace/isaac_sim.
if [ -z "${ISAAC_PATH:-}" ] || [ ! -f "${ISAAC_PATH}/setup_conda_env.sh" ]; then
    _isaac_resolved="$(rlinf_resolve_isaac_sim "${REPO_PATH}" || true)"
    if [ -n "${_isaac_resolved}" ]; then
        ISAAC_PATH="${_isaac_resolved}"
    fi
    unset _isaac_resolved
fi
export ISAAC_PATH="${ISAAC_PATH:-/path/to/isaac-sim}"
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Isaac Kit persists pip envs, user.config.json, and RTX caches here.
# Create them up front so EnvWorkers do not fail on a missing/read-only tree.
if [ -d "${ISAAC_PATH}/kit" ]; then
    mkdir -p \
        "${ISAAC_PATH}/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default" \
        "${ISAAC_PATH}/kit/cache/Kit" \
        "${HOME}/Documents/Kit/shared" || true
fi

# Isaac Lab expects <ISAAC_LAB_PATH>/_isaac_sim -> Isaac Sim root.
if [ -n "${ISAAC_LAB_PATH:-}" ] && [ -d "${ISAAC_LAB_PATH}" ] && [ -d "${ISAAC_PATH}" ]; then
    ln -sfn "${ISAAC_PATH}" "${ISAAC_LAB_PATH}/_isaac_sim"
fi

# Populate PYTHONPATH / CARB paths so `import isaacsim` works in env workers.
# setup_python_env.sh also appends omni.isaac.ml_archive, which shadows the
# OpenPI venv torch. Env workers keep that path; Cluster.allocate strips it
# for actor/rollout so FSDP uses a Blackwell-capable torch.
if [ -f "${ISAAC_PATH}/setup_conda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ISAAC_PATH}/setup_conda_env.sh"
    echo "Using ISAAC_PATH=${ISAAC_PATH}"
    if [[ ":${PYTHONPATH}:" == *omni.isaac.ml_archive* ]]; then
        echo "Isaac ml_archive is on PYTHONPATH (kept for EnvWorkers; stripped for actor/rollout)."
    fi
fi

# Isaac Lab Arena Robolab assets → always resolve to a *local* directory under
# .assets (download from S3 staging if missing). Remote URL roots are ignored.
# shellcheck disable=SC1091
source "${REPO_PATH}/examples/embodiment/scripts/setup_arena_assets.sh"
setup_arena_assets

# Blackwell / torch 2.11 images often ship a flash-attn wheel with a broken ABI.
# Drop it so transformers falls back to PyTorch SDPA (OpenPI uses eager/SDPA anyway).
# Runtime also guards this in rlinf.utils.flash_attn.disable_broken_flash_attn.
if "${PYTHON}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" \
    && ! "${PYTHON}" -c "import flash_attn" >/dev/null 2>&1; then
    echo "Removing broken flash-attn (ABI mismatch with current torch)."
    PY_BIN="${PYTHON}"
    if command -v uv >/dev/null 2>&1; then
        uv pip uninstall -y --python "${PY_BIN}" flash-attn >/dev/null 2>&1 || true
    else
        "${PY_BIN}" -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
    fi
fi

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

# Converted OpenPI CRI weights (Franka/DROID cube pick-place). Prefer
# checkpoint/pi05_droid_cri_rlinf_49999, then checkpoints/.
# shellcheck disable=SC1091
source "${REPO_PATH}/examples/embodiment/scripts/resolve_cri_openpi_ckpt.sh"
if export_cri_openpi_ckpt "${REPO_PATH}"; then
    echo "Using CRI_OPENPI_CKPT=${CRI_OPENPI_CKPT}"
fi

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
fi
shift $(( $# > 0 ? 1 : 0 ))

# NOTE: Set the active robot platform (required for correct action dimension and
# normalization). Supported platforms: LIBERO, ALOHA, BRIDGE (default LIBERO).
# Remaining args that look like Hydra overrides (contain '=') are forwarded to
# the Python entrypoint. A bare platform token may still be passed as $2.
CLI_OVERRIDES=()
if [ "$#" -gt 0 ] && [[ "$1" != *"="* ]]; then
    ROBOT_PLATFORM=$1
    shift
fi
ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
export ROBOT_PLATFORM

while [ "$#" -gt 0 ]; do
    CLI_OVERRIDES+=("$1")
    shift
done

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

if [[ "${CONFIG_NAME}" == *pick_place_cube_plate*cri* ]]; then
    if [ -z "${CRI_OPENPI_CKPT:-}" ] || [ ! -d "${CRI_OPENPI_CKPT}" ]; then
        echo "ERROR: converted OpenPI CRI checkpoint not found." >&2
        echo "Expected: ${REPO_PATH}/checkpoint/pi05_droid_cri_rlinf_49999" >&2
        echo "Or set CRI_OPENPI_CKPT=/path/to/pi05_droid_cri_rlinf_49999" >&2
        echo "Or convert JAX 49999 with:" >&2
        echo "  bash examples/embodiment/scripts/prepare_cri_openpi_ckpt.sh" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${REPO_PATH}/examples/embodiment/scripts/ensure_cri_tensorrt.sh"
    if export_cri_tensorrt "${REPO_PATH}"; then
        echo "Using CRI_EXTRA_LIB_DIRS=${CRI_EXTRA_LIB_DIRS}"
    else
        echo "ERROR: TensorRT 10 (libnvinfer.so.10) is required for online CRI." >&2
        echo "Install tensorrt-cu13-libs==10.16.1.11 or set CRI_EXTRA_LIB_DIRS." >&2
        exit 1
    fi
fi

echo "Using Python at ${PYTHON}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
# Forward optional overrides exported by callers (e.g. tests/parity_tests/run_all.sh).
# Sentinel: "-2" means "do not override, use YAML default". -1 is a legitimate value
# (e.g. runner.max_steps=-1 means unlimited) and is forwarded as-is.
EXTRA_OVERRIDES=""
[ -n "${STEPS:-}" ]      && [ "$STEPS"      != "-2" ] && EXTRA_OVERRIDES+=" runner.max_steps=${STEPS}"
[ -n "${SAVE_INTER:-}" ] && [ "$SAVE_INTER" != "-2" ] && EXTRA_OVERRIDES+=" runner.save_interval=${SAVE_INTER}"
[ -n "${NODES:-}" ]      && [ "$NODES"      != "-2" ] && EXTRA_OVERRIDES+=" cluster.num_nodes=${NODES}"

CMD=("${PYTHON}" "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}")
# shellcheck disable=SC2206
ENV_OVERRIDES=(${EXTRA_OVERRIDES})
if [ "${#ENV_OVERRIDES[@]}" -gt 0 ]; then
    CMD+=("${ENV_OVERRIDES[@]}")
fi
if [ "${#CLI_OVERRIDES[@]}" -gt 0 ]; then
    CMD+=("${CLI_OVERRIDES[@]}")
fi
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
set -o pipefail
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
status=$?
set +o pipefail
exit "${status}"
