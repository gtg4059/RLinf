#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

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
# otherwise probe ./isaac_sim, a sibling tree, /mnt/E/isaac_sim, /workspace/isaac_sim.
if [ -z "${ISAAC_PATH:-}" ] || [ ! -f "${ISAAC_PATH}/setup_conda_env.sh" ]; then
    for _isaac_candidate in \
        "${REPO_PATH}/isaac_sim" \
        "${REPO_PATH}/../isaac_sim" \
        "/mnt/E/isaac_sim" \
        "/workspace/isaac_sim"; do
        if [ -f "${_isaac_candidate}/setup_conda_env.sh" ] && [ -f "${_isaac_candidate}/VERSION" ]; then
            ISAAC_PATH="$(cd "${_isaac_candidate}" && pwd)"
            break
        fi
    done
    unset _isaac_candidate
fi
export ISAAC_PATH="${ISAAC_PATH:-/path/to/isaac-sim}"
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Isaac Lab expects <ISAAC_LAB_PATH>/_isaac_sim -> Isaac Sim root.
if [ -n "${ISAAC_LAB_PATH:-}" ] && [ -d "${ISAAC_LAB_PATH}" ] && [ -d "${ISAAC_PATH}" ]; then
    ln -sfn "${ISAAC_PATH}" "${ISAAC_LAB_PATH}/_isaac_sim"
fi

# Populate PYTHONPATH / CARB paths so `import isaacsim` works in env workers.
if [ -f "${ISAAC_PATH}/setup_conda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ISAAC_PATH}/setup_conda_env.sh"
    echo "Using ISAAC_PATH=${ISAAC_PATH}"
fi

# Isaac Lab Arena Robolab assets → always resolve to a *local* directory under
# .assets (download from S3 staging if missing). Remote URL roots are ignored.
# shellcheck disable=SC1091
source "${REPO_PATH}/examples/embodiment/scripts/setup_arena_assets.sh"
setup_arena_assets

# Blackwell / torch 2.11 images often ship a flash-attn wheel with a broken ABI.
# Drop it so transformers falls back to PyTorch SDPA (OpenPI uses eager/SDPA anyway).
# Runtime also guards this in rlinf.utils.flash_attn.disable_broken_flash_attn.
if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" \
    && ! python -c "import flash_attn" >/dev/null 2>&1; then
    echo "Removing broken flash-attn (ABI mismatch with current torch)."
    PY_BIN="$(command -v python)"
    if command -v uv >/dev/null 2>&1; then
        uv pip uninstall -y --python "${PY_BIN}" flash-attn >/dev/null 2>&1 || true
    else
        "${PY_BIN}" -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
    fi
fi

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

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

echo "Using Python at $(which python)"
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

CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}")
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
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
