#!/usr/bin/env bash
# Train kitchen fridge open-door PPO from the PolarIS CRI-prefix adapter
# (default: checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000).
#
# Host entry: bind-mounts this checkout into the local embodied-isaaclab
# image and launches the named Hydra config. First positional is the config
# (same style as run_embodiment.sh); later tokens are Hydra overrides.
# Collocated actor+env+rollout on all visible GPUs, 32 train / 4 eval
# envs per GPU. Train episodes are 600 steps and two rollout epochs, so
# chunks/rank stay 32*2*(600/15) = 2560.
# This host is 2× Blackwell → 64 train / 8 eval / global_batch 5120.
# Override RLINF_NUM_GPUS or pass Hydra env.train.total_num_envs=… to pin.
# Stop a previous fridge PPO / eval container first (they share GPUs).
# Do not resume a collapsed language-binned 10000 CRI run.
#
# Usage:
#   bash examples/embodiment/scripts/train_fridge_open_polaris.sh
#   POLARIS_OPENPI_CKPT=/path/to/pytorch/dir \
#     bash examples/embodiment/scripts/train_fridge_open_polaris.sh \
#     isaaclab_kitchen_open_fridge_ppo_openpi_pi05
#   bash examples/embodiment/scripts/train_fridge_open_polaris.sh \
#     isaaclab_kitchen_open_fridge_ppo_openpi_pi05 runner.max_epochs=10
#
# Dry-run:
#   RLINF_TRAIN_FRIDGE_DRY_RUN=1 bash examples/embodiment/scripts/train_fridge_open_polaris.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_CONFIG_NAME="isaaclab_kitchen_open_fridge_ppo_openpi_pi05"

_usage() {
  sed -n '2,21p' "$0"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  _usage
  exit 0
fi

# Same argv style as run_embodiment.sh: optional config, then Hydra overrides.
CONFIG_NAME="${DEFAULT_CONFIG_NAME}"
OVERRIDES=()
if [ "$#" -gt 0 ] && [[ "$1" != *"="* ]]; then
  CONFIG_NAME="$1"
  shift
fi
if [ "$#" -gt 0 ]; then
  OVERRIDES=("$@")
fi

CONFIG_YAML="${REPO_ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml"
if [ ! -f "${CONFIG_YAML}" ]; then
  echo "ERROR: Hydra config not found: ${CONFIG_YAML}" >&2
  echo "Pass a name under examples/embodiment/config/ (without .yaml)." >&2
  exit 1
fi

# 32 train / 4 eval envs per GPU. Train rollout_epoch is 2 and episodes
# are 600 steps, so per-rank chunks stay (32)*2*(600/15) = 2560.
# global_batch = 2560 * n_gpus. 64 envs/GPU OOM'd the actor update.
TRAIN_ENVS_PER_GPU=32
EVAL_ENVS_PER_GPU=4
CHUNKS_PER_RANK=2560

_visible_gpu_count() {
  if [ -n "${RLINF_NUM_GPUS:-}" ]; then
    printf '%s\n' "${RLINF_NUM_GPUS}"
    return
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "-1" ]; then
    awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}"
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l
}

_has_hydra_override() {
  local key="$1"
  local o
  for o in "${OVERRIDES[@]}"; do
    case "${o}" in
      "${key}="*) return 0 ;;
    esac
  done
  return 1
}

N_GPUS="$(_visible_gpu_count)"
N_GPUS="${N_GPUS//[[:space:]]/}"
if [ -z "${N_GPUS}" ] || [ "${N_GPUS}" -lt 1 ]; then
  echo "ERROR: no visible GPUs (set RLINF_NUM_GPUS or CUDA_VISIBLE_DEVICES)." >&2
  exit 1
fi
TRAIN_ENVS=$((TRAIN_ENVS_PER_GPU * N_GPUS))
EVAL_ENVS=$((EVAL_ENVS_PER_GPU * N_GPUS))
GLOBAL_BATCH=$((CHUNKS_PER_RANK * N_GPUS))
if ! _has_hydra_override "env.train.total_num_envs"; then
  OVERRIDES+=("env.train.total_num_envs=${TRAIN_ENVS}")
fi
if ! _has_hydra_override "env.eval.total_num_envs"; then
  OVERRIDES+=("env.eval.total_num_envs=${EVAL_ENVS}")
fi
if ! _has_hydra_override "actor.global_batch_size"; then
  OVERRIDES+=("actor.global_batch_size=${GLOBAL_BATCH}")
fi

# Kitchen Isaac + actor share the GPU; avoid reserved-but-unusable fragments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/source_isaaclab_local_env.sh"
source_isaaclab_local_env "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/resolve_polaris_openpi_ckpt.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/docker/runtime_mounts.sh"

if ! export_polaris_openpi_ckpt "${REPO_ROOT}"; then
  echo "ERROR: PolaRiS OpenPI checkpoint not found." >&2
  echo "Expected: ${REPO_ROOT}/checkpoint/pi05_droid_jointpos_polaris_cri_adapter_rlinf_30000" >&2
  echo "Or set POLARIS_OPENPI_CKPT=/path/to/pytorch/dir" >&2
  exit 1
fi

IMAGE_TAG="$(rlinf_resolve_isaaclab_image)"
export IMAGE_TAG
export POLARIS_OPENPI_CKPT
export CONTAINER_NAME="${CONTAINER_NAME:-rlinf-isaaclab-blackwell}"

_in_docker() {
  [ -f /.dockerenv ]
}

_image_present() {
  command -v docker >/dev/null 2>&1 \
    && docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1
}

# Host checkout paths are not visible inside the image; rewrite to the mount.
_container_polaris_ckpt() {
  local ckpt="$1"
  case "${ckpt}" in
    "${REPO_ROOT}"/*)
      printf '%s\n' "/workspace/RLinf/${ckpt#"${REPO_ROOT}"/}"
      ;;
    *)
      printf '%s\n' "${ckpt}"
      ;;
  esac
}

LAUNCH_MODE="local"
INNER=(
  bash examples/embodiment/run_embodiment.sh
  "${CONFIG_NAME}"
)
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
  INNER+=("${OVERRIDES[@]}")
fi

CMD=(
  bash "${REPO_ROOT}/examples/embodiment/run_embodiment.sh"
  "${CONFIG_NAME}"
)
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
  CMD+=("${OVERRIDES[@]}")
fi

if ! _in_docker && [ "${RLINF_NO_DOCKER:-0}" != "1" ] && _image_present; then
  LAUNCH_MODE="docker"
  CONTAINER_CKPT="$(_container_polaris_ckpt "${POLARIS_OPENPI_CKPT}")"
  export POLARIS_OPENPI_CKPT="${CONTAINER_CKPT}"
  if [ -t 0 ] && [ -t 1 ]; then
    CMD=(
      bash "${REPO_ROOT}/docker/run_embodied_isaaclab_blackwell.sh"
      --ipc=host
      --security-opt seccomp=unconfined
      -e "POLARIS_OPENPI_CKPT=${CONTAINER_CKPT}"
      -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
      --
      "${INNER[@]}"
    )
  else
    # No TTY (CI / agent): same mounts as blackwell, without docker -it.
    CMD=(
      docker run --rm --gpus all --shm-size 32g --network host
      --ipc=host --security-opt seccomp=unconfined
      -v "${REPO_ROOT}:/workspace/RLinf"
      -w /workspace/RLinf
      -e ISAAC_LAB_PATH=/opt/envs/isaaclab
      -e ISAAC_PATH=/workspace/RLinf
      -e ISAACSIM_PATH=/workspace/RLinf
      -e OMNI_KIT_ACCEPT_EULA=YES
      -e "POLARIS_OPENPI_CKPT=${CONTAINER_CKPT}"
      -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
      "${IMAGE_TAG}"
      bash -lc "$(printf '%q ' "${INNER[@]}")"
    )
  fi
fi

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "VISIBLE_GPUS=${N_GPUS}"
echo "TRAIN_ENVS_PER_GPU=${TRAIN_ENVS_PER_GPU} train_envs=${TRAIN_ENVS}"
echo "EVAL_ENVS_PER_GPU=${EVAL_ENVS_PER_GPU} eval_envs=${EVAL_ENVS}"
echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
echo "POLARIS_OPENPI_CKPT=${POLARIS_OPENPI_CKPT}"
echo "IMAGE_TAG=${IMAGE_TAG}"
echo "CONTAINER_NAME=${CONTAINER_NAME}"
echo "LAUNCH_MODE=${LAUNCH_MODE}"
printf 'CMD='
printf '%q ' "${CMD[@]}"
echo

if [ "${RLINF_TRAIN_FRIDGE_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

if [ "${LAUNCH_MODE}" = "docker" ]; then
  echo "Launching fridge PPO inside ${IMAGE_TAG} from ${POLARIS_OPENPI_CKPT}"
elif ! _in_docker && [ "${RLINF_NO_DOCKER:-0}" != "1" ]; then
  echo "WARNING: Docker image ${IMAGE_TAG} not found; training on the host." >&2
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"
