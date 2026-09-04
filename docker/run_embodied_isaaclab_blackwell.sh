#!/usr/bin/env bash
# Enter the embodied-isaaclab Blackwell image with the local RLinf checkout mounted.
#
# The Docker image only ships Python venvs under /opt/venv. RLinf source, configs,
# checkpoints, and examples are provided by bind-mounting this repo to
# /workspace/RLinf (same convention as the isaaclab docs).
#
# Usage (from anywhere; defaults to this repo root):
#   bash docker/run_embodied_isaaclab_blackwell.sh
#   bash docker/run_embodied_isaaclab_blackwell.sh --name rlinf-bw
#   bash docker/run_embodied_isaaclab_blackwell.sh -- python examples/embodiment/train_embodied_agent.py --help
#   IMAGE_TAG=rlinf:embodied-isaaclab-u24 bash docker/run_embodied_isaaclab_blackwell.sh
#   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/docker/runtime_mounts.sh"
# Default IMAGE_TAG after sourcing isaaclab_local.env so that file can override.
CONTAINER_NAME="${CONTAINER_NAME:-rlinf-isaaclab-blackwell}"
SHM_SIZE="${SHM_SIZE:-32g}"
WORKDIR="${WORKDIR:-/workspace/RLinf}"
EXTRA_DOCKER_ARGS=()
CMD=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag|--image) IMAGE_TAG="${2:-}"; shift 2 ;;
    --name) CONTAINER_NAME="${2:-}"; shift 2 ;;
    --shm-size) SHM_SIZE="${2:-}"; shift 2 ;;
    --repo) REPO_ROOT="$(cd "${2:-}" && pwd)"; shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      EXTRA_DOCKER_ARGS+=("$1")
      shift
      ;;
  esac
done

# Optional machine-local ISAAC_* / CRI_OPENPI_CKPT / IMAGE_TAG (after --repo).
# shellcheck disable=SC1091
source "${REPO_ROOT}/examples/embodiment/scripts/source_isaaclab_local_env.sh"
source_isaaclab_local_env "${REPO_ROOT}"
# Prefer the locally built u24 tag, then the older blackwell name.
IMAGE_TAG="${IMAGE_TAG:-$(rlinf_resolve_isaaclab_image)}"

if [ ! -d "$REPO_ROOT/rlinf" ] || [ ! -d "$REPO_ROOT/examples" ]; then
  echo "ERROR: '$REPO_ROOT' does not look like an RLinf checkout (missing rlinf/ or examples/)." >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE_TAG' not found. Build it first:" >&2
  echo "  bash docker/build_embodied_isaaclab_u24.sh" >&2
  echo "  # Ubuntu 22.04: bash docker/build_embodied_isaaclab_blackwell.sh" >&2
  exit 1
fi

# Drop a stale stopped container with the same name so --name reuse works.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: container '$CONTAINER_NAME' is already running." >&2
    echo "  docker exec -it $CONTAINER_NAME bash" >&2
    exit 1
  fi
  docker rm "$CONTAINER_NAME" >/dev/null
fi

# Optional: expose a local Isaac Sim 5.1.0 tree to Isaac Lab.
ISAAC_SIM_PATH="$(rlinf_resolve_isaac_sim "$REPO_ROOT" || true)"

RUN_CMD=(
  docker run -it --rm
    --gpus all
    --shm-size "$SHM_SIZE"
    --network host
    --name "$CONTAINER_NAME"
    -v "$REPO_ROOT":/workspace/RLinf
    -w "$WORKDIR"
    -e ISAAC_LAB_PATH=/opt/envs/isaaclab
)
if [ -n "$ISAAC_SIM_PATH" ]; then
  ISAAC_SIM_ABS="$(cd "$ISAAC_SIM_PATH" && pwd -P)"
  REPO_ABS="$(cd "$REPO_ROOT" && pwd -P)"
  if [ "$ISAAC_SIM_ABS" = "$REPO_ABS" ]; then
    # Sim was extracted into this checkout; the repo mount is enough.
    RUN_CMD+=(-e ISAAC_PATH=/workspace/RLinf)
    RUN_CMD+=(-e ISAACSIM_PATH=/workspace/RLinf)
    echo "[run_embodied_isaaclab_blackwell] isaac_sim=$ISAAC_SIM_PATH -> ISAAC_PATH=/workspace/RLinf"
  else
    # Kit writes user.config.json, pip3-envs, and shader cache under kit/data + kit/cache.
    # A read-only bind-mount causes OSError: [Errno 30] Read-only file system.
    mkdir -p "${ISAAC_SIM_PATH}/kit/data" "${ISAAC_SIM_PATH}/kit/cache"
    RUN_CMD+=(-v "$ISAAC_SIM_PATH":/workspace/isaac_sim)
    RUN_CMD+=(-e ISAAC_PATH=/workspace/isaac_sim)
    RUN_CMD+=(-e ISAACSIM_PATH=/workspace/isaac_sim)
    echo "[run_embodied_isaaclab_blackwell] isaac_sim=$ISAAC_SIM_PATH -> /workspace/isaac_sim (rw)"
  fi
fi
CRI_CKPT="$(rlinf_container_cri_ckpt "$REPO_ROOT" || true)"
if [ -n "$CRI_CKPT" ]; then
  RUN_CMD+=(-e CRI_OPENPI_CKPT="$CRI_CKPT")
  echo "[run_embodied_isaaclab_blackwell] CRI_OPENPI_CKPT=$CRI_CKPT"
fi
RUN_CMD+=(
  "${EXTRA_DOCKER_ARGS[@]}"
  "$IMAGE_TAG"
)

if [ "${#CMD[@]}" -gt 0 ]; then
  # Login shell so ~/.bashrc activates /opt/venv/openpi.
  RUN_CMD+=(bash -lc "$(printf '%q ' "${CMD[@]}")")
fi

echo "[run_embodied_isaaclab_blackwell] image=$IMAGE_TAG"
echo "[run_embodied_isaaclab_blackwell] mount=$REPO_ROOT -> /workspace/RLinf"
echo "[run_embodied_isaaclab_blackwell] workdir=$WORKDIR"
echo "[run_embodied_isaaclab_blackwell] running: ${RUN_CMD[*]}"

exec "${RUN_CMD[@]}"
