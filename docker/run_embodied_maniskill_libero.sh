#!/usr/bin/env bash
# Enter the embodied-maniskill_libero image with the local RLinf checkout mounted.
#
# The Docker image only ships Python venvs under /opt/venv. RLinf source, configs,
# checkpoints, and examples are provided by bind-mounting this repo to
# /workspace/RLinf (same convention as the docs' `docker run` examples).
#
# Usage (from anywhere; defaults to this repo root):
#   bash docker/run_embodied_maniskill_libero.sh
#   bash docker/run_embodied_maniskill_libero.sh --name rlinf-msl
#   bash docker/run_embodied_maniskill_libero.sh -- python examples/embodiment/train_embodied_agent.py --help
#   IMAGE_TAG=rlinf:embodied-maniskill_libero bash docker/run_embodied_maniskill_libero.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/docker/runtime_mounts.sh"

IMAGE_TAG="$(rlinf_resolve_image \
  rlinf:embodied-maniskill_libero \
  rlinf/rlinf:agentic-rlinf0.3-maniskill_libero)"
CONTAINER_NAME="${CONTAINER_NAME:-rlinf-maniskill-libero}"
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
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *)
      EXTRA_DOCKER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ! -d "$REPO_ROOT/rlinf" ] || [ ! -d "$REPO_ROOT/examples" ]; then
  echo "ERROR: '$REPO_ROOT' does not look like an RLinf checkout (missing rlinf/ or examples/)." >&2
  exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE_TAG' not found. Pull or build it first:" >&2
  echo "  docker pull rlinf/rlinf:agentic-rlinf0.3-maniskill_libero" >&2
  echo "  bash docker/build_embodied_maniskill_libero.sh" >&2
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

ISAAC_SIM_PATH="$(rlinf_resolve_isaac_sim "$REPO_ROOT" || true)"
CRI_CKPT="$(rlinf_container_cri_ckpt "$REPO_ROOT" || true)"

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
  RUN_CMD+=(-v "$ISAAC_SIM_PATH":/workspace/isaac_sim)
  RUN_CMD+=(-e ISAAC_PATH=/workspace/isaac_sim)
  RUN_CMD+=(-e ISAACSIM_PATH=/workspace/isaac_sim)
  echo "[run_embodied_maniskill_libero] isaac_sim=$ISAAC_SIM_PATH -> /workspace/isaac_sim"
fi
if [ -n "$CRI_CKPT" ]; then
  RUN_CMD+=(-e CRI_OPENPI_CKPT="$CRI_CKPT")
fi
RUN_CMD+=(
  "${EXTRA_DOCKER_ARGS[@]}"
  "$IMAGE_TAG"
)

if [ "${#CMD[@]}" -gt 0 ]; then
  RUN_CMD+=("${CMD[@]}")
fi

echo "[run_embodied_maniskill_libero] image=$IMAGE_TAG"
echo "[run_embodied_maniskill_libero] mount=$REPO_ROOT -> /workspace/RLinf"
echo "[run_embodied_maniskill_libero] workdir=$WORKDIR"
echo "[run_embodied_maniskill_libero] running: ${RUN_CMD[*]}"

exec "${RUN_CMD[@]}"
