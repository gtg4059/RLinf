#!/usr/bin/env bash
# Build the embodied-maniskill_libero image: the multi-model VLA bundle
# (openvla, openvla-oft, openpi, gr00t, gr00t_n1d6, gr00t_n1d7, dexbotic,
# starvla, abot_m0) on the ManiSkill/LIBERO env, one venv per model under
# /opt/venv. This is the same target used to build the reference
# `rlinf/rlinf:agentic-rlinf0.3-maniskill_libero` image.
#
# Usage (from repo root):
#   bash docker/build_embodied_maniskill_libero.sh
#   bash docker/build_embodied_maniskill_libero.sh --tag rlinf:my-maniskill-libero
#   bash docker/build_embodied_maniskill_libero.sh --no-cache
#   PLATFORM=nvidia bash docker/build_embodied_maniskill_libero.sh
#
# After build, run with the RLinf checkout mounted:
#   bash docker/run_embodied_maniskill_libero.sh
#   # or:
#   # docker run --gpus all -it --rm --shm-size 32g --network host \
#   #   -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
#   #   rlinf:embodied-maniskill_libero

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_TARGET="${BUILD_TARGET:-embodied-maniskill_libero}"
PLATFORM="${PLATFORM:-nvidia}"
CUDA_VER="${CUDA_VER:-12.8.1}"
NO_MIRROR="${NO_MIRROR:-1}"
IMAGE_TAG="${IMAGE_TAG:-rlinf:embodied-maniskill_libero}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
EXTRA_ARGS=()
NO_CACHE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) IMAGE_TAG="${2:-}"; shift 2 ;;
    --cuda-ver) CUDA_VER="${2:-}"; shift 2 ;;
    --platform) PLATFORM="${2:-}"; shift 2 ;;
    --dockerfile) DOCKERFILE="${2:-}"; shift 2 ;;
    --no-mirror) NO_MIRROR=1; shift ;;
    --use-mirror) NO_MIRROR=0; shift ;;
    --no-cache) NO_CACHE=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ! -f "$DOCKERFILE" ]; then
  echo "ERROR: Dockerfile not found at $DOCKERFILE" >&2
  exit 1
fi

BUILD_CMD=(
  docker build
  -f "$DOCKERFILE"
  --build-arg "BUILD_TARGET=${BUILD_TARGET}"
  --build-arg "PLATFORM=${PLATFORM}"
  --build-arg "CUDA_VER=${CUDA_VER}"
  --build-arg "NO_MIRROR=${NO_MIRROR}"
  --label "rlinf.build_target=${BUILD_TARGET}"
  --label "rlinf.cuda_ver=${CUDA_VER}"
  -t "$IMAGE_TAG"
)

if [ "$NO_CACHE" -eq 1 ]; then
  BUILD_CMD+=(--no-cache)
fi

BUILD_CMD+=("${EXTRA_ARGS[@]}" .)

echo "[build_embodied_maniskill_libero] repo=$REPO_ROOT"
echo "[build_embodied_maniskill_libero] target=$BUILD_TARGET platform=$PLATFORM"
echo "[build_embodied_maniskill_libero] CUDA_VER=$CUDA_VER NO_MIRROR=$NO_MIRROR tag=$IMAGE_TAG"
echo "[build_embodied_maniskill_libero] running: ${BUILD_CMD[*]}"

"${BUILD_CMD[@]}"

cat <<EOF

[build_embodied_maniskill_libero] done: $IMAGE_TAG

Venvs baked into the image (/opt/venv): openvla, openvla-oft, openpi, gr00t,
gr00t_n1d6, gr00t_n1d7, dexbotic, starvla, abot_m0.

The image has venvs only — mount the RLinf checkout to enter:
  bash docker/run_embodied_maniskill_libero.sh
  # or:
  # docker run --gpus all -it --rm --shm-size 32g --network host \\
  #   -v "$REPO_ROOT":/workspace/RLinf -w /workspace/RLinf \\
  #   $IMAGE_TAG

EOF
