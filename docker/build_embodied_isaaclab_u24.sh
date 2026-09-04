#!/usr/bin/env bash
# Build the Ubuntu 24.04 embodied-isaaclab image used on AWS
# (tag: rlinf:embodied-isaaclab-u24).
#
# Same recipe as the live AWS host:
#   nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
#   TORCH_VERSION=2.11.0  UV_TORCH_BACKEND=cu128  NO_MIRROR=1
#
# Usage (from repo root):
#   bash docker/build_embodied_isaaclab_u24.sh
#   bash docker/build_embodied_isaaclab_u24.sh --no-cache
#   bash docker/build_embodied_isaaclab_u24.sh --tag rlinf:isaaclab-u24
#
# After build:
#   bash docker/run_embodied_isaaclab_blackwell.sh
#   bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export BUILD_TARGET="${BUILD_TARGET:-embodied-isaaclab}"
export PLATFORM="${PLATFORM:-nvidia}"
export CUDA_VER="${CUDA_VER:-12.8.1}"
export UBUNTU_VER="${UBUNTU_VER:-24.04}"
export TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
export NO_MIRROR="${NO_MIRROR:-1}"
export IMAGE_TAG="${IMAGE_TAG:-rlinf:embodied-isaaclab-u24}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile}"
EXTRA_ARGS=()
NO_CACHE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) IMAGE_TAG="${2:-}"; shift 2 ;;
    --torch) TORCH_VERSION="${2:-}"; shift 2 ;;
    --cuda-tag|--uv-torch-backend) UV_TORCH_BACKEND="${2:-}"; shift 2 ;;
    --cuda-ver) CUDA_VER="${2:-}"; shift 2 ;;
    --ubuntu|--ubuntu-ver) UBUNTU_VER="${2:-}"; shift 2 ;;
    --platform) PLATFORM="${2:-}"; shift 2 ;;
    --dockerfile) DOCKERFILE="${2:-}"; shift 2 ;;
    --no-mirror) NO_MIRROR=1; shift ;;
    --use-mirror) NO_MIRROR=0; shift ;;
    --no-cache) NO_CACHE=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$UV_TORCH_BACKEND" =~ ^cu[0-9]+$ ]]; then
  echo "ERROR: UV_TORCH_BACKEND must look like cu128 (got '$UV_TORCH_BACKEND')." >&2
  exit 1
fi

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
  --build-arg "UBUNTU_VER=${UBUNTU_VER}"
  --build-arg "TORCH_VERSION=${TORCH_VERSION}"
  --build-arg "UV_TORCH_BACKEND=${UV_TORCH_BACKEND}"
  --build-arg "NO_MIRROR=${NO_MIRROR}"
  --label "rlinf.build_target=${BUILD_TARGET}"
  --label "rlinf.torch_version=${TORCH_VERSION}"
  --label "rlinf.uv_torch_backend=${UV_TORCH_BACKEND}"
  --label "rlinf.cuda_ver=${CUDA_VER}"
  --label "rlinf.ubuntu_ver=${UBUNTU_VER}"
  --label "rlinf.blackwell=1"
  -t "$IMAGE_TAG"
)

if [ "$NO_CACHE" -eq 1 ]; then
  BUILD_CMD+=(--no-cache)
fi

BUILD_CMD+=("${EXTRA_ARGS[@]}" .)

echo "[build_embodied_isaaclab_u24] repo=$REPO_ROOT"
echo "[build_embodied_isaaclab_u24] target=$BUILD_TARGET platform=$PLATFORM"
echo "[build_embodied_isaaclab_u24] CUDA_VER=$CUDA_VER UBUNTU_VER=$UBUNTU_VER TORCH_VERSION=$TORCH_VERSION UV_TORCH_BACKEND=$UV_TORCH_BACKEND"
echo "[build_embodied_isaaclab_u24] NO_MIRROR=$NO_MIRROR tag=$IMAGE_TAG"
echo "[build_embodied_isaaclab_u24] running: ${BUILD_CMD[*]}"

"${BUILD_CMD[@]}"

cat <<EOF

[build_embodied_isaaclab_u24] done: $IMAGE_TAG

Verify inside the image (GPU host):
  docker run --gpus all --rm $IMAGE_TAG bash -lc \\
    'source switch_env openpi && python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
assert any(a.startswith(\"sm_120\") for a in torch.cuda.get_arch_list())
"'

The image has venvs only (/opt/venv). Mount the RLinf checkout to enter:
  bash docker/run_embodied_isaaclab_blackwell.sh
  bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh

EOF
