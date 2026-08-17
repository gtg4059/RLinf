#!/usr/bin/env bash
# Patch an existing RLinf embodied venv onto a Blackwell-capable PyTorch CUDA wheel.
#
# Isaac Lab installs often leave torch==2.6.0+cu124, which does not include sm_120
# kernels. This script force-reinstalls torch/torchvision/torchaudio from cu128
# (or another --cuda-tag) and rebuilds flash-attn when possible.
#
# Usage:
#   bash requirements/embodied/patch_torch_blackwell.sh --venv openpi
#   TORCH_VERSION=2.8.0 bash requirements/embodied/patch_torch_blackwell.sh --venv openpi
#
# Docker rebuild (same repair runs inside install.sh for isaaclab):
#   bash docker/build_embodied_isaaclab_blackwell.sh
#   # or: TORCH_VERSION=2.11.0 UV_TORCH_BACKEND=cu128 \
#   #     bash docker/build_embodied_isaaclab_blackwell.sh --no-cache

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_NAME=""
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CUDA_TAG="${UV_TORCH_BACKEND:-cu128}"
SKIP_FLASH_ATTN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv) VENV_NAME="${2:-}"; shift 2 ;;
    --torch) TORCH_VERSION="${2:-}"; shift 2 ;;
    --cuda-tag) CUDA_TAG="${2:-}"; shift 2 ;;
    --no-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! "$CUDA_TAG" =~ ^cu[0-9]+$ ]]; then
  echo "ERROR: --cuda-tag / UV_TORCH_BACKEND must look like cu128 (got '$CUDA_TAG')." >&2
  exit 1
fi

activate_venv() {
  local name="$1"
  local candidates=(
    "/opt/venv/${name}/bin/activate"
    "${REPO_ROOT}/.venv/bin/activate"
    "${REPO_ROOT}/${name}/bin/activate"
  )
  local c
  for c in "${candidates[@]}"; do
    if [ -f "$c" ]; then
      # shellcheck disable=SC1090
      source "$c"
      echo "[patch_torch_blackwell] activated ${c}"
      return 0
    fi
  done
  echo "ERROR: venv '${name}' not found." >&2
  return 1
}

if [ -n "$VENV_NAME" ]; then
  activate_venv "$VENV_NAME"
fi

if ! command -v python >/dev/null || ! command -v uv >/dev/null; then
  echo "ERROR: need python + uv on PATH (activate a venv or pass --venv)." >&2
  exit 1
fi

IFS='.' read -r tmaj tmin tpatch <<< "$TORCH_VERSION"
# torchvision minor = torch minor + 15 (torch 2.11.0 -> torchvision 0.26.0).
tv_ver="0.$((tmin + 15)).${tpatch:-0}"
INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "[patch_torch_blackwell] python=$(which python)"
echo "[patch_torch_blackwell] before: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo 'torch missing')"
echo "[patch_torch_blackwell] installing torch==${TORCH_VERSION} torchvision==${tv_ver} torchaudio==${TORCH_VERSION} from ${INDEX_URL}"

# Avoid workspace/pyproject constraint resolution floating the pin: install from /tmp
# with UV_NO_CONFIG so only the CUDA index constraints apply.
(
  cd /tmp
  UV_NO_CONFIG=1 uv pip install --python "$(which python)" --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${tv_ver}" \
    "torchaudio==${TORCH_VERSION}" \
    --index-url "${INDEX_URL}"
)

if [ "$SKIP_FLASH_ATTN" -eq 0 ]; then
  uv pip uninstall -y flash-attn >/dev/null 2>&1 || true
  if ! FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install "flash-attn==2.8.3" --no-build-isolation; then
    echo "[patch_torch_blackwell] WARNING: flash-attn install failed; continuing with PyTorch SDPA." >&2
  fi
fi

python - <<'PY'
import torch

print("[patch_torch_blackwell] after:", torch.__version__, "cuda=", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("[patch_torch_blackwell] ERROR: torch.cuda.is_available() is False")
print("[patch_torch_blackwell] device:", torch.cuda.get_device_name(0))
print("[patch_torch_blackwell] archs:", torch.cuda.get_arch_list())
caps = torch.cuda.get_device_capability(0)
print("[patch_torch_blackwell] capability:", caps)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
print("[patch_torch_blackwell] cuda matmul OK:", float(y[0, 0]))
if caps >= (12, 0) and "sm_120" not in torch.cuda.get_arch_list() and "sm_120a" not in torch.cuda.get_arch_list():
    print("[patch_torch_blackwell] WARNING: GPU is Blackwell but arch list has no sm_120*; check wheel CUDA tag.")
PY

echo "[patch_torch_blackwell] done."
