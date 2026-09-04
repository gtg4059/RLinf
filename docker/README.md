## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning image and the various embodied images. Use the `BUILD_TARGET` build argument to select which image to build:

- `reason` — math reasoning image
- `embodied-<env>` — embodied image for a specific environment (and optionally a specific model when multiple model flavors exist for the same env)

To build the Docker image, run the following command **in the RLinf root directory**:

```shell
export BUILD_TARGET=reason # or one of the embodied-* targets defined in the Dockerfile
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

### Available `BUILD_TARGET` values

Each `BUILD_TARGET` maps to a build stage in [`Dockerfile`](Dockerfile). To see the full, up-to-date list of targets and the venvs each one installs, look at the stage names (`FROM ... AS <target>-image`) and the `install.sh` invocations inside them — the Dockerfile is the source of truth, so this README does not duplicate the list.

### Additional build arguments

- `PLATFORM` (default `nvidia`) — hardware platform: `nvidia` (CUDA), `amd` (ROCm), or `ascend` (CANN). Selects the base image and is also recorded as `RLINF_PLATFORM` in the final image. The `embodied-franka` target ignores `PLATFORM` and always uses a plain `ubuntu:20.04` base.
- Per-platform runtime versions: `CUDA_VER`, `ROCM_VER`, `ROCM_ARCHS`, `CANN_VER`, `UBUNTU_VER`. Override any of these to bump versions without changing the rest of the build. For a fully custom base, set `NVIDIA_BASE_IMAGE`, `AMD_BASE_IMAGE`, or `ASCEND_BASE_IMAGE` directly.
- `NO_MIRROR` — set to `1` to skip the USTC apt/pypi mirror rewrites (recommended outside of mainland China).

Example with non-default args:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-metaworld \
    --build-arg PLATFORM=nvidia \
    --build-arg CUDA_VER=12.4.1 \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-metaworld .
```

### Blackwell (sm_120) / Isaac Lab torch repair

Isaac Lab's installer may leave `torch==2.6.0+cu124`, which does **not** ship
Blackwell kernels. The `embodied-isaaclab` stage repairs torch after Isaac Lab
installs via `requirements/install.sh` (`repair_torch_after_isaaclab`), then
asserts a Blackwell-capable CUDA wheel (`>=2.7`, `cu128+` / CUDA 12.8+).
`sm_120` is checked only when a GPU is visible (docker builds usually have
none). `flash-attn` rebuild is best-effort (missing wheel / nvcc mismatch →
PyTorch SDPA fallback).

Isaac Lab is pinned to **2.3.0** (`ISAAC_LAB_GIT_REF` / `RLinf/IsaacLab`);
pair it with Isaac Sim **5.1.0** mounted at runtime (`ISAAC_PATH`).

**Recommended:** use the helper that locks the working defaults:

```shell
bash docker/build_embodied_isaaclab_u24.sh
# optional: --no-cache --torch 2.11.0 --cuda-tag cu128
# Ubuntu 22.04 tag: bash docker/build_embodied_isaaclab_blackwell.sh
# or: bash docker/bake_isaaclab_into_image.sh rebuild
```

Defaults (overridable build-args / env vars):

- `UBUNTU_VER=24.04` (matches the AWS CRI image `rlinf:embodied-isaaclab-u24`)
- `CUDA_VER=12.8.1` (base image toolkit)
- `TORCH_VERSION=2.11.0` (matches repo `pyproject.toml`)
- `UV_TORCH_BACKEND=cu128` (Blackwell-capable CUDA wheel index; may resolve to `+cu130`)

Equivalent raw `docker build`:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-isaaclab \
    --build-arg UBUNTU_VER=24.04 \
    --build-arg CUDA_VER=12.8.1 \
    --build-arg TORCH_VERSION=2.11.0 \
    --build-arg UV_TORCH_BACKEND=cu128 \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-isaaclab-u24 .
```

To patch an **already running** container/venv without rebuilding:

```shell
source switch_env openpi
bash requirements/embodied/patch_torch_blackwell.sh --venv openpi
# or, if already activated:
bash requirements/embodied/patch_torch_blackwell.sh
```

**Important:** the image does **not** contain the RLinf git tree or Isaac Sim.
Only venvs under `/opt/venv` are installed. Mount the checkout (required for
`rlinf/`, `examples/`, checkpoints). Install Isaac Sim **separately**
(Isaac Sim 5.1.0; see `docs/source-en/rst_source/examples/embodied/isaaclab.rst`),
then set `ISAAC_PATH` or leave the tree where `run_embodiment.sh` probes
(`./isaac_sim`, this checkout if Sim was extracted into it, a sibling
`isaac_sim`, `/workspace/isaac_sim`).

```shell
bash docker/run_embodied_isaaclab_blackwell.sh
# equivalent:
# docker run --gpus all -it --rm --shm-size 32g --network host \
#   -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
#   rlinf:embodied-isaaclab-u24

# CRI PPO from checkpoint/pi05_droid_cri_rlinf_49999 (auto-picks the u24 image):
# bash examples/embodiment/scripts/train_cri_openpi_ckpt.sh
```

If a separately installed Isaac Sim tree is visible on the host, the helper
bind-mounts it at `/workspace/isaac_sim` and sets `ISAAC_PATH`. Otherwise
install Sim inside the container (or on the host) with the IsaacLab recipe,
then `source ./setup_conda_env.sh`.

Inside the container: `cd /workspace/RLinf` (already the workdir with the helper)
and `source switch_env openpi`.

### ManiSkill / Libero (multi-model VLA bundle)

The `embodied-maniskill_libero` target bakes one venv per VLA model onto the
ManiSkill/LIBERO env — `openvla`, `openvla-oft`, `openpi`, `gr00t`,
`gr00t_n1d6`, `gr00t_n1d7`, `dexbotic`, `starvla`, `abot_m0` — matching the
published `rlinf/rlinf:agentic-rlinf0.3-maniskill_libero` image (plus
`gr00t_n1d7`, added since that tag was cut).

```shell
bash docker/build_embodied_maniskill_libero.sh
# optional: --tag rlinf:my-maniskill-libero --no-cache
```

Equivalent raw `docker build`:

```shell
docker build -f docker/Dockerfile \
    --build-arg BUILD_TARGET=embodied-maniskill_libero \
    --build-arg NO_MIRROR=1 \
    -t rlinf:embodied-maniskill_libero .
```

Run with the RLinf checkout mounted (same `--gpus all --shm-size 32g
--network host -v .:/workspace/RLinf` convention as the docs). Isaac Sim
is a separate install; the helper only bind-mounts it when a host tree is
already present (`ISAAC_SIM_PATH` / `./isaac_sim`).

```shell
bash docker/run_embodied_maniskill_libero.sh
# equivalent:
# docker run --gpus all -it --rm --shm-size 32g --network host \
#   -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
#   rlinf:embodied-maniskill_libero
```

Inside the container: `cd /workspace/RLinf` (already the workdir with the
helper) and `source switch_env openpi` (or any other venv listed above).

# Using the Docker Image

The built Docker image contains one or more Python virtual environments (venvs) under `/opt/venv/`. Which venvs are present, and which one is activated by default in new shells, depends on the `BUILD_TARGET` — see the corresponding build stage in the [`Dockerfile`](Dockerfile).

To switch between venvs, use the built-in `switch_env` script:

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env openpi, etc.
```