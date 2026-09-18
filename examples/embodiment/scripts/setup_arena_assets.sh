#!/usr/bin/env bash
# Resolve ARENA_ASSETS_ROOT to a local directory under the repo (.assets).
# Remote http(s) roots are ignored because custom local wrappers are not on
# Nucleus/S3.
#
# Default completeness check is kitchen_bench (Lightwheel one-wall coastal +
# DROID). Maple-table pick-place assets are opt-in via ARENA_ASSET_SET=maple_table.
#
# Source from run_embodiment.sh / run_eval.sh:
#   # shellcheck disable=SC1091
#   source "${REPO_PATH}/examples/embodiment/scripts/setup_arena_assets.sh"
#   setup_arena_assets

setup_arena_assets() {
  local repo_path="${REPO_PATH:-}"
  if [ -z "${repo_path}" ]; then
    echo "setup_arena_assets: REPO_PATH is not set" >&2
    return 1
  fi

  local local_root="${repo_path}/.assets/isaaclab_arena"
  local download_script="${repo_path}/examples/embodiment/scripts/download_isaaclab_arena_assets.sh"
  local kitchen="${local_root}/background_library/lightwheel_kitchen_one_wall_coastal/scene.usd"
  local kitchen_sdk="${local_root}/lightwheel_sdk/floorplan/robocasa-robocasakitchen-1-1/scene.usd"
  local robot="${local_root}/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
  local stand="${local_root}/object_library/srl_robolab_assets/robots/franka_stand_grey.usda"

  # Prefer an explicit *local* override; ignore remote URL env values.
  if [ -n "${ARENA_ASSETS_ROOT:-}" ]; then
    case "${ARENA_ASSETS_ROOT}" in
      http://*|https://*)
        echo "Ignoring remote ARENA_ASSETS_ROOT=${ARENA_ASSETS_ROOT}"
        echo "Using local assets at ${local_root}"
        ;;
      *)
        if [ -d "${ARENA_ASSETS_ROOT}" ]; then
          local_root="${ARENA_ASSETS_ROOT}"
          kitchen="${local_root}/background_library/lightwheel_kitchen_one_wall_coastal/scene.usd"
          kitchen_sdk="${local_root}/lightwheel_sdk/floorplan/robocasa-robocasakitchen-1-1/scene.usd"
          robot="${local_root}/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
          stand="${local_root}/object_library/srl_robolab_assets/robots/franka_stand_grey.usda"
        else
          echo "WARNING: ARENA_ASSETS_ROOT is not a directory (${ARENA_ASSETS_ROOT}); using ${local_root}" >&2
        fi
        ;;
    esac
  fi

  export ARENA_ASSETS_ROOT="${local_root}"

  if [ ! -f "${kitchen}" ] || [ ! -f "${kitchen_sdk}" ] || [ ! -f "${robot}" ] || [ ! -f "${stand}" ]; then
    if [ "${SKIP_ARENA_ASSET_DOWNLOAD:-0}" = "1" ]; then
      echo "WARNING: Arena assets missing under ${ARENA_ASSETS_ROOT} (SKIP_ARENA_ASSET_DOWNLOAD=1)" >&2
    else
      echo "Arena assets incomplete under ${ARENA_ASSETS_ROOT}; installing kitchen_bench (DROID + Lightwheel kitchen)..."
      DEST_DIR="${ARENA_ASSETS_ROOT}" ARENA_ASSET_SET="${ARENA_ASSET_SET:-kitchen_bench}" \
        bash "${download_script}"
    fi
  fi

  if [ -d "${ARENA_ASSETS_ROOT}" ]; then
    echo "Using ARENA_ASSETS_ROOT=${ARENA_ASSETS_ROOT}"
  else
    echo "WARNING: ARENA_ASSETS_ROOT does not exist: ${ARENA_ASSETS_ROOT}" >&2
  fi
}
