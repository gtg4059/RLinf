#!/usr/bin/env bash
# Resolve ARENA_ASSETS_ROOT to a local directory under the repo (.assets).
# Remote http(s) roots are ignored because custom local wrappers (e.g.
# table_maple_arena.usda) are not on Nucleus/S3.
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
  local marker="${local_root}/object_library/srl_robolab_assets/fixtures/table_maple_arena.usda"
  local maple="${local_root}/object_library/srl_robolab_assets/fixtures/table_maple.usda"
  local oak="${local_root}/Materials/Base/Wood/Oak.mdl"
  local hdr="${local_root}/object_library/srl_robolab_assets/backgrounds/default/home_office.exr"
  local stand="${local_root}/object_library/srl_robolab_assets/fixtures/stand_instanceable.usd"
  local franka_table="${local_root}/object_library/srl_robolab_assets/fixtures/franka_table.usd"

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
          marker="${local_root}/object_library/srl_robolab_assets/fixtures/table_maple_arena.usda"
          maple="${local_root}/object_library/srl_robolab_assets/fixtures/table_maple.usda"
          oak="${local_root}/Materials/Base/Wood/Oak.mdl"
          hdr="${local_root}/object_library/srl_robolab_assets/backgrounds/default/home_office.exr"
          stand="${local_root}/object_library/srl_robolab_assets/fixtures/stand_instanceable.usd"
          franka_table="${local_root}/object_library/srl_robolab_assets/fixtures/franka_table.usd"
        else
          echo "WARNING: ARENA_ASSETS_ROOT is not a directory (${ARENA_ASSETS_ROOT}); using ${local_root}" >&2
        fi
        ;;
    esac
  fi

  export ARENA_ASSETS_ROOT="${local_root}"

  if [ ! -f "${maple}" ] || [ ! -f "${marker}" ] || [ ! -f "${oak}" ] || [ ! -f "${hdr}" ] || [ ! -f "${stand}" ] || [ ! -f "${franka_table}" ]; then
    if [ "${SKIP_ARENA_ASSET_DOWNLOAD:-0}" = "1" ]; then
      echo "WARNING: Arena assets missing under ${ARENA_ASSETS_ROOT} (SKIP_ARENA_ASSET_DOWNLOAD=1)" >&2
    else
      echo "Arena assets incomplete under ${ARENA_ASSETS_ROOT}; downloading maple table + pedestal + Oak + HDR..."
      DEST_DIR="${ARENA_ASSETS_ROOT}" bash "${download_script}"
    fi
  fi

  if [ -d "${ARENA_ASSETS_ROOT}" ]; then
    echo "Using ARENA_ASSETS_ROOT=${ARENA_ASSETS_ROOT}"
  else
    echo "WARNING: ARENA_ASSETS_ROOT does not exist: ${ARENA_ASSETS_ROOT}" >&2
  fi
}
