#!/usr/bin/env bash
# Download Isaac Lab Arena assets used by pick_place_cube_plate into
# ${REPO_PATH}/.assets/isaaclab_arena (local paths only — not remote URL roots).
#
# Also fetches the NVIDIA Base Materials referenced by fixtures/table_maple
# (Oak / Walnut / RustedMetal, …) into .assets/isaaclab_arena/Materials and
# rewrites table_maple.usda MDL paths so they resolve under that tree — matching
# the Isaac Lab Arena maple work table appearance offline.
#
# Usage:
#   bash examples/embodiment/scripts/download_isaaclab_arena_assets.sh
#   DEST_DIR=/path/to/.assets/isaaclab_arena bash .../download_isaaclab_arena_assets.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEST_DIR="${DEST_DIR:-${REPO_PATH}/.assets/isaaclab_arena}"
# Staging hosts the 5.1 Arena tree; production often 404s the same paths.
ARENA_S3_BASE="${ARENA_S3_BASE:-https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Arena/assets}"
# NVIDIA Materials live under the Isaac/5.1/NVIDIA prefix on production.
MATERIALS_S3_BASE="${MATERIALS_S3_BASE:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Materials}"

# Arena USD/HDR used by the DROID maple-table pick-place task.
ASSET_PATHS=(
  "robot_library/droid/franka_robotiq_2f_85_flattened.usd"
  "object_library/srl_robolab_assets/fixtures/stand_instanceable.usd"
  "object_library/srl_robolab_assets/fixtures/franka_table.usd"
  "object_library/srl_robolab_assets/fixtures/table_maple.usd"
  "object_library/srl_robolab_assets/fixtures/table_maple.usda"
  "object_library/srl_robolab_assets/scenes/maple_table.usda"
  "object_library/srl_robolab_assets/scenes/maple_table_background.usda"
  "object_library/srl_robolab_assets/objects/hot3d/rubiks_cube.usd"
  "object_library/srl_robolab_assets/objects/ycb/bowl.usd"
  "object_library/srl_robolab_assets/backgrounds/default/brown_photostudio.hdr"
  "object_library/srl_robolab_assets/backgrounds/default/home_office.exr"
)

# Materials needed for the maple work table (top + legs). Paths relative to
# MATERIALS_S3_BASE / DEST_DIR/Materials.
MATERIAL_PATHS=(
  "Base/Wood/Oak.mdl"
  "Base/Wood/Oak/Oak_BaseColor.png"
  "Base/Wood/Oak/Oak_ORM.png"
  "Base/Wood/Oak/Oak_N.png"
  "Base/Wood/Walnut_Planks.mdl"
  "Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png"
  "Base/Wood/Walnut_Planks/Walnut_Planks_ORM.png"
  "Base/Wood/Walnut_Planks/Walnut_Planks_N.png"
  "Base/Metals/RustedMetal.mdl"
  "Base/Metals/RustedMetal/RustedMetal_BaseColor.png"
  "Base/Metals/RustedMetal/RustedMetal_ORM.png"
  "Base/Metals/RustedMetal/RustedMetal_N.png"
  "Base/Wood/Bamboo.mdl"
  "Base/Wood/Bamboo/Bamboo_BaseColor.png"
  "Base/Wood/Bamboo/Bamboo_ORM.png"
  "Base/Wood/Bamboo/Bamboo_N.png"
  "Base/Plastics/Plastic_Clear.mdl"
  "Base/Plastics/Plastic_ABS.mdl"
  "Base/Plastics/Plastic.mdl"
)

WRAPPER_REL="object_library/srl_robolab_assets/fixtures/table_maple_arena.usda"
WRAPPER_SRC="${REPO_PATH}/rlinf/envs/isaaclab/tasks/pick_place_cube_plate/assets/table_maple_arena.usda"
TABLE_USDA_REL="object_library/srl_robolab_assets/fixtures/table_maple.usda"

download_url() {
  local url="$1"
  local out="$2"
  if [ -f "${out}" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
    echo "[arena-assets] skip (exists): ${out#${DEST_DIR}/}"
    return 0
  fi
  mkdir -p "$(dirname "${out}")"
  echo "[arena-assets] download: ${out#${DEST_DIR}/}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 -o "${out}.partial" "${url}"
  else
    wget -O "${out}.partial" "${url}"
  fi
  mv "${out}.partial" "${out}"
}

download_arena_one() {
  local rel="$1"
  download_url "${ARENA_S3_BASE}/${rel}" "${DEST_DIR}/${rel}"
}

download_material_one() {
  local rel="$1"
  download_url "${MATERIALS_S3_BASE}/${rel}" "${DEST_DIR}/Materials/${rel}"
}

# Stock table_maple.usda points at ../../../../../../Materials/... (Nucleus layout).
# Rewrite to ../../../Materials/... so paths resolve under DEST_DIR/Materials.
patch_table_maple_mdl_paths() {
  local usda="${DEST_DIR}/${TABLE_USDA_REL}"
  if [ ! -f "${usda}" ]; then
    echo "[arena-assets] missing ${TABLE_USDA_REL}; cannot patch MDL paths" >&2
    return 1
  fi
  if grep -q '@../../../Materials/' "${usda}"; then
    echo "[arena-assets] MDL paths already patched: ${TABLE_USDA_REL}"
    return 0
  fi
  # Backup once, then rewrite any number of ../ prefixes before Materials/.
  if [ ! -f "${usda}.orig" ]; then
    cp -f "${usda}" "${usda}.orig"
  fi
  sed -E 's|@(\.\./)+Materials/|@../../../Materials/|g' "${usda}.orig" >"${usda}"
  echo "[arena-assets] patched MDL paths in ${TABLE_USDA_REL} → ../../../Materials/"
}

install_wrapper() {
  local out="${DEST_DIR}/${WRAPPER_REL}"
  mkdir -p "$(dirname "${out}")"
  if [ -f "${WRAPPER_SRC}" ]; then
    cp -f "${WRAPPER_SRC}" "${out}"
    echo "[arena-assets] installed Arena maple table wrapper: ${WRAPPER_REL}"
  else
    echo "[arena-assets] ERROR: wrapper source missing: ${WRAPPER_SRC}" >&2
    return 1
  fi
}

main() {
  echo "[arena-assets] DEST_DIR=${DEST_DIR}"
  echo "[arena-assets] ARENA_S3_BASE=${ARENA_S3_BASE}"
  echo "[arena-assets] MATERIALS_S3_BASE=${MATERIALS_S3_BASE}"
  mkdir -p "${DEST_DIR}"

  local rel
  for rel in "${ASSET_PATHS[@]}"; do
    download_arena_one "${rel}"
  done
  for rel in "${MATERIAL_PATHS[@]}"; do
    download_material_one "${rel}" || {
      echo "[arena-assets] WARNING: optional material missing: ${rel}" >&2
    }
  done

  patch_table_maple_mdl_paths
  install_wrapper

  local missing=0
  local required=(
    "robot_library/droid/franka_robotiq_2f_85_flattened.usd"
    "object_library/srl_robolab_assets/fixtures/stand_instanceable.usd"
    "object_library/srl_robolab_assets/fixtures/franka_table.usd"
    "object_library/srl_robolab_assets/fixtures/table_maple.usda"
    "object_library/srl_robolab_assets/objects/hot3d/rubiks_cube.usd"
    "object_library/srl_robolab_assets/objects/ycb/bowl.usd"
    "object_library/srl_robolab_assets/backgrounds/default/home_office.exr"
    "Materials/Base/Wood/Oak.mdl"
    "Materials/Base/Wood/Oak/Oak_BaseColor.png"
    "${WRAPPER_REL}"
  )
  for rel in "${required[@]}"; do
    if [ ! -f "${DEST_DIR}/${rel}" ]; then
      echo "[arena-assets] MISSING: ${DEST_DIR}/${rel}" >&2
      missing=1
    fi
  done
  if [ "${missing}" -ne 0 ]; then
    exit 1
  fi
  echo "[arena-assets] ready at ${DEST_DIR}"
}

main "$@"
