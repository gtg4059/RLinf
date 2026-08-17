# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compose DROID robot + Arena franka_stand_grey pedestal (matches Isaac Lab Arena).

Faithful port of ``isaaclab_arena.embodiments.robot_on_stand_utils.compose_on_stand_usd``
for local ``.assets/isaaclab_arena`` paths. Requires ``pxr`` (available after
Isaac Sim ``AppLauncher``).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

# Arena height/scale; footprint shifted +X vs Arena (−0.05) so the grey
# ``franka_table`` seat stays under ``panda_link0`` toward the maple table.
_STAND_DEFAULT_HEIGHT_M = 1.35
_FOOTPRINT_TRANSLATE_XYZ = (0.08, 0.0, 0.0)
_FOOTPRINT_SCALE_XY = (1.2, 1.2)
_HEIGHT_ATOL = 1e-3
_ALIGN_ATOL = 5e-2


@dataclass(frozen=True)
class _RobotPrimSpec:
    robot_usd_path: str
    root_prim_path: str = "/panda"
    robot_base_prim_name: str = "panda_link0"
    stand_prim_name: str = "stand_instanceable"

    @property
    def robot_base_prim_path(self) -> str:
        return f"{self.root_prim_path}/{self.robot_base_prim_name}"

    @property
    def stand_prim_path(self) -> str:
        return f"{self.robot_base_prim_path}/{self.stand_prim_name}"


@dataclass(frozen=True)
class _StandPrimSpec:
    stand_usd_path: str
    ref_prim_path: str = "/World/franka_table"
    payload_child_name: str = "franka_table"
    footprint_translate_xyz: tuple[float, float, float] = _FOOTPRINT_TRANSLATE_XYZ
    footprint_scale_xy: tuple[float, float] = _FOOTPRINT_SCALE_XY
    stand_default_height: float = _STAND_DEFAULT_HEIGHT_M


def compose_droid_on_stand(
    arena_assets_root: str | Path,
    *,
    stand_height_m: float = _STAND_DEFAULT_HEIGHT_M,
    output_dir: str | Path | None = None,
) -> str:
    """Return a local robot+stand USD path (cached under ``robot_library/droid``).

    Args:
        arena_assets_root: ``.assets/isaaclab_arena`` root.
        stand_height_m: Absolute stand height after align (Arena default 1.35).
        output_dir: Optional override for the composed USD directory.

    Returns:
        Filesystem path to the composed USD.
    """
    arena = Path(arena_assets_root)
    robot_usd = arena / "robot_library/droid/franka_robotiq_2f_85_flattened.usd"
    stand_usd = (
        arena / "object_library/srl_robolab_assets/robots/franka_stand_grey.usda"
    )
    if not robot_usd.is_file():
        raise FileNotFoundError(f"DROID robot USD missing: {robot_usd}")
    if not stand_usd.is_file():
        raise FileNotFoundError(f"Arena stand USD missing: {stand_usd}")

    out_dir = Path(output_dir) if output_dir else robot_usd.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # Include footprint in the cache name so translate/scale tweaks bust stale USDs.
    fx, fy, fz = _FOOTPRINT_TRANSLATE_XYZ
    sx, sy = _FOOTPRINT_SCALE_XY
    out_path = (
        out_dir
        / (
            f"franka_robotiq_2f_85_on_stand_{stand_height_m:.3f}"
            f"_t{fx:.3f}_{fy:.3f}_{fz:.3f}_s{sx:.2f}_{sy:.2f}.usd"
        )
    )
    if out_path.is_file() and out_path.stat().st_mtime >= max(
        robot_usd.stat().st_mtime, stand_usd.stat().st_mtime
    ):
        return str(out_path)

    robot = _RobotPrimSpec(robot_usd_path=str(robot_usd))
    stand = _StandPrimSpec(stand_usd_path=str(stand_usd))

    with tempfile.NamedTemporaryFile(
        suffix=".usd", dir=out_dir, delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        stage = Usd.Stage.CreateNew(str(tmp_path))
        root = stage.DefinePrim(robot.root_prim_path, "Xform")
        root.GetReferences().AddReference(robot.robot_usd_path, robot.root_prim_path)
        stage.SetDefaultPrim(root)
        _mount_stand_normalized(stage, robot, stand, stand_height_m)
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"failed to save composed on-stand USD to {tmp_path}")
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return str(out_path)


def _mount_stand_normalized(
    stage: Usd.Stage,
    robot: _RobotPrimSpec,
    stand: _StandPrimSpec,
    stand_height_m: float,
) -> None:
    """Parent Arena ``franka_table`` under ``panda_link0`` and scale to height."""
    robot_base = stage.GetPrimAtPath(robot.robot_base_prim_path)
    if not robot_base.IsValid():
        raise RuntimeError(
            f"On-stand USD missing robot base prim at {robot.robot_base_prim_path!r}"
        )

    pre_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    robot_range = pre_cache.ComputeWorldBound(robot_base).ComputeAlignedRange()
    if robot_range.IsEmpty():
        raise RuntimeError(f"empty robot base bounds at {robot_base.GetPath()}")
    robot_min_z = float(robot_range.GetMin()[2])

    tx, ty, _tz = stand.footprint_translate_xyz
    sx, sy = stand.footprint_scale_xy
    stand_prim_path = robot.stand_prim_path

    stand_xf = UsdGeom.Xform.Define(stage, stand_prim_path)
    translate_op = stand_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(tx, ty, _tz))
    scale_op = stand_xf.AddScaleOp()
    scale_op.Set(Gf.Vec3d(sx, sy, 1.0))

    payload_prim = stage.DefinePrim(f"{stand_prim_path}/{stand.payload_child_name}")
    payload_prim.GetReferences().AddReference(
        stand.stand_usd_path, stand.ref_prim_path
    )

    stand_prim = stand_xf.GetPrim()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = bbox_cache.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    if stand_range.IsEmpty():
        raise RuntimeError(f"empty stand bounds at {stand_prim.GetPath()}")
    native_height = float(stand_range.GetSize()[2])
    if native_height <= 0.0:
        raise RuntimeError(f"non-positive stand height at {stand_prim.GetPath()}")
    scale_op.Set(Gf.Vec3d(sx, sy, stand_height_m / native_height))
    translate_op.Set(Gf.Vec3d(tx, ty, robot_min_z))

    verify_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = verify_cache.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    stand_height = float(stand_range.GetSize()[2])
    stand_max_z = float(stand_range.GetMax()[2])
    if abs(stand_height - stand_height_m) >= _HEIGHT_ATOL:
        raise RuntimeError(
            f"stand height {stand_height} != requested {stand_height_m}"
        )
    if abs(stand_max_z - robot_min_z) >= _ALIGN_ATOL:
        raise RuntimeError(
            f"stand/robot align failed: stand_max_z={stand_max_z}, "
            f"robot_min_z={robot_min_z}"
        )
