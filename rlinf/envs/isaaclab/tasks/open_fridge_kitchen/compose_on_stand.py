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

"""Compose DROID + Arena stand footprint for kitchen fridge-open."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

# Arena ``StandPrimSpec`` for DROID: translate (−0.05, 0, 0), 1.08 x 0.91032 m.
_ARENA_FOOTPRINT_TRANSLATE_XYZ = (-0.05, 0.0, 0.0)
_ARENA_FOOTPRINT_XY_M = (1.08, 0.91032)
_HEIGHT_ATOL = 1e-3
_FOOTPRINT_ATOL = 1e-3
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


def compose_droid_on_stand_arena(
    arena_assets_root: str | Path,
    *,
    stand_height_m: float,
    output_dir: str | Path | None = None,
) -> str:
    """Return a cached robot+stand USD with Arena fridge footprint."""
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
    tx, ty, tz = _ARENA_FOOTPRINT_TRANSLATE_XYZ
    fx, fy = _ARENA_FOOTPRINT_XY_M
    out_path = out_dir / (
        f"franka_robotiq_2f_85_on_stand_{stand_height_m:.3f}"
        f"_t{tx:.3f}_{ty:.3f}_{tz:.3f}_f{fx:.3f}x{fy:.3f}.usd"
    )
    if out_path.is_file() and out_path.stat().st_mtime >= max(
        robot_usd.stat().st_mtime, stand_usd.stat().st_mtime
    ):
        return str(out_path)

    robot = _RobotPrimSpec(robot_usd_path=str(robot_usd))
    with tempfile.NamedTemporaryFile(
        suffix=".usd", dir=out_dir, delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        stage = Usd.Stage.CreateNew(str(tmp_path))
        root = stage.DefinePrim(robot.root_prim_path, "Xform")
        root.GetReferences().AddReference(robot.robot_usd_path, robot.root_prim_path)
        stage.SetDefaultPrim(root)
        _mount_arena_stand(stage, robot, str(stand_usd), stand_height_m)
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"failed to save composed on-stand USD to {tmp_path}")
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return str(out_path)


def _mount_arena_stand(
    stage: Usd.Stage,
    robot: _RobotPrimSpec,
    stand_usd_path: str,
    stand_height_m: float,
) -> None:
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

    tx, ty, _tz = _ARENA_FOOTPRINT_TRANSLATE_XYZ
    stand_xf = UsdGeom.Xform.Define(stage, robot.stand_prim_path)
    translate_op = stand_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(tx, ty, _tz))
    scale_op = stand_xf.AddScaleOp()
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))

    payload_prim = stage.DefinePrim(f"{robot.stand_prim_path}/franka_table")
    payload_prim.GetReferences().AddReference(stand_usd_path, "/World/franka_table")

    stand_prim = stand_xf.GetPrim()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = bbox_cache.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    if stand_range.IsEmpty():
        raise RuntimeError(f"empty stand bounds at {stand_prim.GetPath()}")
    native_x, native_y, native_height = (float(v) for v in stand_range.GetSize())
    if native_x <= 0.0 or native_y <= 0.0 or native_height <= 0.0:
        raise RuntimeError(f"non-positive stand size at {stand_prim.GetPath()}")
    fx, fy = _ARENA_FOOTPRINT_XY_M
    scale_op.Set(Gf.Vec3d(fx / native_x, fy / native_y, stand_height_m / native_height))
    translate_op.Set(Gf.Vec3d(tx, ty, robot_min_z))

    verify = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = verify.ComputeWorldBound(stand_prim).ComputeAlignedRange()
    stand_x_m, stand_y_m, stand_height = (float(v) for v in stand_range.GetSize())
    stand_max_z = float(stand_range.GetMax()[2])
    if abs(stand_height - stand_height_m) >= _HEIGHT_ATOL:
        raise RuntimeError(f"stand height {stand_height} != requested {stand_height_m}")
    if abs(stand_x_m - fx) >= _FOOTPRINT_ATOL:
        raise RuntimeError(f"stand x {stand_x_m} != requested {fx}")
    if abs(stand_y_m - fy) >= _FOOTPRINT_ATOL:
        raise RuntimeError(f"stand y {stand_y_m} != requested {fy}")
    if abs(stand_max_z - robot_min_z) >= _ALIGN_ATOL:
        raise RuntimeError(
            f"stand/robot align failed: stand_max_z={stand_max_z}, "
            f"robot_min_z={robot_min_z}"
        )
