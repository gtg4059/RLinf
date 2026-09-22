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

"""Arena ``On`` + ``NextTo`` root pose (no Isaac Lab import).

Ports the closed-form zero-loss pose of Isaac Lab Arena's
``NextToLossStrategy`` / ``OnLossStrategy`` so the kitchen fridge layout can
be reconstructed without running the relation solver.

Measured once from the vendored Lightwheel one-wall coastal USD. The
open-fridge YAML does not set ``placement_bbox_stand_only``, so NextTo
uses the full DROID+Arena-stand AABB
(``stand_footprint_xy_m=(1.08, 0.91032)``, translate ``(-0.05, 0, 0)``).
"""

from __future__ import annotations

# ``/world/fridge_main_group`` AABB in ``lightwheel_kitchen_one_wall_coastal``.
FRIDGE_AABB_MIN = (4.214507071646326, -0.8806412921421888, 0.0009619102988488445)
FRIDGE_AABB_MAX = (4.992475768927558, -0.001074445246380673, 1.7490380897011493)
# USD ``ComputeLocalToWorld`` translation of ``Refrigerator032_Body001``.
# PhysX uses this as the rigid-body origin (mesh points are local ±0.38/0.41/0.91).
FRIDGE_VISUAL_POS = (4.603491420286945, -0.44085786869428506, 0.875)

# Arena ``_DROID_STAND_PRIM``: translate −0.05, footprint 1.08 × 0.91032.
# Combined with the folded-arm AABB (placement_bbox_stand_only is unset).
STAND_TRANSLATE_X = -0.05
STAND_FOOTPRINT_XY_M = (1.08, 0.91032)
STAND_LOCAL_MAX_X = STAND_TRANSLATE_X + 0.5 * STAND_FOOTPRINT_XY_M[0]
ROBOT_LOCAL_AABB_MIN = (
    -0.8929012512565141,
    -0.5 * STAND_FOOTPRINT_XY_M[1],
    -0.7999999634007509,
)
ROBOT_LOCAL_AABB_MAX = (
    STAND_LOCAL_MAX_X,
    0.5 * STAND_FOOTPRINT_XY_M[1],
    0.7848902939247407,
)

# Arena ``kitchen_bench_lightwheel_open_fridge`` relations.
NEXT_TO_SIDE = "negative_y"
NEXT_TO_DISTANCE_M = 0.1
NEXT_TO_CROSS_POSITION_RATIO = 0.0
FLOOR_Z = 0.0
ON_CLEARANCE_M = 0.0
# Same marker yaw Arena ObjectPlacer bakes into the DROID AABB before NextTo.
ROTATE_AROUND_YAW_RAD = 1.57
# Extra EE-heading slide. Arena NextTo already owns the 0.1 m gap.
EE_FORWARD_SHIFT_M = 0.0


def yaw_z_aabb(
    child_min: tuple[float, float, float],
    child_max: tuple[float, float, float],
    yaw_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the axis-aligned box enclosing ``child`` after a Z yaw.

    Matches Arena ``AxisAlignedBoundingBox.rotated_by_quat`` for yaw-only
    ``RotateAroundSolution``: ObjectPlacer feeds this enclosing box to NextTo.
    """
    import math

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    xs = (child_min[0], child_max[0])
    ys = (child_min[1], child_max[1])
    corners = tuple(
        (cos_y * x - sin_y * y, sin_y * x + cos_y * y) for x in xs for y in ys
    )
    return (
        (min(p[0] for p in corners), min(p[1] for p in corners), child_min[2]),
        (max(p[0] for p in corners), max(p[1] for p in corners), child_max[2]),
    )


def next_to_root_xy(
    parent_min: tuple[float, float, float],
    parent_max: tuple[float, float, float],
    child_min: tuple[float, float, float],
    child_max: tuple[float, float, float],
    *,
    side: str,
    distance_m: float,
    cross_position_ratio: float = 0.0,
) -> tuple[float, float]:
    """Return the child root ``(x, y)`` that zeros Arena ``NextTo`` loss.

    Args:
        parent_min: Parent world AABB minimum.
        parent_max: Parent world AABB maximum.
        child_min: Child local AABB minimum (root-relative).
        child_max: Child local AABB maximum (root-relative).
        side: ``positive_x`` / ``negative_x`` / ``positive_y`` / ``negative_y``.
        distance_m: Target gap from the parent edge to the child face.
        cross_position_ratio: ``-1`` parent min, ``0`` center, ``1`` parent max
            along the axis perpendicular to ``side``.

    Returns:
        Child root world ``(x, y)``.
    """
    if side.endswith("_x"):
        primary, band = 0, 1
    elif side.endswith("_y"):
        primary, band = 1, 0
    else:
        raise ValueError(f"unsupported NextTo side: {side}")

    if side.startswith("positive_"):
        parent_edge = parent_max[primary]
        child_offset = child_min[primary]
        direction = 1.0
    elif side.startswith("negative_"):
        parent_edge = parent_min[primary]
        child_offset = child_max[primary]
        direction = -1.0
    else:
        raise ValueError(f"unsupported NextTo side: {side}")

    primary_pos = parent_edge + direction * distance_m - child_offset
    valid_band_min = parent_min[band] - child_min[band]
    valid_band_max = parent_max[band] - child_max[band]
    t = (cross_position_ratio + 1.0) / 2.0
    band_pos = valid_band_min + t * (valid_band_max - valid_band_min)

    xy = [0.0, 0.0]
    xy[primary] = primary_pos
    xy[band] = band_pos
    return (xy[0], xy[1])


def on_root_z(
    child_min_z: float,
    *,
    floor_z: float = FLOOR_Z,
    clearance_m: float = ON_CLEARANCE_M,
) -> float:
    """Return the child root ``z`` that sits on ``floor_z`` (Arena ``On``)."""
    return floor_z + clearance_m - child_min_z


def arena_open_fridge_robot_pos(
    *,
    fridge_min: tuple[float, float, float] = FRIDGE_AABB_MIN,
    fridge_max: tuple[float, float, float] = FRIDGE_AABB_MAX,
    robot_min: tuple[float, float, float] = ROBOT_LOCAL_AABB_MIN,
    robot_max: tuple[float, float, float] = ROBOT_LOCAL_AABB_MAX,
    side: str = NEXT_TO_SIDE,
    distance_m: float = NEXT_TO_DISTANCE_M,
    cross_position_ratio: float = NEXT_TO_CROSS_POSITION_RATIO,
    center_on_fridge_x: bool = False,
    yawed_stand_front: bool = False,
    ee_forward_shift_m: float = EE_FORWARD_SHIFT_M,
    yaw_rad: float = ROTATE_AROUND_YAW_RAD,
) -> tuple[float, float, float]:
    """DROID root from Arena ``On`` + ``NextTo`` with yawed AABB.

    Matches ``kitchen_bench_lightwheel_open_fridge``: ObjectPlacer applies
    ``rotate_around_solution(yaw_rad=1.57)`` to the DROID box, then solves
    NextTo fridge ``side=negative_y``, ``distance_m=0.1``. Solving the
    un-yawed box and yawing in place shifts the root ~20 cm in X off the
    fridge center (the view the Arena policy was evaluated on).
    """
    if yaw_rad:
        robot_min, robot_max = yaw_z_aabb(robot_min, robot_max, yaw_rad)
    x, y = next_to_root_xy(
        fridge_min,
        fridge_max,
        robot_min,
        robot_max,
        side=side,
        distance_m=distance_m,
        cross_position_ratio=cross_position_ratio,
    )
    if center_on_fridge_x:
        x = 0.5 * (fridge_min[0] + fridge_max[0])
    if yawed_stand_front:
        y = fridge_min[1] - distance_m - STAND_LOCAL_MAX_X
    z = on_root_z(robot_min[2])
    return (x, y + ee_forward_shift_m, z)


def arena_open_fridge_fridge_root_pos(
    *,
    robot_pos: tuple[float, float, float] | None = None,
    fridge_min: tuple[float, float, float] = FRIDGE_AABB_MIN,
    fridge_max: tuple[float, float, float] = FRIDGE_AABB_MAX,
    distance_m: float = NEXT_TO_DISTANCE_M,
    center_on_robot_x: bool = True,
) -> tuple[float, float, float]:
    """PhysX root for the kitchen fridge (Arena anchor: do not slide it).

    Mesh points are local to ``FRIDGE_VISUAL_POS``. The group origin ``(0,0,0)``
    would draw the body at the world origin. Optional ``robot_pos`` is ignored
    so stand / robot / fridge stay on the Arena layout.
    """
    del robot_pos, fridge_min, fridge_max, distance_m, center_on_robot_x
    return FRIDGE_VISUAL_POS


def placed_fridge_aabb(
    *,
    fridge_root: tuple[float, float, float] | None = None,
    fridge_min: tuple[float, float, float] = FRIDGE_AABB_MIN,
    fridge_max: tuple[float, float, float] = FRIDGE_AABB_MAX,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the fridge body AABB after ``arena_open_fridge_fridge_root_pos``."""
    if fridge_root is None:
        fridge_root = arena_open_fridge_fridge_root_pos()
    dx = fridge_root[0] - FRIDGE_VISUAL_POS[0]
    dy = fridge_root[1] - FRIDGE_VISUAL_POS[1]
    dz = fridge_root[2] - FRIDGE_VISUAL_POS[2]
    new_min = (
        fridge_min[0] + dx,
        fridge_min[1] + dy,
        fridge_min[2] + dz,
    )
    new_max = (
        fridge_max[0] + dx,
        fridge_max[1] + dy,
        fridge_max[2] + dz,
    )
    return new_min, new_max
