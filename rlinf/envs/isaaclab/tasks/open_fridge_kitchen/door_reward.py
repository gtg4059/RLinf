# Copyright 2026 The RLinf Authors.
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

"""Train-time fridge rewards that do not import Isaac Lab.

The success bonus is a terminal 1.0 the first time openness crosses the
Arena threshold. Progress pays only increases in openness, capped so a full
open stays below that bonus. Closing the door, including an Isaac reset,
adds nothing.
"""

from __future__ import annotations

import torch


def positive_openness_delta(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    """Return how much the door opened since the previous step."""
    return (current - previous).clamp(min=0.0)


def openness_level_reward(
    openness: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Pay ``weight * openness`` for the current door angle.

    Holding the door still keeps this payment. A smaller angle pays less
    on that step. Openness is clamped at 0 so a negative reading does not
    become a bonus.

    Args:
        openness: Door openness in ``[0, 1]``, shape ``(N,)``.
        weight: Scale on a fully open door for one step.

    Returns:
        Step reward, shape ``(N,)``.
    """
    return float(weight) * openness.clamp(min=0.0)


def capped_progress_reward(
    previous: torch.Tensor,
    current: torch.Tensor,
    paid: torch.Tensor,
    weight: float,
    cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pay ``weight * max(Δopenness, 0)`` until ``paid`` reaches ``cap``.

    Args:
        previous: Openness at the previous live step, shape ``(N,)``.
        current: Openness now, shape ``(N,)``.
        paid: Progress reward already given in this episode, shape ``(N,)``.
        weight: Scale on a full 0→1 open before the cap.
        cap: Episode ceiling on the sum of progress rewards.

    Returns:
        ``(reward, new_paid)``, each shape ``(N,)``.
    """
    raw = float(weight) * positive_openness_delta(previous, current)
    room = (float(cap) - paid).clamp(min=0.0)
    reward = torch.minimum(raw, room)
    return reward, paid + reward


def fridge_success_disabled(
    env,
    asset_cfg=None,
    joint_name: str = "fridge_door_joint",
    openness_threshold: float = 0.2,
) -> torch.Tensor:
    """Never terminate, while accepting the Isaac success-term parameters.

    Isaac Lab treats ``**kwargs`` as a mandatory parameter named ``kwargs``.
    This signature matches ``fridge_door_is_open`` so the existing term params
    still resolve. Training ends the episode in the RLinf wrapper instead.
    """
    del asset_cfg, joint_name, openness_threshold
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def cri_limit_near_handle(
    distance: torch.Tensor,
    segment_valid: torch.Tensor,
    *,
    base_limit: float,
    near_distance_m: float,
    near_limit: float,
) -> torch.Tensor:
    """Use ``near_limit`` when the gripper is within ``near_distance_m``.

    ``segment_valid`` is false when the handle prim was missing and the
    segment was copied from the gripper. Those rows keep ``base_limit``
    even though the reported distance is zero.

    Args:
        distance: Gripper-to-handle distance in meters, shape ``(N,)``.
        segment_valid: True where the handle segment has nonzero length.
        base_limit: CRI reset level away from the handle.
        near_distance_m: Inclusive radius around the handle bar, in meters.
        near_limit: CRI reset level inside that radius.

    Returns:
        Per-env CRI limits, shape ``(N,)``.
    """
    near = segment_valid.reshape(-1).bool() & (
        distance.reshape(-1) <= float(near_distance_m)
    )
    limits = torch.full_like(distance.reshape(-1), float(base_limit))
    return torch.where(near, limits.new_tensor(float(near_limit)), limits)


def handle_reach_distance(
    reach: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance from the gripper to the handle bar in ``handle_reach``.

    Args:
        reach: ``(N, 9)`` gripper xyz, segment start xyz, segment end xyz.

    Returns:
        ``(distance, segment_valid)``, each shape ``(N,)``. Distance is the
        gap outside the bar: 0 when the fingertip is on or inside the handle
        cross-section, then the extra separation as it moves away. Height
        along the rod does not change it. A segment shorter than 0.1 mm is
        invalid.
    """
    start = reach[:, 3:6]
    end = reach[:, 6:9]
    length = torch.linalg.norm(end - start, dim=-1)
    valid = length > 1e-4
    gap = point_line_distance(reach[:, 0:3], start, end) - FRIDGE_HANDLE_RADIUS_M
    return gap.clamp(min=0.0), valid


# Outer face center of ``Defeatured_2F_85_PAD_OPEN_fingertipsstep``, in each
# inner-finger frame. The finger joint is the body origin; the pad runs along
# local +X to about 14.9 cm. Left and right pads are mirrored in local Y.
LEFT_FINGERTIP_OFFSET_M = (0.1491, 0.0521, 0.0)
RIGHT_FINGERTIP_OFFSET_M = (0.1491, -0.0521, 0.0)

# ``fridge_door_handle_main`` centerline in the PhysX door-body frame.
# The USD door prim is scaled (about 1.015, 1.074, 0.962), and the body pose
# Isaac reports has no scale. These endpoints are the scaled cube axis, so a
# fingertip on the bar stays on it as ``fridge_door_joint`` turns.
FRIDGE_HANDLE_LOCAL_START = (
    -0.2549583329,
    -0.4240630258,
    0.1183077705,
)
FRIDGE_HANDLE_LOCAL_END = (
    -0.2549583329,
    -0.4240630258,
    0.3980461208,
)
# Half-width of ``fridge_door_handle_main`` after the door prim's scale.
# Fingertips on the bar are inside this radius, so their distance is 0.
FRIDGE_HANDLE_RADIUS_M = 0.023


def hinge_handle_segment(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    local_start: tuple[float, float, float] = FRIDGE_HANDLE_LOCAL_START,
    local_end: tuple[float, float, float] = FRIDGE_HANDLE_LOCAL_END,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World centerline of the handle bar on the current door pose.

    Args:
        body_pos: Door-link positions, shape ``(N, 3)``.
        body_quat: Door-link orientations, ``(w, x, y, z)``, shape ``(N, 4)``.
        local_start: Bar start in the door-link frame.
        local_end: Bar end in the door-link frame.

    Returns:
        ``(start, end)`` in world, each shape ``(N, 3)``.
    """
    start = body_pos.new_tensor(local_start).expand(body_pos.shape[0], 3)
    end = body_pos.new_tensor(local_end).expand(body_pos.shape[0], 3)
    world_start = body_pos + quat_apply_wxyz(body_quat, start)
    world_end = body_pos + quat_apply_wxyz(body_quat, end)
    return world_start, world_end


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by quaternion ``quat`` stored as ``(w, x, y, z)``.

    Args:
        quat: Quaternions, shape ``(..., 4)``.
        vec: Vectors, shape ``(..., 3)``.

    Returns:
        Rotated vectors, same shape as ``vec``.
    """
    xyz = quat[..., 1:]
    doubled = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[..., :1] * doubled + torch.cross(xyz, doubled, dim=-1)


def front_face_segment(
    point: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y_front: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertical line on a door front, at the gripper's x clamped to the door.

    Distance from ``point`` to this segment is the distance to the front
    rectangle. The fridge CRI bar uses ``hinge_handle_segment`` instead.

    Args:
        point: Gripper positions, shape ``(N, 3)``.
        x0: Door minimum x, shape ``(N,)``.
        x1: Door maximum x, shape ``(N,)``.
        y_front: Robot-facing plane, shape ``(N,)``.
        z0: Door minimum z, shape ``(N,)``.
        z1: Door maximum z, shape ``(N,)``.

    Returns:
        ``(start, end)`` of the segment, each shape ``(N, 3)``.
    """
    x = torch.maximum(torch.minimum(point[:, 0], x1), x0)
    y = y_front.reshape(-1)
    start = torch.stack((x, y, z0.reshape(-1)), dim=-1)
    end = torch.stack((x, y, z1.reshape(-1)), dim=-1)
    return start, end


def point_line_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Perpendicular distance from each point to the infinite line of a bar.

    Height along the rod is not part of the distance. A fingertip level with
    the top of the handle and one level with the middle are the same when
    their sideways gap matches. A zero-length bar reports distance 0.

    Args:
        point: Query positions, shape ``(N, 3)``.
        start: A point on the line, shape ``(N, 3)``.
        end: Another point on the line, shape ``(N, 3)``.

    Returns:
        Distances, shape ``(N,)``.
    """
    direction = end - start
    length_sq = (direction * direction).sum(dim=-1, keepdim=True)
    safe = length_sq.clamp(min=1e-12)
    scale = ((point - start) * direction).sum(dim=-1, keepdim=True) / safe
    closest = start + scale * direction
    distance = torch.linalg.norm(point - closest, dim=-1)
    missing = length_sq.squeeze(-1) <= 1e-8
    return torch.where(missing, torch.zeros_like(distance), distance)


def point_segment_distance(
    point: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    """Distance from each point to a line segment.

    A long handle is the segment between its two ends. Points beside the
    bar pay only the perpendicular distance. Points past either end pay the
    distance to that end, so the middle of the bar is not a special target.

    Args:
        point: Query positions, shape ``(N, 3)``.
        start: Segment starts, shape ``(N, 3)``.
        end: Segment ends, shape ``(N, 3)``.

    Returns:
        Distances, shape ``(N,)``.
    """
    direction = end - start
    length_sq = (direction * direction).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = ((point - start) * direction).sum(dim=-1, keepdim=True) / length_sq
    scale = scale.clamp(0.0, 1.0)
    closest = start + scale * direction
    return torch.linalg.norm(point - closest, dim=-1)


def segment_from_aabb(
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the centerline of a long bar from its axis-aligned bounds.

    The segment runs along the longest side of the box, through the center
    of the other two sides. A vertical handle therefore becomes a vertical
    segment, and a grasp anywhere along that span is the same distance.

    Args:
        aabb_min: Box minimum corner, shape ``(N, 3)`` or ``(3,)``.
        aabb_max: Box maximum corner, same shape.

    Returns:
        ``(start, end)`` on the long axis, each with the same shape as
        ``aabb_min``.
    """
    extent = aabb_max - aabb_min
    axis = int(extent.reshape(-1, 3).mean(dim=0).argmax().item())
    center = 0.5 * (aabb_min + aabb_max)
    start = center.clone()
    end = center.clone()
    start[..., axis] = aabb_min[..., axis]
    end[..., axis] = aabb_max[..., axis]
    return start, end


def capped_closing_reward(
    previous: torch.Tensor,
    current: torch.Tensor,
    paid: torch.Tensor,
    weight: float,
    cap: float,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pay for getting closer, and nothing for moving away.

    Args:
        previous: Distance at the previous live step, shape ``(N,)``.
        current: Distance now, shape ``(N,)``.
        paid: Reward already given in this episode, shape ``(N,)``.
        weight: Scale on one meter of closing.
        cap: Episode ceiling on the sum of closing rewards.
        active: Steps that still receive the reward, shape ``(N,)``.

    Returns:
        ``(reward, new_paid)``, each shape ``(N,)``.
    """
    raw = float(weight) * (previous - current).clamp(min=0.0)
    raw = torch.where(active, raw, torch.zeros_like(raw))
    room = (float(cap) - paid).clamp(min=0.0)
    reward = torch.minimum(raw, room)
    return reward, paid + reward


# Isaac ``end_effector_position_tracking`` weight −0.4, times step dt 1/15.
ISAAC_EE_POSITION_STEP_WEIGHT = -0.4 / 15.0


def gripper_past_handle(
    distance: torch.Tensor,
    segment_valid: torch.Tensor,
    *,
    max_distance_m: float,
) -> torch.Tensor:
    """True when the gripper is farther than ``max_distance_m`` from the bar.

    A missing handle prim is not past the bar. The comparison is strict, so
    a distance equal to the limit stays in the attempt.

    Args:
        distance: Gripper-to-handle distance in meters, shape ``(N,)``.
        segment_valid: True where the handle segment has nonzero length.
        max_distance_m: Distance that ends the attempt, in meters.

    Returns:
        Bool tensor, shape ``(N,)``.
    """
    return segment_valid.reshape(-1).bool() & (
        distance.reshape(-1) > float(max_distance_m)
    )


# Positive distance reward reaches 0 at 40 cm and stays 0 past that.
EE_BAR_DISTANCE_BONUS_MAX_M = 0.4


def ee_bar_distance_bonus(
    distance: torch.Tensor,
    weight: float,
    max_distance_m: float = EE_BAR_DISTANCE_BONUS_MAX_M,
) -> torch.Tensor:
    """Pay ``weight`` per meter of closeness, and 0 past ``max_distance_m``.

    ``weight`` is the pre-exp linear scale, per meter per step. On the bar
    the step pays ``weight * max_distance_m``. At 40 cm with weight 0.0025
    that is 0.001, so 600 steps on the bar sum to 0.6, below the sparse
    success bonus. A gap of 40 cm or more pays 0.

    Args:
        distance: Meters to the handle segment, shape ``(N,)``.
        weight: Reward per meter inside the cutoff.
        max_distance_m: Gap at which the bonus reaches 0, in meters.

    Returns:
        Step reward, shape ``(N,)``.
    """
    gap = distance.reshape(-1).clamp(min=0.0)
    remaining = (float(max_distance_m) - gap).clamp(min=0.0)
    return float(weight) * remaining


def capped_cri_penalty(
    exceed_chunks: torch.Tensor,
    chunk_penalty: float,
    max_abs: float = 1.0,
) -> torch.Tensor:
    """CRI charge of ``chunk_penalty`` per chunk, stopped at ``-max_abs``.

    Five chunks at −0.2 is −1. Chunks past that do not make it lower.

    Args:
        exceed_chunks: Chunks in this attempt that crossed, shape ``(N,)``.
        chunk_penalty: Amount per chunk. Negative.
        max_abs: Largest charge magnitude. ``1`` with −0.2 stops at five chunks.

    Returns:
        Charge in ``[-max_abs, 0]``, shape ``(N,)``.
    """
    counts = torch.as_tensor(exceed_chunks, dtype=torch.float32).reshape(-1)
    step = abs(float(chunk_penalty))
    if step == 0.0:
        return torch.zeros_like(counts)
    cap = float(max_abs) / step
    used = counts.clamp(min=0.0, max=cap)
    return -step * used


def success_pay_after_cri(
    success_bonus: float | torch.Tensor,
    exceed_chunks: torch.Tensor,
    chunk_penalty: float,
) -> torch.Tensor:
    """Reduce the success payment by each chunk that crossed the CRI limit.

    ``chunk_penalty`` is negative. The result stays at or above 0, so the
    discount changes the success reward and is not written on the
    exceeding step.

    Args:
        success_bonus: Payment before the CRI discount. A scalar or ``(N,)``.
        exceed_chunks: How many chunks in this attempt crossed, shape ``(N,)``.
        chunk_penalty: Amount added per exceeding chunk.

    Returns:
        Success payment, shape ``(N,)``.
    """
    counts = torch.as_tensor(exceed_chunks, dtype=torch.float32).reshape(-1)
    bonus = torch.as_tensor(
        success_bonus, dtype=torch.float32, device=counts.device
    )
    if bonus.ndim == 0:
        bonus = bonus.expand_as(counts)
    else:
        bonus = bonus.reshape(-1)
    return (bonus + float(chunk_penalty) * counts).clamp(min=0.0)


def sparse_success_reward(
    success_bonus: float,
    elapsed_steps: torch.Tensor,
    time_cost: float,
) -> torch.Tensor:
    """Sparse success bonus with the elapsed time taken out of it.

    ``elapsed_steps`` is the count before this step, so the success step
    itself is included. The result stays at or above 0.

    Args:
        success_bonus: Payment before the time deduction.
        elapsed_steps: Steps already taken in this episode, shape ``(N,)``.
        time_cost: Amount removed per step, including the success step.

    Returns:
        Bonus for an env that opens on this step, shape ``(N,)``.
    """
    steps = elapsed_steps.reshape(-1).to(dtype=torch.float32) + 1.0
    return (float(success_bonus) - float(time_cost) * steps).clamp(min=0.0)


def ee_bar_distance_penalty(
    distance: torch.Tensor,
    weight: float = ISAAC_EE_POSITION_STEP_WEIGHT,
) -> torch.Tensor:
    """Per-step penalty proportional to meters from the end-effector to the bar.

    Isaac Lab applies ``position_command_error`` (the L2 distance) with
    weight −0.4 and then multiplies by the 1/15 s step. ``weight`` is that
    product, so −0.4/15 charges the same amount per meter per control step.
    The caller stops the term once the episode has reached its goal.

    Args:
        distance: Meters to the handle segment, shape ``(N,)``.
        weight: Scale on one meter. Negative penalizes being far.

    Returns:
        Step reward, shape ``(N,)``.
    """
    return float(weight) * distance


def repeat_open_crossing(
    openness: torch.Tensor,
    attempt_paid: torch.Tensor,
    incoming_done: torch.Tensor,
    live: torch.Tensor,
    threshold: float = 0.2,
) -> torch.Tensor:
    """True when this attempt newly opens the door.

    ``incoming_done`` covers the step where Isaac already reset the scene,
    so the returned openness is closed even though the open just happened.
    ``attempt_paid`` is cleared after that reset, so a later open pays again.

    Args:
        openness: Door openness, shape ``(N,)``.
        attempt_paid: True after this attempt already received the bonus.
        incoming_done: True where this step is already a success termination.
        live: Rows that are not frozen, shape ``(N,)``.
        threshold: Openness that counts as open.

    Returns:
        ``(N,)`` bool.
    """
    fresh = ~attempt_paid.reshape(-1).bool() & live.reshape(-1).bool()
    opened = openness.reshape(-1) > float(threshold)
    done = incoming_done.reshape(-1).bool()
    return fresh & (opened | done)


# 0 rad pays the full step weight. 0.2 rad pays 0. Past that the factor
# stays 0, so a negative weight cannot flip into a bonus.
CLOSED_HINGE_FULL_RAD = 0.2


def closed_hinge_penalty(
    angle_rad: torch.Tensor,
    weight: float = -0.00025,
    full_open_rad: float = CLOSED_HINGE_FULL_RAD,
) -> torch.Tensor:
    """Pay ``weight * max(0, 1 - angle_rad / full_open_rad)`` per step.

    With the default 0.2 rad span this is ``weight * max(0, 1 - 5 * angle)``.
    ``weight`` is the charge at a shut door (negative). At ``full_open_rad``
    and beyond the term is 0.

    Args:
        angle_rad: Hinge angle from the closed limit, radians, shape ``(N,)``.
        weight: Step scale at 0 rad.
        full_open_rad: Angle where the factor reaches 0.

    Returns:
        Step reward, shape ``(N,)``.
    """
    angle = angle_rad.reshape(-1).to(dtype=torch.float32)
    span = float(full_open_rad)
    if span <= 0.0:
        factor = torch.zeros_like(angle)
    else:
        factor = (1.0 - angle / span).clamp(min=0.0)
    return float(weight) * factor


def unopened_timeout_mask(
    elapsed_steps: torch.Tensor,
    max_episode_steps: int,
    opened: torch.Tensor,
) -> torch.Tensor:
    """True only on the step an unopened episode first hits its own limit.

    Later steps stay past the limit when auto-reset is off, and those do
    not match. An episode that already opened, including an open on this
    step, does not match. A rollout cutoff before the limit does not match.

    Args:
        elapsed_steps: Steps taken in the current episode, shape ``(N,)``.
        max_episode_steps: This split's horizon.
        opened: True where this episode has crossed the open threshold.

    Returns:
        ``(N,)`` bool.
    """
    at_limit = elapsed_steps.reshape(-1) == int(max_episode_steps)
    return at_limit & ~opened.reshape(-1).bool()


def success_crossing(
    openness: torch.Tensor,
    already_succeeded: torch.Tensor,
    threshold: float = 0.2,
    incoming_done: torch.Tensor | None = None,
) -> torch.Tensor:
    """True on the first step that opens the door.

    Isaac's success term resets the hinge shut before this step's
    observation is read, so ``openness`` can already be closed.
    ``incoming_done`` is that success termination, taken before a later
    CRI failure is added. A row that already succeeded does not cross again.
    """
    fresh = ~already_succeeded.reshape(-1).bool()
    opened = openness.reshape(-1) > float(threshold)
    if incoming_done is None:
        return opened & fresh
    done = incoming_done.reshape(-1).bool()
    return fresh & (opened | done)
