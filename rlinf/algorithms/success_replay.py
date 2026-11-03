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

"""Keep recent open trajectories and mix them back when live opens fall."""

import math
from typing import Any

import torch


def _except_batch(values: torch.Tensor) -> list[int]:
    return [0] + list(range(2, values.ndim))


def hold_policy_update(
    live_open_fraction: float,
    buffer_size: int,
    min_open_fraction: float,
    *,
    open_fraction_after_mix: float | None = None,
) -> bool:
    """Stop the policy step only when a dropped batch still has no open.

    A live rate below ``min_open_fraction`` used to zero the update as soon
    as the buffer held anything, which also threw away the opens just mixed
    back in. The step holds when that mix still leaves the batch with no
    open. A batch that regained an open keeps training. ``buffer_size`` is
    accepted for callers that have not mixed yet.

    Args:
        live_open_fraction: Share of the current rollout that opened,
            measured before the mix.
        buffer_size: Stored opens available before this mix.
        min_open_fraction: Live share below which the batch is sparse.
        open_fraction_after_mix: Share that opened after stored opens
            replaced misses. ``None`` keeps the pre-mix check.

    Returns:
        True when the policy surrogate should be scaled to 0.
    """
    below = float(live_open_fraction) < float(min_open_fraction)
    if open_fraction_after_mix is None:
        return int(buffer_size) > 0 and below
    return below and float(open_fraction_after_mix) <= 0.0


def open_columns(rewards: torch.Tensor, success_reward_min: float) -> torch.Tensor:
    """True where any reward in the column exceeds ``success_reward_min``."""
    reduced = rewards.detach().to(dtype=torch.float32)
    return reduced.amax(dim=_except_batch(reduced)) > float(success_reward_min)


def _column_return(rewards: torch.Tensor) -> torch.Tensor:
    reduced = rewards.detach().to(dtype=torch.float32)
    return reduced.sum(dim=_except_batch(reduced))


def _slice_column(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 2 and value.shape[1] == batch_size:
            column = value[:, index : index + 1].detach().clone()
            return column.to(device="cpu").contiguous()
        return None
    if isinstance(value, dict):
        sliced = {}
        for key, item in value.items():
            column = _slice_column(item, index, batch_size)
            if column is not None:
                sliced[key] = column
        return sliced
    return None


def _write_column(dest: Any, source: Any, index: int, batch_size: int) -> None:
    if isinstance(dest, dict) and isinstance(source, dict):
        for key, item in dest.items():
            if key in source:
                _write_column(item, source[key], index, batch_size)
        return
    if (
        isinstance(dest, torch.Tensor)
        and isinstance(source, torch.Tensor)
        and dest.ndim >= 2
        and dest.shape[1] == batch_size
        and source.shape[1] == 1
        and source.shape[0] == dest.shape[0]
        and source.shape[2:] == dest.shape[2:]
    ):
        dest[:, index : index + 1] = source.to(device=dest.device, dtype=dest.dtype)


class SuccessReplay:
    """Ring buffer of open columns, mixed into later rollouts.

    The current batch is stored after the mix, so an empty buffer does not
    change the first update. Replacements are the lowest-return columns that
    did not open.
    """

    def __init__(
        self,
        fraction: float = 0.1,
        capacity: int = 32,
        success_reward_min: float = 0.5,
    ) -> None:
        self.fraction = float(fraction)
        self.capacity = max(int(capacity), 1)
        self.success_reward_min = float(success_reward_min)
        self._columns: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._columns)

    def mix(self, batch: dict[str, Any], *, floor_fraction: float = 0.0) -> int:
        """Replace low-return misses with stored opens. Returns how many.

        Below ``floor_fraction`` the batch is filled back up to that share.
        At or above the floor, ``fraction`` of the batch is still replaced.
        The current batch is stored after the mix, so an empty buffer does
        not change the first update.

        Args:
            batch: Rollout batch with time at dim 0 and envs at dim 1.
            floor_fraction: Open share to restore when the live batch is
                under it. ``0`` leaves only the ``fraction`` cap.

        Returns:
            Number of columns replaced. Zero when the buffer is empty or
            every column in ``batch`` already opened.
        """
        rewards = batch.get("rewards")
        if not isinstance(rewards, torch.Tensor) or rewards.ndim < 2:
            return 0
        batch_size = int(rewards.shape[1])
        opened = open_columns(rewards, self.success_reward_min)
        fresh = [
            _slice_column(batch, index, batch_size)
            for index in opened.nonzero(as_tuple=False).flatten().tolist()
        ]
        mixed = 0
        misses = (~opened).nonzero(as_tuple=False).flatten()
        opened_count = int(opened.sum())
        target = (
            math.ceil(float(floor_fraction) * batch_size - 1e-9)
            if float(floor_fraction) > 0.0
            else 0
        )
        if opened_count < target:
            n_replace = target - opened_count
        else:
            n_replace = int(batch_size * self.fraction)
        n_replace = min(n_replace, len(self._columns), int(misses.numel()))
        if n_replace > 0:
            returns = _column_return(rewards)
            order = torch.argsort(returns[misses])
            destinations = misses[order[:n_replace]].tolist()
            sources = self._columns[-n_replace:]
            for index, column in zip(destinations, sources, strict=True):
                _write_column(batch, column, int(index), batch_size)
            mixed = n_replace
        for column in fresh:
            if column is not None:
                self._columns.append(column)
        if len(self._columns) > self.capacity:
            self._columns = self._columns[-self.capacity :]
        return mixed
