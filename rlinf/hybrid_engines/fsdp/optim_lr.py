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

"""Apply config learning rates after a checkpoint restore."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def apply_cfg_lrs_to_optimizer(
    optimizer: Optimizer | None,
    lr_scheduler: LRScheduler | None,
    actor_lr: float | None,
    value_lr: float | None,
) -> None:
    """Write config learning rates onto an optimizer after checkpoint load.

    Group 0 is the actor and group 1 is the value head, matching
    ``build_optimizer``. DCP restores each group's ``lr`` and the scheduler
    ``base_lrs``. The next ``scheduler.step()`` copies ``base_lrs`` back onto
    the groups, so both have to match the config.

    Args:
        optimizer: Optimizer restored from the checkpoint.
        lr_scheduler: Scheduler restored alongside that optimizer.
        actor_lr: Actor learning rate from config. ``None`` leaves group 0.
        value_lr: Value-head learning rate from config. ``None`` leaves group 1.
    """
    if optimizer is None:
        return
    groups = optimizer.param_groups
    updates: list[tuple[int, float]] = []
    if actor_lr is not None and groups:
        updates.append((0, float(actor_lr)))
    if value_lr is not None and len(groups) > 1:
        updates.append((1, float(value_lr)))
    for idx, lr in updates:
        groups[idx]["lr"] = lr
        if "initial_lr" in groups[idx]:
            groups[idx]["initial_lr"] = lr
    base_lrs = getattr(lr_scheduler, "base_lrs", None)
    if base_lrs is None:
        return
    for idx, lr in updates:
        if idx < len(base_lrs):
            base_lrs[idx] = lr
