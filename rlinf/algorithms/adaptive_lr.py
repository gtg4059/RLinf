# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KL-adaptive learning rate used by rsl_rl PPO (``schedule: adaptive``)."""


def adapt_learning_rate(
    learning_rate: float,
    kl_mean: float,
    desired_kl: float,
    *,
    min_lr: float = 1e-5,
    max_lr: float = 1e-2,
) -> float:
    """Shrink or grow the learning rate from the update's mean KL.

    Above twice ``desired_kl`` the rate is divided by 1.5. Below half of it,
    and still positive, the rate is multiplied by 1.5. Otherwise it stays.
    The result is clamped to ``[min_lr, max_lr]``.

    Args:
        learning_rate: Rate used for the update that produced ``kl_mean``.
        kl_mean: Mean approximate KL of that update.
        desired_kl: Target KL.
        min_lr: Lowest rate the schedule will set.
        max_lr: Highest rate the schedule will set.

    Returns:
        Learning rate for the next update.
    """
    rate = float(learning_rate)
    kl = float(kl_mean)
    target = float(desired_kl)
    if kl > target * 2.0:
        rate = max(float(min_lr), rate / 1.5)
    elif 0.0 < kl < target / 2.0:
        rate = min(float(max_lr), rate * 1.5)
    return rate


def kl_for_adaptive_lr(metrics: dict, kl_beta: float) -> float:
    """Pick the KL the adaptive rate compares with ``desired_kl``.

    A positive ``kl_beta`` logs the distance to the reference policy in
    ``actor/kl_loss``. With that penalty off the entry is 0, so the rate
    uses ``actor/update_kl``, the mean of ``0.5 (log π − log π_old)²``.

    Args:
        metrics: Reduced actor metrics for one update.
        kl_beta: Reference-policy penalty coefficient.

    Returns:
        KL passed to ``adapt_learning_rate``.
    """
    if float(kl_beta) > 0.0 and "actor/kl_loss" in metrics:
        return float(metrics["actor/kl_loss"])
    if "actor/update_kl" in metrics:
        return float(metrics["actor/update_kl"])
    return float(metrics.get("actor/approx_kl", 0.0))
