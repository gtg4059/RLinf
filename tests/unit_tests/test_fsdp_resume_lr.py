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

"""Resume must keep a config learning rate after the scheduler steps."""

import torch
from torch.nn import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from rlinf.algorithms.utils import reference_kl_loss
from rlinf.hybrid_engines.fsdp.optim_lr import apply_cfg_lrs_to_optimizer


def test_resume_lr_survives_scheduler_step() -> None:
    actor = Parameter(torch.zeros(2))
    value = Parameter(torch.zeros(2))
    optimizer = AdamW(
        [
            {"params": [actor], "lr": 3e-7},
            {"params": [value], "lr": 3e-4},
        ]
    )
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    optimizer.param_groups[0]["lr"] = 3e-7
    optimizer.param_groups[0]["initial_lr"] = 3e-7
    scheduler.base_lrs[0] = 3e-7

    apply_cfg_lrs_to_optimizer(
        optimizer,
        scheduler,
        actor_lr=1e-6,
        value_lr=3e-4,
    )
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == 1e-6
    assert optimizer.param_groups[0]["initial_lr"] == 1e-6
    assert scheduler.base_lrs[0] == 1e-6
    assert optimizer.param_groups[1]["lr"] == 3e-4


def test_reference_kl_is_zero_when_policies_match() -> None:
    logprobs = torch.zeros(2, 4, 8)
    mask = torch.ones(2, 4)
    loss = reference_kl_loss(logprobs, logprobs, mask, "kl", action_dim=8)
    assert float(loss) == 0.0


def test_low_var_kl_is_flat_on_the_reference_and_costs_the_observed_drift() -> None:
    """k1 pulls every token. low_var_kl pulls only after the policy leaves."""
    reference = torch.zeros(2, 4, 8)
    mask = torch.ones(2, 4)
    matched = torch.zeros(2, 4, 8, requires_grad=True)
    flat = reference_kl_loss(matched, reference, mask, "low_var_kl", action_dim=8)
    flat.backward()
    assert abs(float(flat)) < 1e-6
    assert float(matched.grad.abs().max()) < 1e-5

    # Mean k1 of 0.38 is where success fell. low_var_kl there is ~0.064,
    # so beta 1 adds ~0.06, and the log-prob gradient is 1 - exp(-0.38).
    drifted = torch.full((2, 4, 8), 0.38, requires_grad=True)
    penalty = reference_kl_loss(
        drifted, reference, mask, "low_var_kl", action_dim=8
    )
    penalty.backward()
    expected = torch.exp(torch.tensor(-0.38)) + 0.38 - 1
    assert abs(float(penalty) - float(expected)) < 1e-4
    # The loss is a mean, so each element's gradient is the derivative / N.
    # 1 - exp(-0.38) is that derivative, about 0.32.
    per_token = drifted.grad.sum()
    assert abs(float(per_token) - (1 - torch.exp(torch.tensor(-0.38)))) < 1e-4


def test_reference_kl_penalizes_leaving_the_reference() -> None:
    reference = torch.zeros(2, 4, 8)
    current = torch.full((2, 4, 8), 0.2)
    mask = torch.ones(2, 4)
    mask[0, 0] = 0
    loss = reference_kl_loss(current, reference, mask, "kl", action_dim=8)
    assert abs(float(loss) - 0.2) < 1e-6
