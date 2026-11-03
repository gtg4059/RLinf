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

"""Match the rollout forward while an actor log-prob replay builds gradients."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def use_rollout_eval_mode(module: nn.Module) -> Iterator[None]:
    """Run ``module`` in eval mode, then restore its training flag.

    Rollout samples the flow-SDE gaussian under ``eval()`` and ``no_grad``.
    The actor replay otherwise stays in ``train()``, where gradient
    checkpointing is on. Those two forwards produce different velocity means
    at the same weights, so the chunk ratio leaves ``[0.8, 1.2]`` before the
    optimizer step. Eval mode uses the rollout kernels. Gradients still flow
    because this does not enter ``no_grad``.

    Args:
        module: Policy whose forward computes the flow-SDE mean.

    Yields:
        None. The caller runs the replay inside the block.
    """
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()
