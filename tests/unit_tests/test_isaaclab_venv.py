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

"""CPU queue transfer for SubProcIsaacLabEnv (avoids pidfd_getfd)."""

import torch

from rlinf.envs.isaaclab.venv import nested_to_device


def test_nested_to_device_cpu_roundtrip() -> None:
    payload = {
        "policy": {"rgb": torch.arange(4, dtype=torch.float32)},
        "flag": True,
    }
    cpu = nested_to_device(payload, None)
    assert cpu["policy"]["rgb"].device.type == "cpu"
    assert torch.equal(cpu["policy"]["rgb"], payload["policy"]["rgb"])
    assert cpu["flag"] is True


def test_nested_to_device_tuple_and_none() -> None:
    obs = torch.ones(2)
    info = {"done": torch.zeros(2, dtype=torch.bool)}
    moved = nested_to_device((obs, info, None), None)
    assert isinstance(moved, tuple)
    assert moved[0].device.type == "cpu"
    assert moved[1]["done"].device.type == "cpu"
    assert moved[2] is None
