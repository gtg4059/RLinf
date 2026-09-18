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

"""DROID abs-joint-pos kitchen fridge open-door task (Isaac Lab gym + RLinf)."""

from .door import compute_door_openness
from .env import (
    GYM_ID,
    IsaaclabOpenFridgeKitchenEnv,
    register_open_fridge_kitchen_env,
)

__all__ = [
    "GYM_ID",
    "IsaaclabOpenFridgeKitchenEnv",
    "compute_door_openness",
    "register_open_fridge_kitchen_env",
]
