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

"""DROID abs-joint-pos cube→plate task (Isaac Lab gym + RLinf wrapper)."""

from .env import (
    GYM_ID,
    IsaaclabPickPlaceCubePlateEnv,
    register_pick_place_cube_plate_env,
    wrap_droid_obs,
)

__all__ = [
    "GYM_ID",
    "IsaaclabPickPlaceCubePlateEnv",
    "register_pick_place_cube_plate_env",
    "wrap_droid_obs",
]
