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

from .tasks.open_fridge_kitchen import IsaaclabOpenFridgeKitchenEnv
from .tasks.pick_place_cube_plate import IsaaclabPickPlaceCubePlateEnv
from .tasks.stack_cube import IsaaclabStackCubeEnv

REGISTER_ISAACLAB_ENVS = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0": IsaaclabStackCubeEnv,
    # DROID abs-joint-pos cube→plate (Arena specs as built-in Isaac Lab task).
    "Isaac-PickPlace-Cube-Plate-Droid-AbsJointPos-v0": IsaaclabPickPlaceCubePlateEnv,
    # DROID abs-joint-pos kitchen fridge open-door (Arena kitchen_bench spec).
    "Isaac-OpenFridge-Kitchen-Droid-AbsJointPos-v0": IsaaclabOpenFridgeKitchenEnv,
}

__all__ = [list(REGISTER_ISAACLAB_ENVS.keys())]
