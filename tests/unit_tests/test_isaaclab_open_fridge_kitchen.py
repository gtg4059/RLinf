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

"""Unit tests for kitchen fridge open-door MDP + registration (no Isaac Sim)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen import (
    GYM_ID,
    IsaaclabOpenFridgeKitchenEnv,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door import compute_door_openness
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.materials import (
    kitchen_fridge_overlay_usda,
    prepare_kitchen_fridge_materials,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.placement import (
    FRIDGE_AABB_MAX,
    FRIDGE_AABB_MIN,
    NEXT_TO_DISTANCE_M,
    ROBOT_LOCAL_AABB_MAX,
    ROBOT_LOCAL_AABB_MIN,
    STAND_LOCAL_MAX_X,
    FRIDGE_VISUAL_POS,
    arena_open_fridge_fridge_root_pos,
    arena_open_fridge_robot_pos,
    next_to_root_xy,
    on_root_z,
    placed_fridge_aabb,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate import wrap_droid_obs
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env import (
    IsaaclabPickPlaceCubePlateEnv,
)


def test_compute_door_openness_closed_and_threshold():
    pos = torch.tensor([0.0, 0.2, 1.0])
    lower = torch.zeros(3)
    upper = torch.ones(3)
    openness = compute_door_openness(pos, lower, upper)
    assert openness[0].item() == pytest.approx(0.0)
    assert openness[1].item() == pytest.approx(0.2)
    assert openness[2].item() == pytest.approx(1.0)
    assert bool((openness >= 0.2)[1])
    assert not bool((openness >= 0.2)[0])


def test_compute_door_openness_clamps_and_zero_span():
    pos = torch.tensor([-1.0, 2.0, 0.5])
    lower = torch.tensor([0.0, 0.0, 0.3])
    upper = torch.tensor([1.0, 1.0, 0.3])
    openness = compute_door_openness(pos, lower, upper)
    assert openness[0].item() == pytest.approx(0.0)
    assert openness[1].item() == pytest.approx(1.0)
    assert 0.0 <= openness[2].item() <= 1.0


def test_wrap_droid_obs_fridge_prompt():
    obs = {
        "policy": {
            "external_camera": np.zeros((224, 224, 3), dtype=np.uint8),
            "external_camera_2": np.full((224, 224, 3), 2, dtype=np.uint8),
            "wrist_camera": np.ones((224, 224, 3), dtype=np.uint8),
            "joint_pos": np.arange(7, dtype=np.float32),
            "gripper_pos": np.array([0.4], dtype=np.float32),
        }
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=1,
        device=torch.device("cpu"),
        task_description="open the fridge door",
    )
    assert wrapped["states"].shape == (1, 8)
    assert wrapped["task_descriptions"] == ["open the fridge door"]
    assert wrapped["main_images"].shape[-1] == 3
    assert wrapped["wrist_images"] is not None


def test_open_fridge_kitchen_registered():
    assert GYM_ID == "Isaac-OpenFridge-Kitchen-Droid-AbsJointPos-v0"
    assert GYM_ID in REGISTER_ISAACLAB_ENVS
    assert REGISTER_ISAACLAB_ENVS[GYM_ID] is IsaaclabOpenFridgeKitchenEnv
    assert issubclass(IsaaclabOpenFridgeKitchenEnv, IsaaclabPickPlaceCubePlateEnv)


def test_pi05_droid_jointpos_polaris_config_name():
    pytest.importorskip("openpi")
    pytest.importorskip("ray")
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT

    assert "pi05_droid_jointpos_polaris" in _CONFIGS_DICT
    cfg = _CONFIGS_DICT["pi05_droid_jointpos_polaris"]
    polaris = _CONFIGS_DICT["pi05_droid_polaris"]
    assert cfg.model.action_horizon == polaris.model.action_horizon
    assert cfg.data.assets.asset_id == polaris.data.assets.asset_id
    assert getattr(cfg.data, "use_cri", False) is False

    cri = _CONFIGS_DICT["pi05_droid_jointpos_polaris_cri"]
    assert cri.model.action_horizon == 15
    assert cri.model.max_token_len == 220
    assert getattr(cri.data, "use_cri", False) is True


def test_kitchen_fridge_overlay_sets_omnipbr_id(tmp_path):
    text = kitchen_fridge_overlay_usda()
    assert 'uniform token info:id = "OmniPBR"' in text
    assert 'uniform token info:id = "UsdPreviewSurface"' in text
    assert "./textures/T_Refrigerator032_BC001_0.png" in text
    assert "subLayers = [@./scene.usd@]" in text
    fake = tmp_path / "scene.usd"
    fake.write_text("#usda 1.0\n", encoding="utf-8")
    out = Path(prepare_kitchen_fridge_materials(fake))
    assert out.name == "scene_fridge_visible.usda"
    assert out.read_text(encoding="utf-8") == text


def test_arena_open_fridge_next_to_negative_y():
    x, y = next_to_root_xy(
        FRIDGE_AABB_MIN,
        FRIDGE_AABB_MAX,
        ROBOT_LOCAL_AABB_MIN,
        ROBOT_LOCAL_AABB_MAX,
        side="negative_y",
        distance_m=NEXT_TO_DISTANCE_M,
    )
    # Child +Y face sits 0.1 m in front of the fridge −Y face.
    assert (y + ROBOT_LOCAL_AABB_MAX[1]) == pytest.approx(
        FRIDGE_AABB_MIN[1] - NEXT_TO_DISTANCE_M
    )
    fridge_cx = 0.5 * (FRIDGE_AABB_MIN[0] + FRIDGE_AABB_MAX[0])
    # Arena stand 1.08×0.91 makes the child wider than the fridge, so NextTo
    # parks the root toward the +X wall (not fridge center).
    assert x == pytest.approx(
        0.5
        * (
            (FRIDGE_AABB_MIN[0] - ROBOT_LOCAL_AABB_MIN[0])
            + (FRIDGE_AABB_MAX[0] - ROBOT_LOCAL_AABB_MAX[0])
        )
    )
    assert y == pytest.approx(
        FRIDGE_AABB_MIN[1] - NEXT_TO_DISTANCE_M - ROBOT_LOCAL_AABB_MAX[1]
    )
    pos = arena_open_fridge_robot_pos()
    assert pos[0] == pytest.approx(x)
    assert pos[0] != pytest.approx(fridge_cx)
    assert pos[1] == pytest.approx(y)
    assert pos[2] == pytest.approx(on_root_z(ROBOT_LOCAL_AABB_MIN[2]))
    assert pos[2] == pytest.approx(0.8, abs=1e-6)
    fridge_root = arena_open_fridge_fridge_root_pos(robot_pos=pos)
    assert fridge_root == pytest.approx(FRIDGE_VISUAL_POS)
    placed_min, placed_max = placed_fridge_aabb(fridge_root=fridge_root)
    assert placed_min == pytest.approx(FRIDGE_AABB_MIN)
    assert placed_max == pytest.approx(FRIDGE_AABB_MAX)
    raw = arena_open_fridge_robot_pos(
        center_on_fridge_x=False, yawed_stand_front=False
    )
    assert raw[0] == pytest.approx(x)
    assert raw[1] == pytest.approx(y)
