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

import numpy as np
import pytest
import torch

from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen import (
    GYM_ID,
    IsaaclabOpenFridgeKitchenEnv,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door import (
    compute_door_openness,
    door_reached_from_rest,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.placement import (
    arena_open_fridge_robot_pos,
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
    # Arena Openable.is_open is strict: openness > threshold.
    assert bool((openness > 0.2)[2])
    assert not bool((openness > 0.2)[1])
    assert not bool((openness > 0.2)[0])


def test_door_reached_from_rest_matches_arena():
    """Reach is Arena is_away_from_rest_openness (min change 0.05)."""
    openness = torch.tensor([0.0, 0.04, 0.05, 0.06, 0.2])
    reached = door_reached_from_rest(openness)
    assert not bool(reached[0])
    assert not bool(reached[1])
    assert not bool(reached[2])
    assert bool(reached[3])
    assert bool(reached[4])


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


def test_arena_open_fridge_robot_pos_is_next_to_solver():
    """Eval pose must use the yawed AABB Arena ObjectPlacer feeds to NextTo."""
    pos = arena_open_fridge_robot_pos()
    assert pos[0] == pytest.approx(4.603651840801024, abs=1e-6)
    assert pos[1] == pytest.approx(-1.4710035928444452, abs=1e-6)
    assert pos[2] == pytest.approx(0.8, abs=1e-6)
    unyawed = arena_open_fridge_robot_pos(yaw_rad=0.0)
    assert unyawed[0] == pytest.approx(4.804942045915199, abs=1e-6)


def test_open_fridge_eval_yaml_uses_wxyz_plus90_for_fridge_view():
    """Converted DROID cams look [+X, −Y]; +90 yaw aims them at the fridge door."""
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[2]
    eval_yaml = (
        repo
        / "examples/embodiment/config/isaaclab_kitchen_open_fridge_openpi_pi05_arena_eval.yaml"
    )
    text = eval_yaml.read_text()
    assert "robot_rot: [0.70710678, 0.0, 0.0, 0.70710678]" in text
    env_yaml = yaml.safe_load(
        (repo / "examples/embodiment/config/env/isaaclab_kitchen_open_fridge.yaml").read_text()
    )
    assert env_yaml["init_params"]["robot_rot"] == [
        0.70710678,
        0.0,
        0.0,
        0.70710678,
    ]
    assert env_yaml["init_params"]["robot_pos"] == [
        4.603651840801024,
        -1.4710035928444452,
        0.8,
    ]


def test_open_fridge_kitchen_registered():
    assert GYM_ID == "Isaac-OpenFridge-Kitchen-Droid-AbsJointPos-v0"
    assert GYM_ID in REGISTER_ISAACLAB_ENVS
    assert REGISTER_ISAACLAB_ENVS[GYM_ID] is IsaaclabOpenFridgeKitchenEnv
    assert issubclass(IsaaclabOpenFridgeKitchenEnv, IsaaclabPickPlaceCubePlateEnv)


def test_open_fridge_env_cfg_matches_arena():
    """Ported scene must keep Arena NextTo pose and USD fridge drives."""
    pytest.importorskip("isaaclab")
    pytest.importorskip("pxr")
    from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg import (
        OpenFridgeKitchenSceneCfg,
        _FRIDGE_POS,
        _ROBOT_POS,
        _ROBOT_ROT,
    )

    expected = arena_open_fridge_robot_pos()
    assert _ROBOT_POS[0] == pytest.approx(expected[0], abs=1e-6)
    assert _ROBOT_POS[1] == pytest.approx(expected[1], abs=1e-6)
    assert _ROBOT_POS[2] == pytest.approx(expected[2], abs=1e-6)
    assert _ROBOT_ROT == (0.70710678, 0.0, 0.0, 0.70710678)
    scene = OpenFridgeKitchenSceneCfg()
    assert scene.fridge.actuators == {}
    assert tuple(scene.fridge.init_state.pos) == pytest.approx(_FRIDGE_POS, abs=1e-6)
    from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.env_cfg import _KITCHEN_USD

    assert "scene_fridge_visible.usda" in _KITCHEN_USD
    # Isaac Lab 2.x camera OffsetCfg.rot is wxyz (Arena DroidCameraCfg is xyzw).
    from isaaclab.sensors import TiledCameraCfg

    assert isinstance(scene.external_camera, TiledCameraCfg)
    assert scene.external_camera.height == 224
    assert scene.external_camera.offset.rot == (-0.393, -0.195, 0.399, 0.805)
    assert scene.external_camera_2.offset.rot == (0.805, 0.399, -0.195, -0.393)
    assert scene.wrist_camera.offset.rot == (-0.420, 0.570, 0.576, -0.409)


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

    adapter = _CONFIGS_DICT["pi05_droid_jointpos_polaris_cri_adapter"]
    assert adapter.model.action_horizon == 15
    assert adapter.model.max_token_len == 200
    assert getattr(adapter.data, "use_cri", False) is True
    assert getattr(adapter.data, "use_cri_prefix", False) is True
