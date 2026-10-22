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

import inspect

import numpy as np
import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen import (
    GYM_ID,
    IsaaclabOpenFridgeKitchenEnv,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door import (
    compute_door_openness,
    door_reached_from_rest,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.door_reward import (
    capped_closing_reward,
    capped_progress_reward,
    cri_limit_near_handle,
    waive_cri_penalty_near_handle,
    LEFT_FINGERTIP_OFFSET_M,
    RIGHT_FINGERTIP_OFFSET_M,
    FRIDGE_HANDLE_LOCAL_END,
    FRIDGE_HANDLE_LOCAL_START,
    ee_bar_distance_penalty,
    gripper_past_handle,
    front_face_segment,
    handle_reach_distance,
    hinge_handle_segment,
    quat_apply_wxyz,
    openness_level_reward,
    fridge_success_disabled,
    point_segment_distance,
    segment_from_aabb,
    repeat_open_crossing,
    success_crossing,
)
from rlinf.envs.isaaclab.tasks.open_fridge_kitchen.placement import (
    arena_open_fridge_robot_pos,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate import wrap_droid_obs
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.env import (
    IsaaclabPickPlaceCubePlateEnv,
)
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate.isaac_reset import (
    zero_paused_reward,
)
from rlinf.utils.metric_utils import compute_loss_mask


def test_fingertip_offset_is_the_outer_pad_face():
    """The query point is the pad tip, 14.9 cm forward of the finger joint."""
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    left = quat_apply_wxyz(quat, torch.tensor([LEFT_FINGERTIP_OFFSET_M]))
    right = quat_apply_wxyz(quat, torch.tensor([RIGHT_FINGERTIP_OFFSET_M]))
    assert left[0, 0].item() == pytest.approx(0.1491)
    assert right[0, 0].item() == pytest.approx(0.1491)
    assert left[0, 1].item() == pytest.approx(0.0521)
    assert right[0, 1].item() == pytest.approx(-0.0521)
    midpoint = 0.5 * (left + right)
    assert midpoint[0, 0].item() == pytest.approx(0.1491)
    assert midpoint[0, 1].item() == pytest.approx(0.0)


def test_handle_bar_follows_the_hinge_and_keeps_a_grasp_on_the_bar():
    """A fingertip on the bar stays within 10 cm after the door rotates.

    The same fingertip, left at the closed-door bar, falls outside 10 cm.
    """
    center = 0.5 * (
        torch.tensor(FRIDGE_HANDLE_LOCAL_START) + torch.tensor(FRIDGE_HANDLE_LOCAL_END)
    )
    origin = torch.zeros(1, 3)
    closed_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    open_quat = torch.tensor([[0.70710678118, 0.70710678118, 0.0, 0.0]])
    closed_start, closed_end = hinge_handle_segment(origin, closed_quat)
    open_start, open_end = hinge_handle_segment(origin, open_quat)
    on_bar = quat_apply_wxyz(open_quat, center.view(1, 3))
    on_bar_distance = point_segment_distance(on_bar, open_start, open_end)
    left_behind = point_segment_distance(center.view(1, 3), open_start, open_end)
    assert on_bar_distance.item() == pytest.approx(0.0, abs=1e-5)
    assert left_behind.item() > 0.10
    closed_span = torch.linalg.norm(closed_end - closed_start, dim=-1)
    open_span = torch.linalg.norm(open_end - open_start, dim=-1)
    assert closed_span.item() == pytest.approx(open_span.item())
    assert closed_span.item() == pytest.approx(0.27973835, abs=1e-4)


def test_front_face_distance_is_the_gap_to_the_door_plane():
    """A point in front of the door pays the y gap, not the distance to a side edge."""
    point = torch.tensor([[0.0, -0.05, 0.5], [0.5, -0.05, 0.5]])
    x0 = torch.tensor([-0.3, -0.3])
    x1 = torch.tensor([0.3, 0.3])
    y_front = torch.zeros(2)
    z0 = torch.zeros(2)
    z1 = torch.ones(2)
    start, end = front_face_segment(point, x0, x1, y_front, z0, z1)
    distance = point_segment_distance(point, start, end)
    assert distance[0].item() == pytest.approx(0.05)
    assert distance[1].item() == pytest.approx((0.2**2 + 0.05**2) ** 0.5)


def test_cri_limit_rises_to_3_within_10cm_of_the_handle():
    """A gripper on the bar uses CRI 3. Farther away, and a missing bar, stay at 0.96."""
    # 5 cm outside the bar, 15 cm outside, and a missing prim (zero-length copy).
    # The bar radius is 2.3 cm, so the centerline gaps are 7.3 cm and 17.3 cm.
    reach = torch.tensor(
        [
            [0.073, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.173, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        ]
    )
    distance, valid = handle_reach_distance(reach)
    limits = cri_limit_near_handle(
        distance,
        valid,
        base_limit=0.96,
        near_distance_m=0.10,
        near_limit=3.0,
    )
    assert distance[0].item() == pytest.approx(0.05)
    assert distance[1].item() == pytest.approx(0.15)
    assert bool(valid[0]) and bool(valid[1]) and not bool(valid[2])
    assert limits[0].item() == pytest.approx(3.0)
    assert limits[1].item() == pytest.approx(0.96)
    assert limits[2].item() == pytest.approx(0.96)


def test_cri_penalty_is_zero_inside_the_handle_radius():
    """Inside 8 cm the penalty is 0 even when CRI is past the raised limit."""
    penalty = torch.tensor([5.0, 5.0, 5.0])
    distance = torch.tensor([0.05, 0.08, 0.09])
    valid = torch.tensor([True, True, True])
    waived = waive_cri_penalty_near_handle(
        penalty, distance, valid, near_distance_m=0.08
    )
    assert waived.tolist() == pytest.approx([0.0, 0.0, 5.0])
    missing = waive_cri_penalty_near_handle(
        penalty[:1],
        torch.zeros(1),
        torch.tensor([False]),
        near_distance_m=0.08,
    )
    assert float(missing[0]) == pytest.approx(5.0)


def test_gripper_past_40cm_ends_the_attempt():
    """40 cm is still in the attempt. Farther, with a real handle, ends it."""
    distance = torch.tensor([0.39, 0.40, 0.41])
    valid = torch.tensor([True, True, True])
    past = gripper_past_handle(distance, valid, max_distance_m=0.40)
    assert past.tolist() == [False, False, True]
    missing = gripper_past_handle(
        torch.tensor([1.0]),
        torch.tensor([False]),
        max_distance_m=0.40,
    )
    assert missing.tolist() == [False]


def test_cri_limit_is_left_out_of_env_metrics():
    """The per-step limit stays on the video and is not averaged."""
    from rlinf.utils.metric_utils import (
        compute_evaluate_metrics,
        omit_cri_limit_metric,
    )

    env_info = {
        "cri_limit": torch.tensor([0.96, 3.0]),
        "cri_limit_outside": torch.tensor([True, False]),
        "success_once": torch.tensor([1.0, 0.0]),
    }
    omit_cri_limit_metric(env_info)
    assert "cri_limit" not in env_info
    assert "cri_limit_outside" not in env_info
    metrics = compute_evaluate_metrics([env_info])
    assert "cri_limit" not in metrics
    assert float(metrics["success_once"]) == pytest.approx(0.5)


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
        (
            repo / "examples/embodiment/config/env/isaaclab_kitchen_open_fridge.yaml"
        ).read_text()
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


def test_openness_level_reward_holds_and_follows_the_angle():
    held = openness_level_reward(torch.tensor([0.19, 0.10]), weight=1.0)
    assert held[0].item() == pytest.approx(0.19)
    assert held[1].item() == pytest.approx(0.10)
    again = openness_level_reward(torch.tensor([0.19, 0.05]), weight=1.0)
    assert again[0].item() == pytest.approx(0.19)
    assert again[1].item() == pytest.approx(0.05)
    assert openness_level_reward(torch.tensor([-0.1]), weight=1.0).item() == 0.0


def test_fridge_level_reward_stops_at_the_success_termination():
    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._success_bonus = 0.0
    env._door_progress_weight = 0.0
    env._door_progress_cap = 0.4
    env._door_openness_weight = 1.0
    env._door_openness_reward_sum = torch.zeros(2)
    env._openness_threshold = 0.2
    env._door_openness_prev = None
    env._door_progress_paid = None
    env._last_door_openness = torch.tensor([0.19, 0.19])
    env.success_once = torch.tensor([False, False])
    env._last_task_success = torch.tensor([False, False])
    terminations = torch.zeros(2, dtype=torch.bool)
    was_frozen = torch.tensor([False, True])
    reward, terminations = env._augment_step_reward(
        torch.zeros(2), terminations, was_frozen
    )
    assert reward[0].item() == pytest.approx(0.19)
    assert reward[1].item() == pytest.approx(0.0)
    assert not bool(terminations[0])
    reward, _ = env._augment_step_reward(torch.zeros(2), terminations, was_frozen)
    assert reward[0].item() == pytest.approx(0.19)
    assert reward[1].item() == pytest.approx(0.0)
    assert env._door_openness_reward_sum[0].item() == pytest.approx(0.38)
    assert env._door_openness_reward_sum[1].item() == pytest.approx(0.0)


def test_progress_reward_ignores_closing_and_caps_at_0_4():
    prev = torch.tensor([0.0, 0.5])
    curr = torch.tensor([1.0, 0.1])
    paid = torch.zeros(2)
    reward, new_paid = capped_progress_reward(prev, curr, paid, weight=0.3, cap=0.4)
    assert reward[0].item() == pytest.approx(0.3)
    assert reward[1].item() == pytest.approx(0.0)
    # Another full close-open only has 0.1 of cap left.
    reward, new_paid = capped_progress_reward(
        torch.zeros(2), torch.ones(2), new_paid, weight=0.3, cap=0.4
    )
    assert reward[0].item() == pytest.approx(0.1)
    assert new_paid[0].item() == pytest.approx(0.4)
    reward, new_paid = capped_progress_reward(
        torch.zeros(2), torch.ones(2), new_paid, weight=0.3, cap=0.4
    )
    assert reward[0].item() == pytest.approx(0.0)


def test_disabled_success_term_keeps_isaac_parameter_names():
    """Isaac rejects ``**kwargs`` and an empty params dict on the success term."""
    params = inspect.signature(fridge_success_disabled).parameters
    assert "kwargs" not in params
    for name in ("asset_cfg", "joint_name", "openness_threshold"):
        assert params[name].default is not inspect.Parameter.empty
        assert params[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    env = type("Env", (), {"num_envs": 3, "device": "cpu"})()
    out = fridge_success_disabled(
        env, asset_cfg=object(), joint_name="fridge_door_joint"
    )
    assert out.shape == (3,)
    assert not bool(out.any())


def test_repeat_open_pays_again_after_the_door_resets():
    """A reset termination pays +40, and the next reset pays +40 again."""
    first = repeat_open_crossing(
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([True]),
        torch.tensor([True]),
        threshold=0.2,
    )
    assert bool(first[0])
    paid = repeat_open_crossing(
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([True]),
        threshold=0.2,
    )
    assert not bool(paid[0])

    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._success_bonus = 40.0
    env._door_progress_weight = 0.0
    env._door_progress_cap = 0.4
    env._door_openness_weight = 0.0
    env._reach_bonus = 0.0
    env._handle_approach_weight = 0.0
    env._openness_threshold = 0.2
    env._repeat_open_after_success = True
    env._door_openness_prev = None
    env._door_progress_paid = None
    env._last_door_openness = torch.tensor([0.0])
    env.success_once = torch.tensor([False])
    env.reach_once = torch.tensor([True])
    env._open_attempt_paid = torch.tensor([False])
    env._opened_once = torch.tensor([False])
    env._last_task_success = torch.tensor([False])
    reward, terminations = env._augment_step_reward(
        torch.zeros(1),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.bool),
    )
    assert reward.item() == pytest.approx(40.0)
    assert bool(terminations[0])
    assert bool(env._opened_once[0])

    env._last_door_openness = torch.tensor([0.5])
    env._start_next_open_attempt(torch.tensor([True]))
    assert bool(env._open_attempt_paid[0])
    assert bool(env.reach_once[0])

    env._last_door_openness = torch.tensor([0.0])
    env._start_next_open_attempt(torch.tensor([True]))
    assert not bool(env._open_attempt_paid[0])
    assert not bool(env.reach_once[0])
    assert bool(env._opened_once[0])

    env._last_task_success = torch.tensor([False])
    reward, _ = env._augment_step_reward(
        torch.zeros(1),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.bool),
    )
    assert reward.item() == pytest.approx(40.0)


def test_success_crossing_is_once_and_strict():
    openness = torch.tensor([0.2, 0.21, 0.9])
    already = torch.tensor([False, False, True])
    crossed = success_crossing(openness, already, threshold=0.2)
    assert not bool(crossed[0])
    assert bool(crossed[1])
    assert not bool(crossed[2])


def test_fridge_augment_adds_bonus_and_terminates_once():
    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._success_bonus = 1.0
    env._door_progress_weight = 0.3
    env._door_progress_cap = 0.4
    env._openness_threshold = 0.2
    env._door_openness_prev = None
    env._door_progress_paid = None
    env._last_door_openness = torch.tensor([0.1, 0.25])
    env.success_once = torch.tensor([False, False])
    env._last_task_success = torch.tensor([False, False])
    reward = torch.zeros(2)
    terminations = torch.zeros(2, dtype=torch.bool)
    was_frozen = torch.zeros(2, dtype=torch.bool)
    reward, terminations = env._augment_step_reward(reward, terminations, was_frozen)
    assert reward[0].item() == pytest.approx(0.03)
    assert reward[1].item() == pytest.approx(1.075)
    assert not bool(terminations[0])
    assert bool(terminations[1])
    assert bool(env._last_task_success[1])

    env.success_once = torch.tensor([False, True])
    env._last_door_openness = torch.tensor([0.0, 0.0])
    reward = torch.zeros(2)
    terminations = torch.tensor([False, True])
    was_frozen = torch.tensor([False, True])
    reward, terminations = env._augment_step_reward(reward, terminations, was_frozen)
    assert reward[0].item() == pytest.approx(0.0)
    assert reward[1].item() == pytest.approx(0.0)
    assert bool(terminations[1])


def test_post_success_gap_is_not_part_of_the_new_episode():
    """Hold steps after a success reset pay nothing and do not reopen latches.

    The next policy chunk clears the attempt, the same way a fresh reset
    does, and keeps the horizon record that the door already opened.
    """
    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._success_bonus = 1.0
    env._door_progress_weight = 0.3
    env._door_progress_cap = 0.4
    env._door_openness_weight = 0.5
    env._reach_bonus = 0.2
    env._handle_approach_weight = 0.0
    env._ee_bar_distance_weight = 0.0
    env._openness_threshold = 0.2
    env._repeat_open_after_success = True
    env._door_openness_prev = torch.tensor([0.0])
    env._door_progress_paid = torch.tensor([0.0])
    env._door_openness_reward_sum = torch.zeros(1)
    env._last_door_openness = torch.tensor([0.5])
    env.success_once = torch.tensor([False])
    env.reach_once = torch.tensor([False])
    env._open_attempt_paid = torch.tensor([False])
    env._opened_once = torch.tensor([True])
    env._last_task_success = torch.tensor([False])
    env._reward_pause = torch.tensor([True])
    reward, terminations = env._augment_step_reward(
        torch.tensor([-0.00025]),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    reward = zero_paused_reward(reward, env._reward_pause)
    assert reward.item() == pytest.approx(0.0)
    assert not bool(terminations[0])
    assert not bool(env.reach_once[0])
    assert env._door_openness_reward_sum.item() == pytest.approx(0.0)
    assert not bool(env._open_attempt_paid[0])

    env.success_once = torch.tensor([True])
    env.reach_once = torch.tensor([True])
    env._open_attempt_paid = torch.tensor([True])
    env._door_openness_prev = torch.tensor([0.4])
    env._door_progress_paid = torch.tensor([0.2])
    env._handle_dist_prev = torch.tensor([0.1])
    env._handle_approach_paid = torch.tensor([0.05])
    env._post_success_gap = torch.tensor([True])
    ids = torch.nonzero(env._post_success_gap.reshape(-1), as_tuple=False).reshape(-1)
    env._reset_attempt_latches(ids)
    assert not bool(env.success_once[0])
    assert not bool(env.reach_once[0])
    assert not bool(env._open_attempt_paid[0])
    assert env._door_openness_prev.item() == pytest.approx(0.0)
    assert env._door_progress_paid.item() == pytest.approx(0.0)
    assert env._handle_approach_paid.item() == pytest.approx(0.0)
    assert bool(env._opened_once[0])


def test_handle_contact_is_zero_and_grows_as_the_tip_leaves():
    """On the bar the distance is 0. Past the surface it is the extra gap."""
    # Axis along z. 1 cm is inside the bar, 2.3 cm is the surface, 10.3 cm
    # is 8 cm outside it. The same gap holds above and below the rod.
    reach = torch.tensor(
        [
            [0.01, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.023, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.103, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.103, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        ]
    )
    distance, valid = handle_reach_distance(reach)
    assert bool(valid.all())
    assert distance[0].item() == pytest.approx(0.0)
    assert distance[1].item() == pytest.approx(0.0)
    assert distance[2].item() == pytest.approx(0.08)
    assert distance[3].item() == pytest.approx(0.08)


def test_vertical_handle_distance_ignores_height_along_the_bar():
    """A grasp beside the bar matches the midpoint. Past the end it does not."""
    aabb_min = torch.tensor([0.0, 0.0, 0.2])
    aabb_max = torch.tensor([0.02, 0.02, 1.2])
    start, end = segment_from_aabb(aabb_min, aabb_max)
    assert start.tolist() == pytest.approx([0.01, 0.01, 0.2])
    assert end.tolist() == pytest.approx([0.01, 0.01, 1.2])
    points = torch.tensor(
        [
            [0.31, 0.01, 0.7],
            [0.01, 0.01, 0.7],
            [0.01, 0.01, 1.5],
        ]
    )
    distance = point_segment_distance(points, start.expand(3, -1), end.expand(3, -1))
    assert distance.tolist() == pytest.approx([0.3, 0.0, 0.3])


def test_handle_approach_pays_only_for_closing_until_the_cap():
    previous = torch.tensor([0.5, 0.4, 0.5])
    current = torch.tensor([0.4, 0.5, 0.2])
    paid = torch.tensor([0.0, 0.0, 0.09])
    active = torch.tensor([True, True, True])
    reward, new_paid = capped_closing_reward(
        previous, current, paid, weight=0.2, cap=0.1, active=active
    )
    assert reward.tolist() == pytest.approx([0.02, 0.0, 0.01])
    assert new_paid.tolist() == pytest.approx([0.02, 0.0, 0.1])
    stopped = torch.tensor([False, True, False])
    reward, _ = capped_closing_reward(
        previous, current, paid, weight=0.2, cap=0.1, active=stopped
    )
    assert reward.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_ee_bar_distance_matches_isaac_position_weight_and_stops_when_frozen():
    """−0.4 × 1/15 per meter, and frozen rows after the open pay nothing."""
    distance = torch.tensor([0.3, 0.0])
    penalty = ee_bar_distance_penalty(distance)
    assert penalty.tolist() == pytest.approx([-0.4 / 15.0 * 0.3, 0.0])

    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._ee_bar_distance_weight = -0.4 / 15.0
    env._ee_bar_distance_sum = torch.zeros(2)
    env._last_handle_reach = torch.tensor(
        [
            [0.323, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.323, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    reward = env._ee_bar_distance_step(torch.tensor([True, False]))
    assert reward.tolist() == pytest.approx([-0.4 / 15.0 * 0.3, 0.0])
    assert env._ee_bar_distance_sum.tolist() == pytest.approx(reward.tolist())


def test_approach_advantage_trains_only_episodes_that_never_reached_the_door():
    """Return below the 0.2 door bonus is handle approach. A near miss stays 0."""
    rewards = torch.tensor(
        [
            [[0.05], [0.2], [1.0]],
        ]
    )
    dones = torch.zeros(2, 3, 1, dtype=torch.bool)
    dones[1, :, 0] = True
    loss_mask = torch.ones(1, 3, 1, dtype=torch.bool)
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="episode_success",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=1,
        success_return_threshold=0.8,
        success_chunk_reward_min=0.0,
        success_advantage=0.5,
        approach_advantage=0.25,
        approach_return_max=0.2,
    )
    advantages = result["advantages"]
    assert advantages[0, :, 0].tolist() == pytest.approx([0.25, 0.0, 0.5])


def test_speed_scale_prefers_shorter_successes_and_keeps_reach():
    """A faster success is reinforced more. A miss is not scaled or negated."""
    from rlinf.algorithms.advantages import compute_episode_success_advantages

    rewards = torch.tensor([1.0, 1.0, 0.05])
    loss_mask = torch.ones(4, 3, dtype=torch.bool)
    loss_mask[2:, 1] = False
    step_rewards = torch.zeros(4, 3)
    step_rewards[1] = torch.tensor([1.0, 1.0, 0.05])
    advantages, returns = compute_episode_success_advantages(
        rewards=rewards,
        loss_mask=loss_mask,
        step_rewards=step_rewards,
        success_return_threshold=0.8,
        success_advantage=0.5,
        approach_advantage=0.25,
        approach_return_max=0.2,
        speed_horizon_chunks=4,
    )
    assert returns is None
    assert advantages[1, 0].item() == pytest.approx(0.0)
    assert advantages[1, 1].item() == pytest.approx(0.25)
    assert advantages[1, 2].item() == pytest.approx(0.25)
    assert torch.all(advantages >= 0)
    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._success_bonus = 0.0
    env._door_progress_weight = 0.0
    env._door_progress_cap = 0.4
    env._reach_bonus = 0.2
    env._openness_threshold = 0.2
    env._door_openness_prev = None
    env._door_progress_paid = None
    env._last_door_openness = torch.tensor([0.06])
    env.success_once = torch.tensor([False])
    env.reach_once = torch.tensor([False])
    env._last_task_success = torch.tensor([False])
    zeros = torch.zeros(1)
    frozen = torch.zeros(1, dtype=torch.bool)
    reward, terminations = env._augment_step_reward(zeros, frozen, frozen)
    assert reward.item() == pytest.approx(0.2)
    assert not bool(terminations[0])
    env.reach_once = torch.tensor([True])
    env._last_door_openness = torch.tensor([0.08])
    reward, _ = env._augment_step_reward(zeros, frozen, frozen)
    assert reward.item() == pytest.approx(0.0)


def test_success_done_stays_in_loss_and_cuts_gae():
    """Done on the success step trains that step and drops the hold after it.

    dones layout is [bootstrap chunk, step chunks...]. The loss mask shifts by
    the leading bootstrap flag, so the success action itself stays trainable.
    """
    # 3 chunks × 2 actions. Success on chunk 1, action 1. Later chunk is hold.
    dones = torch.zeros(4, 1, 2, dtype=torch.bool)
    dones[2, 0, 1] = True
    dones[3, 0, :] = True
    loss_mask, _ = compute_loss_mask(dones)
    assert bool(loss_mask[0, 0, 0]) and bool(loss_mask[0, 0, 1])
    assert bool(loss_mask[1, 0, 0]) and bool(loss_mask[1, 0, 1])
    assert not bool(loss_mask[2, 0, 0]) and not bool(loss_mask[2, 0, 1])
    # chunk_level keeps a chunk if any of its actions are still live.
    chunk_mask = loss_mask.any(dim=-1, keepdim=True)
    assert bool(chunk_mask[1, 0, 0])
    assert not bool(chunk_mask[2, 0, 0])

    rewards = torch.tensor([[0.0], [1.0], [0.0]])
    values = torch.zeros(4, 1)
    gae_dones = torch.tensor([[False], [False], [True], [True]])
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=gae_dones,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    assert returns[1].item() == pytest.approx(1.0)
    assert advantages[2].item() == pytest.approx(0.0)
    assert advantages[0].item() == pytest.approx(0.99 * 0.95)
    assert advantages[0].item() < advantages[1].item()


def test_episode_success_advantage_reinforces_door_motion_on_success():
    """A success reinforces reach, progress, and the terminal bonus.

    The chunk before contact stays zero. Episode 1 returns 0.35, so the
    same shaping is not a negative example.
    """
    rewards = torch.tensor(
        [
            [[-0.00015], [0.0]],
            [[0.2], [0.2]],
            [[0.15], [0.15]],
            [[1.0], [0.0]],
        ]
    )
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    dones[4, :, 0] = True
    loss_mask = torch.ones(4, 2, 1, dtype=torch.bool)
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="episode_success",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=1,
        success_return_threshold=0.8,
        success_chunk_reward_min=0.0,
        success_advantage=0.5,
    )
    advantages = result["advantages"]
    assert advantages.shape == (4, 2, 1)
    assert "returns" not in result
    assert advantages[:, 0, 0].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.5])
    assert advantages[:, 1, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_episode_success_advantage_leaves_a_miss_at_zero():
    """A rollout that never moves the door is not a negative example."""
    rewards = torch.tensor(
        [
            [[0.0], [0.0]],
            [[0.0], [0.0]],
            [[1.0], [0.0]],
            [[0.0], [0.0]],
        ]
    )
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    dones[4, :, 0] = True
    loss_mask = torch.ones(4, 2, 1, dtype=torch.bool)
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="episode_success",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=2,
        success_return_threshold=0.8,
        success_chunk_reward_min=0.0,
        success_advantage=0.5,
    )
    advantages = result["advantages"]
    assert advantages[:, 0, 0].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0])
    assert advantages[:, 1, 0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_cri_boundary_rewind_skips_an_open_door_and_a_frozen_row():
    """Only a live attempt that has not yet opened the door rewinds."""
    from rlinf.envs.isaaclab.spawn_rewind import cri_boundary_rewind_mask

    cri_hit = torch.tensor([True, True, True, False])
    frozen = torch.tensor([False, True, False, False])
    success = torch.tensor([False, False, False, False])
    openness = torch.tensor([0.0, 0.0, 0.25, 0.0])
    mask = cri_boundary_rewind_mask(
        cri_hit, frozen, success, openness, openness_threshold=0.2
    )
    assert mask.tolist() == [True, False, False, False]


def test_cri_boundary_metrics_count_rewinds_and_clean_opens():
    """A rewind is counted. An open counts as clean only with zero rewinds."""
    from rlinf.envs.isaaclab.spawn_rewind import cri_boundary_episode_metrics

    metrics = cri_boundary_episode_metrics(
        success_once=torch.tensor([True, True, False, False]),
        rewind_count=torch.tensor([0.0, 2.0, 1.0, 0.0]),
    )
    assert metrics["cri_rewind_count"].tolist() == [0.0, 2.0, 1.0, 0.0]
    assert metrics["cri_rewind_once"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert metrics["success_without_cri"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_rewind_restores_the_spawn_pose_without_replacing_it():
    """The arm goes back to the reset pose. That stored pose stays put."""
    from rlinf.envs.isaaclab.spawn_rewind import capture_robot_spawn
    from rlinf.envs.isaaclab.spawn_rewind import rewind_robot_to_spawn

    class _Data:
        def __init__(self):
            self.joint_pos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            self.joint_vel = torch.zeros(2, 2)

    class _Robot:
        def __init__(self):
            self.data = _Data()
            self.written = None

        def write_joint_state_to_sim(self, position, velocity, env_ids=None):
            self.written = (position.clone(), velocity.clone(), env_ids.clone())

        def set_joint_position_target(self, position, env_ids=None):
            self.target = position.clone()

        def set_joint_velocity_target(self, velocity, env_ids=None):
            self.vel_target = velocity.clone()

    class _Sim:
        def __init__(self):
            self.renders = 0

        def forward(self):
            self.forwarded = True

        def has_rtx_sensors(self):
            return True

        def render(self):
            self.renders += 1

    class _Obs:
        def compute(self, update_history=False):
            self.update_history = update_history
            return {"policy": "rewound"}

    class _Env:
        def __init__(self):
            self.scene = {"robot": _Robot()}
            self.sim = _Sim()
            self.cfg = type("Cfg", (), {"num_rerenders_on_reset": 2})()
            self.observation_manager = _Obs()

    env = _Env()
    capture_robot_spawn(env)
    env.scene["robot"].data.joint_pos = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    obs = rewind_robot_to_spawn(env, torch.tensor([0]))
    position, velocity, ids = env.scene["robot"].written
    assert position[0].tolist() == pytest.approx([0.1, 0.2])
    assert velocity[0].tolist() == pytest.approx([0.0, 0.0])
    assert ids.tolist() == [0]
    assert obs == {"policy": "rewound"}
    assert env.sim.renders == 2
    assert env.observation_manager.update_history is False
    assert env._rlinf_spawn_q[0].tolist() == pytest.approx([0.1, 0.2])
    assert env._rlinf_spawn_q[1].tolist() == pytest.approx([0.3, 0.4])


def test_zero_bonus_still_ends_the_attempt_at_openness_0_2():
    """A time-only reward still stops when the door crosses 0.2.

    The step reward stays at the incoming penalty. The crossing is recorded
    so the next steps are held and stop paying.
    """
    env = IsaaclabOpenFridgeKitchenEnv.__new__(IsaaclabOpenFridgeKitchenEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._success_bonus = 0.0
    env._door_progress_weight = 0.0
    env._door_progress_cap = 0.4
    env._reach_bonus = 0.0
    env._handle_approach_weight = 0.0
    env._openness_threshold = 0.2
    env._last_door_openness = torch.tensor([0.1, 0.25])
    env.success_once = torch.tensor([False, False])
    env._last_task_success = torch.tensor([False, False])
    reward = torch.tensor([-0.001, -0.001])
    terminations = torch.zeros(2, dtype=torch.bool)
    was_frozen = torch.zeros(2, dtype=torch.bool)
    reward, terminations = env._augment_step_reward(reward, terminations, was_frozen)
    assert reward.tolist() == pytest.approx([-0.001, -0.001])
    assert not bool(terminations[0])
    assert bool(terminations[1])
    assert bool(env._last_task_success[1])
    assert not bool(env._last_task_success[0])


def test_min_time_advantage_prefers_the_shorter_penalty():
    """A two-step open beats a four-step timeout after centering.

    Returns are -2 and -4. The batch mean is -3, so the open is +1 on its
    live chunks and the timeout is -1. The held tail of the open is masked.
    """
    rewards = torch.tensor(
        [
            [[-1.0], [-1.0]],
            [[-1.0], [-1.0]],
            [[0.0], [-1.0]],
            [[0.0], [-1.0]],
        ]
    )
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    dones[2, 0, 0] = True
    dones[3:, 0, 0] = True
    dones[4, 1, 0] = True
    loss_mask = torch.ones(4, 2, 1, dtype=torch.bool)
    loss_mask[2:, 0, 0] = False
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="min_time",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=1,
    )
    advantages = result["advantages"]
    assert advantages[:, 0, 0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert advantages[:, 1, 0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])


def test_min_time_advantage_drops_the_penalty_after_termination():
    """Rewards after the success done are not part of the time sum."""
    rewards = torch.tensor(
        [
            [[-1.0], [-1.0]],
            [[-1.0], [-1.0]],
            [[-1.0], [-1.0]],
            [[-1.0], [-1.0]],
        ]
    )
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    dones[2, 0, 0] = True
    dones[3:, 0, 0] = True
    dones[4, 1, 0] = True
    loss_mask = torch.ones(4, 2, 1, dtype=torch.bool)
    loss_mask[2:, 0, 0] = False
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="min_time",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=1,
    )
    advantages = result["advantages"]
    assert advantages[:, 0, 0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert advantages[:, 1, 0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])


def test_min_time_advantage_scales_by_horizon_and_live_chunks():
    """Horizon 4 and live counts 2 and 4 shrink +1 and -1 equally per episode."""
    rewards = torch.tensor(
        [
            [[-1.0], [-1.0]],
            [[-1.0], [-1.0]],
            [[0.0], [-1.0]],
            [[0.0], [-1.0]],
        ]
    )
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    dones[2, 0, 0] = True
    dones[3:, 0, 0] = True
    dones[4, 1, 0] = True
    loss_mask = torch.ones(4, 2, 1, dtype=torch.bool)
    loss_mask[2:, 0, 0] = False
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="min_time",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        group_size=1,
        advantage_horizon_steps=4,
    )
    advantages = result["advantages"]
    assert advantages[:, 0, 0].tolist() == pytest.approx([0.125, 0.125, 0.0, 0.0])
    assert advantages[:, 1, 0].tolist() == pytest.approx(
        [-0.0625, -0.0625, -0.0625, -0.0625]
    )


def test_trust_ratio_keeps_fast_success_gradient_inside_band():
    """A moderate ratio still trains; a collapsed ratio does not."""
    old_logprobs = torch.zeros(1, dtype=torch.float32)
    advantages = torch.tensor([1.0], dtype=torch.float32)
    # log_ratio -0.5 is outside the ±0.2 clip and inside ±2 nats.
    near = torch.tensor([-0.5], dtype=torch.float32, requires_grad=True)
    near_loss, _ = compute_ppo_actor_loss(
        logprobs=near,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        advantage_trust_ratio=True,
        trust_log_ratio_min=-2.0,
        trust_log_ratio_max=2.0,
    )
    near_loss.backward()
    assert near.grad is not None
    assert near.grad.item() < 0

    slow = torch.tensor([-0.5], dtype=torch.float32, requires_grad=True)
    slow_loss, _ = compute_ppo_actor_loss(
        logprobs=slow,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.tensor([-1.0], dtype=torch.float32),
        advantage_trust_ratio=True,
        trust_log_ratio_min=-2.0,
        trust_log_ratio_max=2.0,
    )
    slow_loss.backward()
    assert slow.grad is not None
    assert slow.grad.item() > 0

    collapsed = torch.tensor([-80.0], dtype=torch.float32, requires_grad=True)
    collapsed_loss, metrics = compute_ppo_actor_loss(
        logprobs=collapsed,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        advantage_trust_ratio=True,
        trust_log_ratio_min=-2.0,
        trust_log_ratio_max=2.0,
    )
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert abs(collapsed.grad.item()) < 1e-6
    assert metrics["actor/ratio"].item() == pytest.approx(0.0, abs=1e-6)

    plain = torch.tensor([-0.5], dtype=torch.float32, requires_grad=True)
    plain_loss, _ = compute_ppo_actor_loss(
        logprobs=plain,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        advantage_trust_ratio=False,
    )
    plain_loss.backward()
    assert plain.grad is not None
    # Advantage > 0 and ratio < 0.8 still backprops through -A * ratio.
    # The trust path uses the clip edge, so its step is larger.
    assert plain.grad.item() < 0
    assert abs(plain.grad.item()) < abs(near.grad.item())
