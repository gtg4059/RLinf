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

import numpy as np
import pytest
import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions_for_isaaclab
from rlinf.envs.isaaclab.tasks.pick_place_cube_plate import wrap_droid_obs


def test_wrap_droid_obs_joint_and_gripper_keys():
    obs = {
        "policy": {
            "table_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.ones((224, 224, 3), dtype=np.uint8),
            "arm_joint_pos": np.arange(7, dtype=np.float32),
            "gripper_pos": np.array([0.3], dtype=np.float32),
        }
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=1,
        device=torch.device("cpu"),
        task_description="Pick up the cube and place it on the plate.",
    )
    assert wrapped["states"].shape == (1, 8)
    assert torch.allclose(wrapped["states"][0, :7], torch.arange(7, dtype=torch.float32))
    assert wrapped["main_images"].shape[-1] == 3
    assert wrapped["wrist_images"].shape[-1] == 3
    assert wrapped["task_descriptions"] == [
        "Pick up the cube and place it on the plate."
    ]


def test_wrap_droid_obs_arena_policy_keys():
    """DROID three cameras: exterior_1, exterior_2, wrist."""
    obs = {
        "policy": {
            "external_camera": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "external_camera_2": np.full((2, 224, 224, 3), 2, dtype=np.uint8),
            "wrist_camera": np.ones((2, 224, 224, 3), dtype=np.uint8),
            "joint_pos": np.stack(
                [np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32) + 1.0]
            ),
            "joint_vel": np.ones((2, 7), dtype=np.float32) * 0.1,
            "gripper_pos": np.array([[0.2], [0.8]], dtype=np.float32),
        }
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=2,
        device=torch.device("cpu"),
        task_description="pick",
    )
    assert wrapped["states"].shape == (2, 8)
    assert wrapped["main_images"].shape == (2, 224, 224, 3)
    assert wrapped["exterior2_images"].shape == (2, 224, 224, 3)
    assert wrapped["extra_view_images"].shape == (2, 1, 224, 224, 3)
    assert wrapped["wrist_images"].shape == (2, 224, 224, 3)
    assert wrapped["joint_vel"].shape == (2, 7)
    assert wrapped["states"][1, -1].item() == pytest.approx(0.8)


def test_wrap_droid_obs_camera_obs_group_rgb_suffix():
    obs = {
        "policy": {
            "joint_pos": np.zeros(7, dtype=np.float32),
            "gripper_pos": np.array([0.0], dtype=np.float32),
        },
        "camera_obs": {
            "external_camera_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_camera_rgb": np.ones((224, 224, 3), dtype=np.uint8),
        },
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=1,
        device=torch.device("cpu"),
        task_description="pick",
    )
    assert wrapped["main_images"].shape[-1] == 3
    assert wrapped["wrist_images"] is not None


def test_wrap_droid_obs_samples_exterior2_into_main():
    """OpenPI DROID RLDS: sampled exterior_2 becomes the policy base image."""
    obs = {
        "policy": {
            "external_camera": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "external_camera_2": np.full((2, 224, 224, 3), 7, dtype=np.uint8),
            "wrist_camera": np.ones((2, 224, 224, 3), dtype=np.uint8),
            "joint_pos": np.zeros((2, 7), dtype=np.float32),
            "gripper_pos": np.zeros((2, 1), dtype=np.float32),
        }
    }
    use_exterior2 = torch.tensor([False, True])
    wrapped = wrap_droid_obs(
        obs,
        num_envs=2,
        device=torch.device("cpu"),
        task_description="pick",
        use_exterior2=use_exterior2,
    )
    assert int(wrapped["main_images"][0].max()) == 0
    assert int(wrapped["main_images"][1].max()) == 7
    # Physical exterior_2 stays in the logging slot.
    assert int(wrapped["exterior2_images"][0].max()) == 7
    assert int(wrapped["exterior2_images"][1].max()) == 7


def test_wrap_droid_obs_sample_mask_ignored_without_exterior2():
    obs = {
        "policy": {
            "external_camera": np.zeros((224, 224, 3), dtype=np.uint8),
            "joint_pos": np.zeros(7, dtype=np.float32),
            "gripper_pos": np.array([0.0], dtype=np.float32),
        }
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=1,
        device=torch.device("cpu"),
        task_description="pick",
        use_exterior2=torch.tensor([True]),
    )
    assert int(wrapped["main_images"].max()) == 0
    assert wrapped["exterior2_images"] is None


def test_wrap_droid_obs_concatenated_state():
    obs = {
        "policy": {
            "exterior_image_1_left": np.zeros((3, 224, 224), dtype=np.uint8),
            "state": np.linspace(0, 1, 8, dtype=np.float32),
        }
    }
    wrapped = wrap_droid_obs(
        obs,
        num_envs=1,
        device=torch.device("cpu"),
        task_description="pick",
    )
    assert wrapped["states"].shape == (1, 8)
    assert wrapped["wrist_images"] is None


def test_env_output_to_dict_keeps_cri():
    """Rollout/actor receive obs via EnvOutput.to_dict(); CRI must survive."""
    cri = torch.linspace(0, 1, 9, dtype=torch.float32).unsqueeze(0)
    env_output = EnvOutput(
        obs={
            "states": torch.zeros((1, 8), dtype=torch.float32),
            "main_images": torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
            "wrist_images": None,
            "extra_view_images": None,
            "task_descriptions": ["pick"],
            "cri": cri,
        },
        dones=torch.zeros((1, 1), dtype=torch.bool),
    )
    payload = env_output.to_dict()
    assert "cri" in payload["obs"]
    assert torch.allclose(payload["obs"]["cri"], cri)


def test_prepare_actions_droid_abs_joint_pos_binarizes_gripper():
    actions = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]], dtype=torch.float32)
    out = prepare_actions_for_isaaclab(
        actions,
        model_type="openpi",
        env_cfg={"action_space": "droid_abs_joint_pos"},
    )
    assert out.shape[-1] == 8
    assert float(out[..., -1]) == 1.0

    actions_low = actions.clone()
    actions_low[..., -1] = 0.2
    out_low = prepare_actions_for_isaaclab(
        actions_low,
        model_type="openpi",
        env_cfg={"init_params": {"action_space": "droid_abs_joint_pos"}},
    )
    assert float(out_low[..., -1]) == 0.0
