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

"""Isaac Lab ManagerBasedRLEnvCfg for DROID kitchen fridge open-door.

Ports Arena ``kitchen_bench_lightwheel_open_fridge`` (DROID abs-joint-pos +
Lightwheel one-wall coastal kitchen + ``OpenDoorTask``) into a standard Isaac
Lab gym task. Asset paths default to ``.assets/isaaclab_arena`` (populate with
``examples/embodiment/scripts/download_isaaclab_arena_assets.sh``; optional
local ``ARENA_ASSETS_ROOT``). No runtime dependency on ``isaaclab_arena``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .materials import prepare_kitchen_fridge_materials
from .placement import arena_open_fridge_fridge_root_pos, arena_open_fridge_robot_pos

# Repo root: rlinf/envs/isaaclab/tasks/open_fridge_kitchen/env_cfg.py → parents[5]
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ARENA_ASSETS = str(_REPO_ROOT / ".assets" / "isaaclab_arena")

# Arena DROID control: sim.dt (1/120 s) x decimation (8) = 15 Hz.
_DROID_CONTROL_HZ = 15.0
_ISAACLAB_DECIMATION = 8
_ISAACLAB_SIM_DT = 1.0 / 120.0


def _resolve_arena_assets_root() -> str:
    """Return a local Arena assets directory.

    Remote http(s) ``ARENA_ASSETS_ROOT`` values are ignored so the vendored
    Lightwheel kitchen USD is used.
    """
    env_root = (os.environ.get("ARENA_ASSETS_ROOT") or "").strip()
    if not env_root or env_root.startswith(("http://", "https://")):
        return _DEFAULT_ARENA_ASSETS
    if Path(env_root).is_dir():
        return env_root
    return _DEFAULT_ARENA_ASSETS


_ARENA = _resolve_arena_assets_root()
_KITCHEN_USD = prepare_kitchen_fridge_materials(
    f"{_ARENA}/background_library/lightwheel_kitchen_one_wall_coastal/scene.usd"
)
_DROID_ROBOT_USD = f"{_ARENA}/robot_library/droid/franka_robotiq_2f_85_flattened.usd"

# Arena kitchen_bench_lightwheel_open_fridge: ObjectPlacer yaws the DROID
# AABB (+90°) then solves NextTo fridge side=negative_y, distance_m=0.1.
# Un-yawed NextTo (~4.80, -1.44) sits 20 cm off the fridge center.
_STAND_HEIGHT_M = 0.8
_ROBOT_POS = arena_open_fridge_robot_pos()
_FRIDGE_POS = arena_open_fridge_fridge_root_pos()
# Isaac Lab InitialStateCfg.rot is (w, x, y, z). Converted DROID cams look
# [+X, −Y] in the parent; +90 deg yaw turns that to [+X, +Y] (fridge door).
_ROBOT_ROT = (0.70710678, 0.0, 0.0, 0.70710678)
_FRIDGE_DOOR_JOINT = "fridge_door_joint"
_OPENNESS_THRESHOLD = 0.2
_RESET_OPENNESS = 0.0


@configclass
class OpenFridgeKitchenSceneCfg(InteractiveSceneCfg):
    """DROID on 0.8 m stand + Lightwheel one-wall coastal kitchen + fridge."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DROID_ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * math.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * math.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * math.pi,
                "panda_joint7": 0.0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=None,
                damping=None,
                velocity_limit=5.0,
            ),
        },
    )

    # Lightwheel kitchen (fridge / counters / floor live in this USD).
    kitchen: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(usd_path=_KITCHEN_USD),
    )

    # Nested fridge articulation already present in the kitchen USD.
    fridge: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen/fridge_main_group",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
            joint_pos={_FRIDGE_DOOR_JOINT: 0.0},
        ),
        # Arena ObjectReference articulations use empty actuators so the USD
        # drive stays in charge. A PD hold (stiffness/damping) fights the
        # policy and keeps the door closed.
        actuators={},
    )

    # Arena ``DroidCameraCfg`` mounts. Isaac Lab 2.x OffsetCfg.rot is wxyz;
    # Arena 3.0 stores xyzw, so convert (x, y, z, w) → (w, x, y, z).
    # Tiled 224 matches pick-place PPO (64 envs/GPU). Wrapper may raise
    # height/width; 1280×720 CameraCfg exhausted RTX descriptors at 64/GPU.
    external_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/external_camera",
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66),
            rot=(-0.393, -0.195, 0.399, 0.805),
            convention="opengl",
        ),
    )

    external_camera_2: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/external_camera_2",
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66),
            rot=(0.805, 0.399, -0.195, -0.393),
            convention="opengl",
        ),
    )

    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_camera",
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )

    eval_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/eval_camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.2908,
        ),
        # Face the fridge (x≈4.60), not the sink/coffee run at x≈2.75.
        offset=CameraCfg.OffsetCfg(
            pos=(4.60, -5.5, 1.5),
            rot=(0.8660, 0.3536, 0.1768, 0.3030),
            convention="opengl",
        ),
    )

    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=GroundPlaneCfg(visible=False),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1500.0,
        ),
    )


@configclass
class ActionsCfg:
    """Arena ``DroidAbsoluteJointPositionActionsCfg``."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    gripper_action: mdp.BinaryJointPositionZeroToOneActionCfg = (
        mdp.BinaryJointPositionZeroToOneActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": math.pi / 4},
        )
    )


@configclass
class ObservationsCfg:
    """Policy obs aligned with Arena DROID / OpenPI wire format."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.arm_joint_pos)
        joint_vel = ObsTerm(func=mdp.arm_joint_vel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        external_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        external_camera_2 = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_camera_2"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        wrist_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        eval_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eval_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        door_openness = ObsTerm(func=mdp.fridge_door_openness)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset robot / kitchen and close the fridge door."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_fridge_door = EventTerm(
        func=mdp.reset_fridge_door,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fridge"),
            "joint_name": _FRIDGE_DOOR_JOINT,
            "reset_openness": _RESET_OPENNESS,
        },
    )


@configclass
class RewardsCfg:
    """Sparse success reward (RLinf-style)."""

    success = RewTerm(
        func=mdp.is_terminated_term,
        weight=1.0,
        params={"term_keys": "success"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.fridge_door_is_open,
        params={
            "asset_cfg": SceneEntityCfg("fridge"),
            "joint_name": _FRIDGE_DOOR_JOINT,
            "openness_threshold": _OPENNESS_THRESHOLD,
        },
    )


@configclass
class OpenFridgeKitchenEnvCfg(ManagerBasedRLEnvCfg):
    """Gym-registered env cfg for ``Isaac-OpenFridge-Kitchen-Droid-AbsJointPos-v0``."""

    scene: OpenFridgeKitchenSceneCfg = OpenFridgeKitchenSceneCfg(
        num_envs=1, env_spacing=10.0, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = _ISAACLAB_DECIMATION
        self.episode_length_s = 10.0
        self.sim.dt = _ISAACLAB_SIM_DT
        self.sim.render_interval = self.decimation
        # Look at the fridge, not the default one-wall kitchen midpoint.
        self.viewer.eye = (4.60, -5.5, 1.5)
        self.viewer.lookat = (4.60, -1.4, 0.9)
