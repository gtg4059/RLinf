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

"""Isaac Lab ManagerBasedRLEnvCfg for DROID cube→bowl pick-and-place.

Ports a *single* (non-varied) layout of Isaac Lab Arena's
``pick_and_place_maple_table`` environment (DROID abs-joint-pos + maple Robolab
table + rubiks cube + YCB bowl) into a standard Isaac Lab gym task. Asset paths
default to the vendored tree at ``.assets/isaaclab_arena`` (populate with
``examples/embodiment/scripts/download_isaaclab_arena_assets.sh``; optional
local ``ARENA_ASSETS_ROOT``). Remote http(s) roots are ignored. No runtime
dependency on ``isaaclab_arena``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

# Repo root: rlinf/envs/isaaclab/tasks/pick_place_cube_plate/env_cfg.py → parents[5]
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ARENA_ASSETS = str(_REPO_ROOT / ".assets" / "isaaclab_arena")


def _resolve_arena_assets_root() -> str:
    """Return a local Arena assets directory.

    Remote http(s) ``ARENA_ASSETS_ROOT`` values are ignored: custom wrappers such
    as ``table_maple_arena.usda`` only exist under the vendored ``.assets`` tree
    (populated by ``download_isaaclab_arena_assets.sh``).
    """
    env_root = (os.environ.get("ARENA_ASSETS_ROOT") or "").strip()
    if not env_root or env_root.startswith(("http://", "https://")):
        return _DEFAULT_ARENA_ASSETS
    if Path(env_root).is_dir():
        return env_root
    return _DEFAULT_ARENA_ASSETS


_ARENA = _resolve_arena_assets_root()
_ROBOLAB = f"{_ARENA}/object_library/srl_robolab_assets"

# Arena ``pick_and_place_maple_table`` / docs ``default_srl_pnp`` stock layout
# (``maple_table_robolab`` = scenes/maple_table_background.usda + DroidSceneCfg):
#   maple work table → fixtures/table_maple_arena.usda
#   grey robot pedestal → fixtures/franka_table.usd
#   thin mount under robot → fixtures/stand_instanceable.usd
#   droid robot → robot_library/droid/franka_robotiq_2f_85_flattened.usd
#   rubiks_cube_hot3d_robolab → objects/hot3d/rubiks_cube.usd
#   bowl_ycb_robolab → objects/ycb/bowl.usd
#   HDR → backgrounds/default/home_office (tiled floor; matches docs hero)
_DROID_ROBOT_USD = f"{_ARENA}/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
_DROID_STAND_USD = f"{_ROBOLAB}/fixtures/stand_instanceable.usd"
_FRANKA_TABLE_USD = f"{_ROBOLAB}/fixtures/franka_table.usd"
_MAPLE_TABLE_USD = f"{_ROBOLAB}/fixtures/table_maple_arena.usda"
_CUBE_USD = f"{_ROBOLAB}/objects/hot3d/rubiks_cube.usd"
_BOWL_USD = f"{_ROBOLAB}/objects/ycb/bowl.usd"
_HDR_TEXTURE = f"{_ROBOLAB}/backgrounds/default/home_office.exr"

# Grey pedestal / maple poses from Arena ``maple_table_background.usda``.
# Arena ``DroidSceneCfg`` stores robot rot=(0,0,0,1) (Z-180 wxyz), but that
# points the arm at the grey pedestal (−X). For maple_table (+X workspace) we
# keep identity root so the arm/cameras face the desk; camera *offsets* still
# match Arena ``DroidCameraCfg`` exactly (robot-relative).
_MAPLE_TABLE_POS = (0.19859090447425842, 0.02206302247941494, 0.003000684082508087)
_FRANKA_TABLE_POS = (-0.087, 0.0, 0.0)
_FRANKA_TABLE_ROT = (0.0, 0.0, 0.0, 1.0)  # USD background Z-180
_ROBOT_POS = (0.0, 0.0, 0.0)
_ROBOT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity → face maple (+X)
_STAND_POS = (-0.05, 0.0, 0.0)  # Arena stand default translation
_STAND_ROT = (1.0, 0.0, 0.0, 0.0)

# Object poses from Arena ``rubiks_cube_bowl.usda`` (on the maple top).
_CUBE_POS = (0.4306747317314148, -0.09746426343917847, 0.03409578651189804)
_BOWL_POS = (0.44257622957229614, 0.1265944093465805, 0.030340319499373436)


@configclass
class PickPlaceCubePlateSceneCfg(InteractiveSceneCfg):
    """DROID robot + Arena maple_table_robolab + cube + YCB bowl (single layout)."""

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
                velocity_limit=1.0,
            ),
        },
    )

    # Thin mount under the robot (Arena ``DroidSceneCfg.stand``).
    stand: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Stand",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_STAND_POS, rot=_STAND_ROT
        ),
        spawn=UsdFileCfg(
            usd_path=_DROID_STAND_USD,
            scale=(1.2, 1.2, 1.7),
            activate_contact_sensors=False,
        ),
    )

    # Grey perforated pedestal from ``maple_table_background`` (separate from
    # the maple desk so the docs ``default_srl_pnp`` gap is preserved).
    franka_table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/franka_table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_FRANKA_TABLE_POS, rot=_FRANKA_TABLE_ROT
        ),
        spawn=UsdFileCfg(usd_path=_FRANKA_TABLE_USD),
    )

    # Arena maple work table. Materials resolve from ``.assets/.../Materials``.
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MapleTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_MAPLE_TABLE_POS),
        spawn=UsdFileCfg(usd_path=_MAPLE_TABLE_USD),
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CUBE_POS, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=_CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Arena default destination ``bowl_ycb_robolab``.
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_BOWL_POS, rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=_BOWL_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Exact Arena ``DroidCameraCfg`` (release/0.2.1), robot-relative mounts.
    # DROID ``exterior_image_1_left`` (OpenPI base / ``main_images``).
    # Resolution overridden in RLinf wrapper (224 for RL).
    external_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/external_camera",
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

    # DROID ``exterior_image_2_left`` (``extra_view_images`` / logging).
    external_camera_2: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/external_camera_2",
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

    # DROID ``wrist_image_left`` (``wrist_images``); Arena gripper mount.
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

    # Per-env (non-tiled) front camera for eval videos. TiledCameraCfg would bake
    # every parallel env into one mosaic frame even when max_envs_in_video=1.
    # See: https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/quickstart/first_arena_env.html
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
        # Top-down 3/4 view matching Arena docs ``default_srl_pnp.png``.
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, -1.3, 1.9),
            rot=(0.9514, 0.3025, 0.0174, 0.0547),
            convention="opengl",
        ),
    )

    # Arena maple_table*_background GroundPlane is collision-only (visibility
    # invisible) so the HDR dome shows through instead of Isaac's grid floor.
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.697)),
        spawn=GroundPlaneCfg(visible=False),
    )

    # Arena pick_and_place_maple_table default: DomeLight intensity=500. HDR from
    # Robolab ``brown_photostudio`` (``--hdr`` in Arena); must be visible in the
    # primary ray so eval cameras see the photostudio panorama.
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=500.0,
            texture_file=_HDR_TEXTURE,
            visible_in_primary_ray=True,
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
        # DROID exterior_image_1_left
        external_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        # DROID exterior_image_2_left
        external_camera_2 = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_camera_2"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        # DROID wrist_image_left
        wrist_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        # Optional third-person viewer (not a DROID dataset camera).
        eval_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eval_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Narrow reset noise only (no Arena layout variations)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube = EventTerm(
        func=mdp.reset_object_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("cube"),
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
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )
    success = DoneTerm(
        func=mdp.object_placed_on_destination,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "destination_cfg": SceneEntityCfg("bowl"),
        },
    )


@configclass
class PickPlaceCubePlateEnvCfg(ManagerBasedRLEnvCfg):
    """Gym-registered env cfg for ``Isaac-PickPlace-Cube-Plate-Droid-AbsJointPos-v0``."""

    scene: PickPlaceCubePlateSceneCfg = PickPlaceCubePlateSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        # Arena pick_and_place_maple_table env_cfg_callback viewer.
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.2, 0.0, 0.0)
