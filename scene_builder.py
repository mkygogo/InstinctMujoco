"""MuJoCo scene construction for the G1 parkour runner.

Builds a complete scene from a robot MJCF: skybox, ground plane,
lights, solver settings, depth camera, terrain, and actuators.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.spec import create_position_actuator
from mjlab.utils.spec_config import MaterialCfg, TextureCfg
from parkour_onnx_policy import G1_JOINT_ORDER

from robot_config import (
    CAMERA_OFFSET_POS,
    CAMERA_OFFSET_QUAT_WXYZ,
    JOINT_ARMATURE,
    JOINT_KD,
    JOINT_KP,
    JOINT_TORQUE_LIMIT,
    TORSO_BODY_NAME,
)

# ── Physics ────────────────────────────────────────────────────────────────
SIM_DT = 0.005
CONTROL_DT = 0.02


# ── Depth camera/image constants ──────────────────────────────────────────
DEPTH_RANGE = (0.0, 2.5)
DEPTH_CROP = (18, 0, 16, 16)
DEPTH_OUTPUT_SHAPE = (18, 32)
DEPTH_HISTORY_SKIP = 5
DEPTH_HISTORY_LEN = 37


def _add_skybox(spec: mujoco.MjSpec) -> None:
    TextureCfg(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=(0.4, 0.6, 0.8),
        rgb2=(0.0, 0.0, 0.0),
        width=512,
        height=3072,
    ).edit_spec(spec)


def _add_ground(spec: mujoco.MjSpec) -> None:
    """Add a visible ground plane with a checker-grid material."""
    TextureCfg(
        name="groundplane",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        markrgb=(0.8, 0.8, 0.8),
        width=300,
        height=300,
    ).edit_spec(spec)
    MaterialCfg(
        name="groundplane",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(5.0, 5.0),
        reflectance=0.2,
        texture="groundplane",
    ).edit_spec(spec)

    ground = spec.worldbody.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size[:] = (0.0, 0.0, 0.05)  # infinite plane
    ground.pos[:] = (0.0, 0.0, 0.0)
    ground.material = "groundplane"
    ground.group = 0  # override default group=3 (hidden in viewer)
    ground.condim = 3
    ground.contype = 1
    ground.conaffinity = 1
    ground.friction[:] = (1.0, 0.5, 0.5)
    ground.priority = 1


def _add_lights(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.ambient[:] = (0.15, 0.15, 0.15)
    spec.visual.headlight.diffuse[:] = (0.4, 0.4, 0.4)
    spec.visual.headlight.specular[:] = (0.1, 0.1, 0.1)

    light = spec.worldbody.add_light()
    light.name = "sun"
    light.pos[:] = (0.0, 0.0, 5.0)
    light.dir[:] = (0.0, 0.2, -1.0)
    light.castshadow = True
    light.diffuse[:] = (0.6, 0.6, 0.6)
    light.specular[:] = (0.2, 0.2, 0.2)


def _add_solver(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.noslip_iterations = 0
    spec.option.ccd_iterations = 128
    spec.nconmax = 128
    spec.njmax = 700


def _add_depth_camera(spec: mujoco.MjSpec) -> None:
    torso_body = spec.body(TORSO_BODY_NAME)
    cam = torso_body.add_camera()
    cam.name = "head_depth"
    cam.pos[:] = CAMERA_OFFSET_POS
    cam.quat[:] = CAMERA_OFFSET_QUAT_WXYZ
    cam.fovy = 58.29


def _add_actuators(spec: mujoco.MjSpec) -> None:
    for idx, joint_name in enumerate(G1_JOINT_ORDER):
        create_position_actuator(
            spec,
            joint_name,
            stiffness=float(JOINT_KP[idx]),
            damping=float(JOINT_KD[idx]),
            effort_limit=float(JOINT_TORQUE_LIMIT[idx]),
            armature=float(JOINT_ARMATURE[idx]),
            transmission_type=TransmissionType.JOINT,
        )


def _add_stairs(
    spec: mujoco.MjSpec,
    num_steps: int = 200,
    step_width: float = 8.0,
    step_depth: float = 0.3,
    step_height: float = 0.12,
    start_x: float = 1.5,
) -> None:
    """Add a staircase made of box geoms to the scene."""
    TextureCfg(
        name="stair_tex",
        type="2d",
        builtin="flat",
        rgb1=(0.55, 0.45, 0.35),
        rgb2=(0.45, 0.35, 0.25),
        width=64,
        height=64,
    ).edit_spec(spec)
    MaterialCfg(
        name="stair_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(2.0, 2.0),
        reflectance=0.1,
        texture="stair_tex",
    ).edit_spec(spec)
    for i in range(num_steps):
        cx = start_x + i * step_depth + step_depth / 2.0
        cz = (i + 1) * step_height / 2.0
        half_h = (i + 1) * step_height / 2.0
        step = spec.worldbody.add_geom()
        step.name = f"stair_{i}"
        step.type = mujoco.mjtGeom.mjGEOM_BOX
        step.size[:] = (step_depth / 2.0, step_width / 2.0, half_h)
        step.pos[:] = (cx, 0.0, cz)
        step.material = "stair_mat"
        step.group = 0  # override default group=3 (hidden in viewer)
        step.condim = 3
        step.contype = 1
        step.conaffinity = 1
        step.friction[:] = (1.0, 0.5, 0.5)
        step.priority = 1


def build_scene_model(robot_xml_path: Path, terrain: str = "flat") -> mujoco.MjModel:
    """Build a complete MuJoCo model from the robot MJCF with scene elements."""
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    _add_skybox(spec)
    _add_ground(spec)
    _add_lights(spec)
    _add_solver(spec)
    _add_depth_camera(spec)
    _add_actuators(spec)

    if terrain == "stairs":
        _add_stairs(spec)

    return spec.compile()
