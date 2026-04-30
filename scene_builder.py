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

    # Visual infinite plane (group 0) – always rendered, no edge artifacts.
    vis = spec.worldbody.add_geom()
    vis.name = "ground_visual"
    vis.type = mujoco.mjtGeom.mjGEOM_PLANE
    vis.size[:] = (0.0, 0.0, 0.05)  # infinite
    vis.pos[:] = (0.0, 0.0, 0.0)
    vis.material = "groundplane"
    vis.group = 0          # rendered (groups 0-2 shown by default)
    vis.contype = 0        # no physics collision
    vis.conaffinity = 0

    # Physics + raycast finite box (group 3 – NOT rendered by default).
    # mj_multiRay (BVH) fails on infinite planes in MuJoCo 3.5.x,
    # so we use a large hidden box for raycasting + collision.
    ground = spec.worldbody.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_BOX
    ground.size[:] = (50.0, 50.0, 0.025)
    ground.pos[:] = (0.0, 0.0, -0.025)
    ground.group = 3       # hidden from viewer; raycast via geomgroup[3]=1
    ground.condim = 3
    ground.contype = 1
    ground.conaffinity = 1
    ground.friction[:] = (1.0, 0.5, 0.5)
    ground.priority = 1


def _add_lights(spec: mujoco.MjSpec) -> None:
    # Reduce headlight (camera-aligned light hurts stereo matching by flattening texture)
    spec.visual.headlight.ambient[:] = (0.15, 0.15, 0.15)
    spec.visual.headlight.diffuse[:] = (0.2, 0.2, 0.2)
    spec.visual.headlight.specular[:] = (0.1, 0.1, 0.1)

    # Primary sun – high angle from front-right, casts shadows
    sun = spec.worldbody.add_light()
    sun.name = "sun"
    sun.pos[:] = (5.0, -3.0, 10.0)
    sun.dir[:] = (-0.4, 0.2, -1.0)
    sun.castshadow = True
    sun.diffuse[:] = (0.9, 0.88, 0.82)
    sun.specular[:] = (0.5, 0.5, 0.5)
    sun.ambient[:] = (0.15, 0.15, 0.15)

    # Fill light – softer, from left side, no shadow (reduces harsh dark areas)
    fill = spec.worldbody.add_light()
    fill.name = "fill"
    fill.pos[:] = (-4.0, 5.0, 6.0)
    fill.dir[:] = (0.3, -0.4, -1.0)
    fill.castshadow = False
    fill.diffuse[:] = (0.4, 0.45, 0.5)
    fill.specular[:] = (0.1, 0.1, 0.1)
    fill.ambient[:] = (0.1, 0.1, 0.1)

    # Back/rim light – creates edge highlights for depth separation
    rim = spec.worldbody.add_light()
    rim.name = "rim"
    rim.pos[:] = (-3.0, 0.0, 8.0)
    rim.dir[:] = (0.3, 0.0, -1.0)
    rim.castshadow = False
    rim.diffuse[:] = (0.35, 0.35, 0.4)
    rim.specular[:] = (0.3, 0.3, 0.3)
    rim.ambient[:] = (0.05, 0.05, 0.05)


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


# ── Stereo camera baseline = 120.114 mm ───────────────────────────────────
_STEREO_MOUNT_POS = (0.15, 0.0, 0.3)   # relative to torso_link
_STEREO_HALF_BASELINE = 0.060057        # half of 120.114 mm
_STEREO_FOVY = 46.8
_STEREO_RES = (1280, 720)
# Look along +X (forward) with ~10° downward pitch, up ≈ +Z.
_STEREO_QUAT_WXYZ = (0.454519, 0.454519, -0.541675, -0.541675)


def _add_stereo_cameras(spec: mujoco.MjSpec) -> None:
    """Add a stereo camera pair (cam_left / cam_right) on a head mount."""
    torso = spec.body(TORSO_BODY_NAME)
    mount = torso.add_body()
    mount.name = "stereo_camera_mount"
    mount.pos[:] = _STEREO_MOUNT_POS

    left = mount.add_camera()
    left.name = "cam_left"
    left.pos[:] = (0.0, _STEREO_HALF_BASELINE, 0.0)
    left.quat[:] = _STEREO_QUAT_WXYZ
    left.fovy = _STEREO_FOVY
    left.resolution[:] = _STEREO_RES

    right = mount.add_camera()
    right.name = "cam_right"
    right.pos[:] = (0.0, -_STEREO_HALF_BASELINE, 0.0)
    right.quat[:] = _STEREO_QUAT_WXYZ
    right.fovy = _STEREO_FOVY
    right.resolution[:] = _STEREO_RES


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


def _add_pyramid_stairs(
    spec: mujoco.MjSpec,
    up_steps: int = 5,
    down_steps: int = 5,
    step_width: float = 8.0,
    step_depth: float = 0.3,
    step_height: float = 0.08,
    start_x: float = 4.5,
) -> None:
    """Add a pyramid staircase: *up_steps* up, a flat platform, then *down_steps* down."""
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

    idx = 0
    x = start_x

    # ── Ramp approach: a gentle slope to the first step edge ──
    ramp_len = 0.4
    ramp = spec.worldbody.add_geom()
    ramp.name = f"pyr_{idx}"
    ramp.type = mujoco.mjtGeom.mjGEOM_BOX
    ramp.size[:] = (ramp_len / 2.0, step_width / 2.0, step_height / 4.0)
    ramp.pos[:] = (x + ramp_len / 2.0, 0.0, step_height / 4.0)
    ramp.material = "stair_mat"
    ramp.group = 0
    ramp.condim = 3
    ramp.contype = 1
    ramp.conaffinity = 1
    ramp.friction[:] = (1.0, 0.5, 0.5)
    ramp.priority = 1
    x += ramp_len
    idx += 1

    # ── Ascending ──
    for i in range(up_steps):
        height = (i + 1) * step_height
        half_h = height / 2.0
        cx = x + step_depth / 2.0
        step = spec.worldbody.add_geom()
        step.name = f"pyr_{idx}"
        step.type = mujoco.mjtGeom.mjGEOM_BOX
        step.size[:] = (step_depth / 2.0, step_width / 2.0, half_h)
        step.pos[:] = (cx, 0.0, half_h)
        step.material = "stair_mat"
        step.group = 0
        step.condim = 3
        step.contype = 1
        step.conaffinity = 1
        step.friction[:] = (1.0, 0.5, 0.5)
        step.priority = 1
        x += step_depth
        idx += 1

    # ── Flat platform at the top ──
    platform_len = 0.8
    top_h = up_steps * step_height
    plat = spec.worldbody.add_geom()
    plat.name = f"pyr_{idx}"
    plat.type = mujoco.mjtGeom.mjGEOM_BOX
    plat.size[:] = (platform_len / 2.0, step_width / 2.0, top_h / 2.0)
    plat.pos[:] = (x + platform_len / 2.0, 0.0, top_h / 2.0)
    plat.material = "stair_mat"
    plat.group = 0
    plat.condim = 3
    plat.contype = 1
    plat.conaffinity = 1
    plat.friction[:] = (1.0, 0.5, 0.5)
    plat.priority = 1
    x += platform_len
    idx += 1

    # ── Descending ──
    for i in range(down_steps):
        height = (down_steps - i) * step_height
        half_h = height / 2.0
        cx = x + step_depth / 2.0
        step = spec.worldbody.add_geom()
        step.name = f"pyr_{idx}"
        step.type = mujoco.mjtGeom.mjGEOM_BOX
        step.size[:] = (step_depth / 2.0, step_width / 2.0, half_h)
        step.pos[:] = (cx, 0.0, half_h)
        step.material = "stair_mat"
        step.group = 0
        step.condim = 3
        step.contype = 1
        step.conaffinity = 1
        step.friction[:] = (1.0, 0.5, 0.5)
        step.priority = 1
        x += step_depth
        idx += 1

    # ── Ramp exit: gentle slope from last step to ground ──
    ramp_exit = spec.worldbody.add_geom()
    ramp_exit.name = f"pyr_{idx}"
    ramp_exit.type = mujoco.mjtGeom.mjGEOM_BOX
    ramp_exit.size[:] = (ramp_len / 2.0, step_width / 2.0, step_height / 4.0)
    ramp_exit.pos[:] = (x + ramp_len / 2.0, 0.0, step_height / 4.0)
    ramp_exit.material = "stair_mat"
    ramp_exit.group = 0
    ramp_exit.condim = 3
    ramp_exit.contype = 1
    ramp_exit.conaffinity = 1
    ramp_exit.friction[:] = (1.0, 0.5, 0.5)
    ramp_exit.priority = 1


def _add_landmarks(spec: mujoco.MjSpec) -> None:
    """Add visual reference objects so movement is obvious in the scene."""
    # Materials for landmarks
    TextureCfg(
        name="crate_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.6, 0.4, 0.2),
        rgb2=(0.45, 0.28, 0.12),
        width=64,
        height=64,
    ).edit_spec(spec)
    MaterialCfg(
        name="crate_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(3.0, 3.0),
        reflectance=0.05,
        texture="crate_tex",
    ).edit_spec(spec)

    # Pillar texture – gradient gives stereo matcher something to latch onto
    TextureCfg(
        name="pillar_tex",
        type="2d",
        builtin="gradient",
        rgb1=(0.75, 0.75, 0.8),
        rgb2=(0.45, 0.45, 0.5),
        width=32,
        height=64,
    ).edit_spec(spec)
    MaterialCfg(
        name="pillar_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=False,
        texrepeat=(1.0, 2.0),
        reflectance=0.3,
        texture="pillar_tex",
    ).edit_spec(spec)

    # ── Colored pillars at cardinal directions ──
    pillar_positions = [
        (3.0, 0.0, "pillar_front", (0.2, 0.6, 0.9, 1.0)),    # blue, ahead
        (-3.0, 0.0, "pillar_back", (0.9, 0.3, 0.2, 1.0)),    # red, behind
        (0.0, 3.0, "pillar_left", (0.2, 0.8, 0.3, 1.0)),     # green, left
        (0.0, -3.0, "pillar_right", (0.9, 0.8, 0.2, 1.0)),   # yellow, right
    ]
    for x, y, name, rgba in pillar_positions:
        mat_name = f"{name}_mat"
        MaterialCfg(
            name=mat_name,
            rgba=rgba,
            reflectance=0.2,
        ).edit_spec(spec)
        p = spec.worldbody.add_geom()
        p.name = name
        p.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        p.size[:2] = (0.08, 0.6)  # radius, half-height
        p.pos[:] = (x, y, 0.6)
        p.material = mat_name
        p.group = 0
        p.contype = 1
        p.conaffinity = 1

    # ── Crates scattered around ──
    crate_positions = [
        (2.0, 1.5, "crate_1"),
        (4.0, -1.0, "crate_2"),
        (-1.5, 2.5, "crate_3"),
        (5.0, 2.0, "crate_4"),
        (-2.0, -2.0, "crate_5"),
    ]
    for x, y, name in crate_positions:
        c = spec.worldbody.add_geom()
        c.name = name
        c.type = mujoco.mjtGeom.mjGEOM_BOX
        c.size[:] = (0.2, 0.2, 0.2)
        c.pos[:] = (x, y, 0.2)
        c.material = "crate_mat"
        c.group = 0
        c.contype = 1
        c.conaffinity = 1

    # ── Spheres as colorful landmarks at farther distances ──
    sphere_positions = [
        (6.0, 0.0, "sphere_6m", (1.0, 0.2, 0.5, 1.0)),
        (0.0, 6.0, "sphere_left6", (0.3, 0.9, 0.9, 1.0)),
        (8.0, 3.0, "sphere_far", (1.0, 0.6, 0.0, 1.0)),
    ]
    for x, y, name, rgba in sphere_positions:
        mat_name = f"{name}_mat"
        MaterialCfg(
            name=mat_name,
            rgba=rgba,
            reflectance=0.4,
        ).edit_spec(spec)
        s = spec.worldbody.add_geom()
        s.name = name
        s.type = mujoco.mjtGeom.mjGEOM_SPHERE
        s.size[0] = 0.25
        s.pos[:] = (x, y, 0.25)
        s.material = mat_name
        s.group = 0
        s.contype = 1
        s.conaffinity = 1


def build_scene_model(robot_xml_path: Path, terrain: str = "flat") -> mujoco.MjModel:
    """Build a complete MuJoCo model from the robot MJCF with scene elements."""
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    _add_skybox(spec)
    _add_ground(spec)
    _add_lights(spec)
    _add_solver(spec)
    _add_depth_camera(spec)
    _add_stereo_cameras(spec)
    _add_actuators(spec)
    _add_landmarks(spec)

    # Ensure offscreen framebuffer is large enough for stereo camera rendering
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, _STEREO_RES[0])
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, _STEREO_RES[1])

    # znear is a ratio of model extent; keep it tiny so close-up views work
    spec.visual.map.znear = 0.001

    if terrain == "stairs":
        _add_stairs(spec)
    elif terrain == "pyramid":
        _add_pyramid_stairs(spec)

    return spec.compile()
