"""Indoor room scene for MuJoCo frame server.

Builds a realistic indoor environment (living room / hallway) optimised for
FoundationStereo depth estimation.  Separate from the parkour scene_builder.py
so that other demos remain unaffected.

Scene layout (MuJoCo coords: X = forward, Y = left, Z = up):
  - A rectangular room ~8 × 6 m with textured walls and wood floor
  - Furniture: sofa, table, bookshelf, cabinet, potted plant stands
  - Natural 3-point lighting (window-like key, overhead fill, warm accent)
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

# ── Physics (same as parkour scene) ───────────────────────────────────────
SIM_DT = 0.005
CONTROL_DT = 0.02

# ── Stereo camera (same as parkour scene) ─────────────────────────────────
_STEREO_MOUNT_POS = (0.15, 0.0, 0.3)
_STEREO_HALF_BASELINE = 0.060057
_STEREO_FOVY = 46.8
_STEREO_RES = (1280, 720)
_STEREO_QUAT_WXYZ = (0.454519, 0.454519, -0.541675, -0.541675)

# ── Texture directory (reserved for future PNG-based textures) ────────────
_TEX_DIR = Path(__file__).parent / "textures"


# ═══════════════════════════════════════════════════════════════════════════
#  Textures & Materials
# ═══════════════════════════════════════════════════════════════════════════

def _add_textures_and_materials(spec: mujoco.MjSpec) -> None:
    """Register all PNG-based textures and their materials."""
    # -- Skybox (indoor: subtle grey gradient, like a room ceiling) --
    TextureCfg(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=(0.85, 0.85, 0.88),
        rgb2=(0.55, 0.55, 0.6),
        width=512,
        height=3072,
    ).edit_spec(spec)

    # -- Wood floor (checker pattern simulates plank seams) --
    TextureCfg(
        name="wood_floor_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.62, 0.47, 0.30),
        rgb2=(0.55, 0.40, 0.24),
        width=128,
        height=128,
        mark="edge",
        markrgb=(0.42, 0.30, 0.16),
    ).edit_spec(spec)
    MaterialCfg(
        name="wood_floor_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(8.0, 8.0),
        reflectance=0.15,
        texture="wood_floor_tex",
    ).edit_spec(spec)

    # -- Plaster wall (subtle gradient) --
    TextureCfg(
        name="plaster_wall_tex",
        type="2d",
        builtin="gradient",
        rgb1=(0.92, 0.89, 0.84),
        rgb2=(0.88, 0.85, 0.80),
        width=128,
        height=128,
    ).edit_spec(spec)
    MaterialCfg(
        name="plaster_wall_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(3.0, 2.0),
        reflectance=0.05,
        texture="plaster_wall_tex",
    ).edit_spec(spec)

    # -- Brick wall (accent wall – checker simulates mortar pattern) --
    TextureCfg(
        name="brick_wall_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.65, 0.30, 0.22),
        rgb2=(0.55, 0.25, 0.18),
        width=64,
        height=32,
        mark="edge",
        markrgb=(0.70, 0.68, 0.64),
    ).edit_spec(spec)
    MaterialCfg(
        name="brick_wall_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(8.0, 4.0),
        reflectance=0.03,
        texture="brick_wall_tex",
    ).edit_spec(spec)

    # -- Wood panel (furniture – gradient simulates wood grain) --
    TextureCfg(
        name="wood_panel_tex",
        type="2d",
        builtin="gradient",
        rgb1=(0.45, 0.30, 0.16),
        rgb2=(0.35, 0.22, 0.10),
        width=64,
        height=128,
    ).edit_spec(spec)
    MaterialCfg(
        name="wood_panel_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(2.0, 2.0),
        reflectance=0.1,
        texture="wood_panel_tex",
    ).edit_spec(spec)

    # -- Carpet (checker weave) --
    TextureCfg(
        name="carpet_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.32, 0.18, 0.20),
        rgb2=(0.28, 0.15, 0.17),
        width=32,
        height=32,
    ).edit_spec(spec)
    MaterialCfg(
        name="carpet_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(12.0, 8.0),
        reflectance=0.02,
        texture="carpet_tex",
    ).edit_spec(spec)

    # -- Fabric (sofa cushions) --
    TextureCfg(
        name="fabric_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.45, 0.50, 0.55),
        rgb2=(0.40, 0.45, 0.50),
        width=64,
        height=64,
    ).edit_spec(spec)
    MaterialCfg(
        name="fabric_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(4.0, 4.0),
        reflectance=0.02,
        texture="fabric_tex",
    ).edit_spec(spec)

    # -- Metal (table legs, lamp) --
    TextureCfg(
        name="metal_tex",
        type="2d",
        builtin="gradient",
        rgb1=(0.55, 0.55, 0.58),
        rgb2=(0.35, 0.35, 0.38),
        width=32,
        height=64,
    ).edit_spec(spec)
    MaterialCfg(
        name="metal_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=False,
        reflectance=0.4,
        texture="metal_tex",
    ).edit_spec(spec)

    # -- White trim (baseboard, door frames) --
    MaterialCfg(
        name="trim_mat",
        rgba=(0.95, 0.95, 0.93, 1.0),
        reflectance=0.2,
    ).edit_spec(spec)

    # -- Dark green (potted plants) --
    TextureCfg(
        name="plant_tex",
        type="2d",
        builtin="checker",
        rgb1=(0.15, 0.45, 0.15),
        rgb2=(0.10, 0.35, 0.10),
        width=32,
        height=32,
    ).edit_spec(spec)
    MaterialCfg(
        name="plant_mat",
        rgba=(1.0, 1.0, 1.0, 1.0),
        texuniform=True,
        texrepeat=(3.0, 3.0),
        reflectance=0.05,
        texture="plant_tex",
    ).edit_spec(spec)

    # -- Terracotta (plant pot) --
    MaterialCfg(
        name="pot_mat",
        rgba=(0.72, 0.42, 0.28, 1.0),
        reflectance=0.1,
    ).edit_spec(spec)


# ═══════════════════════════════════════════════════════════════════════════
#  Room Structure
# ═══════════════════════════════════════════════════════════════════════════

def _add_floor(spec: mujoco.MjSpec) -> None:
    """Add a wood-textured floor plane."""
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = (6.0, 4.0, 0.01)
    floor.pos[:] = (3.0, 0.0, 0.0)
    floor.material = "wood_floor_mat"
    floor.contype = 1
    floor.conaffinity = 1
    floor.friction[:] = (1.0, 0.5, 0.5)
    floor.priority = 1


def _add_walls(spec: mujoco.MjSpec) -> None:
    """Add textured walls around the room perimeter.

    Room: X ∈ [-1, 7], Y ∈ [-3, 3], wall height = 2.8 m
    """
    wall_h = 2.8
    half_h = wall_h / 2.0
    wall_thick = 0.08

    walls = [
        # (name, pos, size, material)
        # Front wall (X=7) - brick accent wall
        ("wall_front", (7.0, 0.0, half_h), (wall_thick/2, 3.0, half_h), "brick_wall_mat"),
        # Back wall (X=-1) - plaster
        ("wall_back", (-1.0, 0.0, half_h), (wall_thick/2, 3.0, half_h), "plaster_wall_mat"),
        # Left wall (Y=3) - plaster
        ("wall_left", (3.0, 3.0, half_h), (4.0, wall_thick/2, half_h), "plaster_wall_mat"),
        # Right wall (Y=-3) - plaster
        ("wall_right", (3.0, -3.0, half_h), (4.0, wall_thick/2, half_h), "plaster_wall_mat"),
    ]
    for name, pos, size, mat in walls:
        w = spec.worldbody.add_geom()
        w.name = name
        w.type = mujoco.mjtGeom.mjGEOM_BOX
        w.pos[:] = pos
        w.size[:] = size
        w.material = mat
        w.contype = 1
        w.conaffinity = 1
        w.group = 0

    # Baseboard trim (thin white strip at wall base)
    trim_h = 0.08
    trims = [
        ("trim_front", (7.0 - wall_thick, 0.0, trim_h/2), (0.01, 2.95, trim_h/2)),
        ("trim_back", (-1.0 + wall_thick, 0.0, trim_h/2), (0.01, 2.95, trim_h/2)),
        ("trim_left", (3.0, 3.0 - wall_thick, trim_h/2), (3.95, 0.01, trim_h/2)),
        ("trim_right", (3.0, -3.0 + wall_thick, trim_h/2), (3.95, 0.01, trim_h/2)),
    ]
    for name, pos, size in trims:
        t = spec.worldbody.add_geom()
        t.name = name
        t.type = mujoco.mjtGeom.mjGEOM_BOX
        t.pos[:] = pos
        t.size[:] = size
        t.material = "trim_mat"
        t.contype = 0
        t.conaffinity = 0
        t.group = 0


def _add_carpet(spec: mujoco.MjSpec) -> None:
    """Add an area rug in the center of the room."""
    rug = spec.worldbody.add_geom()
    rug.name = "area_rug"
    rug.type = mujoco.mjtGeom.mjGEOM_BOX
    rug.pos[:] = (3.5, 0.0, 0.005)
    rug.size[:] = (1.5, 1.2, 0.005)
    rug.material = "carpet_mat"
    rug.contype = 0
    rug.conaffinity = 0
    rug.group = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Furniture
# ═══════════════════════════════════════════════════════════════════════════

def _add_sofa(spec: mujoco.MjSpec) -> None:
    """Add a sofa against the left wall."""
    # Sofa base (seat)
    seat = spec.worldbody.add_geom()
    seat.name = "sofa_seat"
    seat.type = mujoco.mjtGeom.mjGEOM_BOX
    seat.pos[:] = (4.0, 2.3, 0.25)
    seat.size[:] = (0.45, 0.4, 0.25)
    seat.material = "fabric_mat"
    seat.contype = 1
    seat.conaffinity = 1

    # Sofa backrest
    back = spec.worldbody.add_geom()
    back.name = "sofa_back"
    back.type = mujoco.mjtGeom.mjGEOM_BOX
    back.pos[:] = (4.0, 2.65, 0.55)
    back.size[:] = (0.45, 0.08, 0.3)
    back.material = "fabric_mat"
    back.contype = 1
    back.conaffinity = 1

    # Armrests
    for side, y_off, name in [("left", 0.35, "sofa_arm_l"), ("right", -0.35, "sofa_arm_r")]:
        arm = spec.worldbody.add_geom()
        arm.name = name
        arm.type = mujoco.mjtGeom.mjGEOM_BOX
        arm.pos[:] = (4.0, 2.3 + y_off, 0.4)
        arm.size[:] = (0.08, 0.08, 0.15)
        arm.material = "fabric_mat"
        arm.contype = 1
        arm.conaffinity = 1


def _add_coffee_table(spec: mujoco.MjSpec) -> None:
    """Add a coffee table in front of the sofa."""
    # Table top
    top = spec.worldbody.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.pos[:] = (4.0, 1.2, 0.42)
    top.size[:] = (0.4, 0.3, 0.02)
    top.material = "wood_panel_mat"
    top.contype = 1
    top.conaffinity = 1

    # Table legs
    leg_positions = [
        (3.65, 0.95, "table_leg_1"),
        (3.65, 1.45, "table_leg_2"),
        (4.35, 0.95, "table_leg_3"),
        (4.35, 1.45, "table_leg_4"),
    ]
    for x, y, name in leg_positions:
        leg = spec.worldbody.add_geom()
        leg.name = name
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos[:] = (x, y, 0.2)
        leg.size[:2] = (0.02, 0.2)
        leg.material = "metal_mat"
        leg.contype = 1
        leg.conaffinity = 1


def _add_bookshelf(spec: mujoco.MjSpec) -> None:
    """Add a tall bookshelf against the front (brick) wall."""
    # Main frame
    frame = spec.worldbody.add_geom()
    frame.name = "bookshelf_frame"
    frame.type = mujoco.mjtGeom.mjGEOM_BOX
    frame.pos[:] = (6.6, -1.5, 0.9)
    frame.size[:] = (0.2, 0.5, 0.9)
    frame.material = "wood_panel_mat"
    frame.contype = 1
    frame.conaffinity = 1

    # Shelf boards (horizontal dividers)
    for i, z in enumerate([0.35, 0.7, 1.05, 1.4]):
        shelf = spec.worldbody.add_geom()
        shelf.name = f"shelf_board_{i}"
        shelf.type = mujoco.mjtGeom.mjGEOM_BOX
        shelf.pos[:] = (6.6, -1.5, z)
        shelf.size[:] = (0.18, 0.48, 0.01)
        shelf.material = "wood_panel_mat"
        shelf.contype = 0
        shelf.conaffinity = 0

    # Some "books" (thin tall boxes with varied colors)
    book_colors = [
        (0.7, 0.2, 0.2, 1.0),  # red
        (0.2, 0.3, 0.7, 1.0),  # blue
        (0.2, 0.6, 0.3, 1.0),  # green
        (0.8, 0.7, 0.2, 1.0),  # yellow
        (0.5, 0.2, 0.6, 1.0),  # purple
    ]
    for i, (color, z_base) in enumerate(zip(book_colors, [0.38, 0.38, 0.73, 0.73, 1.08])):
        mat_name = f"book_{i}_mat"
        MaterialCfg(
            name=mat_name,
            rgba=color,
            reflectance=0.1,
        ).edit_spec(spec)
        book = spec.worldbody.add_geom()
        book.name = f"book_{i}"
        book.type = mujoco.mjtGeom.mjGEOM_BOX
        book.pos[:] = (6.6 + (i % 3 - 1) * 0.07, -1.5 + (i - 2) * 0.15, z_base + 0.12)
        book.size[:] = (0.06, 0.02, 0.12)
        book.material = mat_name
        book.contype = 0
        book.conaffinity = 0


def _add_cabinet(spec: mujoco.MjSpec) -> None:
    """Add a TV cabinet / sideboard against the right wall."""
    cab = spec.worldbody.add_geom()
    cab.name = "cabinet"
    cab.type = mujoco.mjtGeom.mjGEOM_BOX
    cab.pos[:] = (2.5, -2.6, 0.35)
    cab.size[:] = (0.6, 0.25, 0.35)
    cab.material = "wood_panel_mat"
    cab.contype = 1
    cab.conaffinity = 1

    # Decorative box on top
    deco = spec.worldbody.add_geom()
    deco.name = "deco_box"
    deco.type = mujoco.mjtGeom.mjGEOM_BOX
    deco.pos[:] = (2.2, -2.6, 0.78)
    deco.size[:] = (0.1, 0.08, 0.08)
    MaterialCfg(
        name="deco_mat",
        rgba=(0.85, 0.75, 0.6, 1.0),
        reflectance=0.15,
    ).edit_spec(spec)
    deco.material = "deco_mat"
    deco.contype = 0
    deco.conaffinity = 0


def _add_dining_table(spec: mujoco.MjSpec) -> None:
    """Add a small dining table with chairs."""
    # Table
    dtop = spec.worldbody.add_geom()
    dtop.name = "dining_table_top"
    dtop.type = mujoco.mjtGeom.mjGEOM_BOX
    dtop.pos[:] = (1.5, 0.5, 0.72)
    dtop.size[:] = (0.5, 0.4, 0.025)
    dtop.material = "wood_panel_mat"
    dtop.contype = 1
    dtop.conaffinity = 1

    # 4 legs
    for i, (dx, dy) in enumerate([(-0.42, -0.32), (-0.42, 0.32), (0.42, -0.32), (0.42, 0.32)]):
        leg = spec.worldbody.add_geom()
        leg.name = f"dining_leg_{i}"
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.pos[:] = (1.5 + dx, 0.5 + dy, 0.35)
        leg.size[:2] = (0.025, 0.35)
        leg.material = "wood_panel_mat"
        leg.contype = 1
        leg.conaffinity = 1

    # 2 chairs
    for ci, (cx, cy, cyaw) in enumerate([(1.5, -0.1, 0.0), (1.5, 1.1, np.pi)]):
        # Chair seat
        cs = spec.worldbody.add_geom()
        cs.name = f"chair_seat_{ci}"
        cs.type = mujoco.mjtGeom.mjGEOM_BOX
        cs.pos[:] = (cx, cy, 0.42)
        cs.size[:] = (0.2, 0.2, 0.02)
        cs.material = "wood_panel_mat"
        cs.contype = 1
        cs.conaffinity = 1

        # Chair back
        cb = spec.worldbody.add_geom()
        cb.name = f"chair_back_{ci}"
        cb.type = mujoco.mjtGeom.mjGEOM_BOX
        back_y = cy + 0.18 * np.cos(cyaw)
        cb.pos[:] = (cx, back_y, 0.7)
        cb.size[:] = (0.18, 0.015, 0.26)
        cb.material = "wood_panel_mat"
        cb.contype = 1
        cb.conaffinity = 1

        # Chair legs
        for li, (ldx, ldy) in enumerate([(-0.15, -0.15), (-0.15, 0.15), (0.15, -0.15), (0.15, 0.15)]):
            cl = spec.worldbody.add_geom()
            cl.name = f"chair_{ci}_leg_{li}"
            cl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            cl.pos[:] = (cx + ldx, cy + ldy, 0.2)
            cl.size[:2] = (0.015, 0.2)
            cl.material = "metal_mat"
            cl.contype = 1
            cl.conaffinity = 1


def _add_plants(spec: mujoco.MjSpec) -> None:
    """Add potted plants for visual variety."""
    plants = [
        (5.5, 2.5, "plant_corner"),
        (0.0, -2.5, "plant_entry"),
    ]
    for x, y, name in plants:
        # Pot
        pot = spec.worldbody.add_geom()
        pot.name = f"{name}_pot"
        pot.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        pot.pos[:] = (x, y, 0.15)
        pot.size[:2] = (0.12, 0.15)
        pot.material = "pot_mat"
        pot.contype = 1
        pot.conaffinity = 1

        # Plant foliage (sphere cluster)
        foliage = spec.worldbody.add_geom()
        foliage.name = f"{name}_foliage"
        foliage.type = mujoco.mjtGeom.mjGEOM_SPHERE
        foliage.pos[:] = (x, y, 0.55)
        foliage.size[0] = 0.2
        foliage.material = "plant_mat"
        foliage.contype = 0
        foliage.conaffinity = 0

        # Second foliage ball (slightly offset for natural look)
        f2 = spec.worldbody.add_geom()
        f2.name = f"{name}_foliage2"
        f2.type = mujoco.mjtGeom.mjGEOM_SPHERE
        f2.pos[:] = (x + 0.08, y - 0.05, 0.65)
        f2.size[0] = 0.15
        f2.material = "plant_mat"
        f2.contype = 0
        f2.conaffinity = 0


def _add_floor_lamp(spec: mujoco.MjSpec) -> None:
    """Add a floor lamp near the sofa."""
    # Lamp pole
    pole = spec.worldbody.add_geom()
    pole.name = "lamp_pole"
    pole.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pole.pos[:] = (5.0, 2.5, 0.7)
    pole.size[:2] = (0.02, 0.7)
    pole.material = "metal_mat"
    pole.contype = 1
    pole.conaffinity = 1

    # Lamp shade (cone approximated by capsule)
    MaterialCfg(
        name="lampshade_mat",
        rgba=(0.95, 0.9, 0.75, 0.9),
        reflectance=0.05,
    ).edit_spec(spec)
    shade = spec.worldbody.add_geom()
    shade.name = "lamp_shade"
    shade.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    shade.pos[:] = (5.0, 2.5, 1.45)
    shade.size[:2] = (0.12, 0.1)
    shade.material = "lampshade_mat"
    shade.contype = 0
    shade.conaffinity = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Lighting
# ═══════════════════════════════════════════════════════════════════════════

def _add_lights(spec: mujoco.MjSpec) -> None:
    """Natural indoor lighting optimised for stereo matching."""
    # Minimal headlight to avoid flat camera-direction illumination
    spec.visual.headlight.ambient[:] = (0.1, 0.1, 0.1)
    spec.visual.headlight.diffuse[:] = (0.15, 0.15, 0.15)
    spec.visual.headlight.specular[:] = (0.05, 0.05, 0.05)

    # Key light: simulates window on left wall (warm, directional)
    key = spec.worldbody.add_light()
    key.name = "window_key"
    key.pos[:] = (3.0, 2.8, 2.4)
    key.dir[:] = (0.2, -0.8, -0.3)
    key.castshadow = True
    key.diffuse[:] = (0.95, 0.9, 0.8)
    key.specular[:] = (0.5, 0.5, 0.5)
    key.ambient[:] = (0.1, 0.1, 0.1)

    # Fill: overhead ceiling light (neutral, broad)
    fill = spec.worldbody.add_light()
    fill.name = "ceiling_fill"
    fill.pos[:] = (3.0, 0.0, 2.7)
    fill.dir[:] = (0.0, 0.0, -1.0)
    fill.castshadow = False
    fill.diffuse[:] = (0.5, 0.5, 0.52)
    fill.specular[:] = (0.2, 0.2, 0.2)
    fill.ambient[:] = (0.15, 0.15, 0.15)

    # Accent: warm lamp light near sofa
    accent = spec.worldbody.add_light()
    accent.name = "lamp_accent"
    accent.pos[:] = (5.0, 2.5, 1.5)
    accent.dir[:] = (0.0, 0.0, -1.0)
    accent.castshadow = True
    accent.diffuse[:] = (0.6, 0.5, 0.35)
    accent.specular[:] = (0.2, 0.15, 0.1)
    accent.ambient[:] = (0.05, 0.04, 0.03)

    # Rim: back wall bounce (cool, subtle edge separation)
    rim = spec.worldbody.add_light()
    rim.name = "back_bounce"
    rim.pos[:] = (-0.5, 0.0, 2.0)
    rim.dir[:] = (1.0, 0.0, -0.3)
    rim.castshadow = False
    rim.diffuse[:] = (0.25, 0.28, 0.32)
    rim.specular[:] = (0.1, 0.1, 0.1)
    rim.ambient[:] = (0.05, 0.05, 0.06)


# ═══════════════════════════════════════════════════════════════════════════
#  Cameras & Actuators (shared with parkour scene)
# ═══════════════════════════════════════════════════════════════════════════

def _add_depth_camera(spec: mujoco.MjSpec) -> None:
    torso_body = spec.body(TORSO_BODY_NAME)
    cam = torso_body.add_camera()
    cam.name = "head_depth"
    cam.pos[:] = CAMERA_OFFSET_POS
    cam.quat[:] = CAMERA_OFFSET_QUAT_WXYZ
    cam.fovy = 58.29


def _add_stereo_cameras(spec: mujoco.MjSpec) -> None:
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


def _add_solver(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.noslip_iterations = 0
    spec.option.ccd_iterations = 128
    spec.nconmax = 128
    spec.njmax = 700


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


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def build_indoor_scene(robot_xml_path: Path) -> mujoco.MjModel:
    """Build a realistic indoor room scene with the G1 robot.

    This is independent from scene_builder.build_scene_model() and is
    intended for the frame server / stereo vision pipeline only.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    # Materials & textures
    _add_textures_and_materials(spec)

    # Room structure
    _add_floor(spec)
    _add_walls(spec)
    _add_carpet(spec)

    # Furniture
    _add_sofa(spec)
    _add_coffee_table(spec)
    _add_bookshelf(spec)
    _add_cabinet(spec)
    _add_dining_table(spec)
    _add_plants(spec)
    _add_floor_lamp(spec)

    # Lighting
    _add_lights(spec)

    # Robot subsystems
    _add_solver(spec)
    _add_depth_camera(spec)
    _add_stereo_cameras(spec)
    _add_actuators(spec)

    # Ensure offscreen framebuffer is large enough
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, _STEREO_RES[0])
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, _STEREO_RES[1])
    spec.visual.map.znear = 0.001

    return spec.compile()
