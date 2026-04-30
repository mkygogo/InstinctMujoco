"""两室一厅 apartment scene for MuJoCo frame server.

Builds a realistic Chinese apartment with:
  - Living room (客厅) — largest room, robot starts here
  - Bedroom 1 (主卧) — north-west
  - Bedroom 2 (次卧) — north-east
  - Kitchen (厨房) — south-east
  - Central hallway connecting all rooms via door openings

Uses RoboCasa PNG textures for realistic appearance.

Floor plan (MuJoCo coords: X = forward, Y = left, Z = up):

  Y=4.5 ┌──────────────────┬────────────────────────┐
        │   Bedroom 1      │      Bedroom 2          │
        │   (5m × 3.5m)    │      (5m × 3.5m)        │
        │                   │                         │
  Y=1.0 ├──── ─ ───────────┴──────── ─ ──────────────┤
        │            Hallway (11m × 1.5m)             │
  Y=-0.5├──── ─ ────────────┬─────── ─ ──────────────┤
        │                    │                         │
        │  Living Room       │      Kitchen            │
        │  (6.5m × 4m)      │      (4m × 4m)          │
        │                    │                         │
  Y=-4.5└────────────────────┴────────────────────────┘
        X=0                  X=6.75                   X=11
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.spec import create_position_actuator
from mjlab.utils.spec_config import MaterialCfg
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

# ── Physics ───────────────────────────────────────────────────────────────
SIM_DT = 0.005
CONTROL_DT = 0.02

# ── Stereo camera ─────────────────────────────────────────────────────────
_STEREO_MOUNT_POS = (0.15, 0.0, 0.3)
_STEREO_HALF_BASELINE = 0.060057
_STEREO_FOVY = 46.8
_STEREO_RES = (1280, 720)
_STEREO_QUAT_WXYZ = (0.454519, 0.454519, -0.541675, -0.541675)

# ── RoboCasa texture paths ────────────────────────────────────────────────
_ROBOCASA_TEX = Path.home() / "robocasa/robocasa/models/assets/textures"

# ── Room layout constants ─────────────────────────────────────────────────
_WALL_H = 2.8       # wall height
_WALL_THICK = 0.08  # wall thickness
_DOOR_W = 1.0       # door opening width
_DOOR_H = 2.1       # door height (for visual framing)

# Outer bounds
_X_MIN, _X_MAX = 0.0, 11.0
_Y_MIN, _Y_MAX = -4.5, 4.5

# Hallway strip
_HALL_Y_MIN = -0.5
_HALL_Y_MAX = 1.0

# Living room / kitchen divider
_LK_DIVIDER_X = 6.75

# Bedroom divider
_BR_DIVIDER_X = 5.5


# ═══════════════════════════════════════════════════════════════════════════
#  Textures & Materials (file-based from RoboCasa)
# ═══════════════════════════════════════════════════════════════════════════

def _add_file_texture(spec: mujoco.MjSpec, name: str, filepath: Path,
                      tex_type: str = "2d") -> None:
    """Add a file-based PNG texture to the spec."""
    tex = spec.add_texture()
    tex.name = name
    type_map = {"2d": mujoco.mjtTexture.mjTEXTURE_2D,
                "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
                "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX}
    tex.type = type_map[tex_type]
    tex.file = str(filepath)


def _add_material(spec: mujoco.MjSpec, name: str, texture: str,
                  texrepeat: tuple = (4.0, 4.0), reflectance: float = 0.05,
                  texuniform: bool = True) -> None:
    """Add a material referencing a named texture."""
    mat = spec.add_material()
    mat.name = name
    mat.textures[0] = texture
    mat.texrepeat[:] = texrepeat
    mat.reflectance = reflectance
    mat.texuniform = texuniform


def _add_textures_and_materials(spec: mujoco.MjSpec) -> None:
    """Register all textures and materials for the apartment."""
    # Skybox (indoor ceiling-like gradient)
    from mjlab.utils.spec_config import TextureCfg
    TextureCfg(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=(0.82, 0.82, 0.85),
        rgb2=(0.50, 0.50, 0.55),
        width=512,
        height=3072,
    ).edit_spec(spec)

    # ── File-based textures from RoboCasa ──
    # Living room floor: warm wood planks
    _add_file_texture(spec, "living_floor_tex",
                      _ROBOCASA_TEX / "wood/light_wood_planks.png")
    _add_material(spec, "living_floor_mat", "living_floor_tex",
                  texrepeat=(6.0, 6.0), reflectance=0.12)

    # Bedroom floor: darker parquet
    _add_file_texture(spec, "bedroom_floor_tex",
                      _ROBOCASA_TEX / "wood/warm_wood_parquet.png")
    _add_material(spec, "bedroom_floor_mat", "bedroom_floor_tex",
                  texrepeat=(5.0, 5.0), reflectance=0.10)

    # Kitchen floor: tiles
    _add_file_texture(spec, "kitchen_floor_tex",
                      _ROBOCASA_TEX / "tiles/white_tiles.png")
    _add_material(spec, "kitchen_floor_mat", "kitchen_floor_tex",
                  texrepeat=(6.0, 6.0), reflectance=0.15)

    # Hallway floor: wood planks (different from living room)
    _add_file_texture(spec, "hall_floor_tex",
                      _ROBOCASA_TEX / "wood/wood_planks.png")
    _add_material(spec, "hall_floor_mat", "hall_floor_tex",
                  texrepeat=(8.0, 2.0), reflectance=0.10)

    # Wall textures
    _add_file_texture(spec, "plaster_cream_tex",
                      _ROBOCASA_TEX / "cream-plaster.png")
    _add_material(spec, "wall_cream_mat", "plaster_cream_tex",
                  texrepeat=(3.0, 2.0), reflectance=0.04)

    _add_file_texture(spec, "plaster_white_tex",
                      _ROBOCASA_TEX / "prev/white-plaster.png")
    _add_material(spec, "wall_white_mat", "plaster_white_tex",
                  texrepeat=(3.0, 2.0), reflectance=0.05)

    _add_file_texture(spec, "plaster_gray_tex",
                      _ROBOCASA_TEX / "prev/gray-plaster.png")
    _add_material(spec, "wall_gray_mat", "plaster_gray_tex",
                  texrepeat=(3.0, 2.0), reflectance=0.04)

    # Accent wall (brick / wood panels)
    _add_file_texture(spec, "wood_wall_tex",
                      _ROBOCASA_TEX / "wood/wood_planks_wall.png")
    _add_material(spec, "wall_wood_mat", "wood_wall_tex",
                  texrepeat=(4.0, 2.0), reflectance=0.08)

    # Furniture wood
    _add_file_texture(spec, "furniture_wood_tex",
                      _ROBOCASA_TEX / "wood/walnut_wood_grain.png")
    _add_material(spec, "furniture_wood_mat", "furniture_wood_tex",
                  texrepeat=(2.0, 2.0), reflectance=0.10)

    # Ceiling (flat white)
    _add_file_texture(spec, "ceiling_tex",
                      _ROBOCASA_TEX / "flat/lighter_gray.png")
    _add_material(spec, "ceiling_mat", "ceiling_tex",
                  texrepeat=(1.0, 1.0), reflectance=0.02)

    # ── Builtin materials (no texture file needed) ──
    MaterialCfg(name="trim_mat", rgba=(0.94, 0.94, 0.92, 1.0),
                reflectance=0.15).edit_spec(spec)
    MaterialCfg(name="metal_mat", rgba=(0.50, 0.50, 0.53, 1.0),
                reflectance=0.35).edit_spec(spec)
    MaterialCfg(name="fabric_gray_mat", rgba=(0.55, 0.55, 0.58, 1.0),
                reflectance=0.03).edit_spec(spec)
    MaterialCfg(name="fabric_blue_mat", rgba=(0.35, 0.45, 0.60, 1.0),
                reflectance=0.03).edit_spec(spec)
    MaterialCfg(name="pot_mat", rgba=(0.72, 0.42, 0.28, 1.0),
                reflectance=0.08).edit_spec(spec)
    MaterialCfg(name="plant_mat", rgba=(0.18, 0.50, 0.20, 1.0),
                reflectance=0.05).edit_spec(spec)
    MaterialCfg(name="counter_mat", rgba=(0.90, 0.88, 0.85, 1.0),
                reflectance=0.20).edit_spec(spec)
    MaterialCfg(name="appliance_mat", rgba=(0.85, 0.85, 0.87, 1.0),
                reflectance=0.25).edit_spec(spec)
    MaterialCfg(name="door_frame_mat", rgba=(0.75, 0.60, 0.42, 1.0),
                reflectance=0.10).edit_spec(spec)
    MaterialCfg(name="bed_mat", rgba=(0.92, 0.90, 0.85, 1.0),
                reflectance=0.03).edit_spec(spec)
    MaterialCfg(name="pillow_mat", rgba=(0.95, 0.95, 0.92, 1.0),
                reflectance=0.02).edit_spec(spec)


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: wall segment with optional door opening
# ═══════════════════════════════════════════════════════════════════════════

_geom_counter = 0


def _next_name(prefix: str) -> str:
    global _geom_counter
    _geom_counter += 1
    return f"{prefix}_{_geom_counter}"


def _add_wall_segment(spec: mujoco.MjSpec, p1: tuple, p2: tuple,
                      material: str, height: float = _WALL_H,
                      thick: float = _WALL_THICK) -> None:
    """Add a single wall segment from p1=(x1,y1) to p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_h = height / 2
    # Wall is axis-aligned (only horizontal or vertical)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > dy:
        # horizontal wall (along X)
        size = (dx / 2, thick / 2, half_h)
    else:
        # vertical wall (along Y)
        size = (thick / 2, dy / 2, half_h)

    w = spec.worldbody.add_geom()
    w.name = _next_name("wall")
    w.type = mujoco.mjtGeom.mjGEOM_BOX
    w.pos[:] = (cx, cy, half_h)
    w.size[:] = size
    w.material = material
    w.contype = 1
    w.conaffinity = 1
    w.group = 0


def _add_wall_with_door(spec: mujoco.MjSpec, start: tuple, end: tuple,
                        door_center: float, door_width: float,
                        material: str, is_x_wall: bool) -> None:
    """Add a wall with a door opening.

    is_x_wall: True if wall runs along X axis (constant Y).
    door_center: position along the wall's running axis where door is.
    """
    x1, y1 = start
    x2, y2 = end

    if is_x_wall:
        # Wall at constant Y, running from x1 to x2
        door_left = door_center - door_width / 2
        door_right = door_center + door_width / 2
        # Left segment
        if door_left > x1 + 0.01:
            _add_wall_segment(spec, (x1, y1), (door_left, y2), material)
        # Right segment
        if door_right < x2 - 0.01:
            _add_wall_segment(spec, (door_right, y1), (x2, y2), material)
        # Door frame (lintel above door)
        lintel = spec.worldbody.add_geom()
        lintel.name = _next_name("lintel")
        lintel.type = mujoco.mjtGeom.mjGEOM_BOX
        lintel.pos[:] = (door_center, (y1 + y2) / 2,
                         _DOOR_H + (_WALL_H - _DOOR_H) / 2)
        lintel.size[:] = (door_width / 2, _WALL_THICK / 2,
                          (_WALL_H - _DOOR_H) / 2)
        lintel.material = material
        lintel.contype = 1
        lintel.conaffinity = 1
    else:
        # Wall at constant X, running from y1 to y2
        door_bottom = door_center - door_width / 2
        door_top = door_center + door_width / 2
        # Bottom segment
        if door_bottom > y1 + 0.01:
            _add_wall_segment(spec, (x1, y1), (x2, door_bottom), material)
        # Top segment
        if door_top < y2 - 0.01:
            _add_wall_segment(spec, (x1, door_top), (x2, y2), material)
        # Lintel
        lintel = spec.worldbody.add_geom()
        lintel.name = _next_name("lintel")
        lintel.type = mujoco.mjtGeom.mjGEOM_BOX
        lintel.pos[:] = ((x1 + x2) / 2, door_center,
                         _DOOR_H + (_WALL_H - _DOOR_H) / 2)
        lintel.size[:] = (_WALL_THICK / 2, door_width / 2,
                          (_WALL_H - _DOOR_H) / 2)
        lintel.material = material
        lintel.contype = 1
        lintel.conaffinity = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Room Structure
# ═══════════════════════════════════════════════════════════════════════════

def _add_floors(spec: mujoco.MjSpec) -> None:
    """Add floor planes for each room with different materials."""
    floors = [
        # (name, pos, size, material)
        # Living room
        ("floor_living", ((_X_MIN + _LK_DIVIDER_X) / 2, (_Y_MIN + _HALL_Y_MIN) / 2, 0.0),
         ((_LK_DIVIDER_X - _X_MIN) / 2, (_HALL_Y_MIN - _Y_MIN) / 2, 0.01),
         "living_floor_mat"),
        # Kitchen
        ("floor_kitchen", ((_LK_DIVIDER_X + _X_MAX) / 2, (_Y_MIN + _HALL_Y_MIN) / 2, 0.0),
         ((_X_MAX - _LK_DIVIDER_X) / 2, (_HALL_Y_MIN - _Y_MIN) / 2, 0.01),
         "kitchen_floor_mat"),
        # Hallway
        ("floor_hallway", ((_X_MIN + _X_MAX) / 2, (_HALL_Y_MIN + _HALL_Y_MAX) / 2, 0.0),
         ((_X_MAX - _X_MIN) / 2, (_HALL_Y_MAX - _HALL_Y_MIN) / 2, 0.01),
         "hall_floor_mat"),
        # Bedroom 1 (NW)
        ("floor_bedroom1", ((_X_MIN + _BR_DIVIDER_X) / 2, (_HALL_Y_MAX + _Y_MAX) / 2, 0.0),
         ((_BR_DIVIDER_X - _X_MIN) / 2, (_Y_MAX - _HALL_Y_MAX) / 2, 0.01),
         "bedroom_floor_mat"),
        # Bedroom 2 (NE)
        ("floor_bedroom2", ((_BR_DIVIDER_X + _X_MAX) / 2, (_HALL_Y_MAX + _Y_MAX) / 2, 0.0),
         ((_X_MAX - _BR_DIVIDER_X) / 2, (_Y_MAX - _HALL_Y_MAX) / 2, 0.01),
         "bedroom_floor_mat"),
    ]
    for name, pos, size, mat in floors:
        f = spec.worldbody.add_geom()
        f.name = name
        f.type = mujoco.mjtGeom.mjGEOM_BOX
        f.pos[:] = pos
        f.size[:] = size
        f.material = mat
        f.contype = 1
        f.conaffinity = 1
        f.friction[:] = (1.0, 0.5, 0.5)
        f.priority = 1


def _add_ceilings(spec: mujoco.MjSpec) -> None:
    """Add ceiling panels."""
    ceil = spec.worldbody.add_geom()
    ceil.name = "ceiling"
    ceil.type = mujoco.mjtGeom.mjGEOM_BOX
    ceil.pos[:] = ((_X_MIN + _X_MAX) / 2, (_Y_MIN + _Y_MAX) / 2, _WALL_H)
    ceil.size[:] = ((_X_MAX - _X_MIN) / 2, (_Y_MAX - _Y_MIN) / 2, 0.02)
    ceil.material = "ceiling_mat"
    ceil.contype = 0
    ceil.conaffinity = 0
    ceil.group = 0


def _add_outer_walls(spec: mujoco.MjSpec) -> None:
    """Add the four outer walls of the apartment."""
    # South wall (Y = _Y_MIN)
    _add_wall_segment(spec, (_X_MIN, _Y_MIN), (_X_MAX, _Y_MIN), "wall_cream_mat")
    # North wall (Y = _Y_MAX)
    _add_wall_segment(spec, (_X_MIN, _Y_MAX), (_X_MAX, _Y_MAX), "wall_cream_mat")
    # West wall (X = _X_MIN)
    _add_wall_segment(spec, (_X_MIN, _Y_MIN), (_X_MIN, _Y_MAX), "wall_cream_mat")
    # East wall (X = _X_MAX) — wood accent
    _add_wall_segment(spec, (_X_MAX, _Y_MIN), (_X_MAX, _Y_MAX), "wall_wood_mat")


def _add_interior_walls(spec: mujoco.MjSpec) -> None:
    """Add interior walls with door openings for navigation."""
    # ── South hallway wall (Y = _HALL_Y_MIN) ──
    # Door to living room at X=3.0
    # Door to kitchen at X=8.5
    _add_wall_with_door(spec,
                        start=(_X_MIN, _HALL_Y_MIN), end=(_LK_DIVIDER_X, _HALL_Y_MIN),
                        door_center=3.0, door_width=_DOOR_W,
                        material="wall_white_mat", is_x_wall=True)
    _add_wall_with_door(spec,
                        start=(_LK_DIVIDER_X, _HALL_Y_MIN), end=(_X_MAX, _HALL_Y_MIN),
                        door_center=8.8, door_width=_DOOR_W,
                        material="wall_white_mat", is_x_wall=True)

    # ── North hallway wall (Y = _HALL_Y_MAX) ──
    # Door to bedroom 1 at X=2.5
    # Door to bedroom 2 at X=8.0
    _add_wall_with_door(spec,
                        start=(_X_MIN, _HALL_Y_MAX), end=(_BR_DIVIDER_X, _HALL_Y_MAX),
                        door_center=2.5, door_width=_DOOR_W,
                        material="wall_white_mat", is_x_wall=True)
    _add_wall_with_door(spec,
                        start=(_BR_DIVIDER_X, _HALL_Y_MAX), end=(_X_MAX, _HALL_Y_MAX),
                        door_center=8.0, door_width=_DOOR_W,
                        material="wall_white_mat", is_x_wall=True)

    # ── Bedroom divider (X = _BR_DIVIDER_X, Y from hallway to north wall) ──
    _add_wall_segment(spec,
                      (_BR_DIVIDER_X, _HALL_Y_MAX), (_BR_DIVIDER_X, _Y_MAX),
                      "wall_gray_mat")

    # ── Living/Kitchen divider (X = _LK_DIVIDER_X, Y from south wall to hallway) ──
    # Door at Y=-2.5
    _add_wall_with_door(spec,
                        start=(_LK_DIVIDER_X, _Y_MIN), end=(_LK_DIVIDER_X, _HALL_Y_MIN),
                        door_center=-2.5, door_width=_DOOR_W,
                        material="wall_gray_mat", is_x_wall=False)


def _add_baseboards(spec: mujoco.MjSpec) -> None:
    """Add white baseboard trim along outer walls."""
    trim_h = 0.08
    ht = trim_h / 2
    off = _WALL_THICK
    cx = (_X_MIN + _X_MAX) / 2
    cy = (_Y_MIN + _Y_MAX) / 2
    hx = (_X_MAX - _X_MIN) / 2 - off
    hy = (_Y_MAX - _Y_MIN) / 2 - off

    trims = [
        ("trim_s", (cx, _Y_MIN + off, ht), (hx, 0.01, ht)),
        ("trim_n", (cx, _Y_MAX - off, ht), (hx, 0.01, ht)),
        ("trim_w", (_X_MIN + off, cy, ht), (0.01, hy, ht)),
        ("trim_e", (_X_MAX - off, cy, ht), (0.01, hy, ht)),
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


# ═══════════════════════════════════════════════════════════════════════════
#  Furniture
# ═══════════════════════════════════════════════════════════════════════════

def _add_living_room_furniture(spec: mujoco.MjSpec) -> None:
    """Sofa, coffee table, TV cabinet in the living room."""
    # ── Sofa (against south wall) ──
    sofa_x, sofa_y = 3.5, -4.0
    # Seat
    g = spec.worldbody.add_geom()
    g.name = "sofa_seat"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (sofa_x, sofa_y, 0.25); g.size[:] = (0.9, 0.35, 0.25)
    g.material = "fabric_gray_mat"; g.contype = 1; g.conaffinity = 1
    # Backrest
    g = spec.worldbody.add_geom()
    g.name = "sofa_back"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (sofa_x, -4.3, 0.55); g.size[:] = (0.9, 0.08, 0.30)
    g.material = "fabric_gray_mat"; g.contype = 1; g.conaffinity = 1
    # Armrests
    for i, dx in enumerate([-0.85, 0.85]):
        g = spec.worldbody.add_geom()
        g.name = f"sofa_arm_{i}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos[:] = (sofa_x + dx, sofa_y, 0.38); g.size[:] = (0.08, 0.30, 0.12)
        g.material = "fabric_gray_mat"; g.contype = 1; g.conaffinity = 1

    # ── Coffee table ──
    g = spec.worldbody.add_geom()
    g.name = "coffee_table_top"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (3.5, -2.8, 0.40); g.size[:] = (0.55, 0.30, 0.02)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
    for i, (dx, dy) in enumerate([(-0.45, -0.22), (-0.45, 0.22),
                                   (0.45, -0.22), (0.45, 0.22)]):
        g = spec.worldbody.add_geom()
        g.name = f"coffee_leg_{i}"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos[:] = (3.5 + dx, -2.8 + dy, 0.19); g.size[:2] = (0.02, 0.19)
        g.material = "metal_mat"; g.contype = 1; g.conaffinity = 1

    # ── TV cabinet (against west wall) ──
    g = spec.worldbody.add_geom()
    g.name = "tv_cabinet"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (0.4, -2.5, 0.30); g.size[:] = (0.22, 0.8, 0.30)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1

    # ── Bookshelf (against divider wall) ──
    g = spec.worldbody.add_geom()
    g.name = "bookshelf"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (6.3, -1.2, 0.85); g.size[:] = (0.20, 0.4, 0.85)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1

    # ── Potted plant (corner) ──
    g = spec.worldbody.add_geom()
    g.name = "plant_living_pot"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.pos[:] = (0.5, -0.9, 0.15); g.size[:2] = (0.12, 0.15)
    g.material = "pot_mat"; g.contype = 1; g.conaffinity = 1
    g = spec.worldbody.add_geom()
    g.name = "plant_living_top"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.pos[:] = (0.5, -0.9, 0.55); g.size[0] = 0.22
    g.material = "plant_mat"; g.contype = 0; g.conaffinity = 0


def _add_kitchen_furniture(spec: mujoco.MjSpec) -> None:
    """Kitchen counter, fridge-like box, small table."""
    # Counter along east wall
    g = spec.worldbody.add_geom()
    g.name = "kitchen_counter"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (10.3, -2.5, 0.45); g.size[:] = (0.22, 1.5, 0.45)
    g.material = "counter_mat"; g.contype = 1; g.conaffinity = 1

    # Fridge (tall box, NE corner of kitchen)
    g = spec.worldbody.add_geom()
    g.name = "fridge"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (10.3, -0.8, 0.90); g.size[:] = (0.30, 0.35, 0.90)
    g.material = "appliance_mat"; g.contype = 1; g.conaffinity = 1

    # Small dining table
    g = spec.worldbody.add_geom()
    g.name = "kitchen_table"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (8.5, -3.0, 0.72); g.size[:] = (0.40, 0.40, 0.025)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
    for i, (dx, dy) in enumerate([(-0.32, -0.32), (-0.32, 0.32),
                                   (0.32, -0.32), (0.32, 0.32)]):
        g = spec.worldbody.add_geom()
        g.name = f"ktable_leg_{i}"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos[:] = (8.5 + dx, -3.0 + dy, 0.35); g.size[:2] = (0.02, 0.35)
        g.material = "metal_mat"; g.contype = 1; g.conaffinity = 1

    # Two chairs
    for ci, cx in enumerate([8.1, 8.9]):
        g = spec.worldbody.add_geom()
        g.name = f"kchair_seat_{ci}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos[:] = (cx, -3.0, 0.42); g.size[:] = (0.20, 0.20, 0.02)
        g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
        g = spec.worldbody.add_geom()
        g.name = f"kchair_back_{ci}"; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos[:] = (cx, -3.0 - 0.18, 0.65); g.size[:] = (0.18, 0.015, 0.22)
        g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1


def _add_bedroom1_furniture(spec: mujoco.MjSpec) -> None:
    """Bed, nightstand, wardrobe in bedroom 1 (NW)."""
    # Bed (against north wall)
    g = spec.worldbody.add_geom()
    g.name = "bed1_frame"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (2.5, 3.8, 0.25); g.size[:] = (0.95, 0.45, 0.25)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
    # Mattress
    g = spec.worldbody.add_geom()
    g.name = "bed1_mattress"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (2.5, 3.8, 0.55); g.size[:] = (0.90, 0.42, 0.08)
    g.material = "bed_mat"; g.contype = 1; g.conaffinity = 1
    # Pillow
    g = spec.worldbody.add_geom()
    g.name = "bed1_pillow"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (2.5, 4.1, 0.66); g.size[:] = (0.25, 0.12, 0.04)
    g.material = "pillow_mat"; g.contype = 0; g.conaffinity = 0

    # Nightstand
    g = spec.worldbody.add_geom()
    g.name = "nightstand1"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (3.8, 3.8, 0.30); g.size[:] = (0.20, 0.20, 0.30)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1

    # Wardrobe (against west wall)
    g = spec.worldbody.add_geom()
    g.name = "wardrobe1"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (0.4, 2.75, 1.0); g.size[:] = (0.30, 0.8, 1.0)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1


def _add_bedroom2_furniture(spec: mujoco.MjSpec) -> None:
    """Bed, desk, chair in bedroom 2 (NE)."""
    # Bed (against east wall)
    g = spec.worldbody.add_geom()
    g.name = "bed2_frame"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (8.5, 3.8, 0.25); g.size[:] = (0.85, 0.45, 0.25)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
    g = spec.worldbody.add_geom()
    g.name = "bed2_mattress"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (8.5, 3.8, 0.55); g.size[:] = (0.80, 0.42, 0.08)
    g.material = "bed_mat"; g.contype = 1; g.conaffinity = 1
    g = spec.worldbody.add_geom()
    g.name = "bed2_pillow"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (8.5, 4.1, 0.66); g.size[:] = (0.22, 0.12, 0.04)
    g.material = "pillow_mat"; g.contype = 0; g.conaffinity = 0

    # Desk (study desk along north wall)
    g = spec.worldbody.add_geom()
    g.name = "desk2_top"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (10.0, 2.5, 0.72); g.size[:] = (0.35, 0.55, 0.02)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1
    for i, (dx, dy) in enumerate([(-0.28, -0.48), (-0.28, 0.48),
                                   (0.28, -0.48), (0.28, 0.48)]):
        g = spec.worldbody.add_geom()
        g.name = f"desk2_leg_{i}"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos[:] = (10.0 + dx, 2.5 + dy, 0.35); g.size[:2] = (0.02, 0.35)
        g.material = "metal_mat"; g.contype = 1; g.conaffinity = 1

    # Desk chair
    g = spec.worldbody.add_geom()
    g.name = "desk_chair_seat"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (10.0, 1.7, 0.42); g.size[:] = (0.20, 0.20, 0.02)
    g.material = "fabric_blue_mat"; g.contype = 1; g.conaffinity = 1

    # Potted plant (corner)
    g = spec.worldbody.add_geom()
    g.name = "plant_br2_pot"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.pos[:] = (6.0, 4.0, 0.12); g.size[:2] = (0.10, 0.12)
    g.material = "pot_mat"; g.contype = 1; g.conaffinity = 1
    g = spec.worldbody.add_geom()
    g.name = "plant_br2_top"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.pos[:] = (6.0, 4.0, 0.45); g.size[0] = 0.18
    g.material = "plant_mat"; g.contype = 0; g.conaffinity = 0


def _add_hallway_furniture(spec: mujoco.MjSpec) -> None:
    """Shoe cabinet and coat rack in hallway."""
    # Shoe cabinet near west wall
    g = spec.worldbody.add_geom()
    g.name = "shoe_cabinet"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos[:] = (0.4, 0.25, 0.40); g.size[:] = (0.25, 0.50, 0.40)
    g.material = "furniture_wood_mat"; g.contype = 1; g.conaffinity = 1

    # Hall plant
    g = spec.worldbody.add_geom()
    g.name = "plant_hall_pot"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.pos[:] = (10.2, 0.25, 0.15); g.size[:2] = (0.10, 0.15)
    g.material = "pot_mat"; g.contype = 1; g.conaffinity = 1
    g = spec.worldbody.add_geom()
    g.name = "plant_hall_top"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.pos[:] = (10.2, 0.25, 0.50); g.size[0] = 0.18
    g.material = "plant_mat"; g.contype = 0; g.conaffinity = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Lighting
# ═══════════════════════════════════════════════════════════════════════════

def _add_lights(spec: mujoco.MjSpec) -> None:
    """Multi-room lighting optimised for stereo matching."""
    spec.visual.headlight.ambient[:] = (0.08, 0.08, 0.08)
    spec.visual.headlight.diffuse[:] = (0.12, 0.12, 0.12)
    spec.visual.headlight.specular[:] = (0.04, 0.04, 0.04)

    # Living room: bright ceiling light
    light = spec.worldbody.add_light()
    light.name = "living_ceiling"
    light.pos[:] = (3.5, -2.5, 2.6)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.castshadow = True
    light.diffuse[:] = (0.8, 0.78, 0.72)
    light.specular[:] = (0.3, 0.3, 0.3)
    light.ambient[:] = (0.12, 0.12, 0.12)

    # Living room: window simulation (warm side light)
    light = spec.worldbody.add_light()
    light.name = "living_window"
    light.pos[:] = (3.5, -4.2, 2.0)
    light.dir[:] = (0.0, 1.0, -0.3)
    light.castshadow = True
    light.diffuse[:] = (0.7, 0.65, 0.55)
    light.specular[:] = (0.3, 0.3, 0.3)
    light.ambient[:] = (0.05, 0.05, 0.05)

    # Kitchen ceiling light
    light = spec.worldbody.add_light()
    light.name = "kitchen_ceiling"
    light.pos[:] = (8.8, -2.5, 2.6)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.castshadow = False
    light.diffuse[:] = (0.75, 0.75, 0.78)
    light.specular[:] = (0.25, 0.25, 0.25)
    light.ambient[:] = (0.10, 0.10, 0.10)

    # Hallway ceiling light
    light = spec.worldbody.add_light()
    light.name = "hallway_ceiling"
    light.pos[:] = (5.5, 0.25, 2.6)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.castshadow = False
    light.diffuse[:] = (0.55, 0.55, 0.58)
    light.specular[:] = (0.15, 0.15, 0.15)
    light.ambient[:] = (0.10, 0.10, 0.10)

    # Bedroom 1 ceiling light
    light = spec.worldbody.add_light()
    light.name = "bedroom1_ceiling"
    light.pos[:] = (2.5, 2.75, 2.6)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.castshadow = True
    light.diffuse[:] = (0.65, 0.60, 0.52)
    light.specular[:] = (0.2, 0.2, 0.2)
    light.ambient[:] = (0.10, 0.10, 0.10)

    # Bedroom 2 ceiling light
    light = spec.worldbody.add_light()
    light.name = "bedroom2_ceiling"
    light.pos[:] = (8.5, 2.75, 2.6)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.castshadow = True
    light.diffuse[:] = (0.60, 0.62, 0.65)
    light.specular[:] = (0.2, 0.2, 0.2)
    light.ambient[:] = (0.10, 0.10, 0.10)


# ═══════════════════════════════════════════════════════════════════════════
#  Cameras & Actuators
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

def build_apartment_scene(robot_xml_path: Path) -> mujoco.MjModel:
    """Build a 两室一厅 apartment scene with the G1 robot.

    Robot spawns in the living room at approximately (2, -2.5).
    Navigation targets: kitchen, bedroom 1, bedroom 2, hallway.
    """
    global _geom_counter
    _geom_counter = 0

    spec = mujoco.MjSpec.from_file(str(robot_xml_path))

    # Move robot spawn into the living room center
    torso = spec.body(TORSO_BODY_NAME)
    torso.pos[:] = (3.0, -2.5, 0.9)
    # Materials & textures
    _add_textures_and_materials(spec)

    # Room structure
    _add_floors(spec)
    _add_ceilings(spec)
    _add_outer_walls(spec)
    _add_interior_walls(spec)
    _add_baseboards(spec)

    # Furniture
    _add_living_room_furniture(spec)
    _add_kitchen_furniture(spec)
    _add_bedroom1_furniture(spec)
    _add_bedroom2_furniture(spec)
    _add_hallway_furniture(spec)

    # Lighting
    _add_lights(spec)

    # Robot subsystems
    _add_solver(spec)
    _add_depth_camera(spec)
    _add_stereo_cameras(spec)
    _add_actuators(spec)

    # Offscreen framebuffer
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, _STEREO_RES[0])
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, _STEREO_RES[1])
    spec.visual.map.znear = 0.001

    return spec.compile()
