"""G1 29-DOF robot configuration constants.

Joint-level actuator parameters (BeyondMimic), nominal pose,
camera offsets, and body/geom names used across the runner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = _SCRIPT_DIR / "mjcf" / "g1_29dof_torsoBase_popsicle_with_shoe.xml"
MODEL_DIR = _SCRIPT_DIR / "models" / "parkour"

# ── Body / geom names ─────────────────────────────────────────────────────
HEAD_BODY_NAME = "head_link"
TORSO_BODY_NAME = "torso_link"
FOOT_GEOM_NAMES = (
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
    "left_foot4_collision",
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
    "right_foot4_collision",
)

# ── Camera (head depth) ───────────────────────────────────────────────────
CAMERA_OFFSET_POS = np.asarray(
    [0.0487988662332928, 0.01, 0.4378029937970051], dtype=np.float64
)
CAMERA_OFFSET_QUAT_WXYZ = np.asarray(
    [0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0],
    dtype=np.float64,
)

# ── Nominal joint pose ────────────────────────────────────────────────────
NOMINAL_JOINT_POS = np.asarray(
    [
        0.0, 0.0, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

# ── Per-joint actuator parameters (BeyondMimic config) ────────────────────
JOINT_KP = np.asarray(
    [
        28.50124619574858,
        28.50124619574858,
        40.17923847137318,
        40.17923847137318,
        99.09842777666113,
        40.17923847137318,
        99.09842777666113,
        28.50124619574858,
        28.50124619574858,
        40.17923847137318,
        99.09842777666113,
        40.17923847137318,
        99.09842777666113,
        28.50124619574858,
        28.50124619574858,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        16.77832748089279,
        16.77832748089279,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        16.77832748089279,
        16.77832748089279,
    ],
    dtype=np.float32,
)

JOINT_KD = np.asarray(
    [
        1.814445686584846,
        1.814445686584846,
        2.5578897650279457,
        2.5578897650279457,
        6.3088018534966395,
        2.5578897650279457,
        6.3088018534966395,
        1.814445686584846,
        1.814445686584846,
        2.5578897650279457,
        6.3088018534966395,
        2.5578897650279457,
        6.3088018534966395,
        1.814445686584846,
        1.814445686584846,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        1.06814150219,
        1.06814150219,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        1.06814150219,
        1.06814150219,
    ],
    dtype=np.float32,
)

JOINT_TORQUE_LIMIT = np.asarray(
    [
        50.0, 50.0, 88.0, 88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ],
    dtype=np.float32,
)

JOINT_ARMATURE = np.asarray(
    [
        0.00721945,    # waist_pitch  (2*ARMATURE_5020)
        0.00721945,    # waist_roll
        0.010177520,   # waist_yaw    (ARMATURE_7520_14)
        0.010177520,   # left_hip_pitch
        0.025101925,   # left_hip_roll  (ARMATURE_7520_22)
        0.010177520,   # left_hip_yaw
        0.025101925,   # left_knee
        0.00721945,    # left_ankle_pitch
        0.00721945,    # left_ankle_roll
        0.010177520,   # right_hip_pitch
        0.025101925,   # right_hip_roll
        0.010177520,   # right_hip_yaw
        0.025101925,   # right_knee
        0.00721945,    # right_ankle_pitch
        0.00721945,    # right_ankle_roll
        0.003609725,   # left_shoulder_pitch (ARMATURE_5020)
        0.003609725,   # left_shoulder_roll
        0.003609725,   # left_shoulder_yaw
        0.003609725,   # left_elbow
        0.003609725,   # left_wrist_roll
        0.00425,       # left_wrist_pitch (ARMATURE_4010)
        0.00425,       # left_wrist_yaw
        0.003609725,   # right_shoulder_pitch
        0.003609725,   # right_shoulder_roll
        0.003609725,   # right_shoulder_yaw
        0.003609725,   # right_elbow
        0.003609725,   # right_wrist_roll
        0.00425,       # right_wrist_pitch
        0.00425,       # right_wrist_yaw
    ],
    dtype=np.float64,
)
