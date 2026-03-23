"""Pure math utilities: quaternion ops & depth image processing."""
from __future__ import annotations

import numpy as np

from scene_builder import DEPTH_CROP, DEPTH_OUTPUT_SHAPE, DEPTH_RANGE

# ── Constants ──────────────────────────────────────────────────────────────
GRAVITY_WORLD = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
DEFAULT_COMMAND = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

# ── Quaternion helpers (wxyz convention) ───────────────────────────────────

def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_wxyz(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    pure = np.asarray([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    rotated = quat_mul_wxyz(quat_mul_wxyz(q, pure), quat_conjugate_wxyz(q))
    return rotated[1:]


def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_base(vec_world: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_wxyz(quat_conjugate_wxyz(base_quat_wxyz), vec_world)


# ── Depth image utilities ─────────────────────────────────────────────────

def normalize_depth_image(depth: np.ndarray) -> np.ndarray:
    min_depth, max_depth = DEPTH_RANGE
    depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / max(max_depth - min_depth, 1.0e-6)
    return depth.astype(np.float32)


def crop_and_resize_depth(depth: np.ndarray) -> np.ndarray:
    import cv2

    crop_top, crop_bottom, crop_left, crop_right = DEPTH_CROP
    row_start = min(max(crop_top, 0), depth.shape[0])
    row_stop = max(row_start, depth.shape[0] - max(crop_bottom, 0))
    col_start = min(max(crop_left, 0), depth.shape[1])
    col_stop = max(col_start, depth.shape[1] - max(crop_right, 0))
    cropped = depth[row_start:row_stop, col_start:col_stop]
    resized = cv2.resize(
        cropped,
        (DEPTH_OUTPUT_SHAPE[1], DEPTH_OUTPUT_SHAPE[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32)


def gaussian_blur_depth(depth: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(depth, (3, 3), sigmaX=1.0).astype(np.float32)
