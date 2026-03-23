"""Standalone MuJoCo runner for the G1 parkour ONNX policy.

Usage:
    python run_parkour_mujoco.py --command-x 0.5
    python run_parkour_mujoco.py --command-x 0.5 --terrain stairs --use-depth
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import time
from dataclasses import dataclass
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from math_utils import (
    GRAVITY_WORLD,
    DEFAULT_COMMAND,
    crop_and_resize_depth,
    gaussian_blur_depth,
    normalize_depth_image,
    quat_conjugate_wxyz,
    quat_mul_wxyz,
    quat_to_mat_wxyz,
    world_to_base,
)
from parkour_onnx_policy import (
    G1_JOINT_ORDER,
    ParkourOnnxPolicy,
    PolicyFrame,
    PolicyHistory,
)
from robot_config import (
    CAMERA_OFFSET_POS,
    CAMERA_OFFSET_QUAT_WXYZ,
    FOOT_GEOM_NAMES,
    HEAD_BODY_NAME,
    MODEL_DIR,
    NOMINAL_JOINT_POS,
    TORSO_BODY_NAME,
    XML_PATH,
)
from scene_builder import (
    CONTROL_DT,
    DEPTH_RANGE,
    DEPTH_HISTORY_LEN,
    DEPTH_HISTORY_SKIP,
    DEPTH_OUTPUT_SHAPE,
    build_scene_model,
)

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_ACTION_CLIP = 100.0
DEFAULT_STARTUP_HOLD_STEPS = 0
DEFAULT_TARGET_SMOOTHING = 1.0


@dataclass
class JointIndex:
    qpos: int
    qvel: int


class G1MujocoRunner:
    def __init__(
        self,
        model_dir: Path,
        xml_path: Path,
        command: np.ndarray,
        passive: bool,
        real_time: bool,
        action_clip: float | None,
        startup_hold_steps: int,
        target_smoothing: float,
        stand_only: bool,
        log_interval: int,
        use_depth: bool = False,
        terrain: str = "flat",
        yaw_correction_gain: float = 0.0,
    ):
        self.policy = ParkourOnnxPolicy(model_dir)
        self.model = build_scene_model(xml_path, terrain=terrain)
        self.data = mujoco.MjData(self.model)
        self.command = np.asarray(command, dtype=np.float32)
        self.passive = passive
        self.real_time = real_time
        self.action_clip = action_clip
        self.startup_hold_steps = max(int(startup_hold_steps), 0)
        self.target_smoothing = float(np.clip(target_smoothing, 0.0, 1.0))
        self.stand_only = stand_only
        self.log_interval = max(int(log_interval), 0)
        self.use_depth = use_depth
        self.yaw_correction_gain = float(yaw_correction_gain)
        self.control_steps = max(int(round(CONTROL_DT / self.model.opt.timestep)), 1)
        self.depth_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_depth"
        )
        self.head_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY_NAME
        )
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME
        )
        self.joint_index = self._build_joint_index()
        self.actuator_ids = self._build_actuator_ids()
        self.history = PolicyHistory()
        self.depth_frame_history: deque[np.ndarray] = deque(maxlen=DEPTH_HISTORY_LEN)
        self.step_count = 0
        self.last_action = np.zeros(29, dtype=np.float32)
        self.target_qpos = NOMINAL_JOINT_POS.copy()
        self.renderer: mujoco.Renderer | None = None
        self._raycast_local_directions = self._build_raycast_local_directions(
            width=64,
            height=36,
            fovy=58.29,
        )
        self._depth_geomgroup = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.foot_geom_ids = self._resolve_foot_geom_ids()
        self._neutral_depth = self._compute_neutral_depth()
        self._neutral_depth_stack = self._make_constant_depth_stack(self._neutral_depth)
        self._initialize_pose()

    # ── Initialization helpers ─────────────────────────────────────────────

    def _compute_neutral_depth(self) -> np.ndarray:
        """Normalizer-mean depth -> zero after normalization (proprioception only)."""
        depth_start = self.policy._flat_slices["depth_image"].start
        frame_size = DEPTH_OUTPUT_SHAPE[0] * DEPTH_OUTPUT_SHAPE[1]
        return self.policy.normalizer.mean[
            0, depth_start : depth_start + frame_size
        ].reshape(DEPTH_OUTPUT_SHAPE).copy()

    def _make_constant_depth_stack(self, depth_frame: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(depth_frame, dtype=np.float32)[None, :, :], 8, axis=0)

    def _bootstrap_depth_history(self, depth_frame: np.ndarray) -> None:
        self.depth_frame_history.clear()
        for _ in range(DEPTH_HISTORY_LEN):
            self.depth_frame_history.append(np.asarray(depth_frame, dtype=np.float32).copy())

    def _build_raycast_local_directions(
        self,
        *,
        width: int,
        height: int,
        fovy: float,
    ) -> np.ndarray:
        v_fov_rad = np.deg2rad(fovy)
        aspect = width / height
        h_fov_rad = 2.0 * np.arctan(np.tan(v_fov_rad / 2.0) * aspect)

        u = np.linspace(0.5, width - 0.5, width, dtype=np.float64)
        v = np.linspace(0.5, height - 0.5, height, dtype=np.float64)
        grid_u, grid_v = np.meshgrid(u, v, indexing="xy")

        c_x = width / 2.0
        c_y = height / 2.0
        f_y = 0.5 * height / max(np.tan(v_fov_rad / 2.0), 1.0e-8)
        f_x = f_y

        pix_x = (grid_u - c_x) / f_x
        pix_y = (grid_v - c_y) / f_y
        pix_z = np.ones_like(pix_x)

        # Convert image camera frame (x-right, y-down, z-forward)
        # to training world camera frame (x-forward, y-left, z-up).
        directions = np.stack([pix_z, -pix_x, -pix_y], axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        return directions.reshape(-1, 3)

    def _build_depth_stack_from_history(self) -> np.ndarray:
        if not self.depth_frame_history:
            return self._neutral_depth_stack.copy()

        frames = list(self.depth_frame_history)
        last_idx = len(frames) - 1
        sampled: list[np.ndarray] = []
        for history_idx in range(8):
            frame_idx = max(last_idx - (7 - history_idx) * DEPTH_HISTORY_SKIP, 0)
            sampled.append(np.asarray(frames[frame_idx], dtype=np.float32))
        return np.stack(sampled, axis=0).astype(np.float32)

    def _create_renderer(self) -> mujoco.Renderer:
        try:
            return mujoco.Renderer(self.model, height=36, width=64)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create MuJoCo renderer. "
                "In headless mode set MUJOCO_GL=egl."
            ) from exc

    def _build_joint_index(self) -> dict[str, JointIndex]:
        mapping: dict[str, JointIndex] = {}
        for name in G1_JOINT_ORDER:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            mapping[name] = JointIndex(
                qpos=int(self.model.jnt_qposadr[jid]),
                qvel=int(self.model.jnt_dofadr[jid]),
            )
        return mapping

    def _build_actuator_ids(self) -> np.ndarray:
        ids = np.zeros(len(G1_JOINT_ORDER), dtype=np.int32)
        for idx, name in enumerate(G1_JOINT_ORDER):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator not found: {name}")
            ids[idx] = int(aid)
        return ids

    def _resolve_foot_geom_ids(self) -> list[int]:
        ids: list[int] = []
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.append(int(gid))
        if not ids:
            raise RuntimeError("No foot geoms found in the model.")
        return ids

    def _initialize_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.82]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for jname, jtarget in zip(G1_JOINT_ORDER, NOMINAL_JOINT_POS, strict=True):
            self.data.qpos[self.joint_index[jname].qpos] = float(jtarget)
        mujoco.mj_forward(self.model, self.data)
        self._place_feet_above_ground(clearance=0.01)
        initial_depth = self._render_head_depth() if self.use_depth else self._neutral_depth
        self._bootstrap_depth_history(initial_depth)
        self.history.reset()
        frame = self._build_policy_frame()
        for _ in range(self.history.history_len):
            self.history.append(frame)

    def _place_feet_above_ground(self, clearance: float) -> None:
        min_z = min(
            float(self.data.geom_xpos[gid][2]) for gid in self.foot_geom_ids
        )
        self.data.qpos[2] += clearance - min_z
        mujoco.mj_forward(self.model, self.data)

    # ── Runtime helpers ────────────────────────────────────────────────────

    def _joint_array(self, source: np.ndarray, field: str) -> np.ndarray:
        values = np.zeros(29, dtype=np.float32)
        for idx, name in enumerate(G1_JOINT_ORDER):
            j = self.joint_index[name]
            values[idx] = source[j.qpos if field == "qpos" else j.qvel]
        return values

    def _base_quat_wxyz(self) -> np.ndarray:
        return np.asarray(self.data.qpos[3:7], dtype=np.float64)

    def _yaw_only_quat_wxyz(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        half_yaw = 0.5 * yaw
        return np.asarray([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)

    def _depth_camera_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body_quat = self._base_quat_wxyz()
        body_rot = quat_to_mat_wxyz(body_quat)
        body_pos = np.asarray(self.data.xpos[self.torso_body_id], dtype=np.float64)
        camera_pos = body_pos + body_rot @ CAMERA_OFFSET_POS
        yaw_quat = self._yaw_only_quat_wxyz(body_quat)
        camera_quat = quat_mul_wxyz(yaw_quat, CAMERA_OFFSET_QUAT_WXYZ)
        return camera_pos, camera_quat

    def _set_depth_camera_yaw_aligned(self) -> None:
        body_quat = self._base_quat_wxyz()
        yaw_quat = self._yaw_only_quat_wxyz(body_quat)
        world_camera_quat = quat_mul_wxyz(yaw_quat, CAMERA_OFFSET_QUAT_WXYZ)
        local_camera_quat = quat_mul_wxyz(quat_conjugate_wxyz(body_quat), world_camera_quat)
        self.model.cam_quat[self.depth_camera_id] = local_camera_quat

    def _raycast_head_depth(self) -> np.ndarray:
        camera_pos, camera_quat = self._depth_camera_world_pose()
        camera_rot = quat_to_mat_wxyz(camera_quat)
        ray_dirs_world = self._raycast_local_directions @ camera_rot.T

        nray = ray_dirs_world.shape[0]
        dist = np.zeros(nray, dtype=np.float64)
        geomid = np.full(nray, -1, dtype=np.int32)
        mujoco.mj_multiRay(
            m=self.model,
            d=self.data,
            pnt=camera_pos,
            vec=ray_dirs_world.reshape(-1),
            geomgroup=self._depth_geomgroup,
            flg_static=1,
            bodyexclude=self.torso_body_id,
            geomid=geomid,
            dist=dist,
            normal=None,
            nray=nray,
            cutoff=DEPTH_RANGE[1],
        )

        depth = dist * self._raycast_local_directions[:, 0]
        invalid = (dist < 0.0) | (dist <= 0.1)
        depth[invalid] = DEPTH_RANGE[1]
        depth = np.clip(depth, DEPTH_RANGE[0], DEPTH_RANGE[1])
        depth = depth.reshape(36, 64).astype(np.float32)
        depth = crop_and_resize_depth(depth)
        depth = gaussian_blur_depth(depth)
        depth = normalize_depth_image(depth)
        return depth

    def _render_head_depth(self) -> np.ndarray:
        return self._raycast_head_depth()

    def _current_yaw(self) -> float:
        """Extract yaw angle (radians) from the base quaternion."""
        w, x, y, z = self._base_quat_wxyz()
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _corrected_command(self) -> np.ndarray:
        """Apply yaw and lateral correction to keep the robot on track."""
        cmd = self.command.copy()
        if self.yaw_correction_gain > 0.0:
            yaw = self._current_yaw()
            y = float(self.data.qpos[1])
            # Combined heading correction: correct for both yaw drift
            # and lateral offset (steer back toward y=0).
            desired_yaw = np.arctan2(-y, 2.0)  # aim back toward centerline
            cmd[2] = cmd[2] + self.yaw_correction_gain * (desired_yaw - yaw)
        return cmd

    def _build_policy_frame(self) -> PolicyFrame:
        base_quat = self._base_quat_wxyz()
        base_ang_vel_world = np.asarray(
            self.data.cvel[self.torso_body_id][0:3], dtype=np.float64
        )
        base_ang_vel_body = world_to_base(base_ang_vel_world, base_quat).astype(
            np.float32
        )
        projected_gravity = world_to_base(GRAVITY_WORLD, base_quat).astype(np.float32)
        joint_pos = self._joint_array(self.data.qpos, "qpos") - NOMINAL_JOINT_POS
        joint_vel = self._joint_array(self.data.qvel, "qvel")
        if self.use_depth:
            depth_frame = self._render_head_depth()
            self.depth_frame_history.append(depth_frame)
            depth_image = self._build_depth_stack_from_history()
        else:
            depth_image = self._neutral_depth_stack
        return PolicyFrame.from_raw_terms(
            base_ang_vel=base_ang_vel_body,
            projected_gravity=projected_gravity,
            velocity_commands=self._corrected_command(),
            joint_pos_rel=joint_pos,
            joint_vel_rel=joint_vel,
            last_action=self.last_action,
            depth_image=depth_image,
        )

    def _apply_position_targets(self) -> None:
        self.data.ctrl[self.actuator_ids] = self.target_qpos.astype(np.float64)

    def _configure_viewer_camera(self, viewer) -> None:
        viewer.cam.lookat[:] = np.asarray(self.data.qpos[0:3], dtype=np.float64)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

    def _log_status(self) -> None:
        q = self.data.qpos[3:7]
        print(
            f"step={self.step_count:05d} base_z={self.data.qpos[2]:.4f} "
            f"ncon={self.data.ncon:02d} "
            f"quat=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}] "
            f"action=[{self.last_action.min():.3f}, {self.last_action.max():.3f}]"
        )

    # ── Step / run ─────────────────────────────────────────────────────────

    def step(self) -> None:
        if self.stand_only:
            action = np.zeros(29, dtype=np.float32)
            desired = NOMINAL_JOINT_POS.copy()
        else:
            frame = self._build_policy_frame()
            self.history.append(frame)
            if self.step_count < self.startup_hold_steps:
                action = np.zeros(29, dtype=np.float32)
                desired = NOMINAL_JOINT_POS.copy()
            else:
                action = self.policy.act(self.history)
                if self.action_clip is not None:
                    action = np.clip(action, -self.action_clip, self.action_clip)
                desired = self.policy.action_to_joint_targets(
                    action, nominal_joint_pos=NOMINAL_JOINT_POS
                )

        self.last_action = action.astype(np.float32)
        self.target_qpos = (
            (1.0 - self.target_smoothing) * self.target_qpos
            + self.target_smoothing * desired
        ).astype(np.float32)
        for _ in range(self.control_steps):
            self._apply_position_targets()
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if self.log_interval and self.step_count % self.log_interval == 0:
            self._log_status()

    def run(self, steps: int) -> None:
        if self.passive:
            if not os.environ.get("DISPLAY"):
                raise RuntimeError(
                    "Passive viewer requires DISPLAY. Use --headless."
                )
            from mujoco import viewer as mujoco_viewer

            with mujoco_viewer.launch_passive(self.model, self.data) as viewer:
                self._configure_viewer_camera(viewer)
                while viewer.is_running() and steps > 0:
                    start = time.time()
                    self.step()
                    viewer.cam.lookat[:] = np.asarray(
                        self.data.qpos[0:3], dtype=np.float64
                    )
                    viewer.sync()
                    steps -= 1
                    if self.real_time:
                        elapsed = time.time() - start
                        remaining = max(CONTROL_DT - elapsed, 0.0)
                        if remaining > 0.0:
                            time.sleep(remaining)
            return

        for _ in range(steps):
            start = time.time()
            self.step()
            if self.real_time:
                elapsed = time.time() - start
                remaining = max(CONTROL_DT - elapsed, 0.0)
                if remaining > 0.0:
                    time.sleep(remaining)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the exported parkour ONNX policy in a pure MuJoCo loop."
    )
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--xml-path", type=Path, default=XML_PATH)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--command-x", type=float, default=float(DEFAULT_COMMAND[0]))
    p.add_argument("--command-y", type=float, default=float(DEFAULT_COMMAND[1]))
    p.add_argument("--command-yaw", type=float, default=float(DEFAULT_COMMAND[2]))
    p.add_argument("--action-clip", type=float, default=DEFAULT_ACTION_CLIP)
    p.add_argument(
        "--startup-hold-steps", type=int, default=DEFAULT_STARTUP_HOLD_STEPS
    )
    p.add_argument(
        "--target-smoothing", type=float, default=DEFAULT_TARGET_SMOOTHING
    )
    p.add_argument("--stand-only", action="store_true")
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument(
        "--use-depth",
        action="store_true",
        help="Use rendered depth instead of neutral depth",
    )
    p.add_argument(
        "--terrain",
        type=str,
        default="flat",
        choices=["flat", "stairs"],
        help="Terrain type: flat (default) or stairs",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-real-time", action="store_true")
    p.add_argument(
        "--yaw-correction-gain",
        type=float,
        default=0.5,
        help="P-gain for heading correction (0 to disable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    command = np.asarray(
        [args.command_x, args.command_y, args.command_yaw], dtype=np.float32
    )
    runner = G1MujocoRunner(
        model_dir=args.model_dir,
        xml_path=args.xml_path,
        command=command,
        passive=not args.headless,
        real_time=not args.no_real_time,
        action_clip=args.action_clip,
        startup_hold_steps=args.startup_hold_steps,
        target_smoothing=args.target_smoothing,
        stand_only=args.stand_only,
        log_interval=args.log_interval,
        use_depth=args.use_depth,
        terrain=args.terrain,
        yaw_correction_gain=args.yaw_correction_gain,
    )
    try:
        runner.run(args.steps)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
