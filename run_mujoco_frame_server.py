#!/usr/bin/env python3
"""MuJoCo stereo frame server for StereoSpatial / FFS integration.

Runs the G1 humanoid simulation (with or without GUI), renders stereo camera
images, and serves them over a TCP socket to the FFS perception service.

Architecture:
    MuJoCo (this script)  ←TCP socket→  FFS camera_mujoco.py driver
                                              ↓
                                         FFS perception pipeline
                                              ↓
                                         WebSocket :8765

Wire protocol (little-endian):
  Server → Client (per frame):
    [4B] left_size   (uint32, JPEG bytes)
    [4B] right_size  (uint32, JPEG bytes)
    [4B] meta_size   (uint32, JSON metadata bytes)
    [left_size B]  left JPEG
    [right_size B] right JPEG
    [meta_size B]  JSON: {"x","y","z","yaw","nav_state","step"}

  Client → Server (newline-delimited JSON commands):
    {"cmd":"nav_goal","x":1.0,"y":2.0,"z":0.0}
    {"cmd":"nav_cancel"}

Usage:
    cd ~/ProjectInstinct/mujoco/InstinctMujoco
    conda run -n instinct python run_mujoco_frame_server.py --port 9876
    conda run -n instinct python run_mujoco_frame_server.py --port 9876 --headless
"""
from __future__ import annotations

import argparse
import json
import logging
import select
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from robot_config import MODEL_DIR, XML_PATH
from run_parkour_mujoco import G1MujocoRunner
from scene_builder import CONTROL_DT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stereo camera parameters (mirrors scene_builder.py)
# --------------------------------------------------------------------------
_STEREO_WIDTH = 1280
_STEREO_HEIGHT = 720
_JPEG_QUALITY = 80


# --------------------------------------------------------------------------
# NavController (same as run_goto_demo.py)
# --------------------------------------------------------------------------

class NavController:
    """Simple P-controller for point-to-point ground navigation."""

    def __init__(
        self,
        arrive_dist: float = 0.3,
        max_forward: float = 0.5,
        kp_yaw: float = 0.8,
        turn_thresh: float = 0.5,
        smoothing: float = 0.15,
    ):
        self.arrive_dist = arrive_dist
        self.max_forward = max_forward
        self.kp_yaw = kp_yaw
        self.turn_thresh = turn_thresh
        self.smoothing = smoothing
        self.target: tuple[float, float] | None = None
        self.target_z: float = 0.0
        self._prev_cmd = np.zeros(3, dtype=np.float32)
        self._arrived = False

    def set(self, x: float, y: float, z: float = 0.0) -> None:
        self.target = (x, y)
        self.target_z = z
        self._arrived = False
        logger.info("Nav goal set: (%.2f, %.2f, %.2f)", x, y, z)

    def clear(self) -> None:
        self.target = None
        self._prev_cmd[:] = 0.0
        self._arrived = False
        logger.info("Nav goal cleared")

    @property
    def state(self) -> str:
        if self._arrived:
            return "arrived"
        if self.target is not None:
            return "moving"
        return "idle"

    def command(self, rx: float, ry: float, ryaw: float) -> np.ndarray:
        if self.target is None:
            self._prev_cmd *= (1.0 - self.smoothing)
            return self._prev_cmd.copy()

        dx = self.target[0] - rx
        dy = self.target[1] - ry
        dist = float(np.hypot(dx, dy))

        if dist < self.arrive_dist:
            self.target = None
            self._prev_cmd[:] = 0.0
            self._arrived = True
            logger.info("  ✓ Arrived!")
            return np.zeros(3, dtype=np.float32)

        heading = float(np.arctan2(dy, dx))
        err = (heading - ryaw + np.pi) % (2 * np.pi) - np.pi
        yaw_cmd = float(np.clip(err * self.kp_yaw, -0.6, 0.6))

        if abs(err) > self.turn_thresh:
            fwd_cmd = 0.1
        else:
            speed = self.max_forward * min(1.0, abs(err) / self.turn_thresh)
            fwd_cmd = float(np.clip(
                self.max_forward - speed * 0.5, 0.15, self.max_forward,
            ))

        raw = np.array([fwd_cmd, 0.0, yaw_cmd], dtype=np.float32)
        alpha = self.smoothing
        self._prev_cmd = (1.0 - alpha) * self._prev_cmd + alpha * raw
        self._arrived = False
        return self._prev_cmd.copy()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get_yaw(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# --------------------------------------------------------------------------
# Frame server
# --------------------------------------------------------------------------

class MujocoFrameServer:
    """TCP server that pushes stereo frames to a single FFS client."""

    def __init__(self, port: int, headless: bool, terrain: str, fps: float):
        self._port = port
        self._headless = headless
        self._terrain = terrain
        self._target_dt = 1.0 / fps
        self._nav = NavController()
        self._lock = threading.Lock()
        self._client_sock: socket.socket | None = None
        self._running = False

    def run(self) -> None:
        if not self._headless:
            self._run_gui()
        else:
            self._run_headless()

    # ------------------------------------------------------------------
    # GUI mode: custom GLFW window with double-click navigation
    # ------------------------------------------------------------------
    def _run_gui(self) -> None:
        import glfw

        logger.info("Initializing G1MujocoRunner (headless=False, terrain=%s)", self._terrain)
        runner = G1MujocoRunner(
            model_dir=MODEL_DIR,
            xml_path=XML_PATH,
            command=np.zeros(3, dtype=np.float32),
            passive=False,
            real_time=False,
            action_clip=100.0,
            startup_hold_steps=0,
            target_smoothing=1.0,
            stand_only=False,
            log_interval=0,
            use_depth=False,
            terrain=self._terrain,
            yaw_correction_gain=0.0,
        )
        model = runner.model
        data = runner.data

        # Camera IDs for stereo
        cam_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left")
        cam_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_right")
        if cam_left_id < 0 or cam_right_id < 0:
            raise RuntimeError("cam_left / cam_right not found in MuJoCo model")

        # Set offscreen buffer size for stereo frame capture
        model.vis.global_.offwidth = _STEREO_WIDTH
        model.vis.global_.offheight = _STEREO_HEIGHT

        # GLFW window (no side panels)
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")
        window = glfw.create_window(1280, 720, "MuJoCo Frame Server", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # MuJoCo visualisation
        scn = mujoco.MjvScene(model, maxgeom=10000)
        ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        cam = mujoco.MjvCamera()
        vopt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()

        # Offscreen scene for stereo frame capture (reuses same ctx)
        offscreen_scn = mujoco.MjvScene(model, maxgeom=10000)

        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = data.qpos[:3]
        cam.distance = 5.0
        cam.azimuth = 135.0
        cam.elevation = -20.0

        # Stereo camera overlays
        stereo_cam_left = mujoco.MjvCamera()
        stereo_cam_left.type = mujoco.mjtCamera.mjCAMERA_FIXED
        stereo_cam_left.fixedcamid = cam_left_id
        stereo_cam_right = mujoco.MjvCamera()
        stereo_cam_right.type = mujoco.mjtCamera.mjCAMERA_FIXED
        stereo_cam_right.fixedcamid = cam_right_id
        stereo_scn = mujoco.MjvScene(model, maxgeom=10000)
        STEREO_W, STEREO_H = 320, 180

        # Input state
        btn_left = [False]
        btn_right = [False]
        btn_mid = [False]
        last_mx = [0.0]
        last_my = [0.0]
        last_click_time = [0.0]

        # Target marker helpers
        _EYE3 = np.eye(3, dtype=np.float64).ravel()
        _RGBA_DISC = np.array([0.9, 0.15, 0.15, 0.7], dtype=np.float32)
        _RGBA_PIN = np.array([0.9, 0.15, 0.15, 0.5], dtype=np.float32)

        def _add_target_marker(tx, ty, tz):
            if scn.ngeom >= scn.maxgeom - 2:
                return
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([0.15, 0.003, 0.0]), np.array([tx, ty, tz + 0.004]),
                _EYE3, _RGBA_DISC,
            )
            scn.ngeom += 1
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([0.012, 0.25, 0.0]), np.array([tx, ty, tz + 0.27]),
                _EYE3, _RGBA_PIN,
            )
            scn.ngeom += 1

        def _pick_target(cx, cy):
            ww, wh = glfw.get_window_size(window)
            fw, fh = glfw.get_framebuffer_size(window)
            if ww == 0 or wh == 0:
                return
            relx = cx / ww
            rely = 1.0 - cy / wh
            aspect = fw / fh
            selpnt = np.zeros(3, dtype=np.float64)
            geomid = np.array([-1], dtype=np.int32)
            flexid = np.array([-1], dtype=np.int32)
            skinid = np.array([-1], dtype=np.int32)
            body = mujoco.mjv_select(
                model, data, vopt, aspect, relx, rely,
                scn, selpnt, geomid, flexid, skinid,
            )
            if body >= 0:
                with self._lock:
                    self._nav.set(float(selpnt[0]), float(selpnt[1]), float(selpnt[2]))

        def on_mouse_button(win, button, act, mods):
            btn_left[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            btn_right[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            btn_mid[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            mx, my = glfw.get_cursor_pos(win)
            last_mx[0], last_my[0] = mx, my
            if button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS:
                now = time.time()
                if now - last_click_time[0] < 0.3:
                    _pick_target(mx, my)
                last_click_time[0] = now

        def on_cursor_pos(win, xpos, ypos):
            dx = xpos - last_mx[0]
            dy = ypos - last_my[0]
            last_mx[0], last_my[0] = xpos, ypos
            if not (btn_left[0] or btn_right[0] or btn_mid[0]):
                return
            _, wh = glfw.get_window_size(win)
            if wh == 0:
                return
            shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            if btn_right[0]:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif btn_left[0]:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            mujoco.mjv_moveCamera(model, action, dx / wh, dy / wh, scn, cam)

        def on_scroll(win, xoff, yoff):
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff, scn, cam)

        def on_key(win, key, scancode, act, mods):
            if act != glfw.PRESS:
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_BACKSPACE:
                with self._lock:
                    self._nav.clear()

        glfw.set_mouse_button_callback(window, on_mouse_button)
        glfw.set_cursor_pos_callback(window, on_cursor_pos)
        glfw.set_scroll_callback(window, on_scroll)
        glfw.set_key_callback(window, on_key)

        # TCP server
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", self._port))
        server_sock.listen(1)
        server_sock.setblocking(False)
        logger.info("Frame server listening on tcp://0.0.0.0:%d", self._port)

        self._running = True
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]

        logger.info("GUI ready — double-click to set target, Backspace to cancel, Esc to quit")

        try:
            while not glfw.window_should_close(window):
                t0 = time.perf_counter()

                # Accept new client (non-blocking)
                self._accept_client(server_sock)
                # Process incoming commands from client
                self._process_commands()

                # Update nav → runner command
                rx, ry = float(data.qpos[0]), float(data.qpos[1])
                ryaw = _get_yaw(data.qpos)
                with self._lock:
                    runner.command[:] = self._nav.command(rx, ry, ryaw)

                # Step simulation
                runner.step()

                # Smooth camera follow
                cam.lookat[:] = 0.9 * cam.lookat + 0.1 * np.asarray(data.qpos[:3])

                # Build scene
                mujoco.mjv_updateScene(
                    model, data, vopt, pert, cam,
                    mujoco.mjtCatBit.mjCAT_ALL, scn,
                )

                # Draw target marker
                with self._lock:
                    if self._nav.target is not None:
                        _add_target_marker(self._nav.target[0], self._nav.target[1], self._nav.target_z)

                # Render main viewport
                viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
                mujoco.mjr_render(viewport, scn, ctx)

                # Stereo camera overlays (top-right)
                fw, fh = glfw.get_framebuffer_size(window)
                pad = 8
                left_vp = mujoco.MjrRect(fw - 2 * STEREO_W - 2 * pad, fh - STEREO_H - pad, STEREO_W, STEREO_H)
                right_vp = mujoco.MjrRect(fw - STEREO_W - pad, fh - STEREO_H - pad, STEREO_W, STEREO_H)
                for stereo_cam, svp, label in (
                    (stereo_cam_left, left_vp, "Left"),
                    (stereo_cam_right, right_vp, "Right"),
                ):
                    mujoco.mjv_updateScene(model, data, vopt, pert, stereo_cam, mujoco.mjtCatBit.mjCAT_ALL, stereo_scn)
                    mujoco.mjr_render(svp, stereo_scn, ctx)
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, svp, label, "", ctx)

                # Status overlay
                with self._lock:
                    if self._nav.target is not None:
                        dist = np.hypot(self._nav.target[0] - rx, self._nav.target[1] - ry)
                        status = f"Target: ({self._nav.target[0]:.1f}, {self._nav.target[1]:.1f})  dist={dist:.1f}m"
                    else:
                        status = "Double-click to set target"
                client_status = "Client: connected" if self._client_sock else "Client: waiting..."
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    viewport, status + "\n" + client_status, "", ctx,
                )

                glfw.swap_buffers(window)
                glfw.poll_events()

                # Send stereo frame to TCP client
                if self._client_sock is not None:
                    offscreen_vp = mujoco.MjrRect(0, 0, _STEREO_WIDTH, _STEREO_HEIGHT)

                    # Render left camera offscreen
                    stereo_cam_left_off = mujoco.MjvCamera()
                    stereo_cam_left_off.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    stereo_cam_left_off.fixedcamid = cam_left_id
                    mujoco.mjv_updateScene(model, data, vopt, pert, stereo_cam_left_off, mujoco.mjtCatBit.mjCAT_ALL, offscreen_scn)
                    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
                    mujoco.mjr_render(offscreen_vp, offscreen_scn, ctx)
                    left_rgb = np.empty((_STEREO_HEIGHT, _STEREO_WIDTH, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(left_rgb, None, offscreen_vp, ctx)
                    left_rgb = np.flipud(left_rgb)

                    # Render right camera offscreen
                    stereo_cam_right_off = mujoco.MjvCamera()
                    stereo_cam_right_off.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    stereo_cam_right_off.fixedcamid = cam_right_id
                    mujoco.mjv_updateScene(model, data, vopt, pert, stereo_cam_right_off, mujoco.mjtCatBit.mjCAT_ALL, offscreen_scn)
                    mujoco.mjr_render(offscreen_vp, offscreen_scn, ctx)
                    right_rgb = np.empty((_STEREO_HEIGHT, _STEREO_WIDTH, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(right_rgb, None, offscreen_vp, ctx)
                    right_rgb = np.flipud(right_rgb)

                    # Switch back to window buffer for next GUI frame
                    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)

                    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
                    _, left_jpeg = cv2.imencode(".jpg", left_bgr, encode_params)
                    _, right_jpeg = cv2.imencode(".jpg", right_bgr, encode_params)

                    meta = json.dumps({
                        "x": float(data.qpos[0]),
                        "y": float(data.qpos[1]),
                        "z": float(data.qpos[2]),
                        "yaw": ryaw,
                        "nav_state": self._nav.state,
                        "step": runner.step_count,
                    }).encode()
                    self._send_frame(left_jpeg.tobytes(), right_jpeg.tobytes(), meta)

                # Real-time throttle
                elapsed = time.perf_counter() - t0
                remaining = CONTROL_DT - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            if self._client_sock:
                self._client_sock.close()
            server_sock.close()
            ctx.free()
            glfw.terminate()
            runner.close()

    # ------------------------------------------------------------------
    # Headless mode: no GUI, just TCP frame serving
    # ------------------------------------------------------------------
    def _run_headless(self) -> None:
        logger.info("Initializing G1MujocoRunner (headless=True, terrain=%s)", self._terrain)
        runner = G1MujocoRunner(
            model_dir=MODEL_DIR,
            xml_path=XML_PATH,
            command=np.zeros(3, dtype=np.float32),
            passive=False,
            real_time=False,
            action_clip=100.0,
            startup_hold_steps=0,
            target_smoothing=1.0,
            stand_only=False,
            log_interval=0,
            use_depth=False,
            terrain=self._terrain,
            yaw_correction_gain=0.0,
        )
        model = runner.model
        data = runner.data

        cam_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_left")
        cam_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_right")
        if cam_left_id < 0 or cam_right_id < 0:
            raise RuntimeError("cam_left / cam_right not found in MuJoCo model")

        renderer_left = mujoco.Renderer(model, height=_STEREO_HEIGHT, width=_STEREO_WIDTH)
        renderer_right = mujoco.Renderer(model, height=_STEREO_HEIGHT, width=_STEREO_WIDTH)

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", self._port))
        server_sock.listen(1)
        server_sock.setblocking(False)
        logger.info("Frame server listening on tcp://0.0.0.0:%d", self._port)

        self._running = True
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]

        try:
            while self._running:
                t0 = time.perf_counter()

                self._accept_client(server_sock)
                self._process_commands()

                rx, ry = float(data.qpos[0]), float(data.qpos[1])
                ryaw = _get_yaw(data.qpos)
                with self._lock:
                    runner.command[:] = self._nav.command(rx, ry, ryaw)

                runner.step()

                if self._client_sock is not None:
                    renderer_left.update_scene(data, camera=cam_left_id)
                    left_rgb = renderer_left.render()
                    renderer_right.update_scene(data, camera=cam_right_id)
                    right_rgb = renderer_right.render()

                    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
                    _, left_jpeg = cv2.imencode(".jpg", left_bgr, encode_params)
                    _, right_jpeg = cv2.imencode(".jpg", right_bgr, encode_params)

                    meta = json.dumps({
                        "x": float(data.qpos[0]),
                        "y": float(data.qpos[1]),
                        "z": float(data.qpos[2]),
                        "yaw": ryaw,
                        "nav_state": self._nav.state,
                        "step": runner.step_count,
                    }).encode()
                    self._send_frame(left_jpeg.tobytes(), right_jpeg.tobytes(), meta)

                elapsed = time.perf_counter() - t0
                sleep_time = self._target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            if self._client_sock:
                self._client_sock.close()
            server_sock.close()
            renderer_left.close()
            renderer_right.close()
            runner.close()

    def _accept_client(self, server_sock: socket.socket) -> None:
        """Accept a new client connection (replaces existing)."""
        try:
            readable, _, _ = select.select([server_sock], [], [], 0)
            if readable:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self._client_sock is not None:
                    logger.info("Replacing existing client")
                    try:
                        self._client_sock.close()
                    except OSError:
                        pass
                self._client_sock = conn
                self._cmd_buffer = b""
                logger.info("FFS client connected from %s:%d", addr[0], addr[1])
        except OSError:
            pass

    def _process_commands(self) -> None:
        """Read and handle incoming commands from the client (non-blocking)."""
        if self._client_sock is None:
            return
        try:
            readable, _, _ = select.select([self._client_sock], [], [], 0)
            if not readable:
                return
            data = self._client_sock.recv(4096)
            if not data:
                logger.info("Client disconnected")
                self._client_sock.close()
                self._client_sock = None
                return
            self._cmd_buffer += data
            # Process newline-delimited commands
            while b"\n" in self._cmd_buffer:
                line, self._cmd_buffer = self._cmd_buffer.split(b"\n", 1)
                self._handle_command(line)
        except (OSError, ConnectionError):
            self._client_sock = None

    def _handle_command(self, line: bytes) -> None:
        """Parse and execute a single command."""
        try:
            cmd = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        with self._lock:
            if cmd.get("cmd") == "nav_goal":
                self._nav.set(
                    float(cmd.get("x", 0)),
                    float(cmd.get("y", 0)),
                    float(cmd.get("z", 0)),
                )
            elif cmd.get("cmd") == "nav_cancel":
                self._nav.clear()

    def _send_frame(self, left_jpeg: bytes, right_jpeg: bytes, meta: bytes) -> None:
        """Send a frame to the connected client."""
        if self._client_sock is None:
            return
        header = struct.pack("<III", len(left_jpeg), len(right_jpeg), len(meta))
        try:
            self._client_sock.sendall(header + left_jpeg + right_jpeg + meta)
        except (OSError, BrokenPipeError):
            logger.info("Client send failed, disconnecting")
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo stereo frame server for FFS")
    parser.add_argument("--port", type=int, default=9876, help="TCP port to listen on")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--terrain", default="flat", choices=["flat", "stairs", "pyramid"])
    parser.add_argument("--fps", type=float, default=25.0, help="Target frame rate for stereo rendering")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG compression quality (0-100)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global _JPEG_QUALITY
    _JPEG_QUALITY = args.jpeg_quality

    server = MujocoFrameServer(
        port=args.port,
        headless=args.headless,
        terrain=args.terrain,
        fps=args.fps,
    )
    server.run()


if __name__ == "__main__":
    main()
