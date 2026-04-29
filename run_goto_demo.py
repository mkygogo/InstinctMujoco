#!/usr/bin/env python3
"""Interactive go-to demo: double-click on terrain to set a navigation target.

The robot stands in place until you double-click on the ground or stairs.
It then walks to that point and stops, waiting for the next target.

Controls
--------
  Double-click     Set navigation target on terrain
  Backspace        Cancel current target
  Esc              Quit
  Left-drag        Orbit camera
  Right-drag       Pan camera
  Scroll           Zoom
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from robot_config import MODEL_DIR, XML_PATH
from run_parkour_mujoco import G1MujocoRunner
from scene_builder import CONTROL_DT


# ── Navigation controller ─────────────────────────────────────────────────


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

    def set(self, x: float, y: float, z: float = 0.0) -> None:
        self.target = (x, y)
        self.target_z = z

    def clear(self) -> None:
        self.target = None
        self._prev_cmd[:] = 0.0

    @property
    def active(self) -> bool:
        return self.target is not None

    def command(self, rx: float, ry: float, ryaw: float) -> np.ndarray:
        """Return (vx, vy, vyaw) velocity command with smoothing."""
        if self.target is None:
            self._prev_cmd *= (1.0 - self.smoothing)
            return self._prev_cmd.copy()

        dx = self.target[0] - rx
        dy = self.target[1] - ry
        dist = np.hypot(dx, dy)

        if dist < self.arrive_dist:
            self.target = None
            self._prev_cmd[:] = 0.0
            print("  ✓ Arrived!")
            return np.zeros(3, dtype=np.float32)

        heading = np.arctan2(dy, dx)
        err = (heading - ryaw + np.pi) % (2 * np.pi) - np.pi
        yaw_cmd = float(np.clip(err * self.kp_yaw, -0.6, 0.6))

        if abs(err) > self.turn_thresh:
            fwd_cmd = 0.1
        else:
            speed = self.max_forward * min(1.0, abs(err) / self.turn_thresh)
            fwd_cmd = float(np.clip(
                self.max_forward - speed * 0.5,
                0.15, self.max_forward,
            ))

        raw = np.array([fwd_cmd, 0.0, yaw_cmd], dtype=np.float32)
        # Exponential moving average for smooth command transitions
        alpha = self.smoothing
        self._prev_cmd = (1.0 - alpha) * self._prev_cmd + alpha * raw
        return self._prev_cmd.copy()


# ── Helpers ────────────────────────────────────────────────────────────────

_EYE3 = np.eye(3, dtype=np.float64).ravel()
_RGBA_DISC = np.array([0.9, 0.15, 0.15, 0.7], dtype=np.float32)
_RGBA_PIN = np.array([0.9, 0.15, 0.15, 0.5], dtype=np.float32)


def _get_yaw(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _add_target_marker(scn: mujoco.MjvScene, tx: float, ty: float, tz: float) -> None:
    """Draw a red pin at the target location."""
    if scn.ngeom >= scn.maxgeom - 2:
        return
    # Ground disc
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.15, 0.003, 0.0]),
        np.array([tx, ty, tz + 0.004]),
        _EYE3,
        _RGBA_DISC,
    )
    scn.ngeom += 1
    # Vertical capsule (pin)
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([0.012, 0.25, 0.0]),
        np.array([tx, ty, tz + 0.27]),
        _EYE3,
        _RGBA_PIN,
    )
    scn.ngeom += 1


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    if not os.environ.get("DISPLAY"):
        print("Error: this demo needs a display (DISPLAY not set).", file=sys.stderr)
        sys.exit(1)

    import glfw

    ap = argparse.ArgumentParser(description="G1 go-to navigation demo")
    ap.add_argument("--terrain", default="pyramid", choices=["flat", "stairs", "pyramid"])
    ap.add_argument("--use-depth", action="store_true", default=True)
    ap.add_argument("--no-depth", dest="use_depth", action="store_false")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--xml-path", type=Path, default=XML_PATH)
    args = ap.parse_args()

    # ── Simulation runner (reuse policy + physics from parkour runner) ──
    runner = G1MujocoRunner(
        model_dir=args.model_dir,
        xml_path=args.xml_path,
        command=np.zeros(3, dtype=np.float32),
        passive=False,
        real_time=False,
        action_clip=100.0,
        startup_hold_steps=0,
        target_smoothing=1.0,
        stand_only=False,
        log_interval=0,
        use_depth=args.use_depth,
        terrain=args.terrain,
        yaw_correction_gain=0.0,  # we drive yaw via nav controller
    )
    model, data = runner.model, runner.data
    nav = NavController()

    # ── GLFW window ──
    if not glfw.init():
        raise RuntimeError("Failed to initialise GLFW")
    window = glfw.create_window(1280, 720, "G1 Go-To Demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ── MuJoCo visualisation ──
    scn = mujoco.MjvScene(model, maxgeom=10000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.qpos[:3]
    cam.distance = 5.0
    cam.azimuth = 135.0
    cam.elevation = -20.0

    # ── Input state (captured by closures) ──
    btn_left = [False]
    btn_right = [False]
    btn_mid = [False]
    last_mx = [0.0]
    last_my = [0.0]
    last_click_time = [0.0]

    def _pick_target(cx: float, cy: float) -> None:
        """Raycast from cursor into the scene and set nav target."""
        ww, wh = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        if ww == 0 or wh == 0:
            return
        relx = cx / ww
        rely = 1.0 - cy / wh  # GLFW y is top-down, MuJoCo is bottom-up
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
            nav.set(float(selpnt[0]), float(selpnt[1]), float(selpnt[2]))
            print(f"  → Target: ({selpnt[0]:.2f}, {selpnt[1]:.2f})")

    def on_mouse_button(win, button, act, mods):
        btn_left[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        btn_right[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        btn_mid[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        mx, my = glfw.get_cursor_pos(win)
        last_mx[0], last_my[0] = mx, my

        # Double-click detection
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
        mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff, scn, cam,
        )

    def on_key(win, key, scancode, act, mods):
        if act != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_BACKSPACE:
            nav.clear()
            print("  → Target cancelled")

    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    print()
    print("  G1 Go-To Demo")
    print("  ─────────────────────────────────────")
    print("  Double-click on terrain → set target")
    print("  Backspace → cancel  |  Esc → quit")
    print()

    # ── Main loop ──
    try:
        while not glfw.window_should_close(window):
            t0 = time.time()

            # Compute navigation command and feed to runner
            rx, ry = float(data.qpos[0]), float(data.qpos[1])
            ryaw = _get_yaw(data.qpos)
            runner.command[:] = nav.command(rx, ry, ryaw)

            # Step physics + policy
            runner.step()

            # Smooth camera follow
            cam.lookat[:] = 0.9 * cam.lookat + 0.1 * np.asarray(data.qpos[:3])

            # Build visualisation scene
            mujoco.mjv_updateScene(
                model, data, vopt, pert, cam,
                mujoco.mjtCatBit.mjCAT_ALL, scn,
            )

            # Draw target marker (after updateScene so it appears on top)
            if nav.active:
                _add_target_marker(scn, nav.target[0], nav.target[1], nav.target_z)

            # Render
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjr_render(viewport, scn, ctx)

            # Status overlay
            if nav.active:
                dist = np.hypot(nav.target[0] - rx, nav.target[1] - ry)
                status = f"Target: ({nav.target[0]:.1f}, {nav.target[1]:.1f})  dist={dist:.1f}m"
            else:
                status = "Double-click to set target"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                status,
                "",
                ctx,
            )

            glfw.swap_buffers(window)
            glfw.poll_events()

            # Real-time throttle
            elapsed = time.time() - t0
            remaining = CONTROL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
        ctx.free()
        glfw.terminate()


if __name__ == "__main__":
    main()
