"""AR Hand-Controlled 3D Object System — Test 3 entry point.

Pipeline (blueprint Sec. 2):
  Live Camera (720p, mirrored) -> MediaPipe Hands (x2) -> Left=anchor / Right=trigger
  -> Bounding-Box Transform + Transition Controller -> 3D Render -> Composited AR.

Gestures (trigger hand):
  INDEX only  -> Blooming Lilies    FIST        -> Red Dragon
  L (thumb+index 90deg) -> Blue Morpho         PEACE (index+middle) -> Cosmic Bonsai
  OPEN PALM   -> particle dissolve  BOTH FISTS  -> scene clear

Keys: ESC/Q quit | F fullscreen | C switch camera | R/X clear scene
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from config.settings import SystemSettings
from core.camera import CameraGrabber
from core.state_machine import GestureStateMachine, SceneState, MODEL_LABEL
from vision.hand_tracker import DualHandTracker
from vision.gestures import Gesture
from math3d.transforms import transform_vertices
from models.procedural import BUILDERS
from simulation.particles import ParticleDissolver
from rendering.renderer import ARRenderer


def gesture_name(g: Gesture) -> str:
    return {
        Gesture.INDEX: "INDEX->Lilies",
        Gesture.FIST: "FIST->Dragon",
        Gesture.L_SHAPE: "L->Morpho",
        Gesture.PEACE: "PEACE->Bonsai",
        Gesture.OPEN_PALM: "PALM->Dissolve",
        Gesture.UNKNOWN: "--",
    }[g]


def main() -> None:
    s = SystemSettings()
    cam = CameraGrabber(0, s.display.cam_width, s.display.cam_height)
    cam.start()
    tracker = DualHandTracker(ext_ratio=s.gestures.finger_extension_ratio,
                              thumb_ratio=s.gestures.thumb_extension_ratio)
    sm = GestureStateMachine(s.gestures.debounce_frames,
                             s.gestures.dissolve_trigger_frames)
    dissolver = ParticleDissolver(s.particles.max_particles, s.particles.dissolve_time,
                                  s.particles.radial_speed, s.particles.curl_strength,
                                  s.particles.curl_freq, s.particles.damping_gamma,
                                  s.particles.gravity)
    rnd = ARRenderer(s)

    win = s.display.window_name
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    fullscreen = False

    print("=" * 62)
    print("  AR HAND-CONTROLLED 3D OBJECTS — Test 3")
    print("=" * 62)
    print("  INDEX (point)  -> Blooming Lilies")
    print("  FIST           -> Red Dragon")
    print("  L (thumb+idx)  -> Blue Morpho Butterfly")
    print("  PEACE (2 fing) -> Cosmic Bonsai")
    print("  OPEN PALM      -> Particle Dissolve")
    print("  BOTH FISTS     -> Scene Clear")
    print("  Left hand anchors/orients, right hand triggers.")
    print("  [C] camera  [F] fullscreen  [R/X] clear  [Q/ESC] quit")
    print("=" * 62)

    fps = 60.0
    t0 = time.perf_counter()
    prev = t0
    last_world = np.zeros((0, 3))
    last_colors = np.zeros((0, 3), dtype=np.uint8)
    fallback_ax, fallback_ay = None, None
    ax, ay = 640.0, 300.0

    try:
        while True:
            now = time.perf_counter()
            dt = min(now - prev, 0.05)
            prev = now
            t = now - t0

            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue
            if s.display.mirror:
                frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            # ── vision ──
            dual = tracker.process(frame)
            trig = dual.trigger_gesture
            # ── state machine ──
            ev = sm.update(trig, dual.dual_fists)
            if ev is not None:
                if ev.kind == "dissolve_start":
                    if last_world.shape[0] > 0:
                        dissolver.spawn_from_vertices(last_world, last_colors)
                        last_world = np.zeros((0, 3))
                    else:
                        sm.finish_dissolve()
                elif ev.kind == "clear":
                    dissolver.clear()
                    last_world = np.zeros((0, 3))

            # ── anchor ──
            if dual.has_anchor:
                pose = dual.anchor.pose
                cx, cy, cz = (float(pose.center_3d[0]), float(pose.center_3d[1]),
                              float(pose.center_3d[2]))
                cx = min(0.98, max(0.02, cx))
                cy = min(0.98, max(0.02, cy))
                ax, ay = rnd.anchor_pixel(cx, cy, W, H)
                fallback_ax, fallback_ay = ax, ay
                R = pose.rotation
                zoff = rnd.depth_from_z(cz)
                C = np.array([0.0, 0.0, zoff])
                anchor_ok = True
            else:
                R = np.eye(3)
                C = np.array([0.0, 0.0, 0.0])
                if fallback_ax is None:
                    ax, ay = W * 0.5, H * 0.42
                else:
                    ax, ay = fallback_ax, fallback_ay
                anchor_ok = False

            dissolving = (sm.state == SceneState.DISSOLVING)

            # ── transition progress ──
            if dissolving:
                dissolver.update(dt)
                if not dissolver.active:
                    sm.finish_dissolve()
                    dissolving = False

            # ── render 3D ──
            active = sm.active_model if not dissolving else None
            if anchor_ok and (active or dissolving):
                rnd.draw_box(frame, transform_vertices(rnd._box_local, R, 1.0, C), ax, ay)
            if active and dual.has_anchor:
                builder = BUILDERS.get(active)
                if builder is not None:
                    verts, edges, colors = builder(t)
                    world = transform_vertices(verts, R, rnd.model_scale, C)
                    last_world = world.copy()
                    last_colors = np.asarray(colors, dtype=np.uint8).copy()
                    rnd.draw_mesh(frame, world, edges, colors, ax, ay)
            if dissolver.active:
                rnd.draw_particles(frame, dissolver.pos, dissolver.color,
                                   dissolver.alphas(), ax, ay)

            # ── HUD ──
            n_hands = len(dual.hands)
            model_lbl = MODEL_LABEL.get(active, "--") if active else (
                f"DISSOLVING->{MODEL_LABEL.get(sm.pending, '...')}" if dissolving and sm.pending
                else ("DISSOLVING..." if dissolving else "--"))
            lines = [
                "AR HAND 3D OBJECTS",
                f"FPS {fps:4.0f}  Hands {n_hands}  Anchor {'L' if anchor_ok else '--'}",
                f"Trigger {gesture_name(trig)}",
                f"Model {model_lbl}",
                f"Parts {dissolver.count}" + ("  DUAL-FIST CLEAR!" if dual.dual_fists else ""),
                "[C]am [F]ull [R]eset [Q]uit",
            ]
            rnd.draw_hud(frame, lines, accent_idx=(0,))
            rnd.draw_control_bar(frame, active, gesture_name(trig), dissolving)

            cv2.imshow(win, frame)
            inst = 1.0 / max(dt, 1e-6)
            fps = 0.92 * fps + 0.08 * min(inst, 240.0)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key in (ord("c"), ord("C")):
                idx = cam.switch()
                print(f"[Controls] Camera -> index {idx}")
            elif key in (ord("r"), ord("R"), ord("x"), ord("X")):
                sm.force_clear()
                dissolver.clear()
                last_world = np.zeros((0, 3))
                print("[Controls] Scene cleared.")
    finally:
        cam.stop()
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
