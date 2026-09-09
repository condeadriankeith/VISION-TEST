"""Main application entry point for Hand Gesture 3D Hologram.

High-performance real-time loop utilizing non-blocking threaded camera acquisition,
downsampled neural network inference, temporal jitter filtering, and localized
ROI rendering for ultra-low latency.
"""

import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from camera import CameraManager, ThreadedCamera
import config
from cube_renderer import CubeHologramRenderer
from hand_tracker import HandPose, HandTracker


class HologramApplication:
    """Master controller application managing camera, state, and rendering loop."""

    def __init__(self, camera_index: Optional[int] = None) -> None:
        """Initialize tracker, 3D renderer, camera, and application state."""
        print("[HologramApp] Initializing Hand Gesture 3D Hologram system...")

        # Initialize Hand Tracker & 3D Renderer
        self.tracker: HandTracker = HandTracker()
        self.renderer: CubeHologramRenderer = CubeHologramRenderer()

        # Application state
        self.show_skeleton: bool = False

        # Frame rate calculation
        self.prev_frame_time: float = time.time()
        self.current_fps: float = 0.0

        # Initialize Threaded Camera Stream (non-blocking, zero-latency buffer)
        target_idx = config.CAMERA_INDEX if camera_index is None else camera_index
        print(f"[HologramApp] Connecting to camera {target_idx} (main camera)...")
        self.camera: ThreadedCamera = CameraManager.open_best_camera(preferred_index=target_idx)

    @property
    def current_cam_index(self) -> int:
        """Returns active camera index."""
        return self.camera.index

    def switch_camera(self) -> None:
        """Toggles between Camera 0 (main) and Camera 1 (secondary)."""
        next_idx = 1 if self.camera.index == 0 else 0
        print(f"[HologramApp] Switching camera: {self.camera.index} -> {next_idx}...")
        try:
            old_cam = self.camera
            self.camera = CameraManager.open_camera(next_idx)
            old_cam.release()
            print(f"[HologramApp] Switched to Camera {self.camera.index}.")
        except Exception as err:
            print(f"[HologramApp] Could not switch to camera {next_idx}: {err}")

    @property
    def current_material(self) -> config.GreyMaterial:
        """Returns the single default prismatic material."""
        return config.MATERIALS["Prismatic"]

    def _draw_hud(
        self,
        frame: np.ndarray,
        poses: List[HandPose],
    ) -> None:
        """Draws a clean, futuristic glassmorphism HUD with status and hotkeys.

        Optimized with slice-based alpha blending (no full-frame copies).
        """
        h, w = frame.shape[:2]
        material = self.current_material

        # 1. Top HUD Bar (Blend only the top slice, 40x faster than full frame copy)
        top_bar_height = 54
        top_slice = frame[0:top_bar_height, :]
        top_tint = np.full_like(top_slice, (18, 20, 26))
        cv2.addWeighted(top_slice, 0.25, top_tint, 0.75, 0, top_slice)
        cv2.line(frame, (0, top_bar_height), (w, top_bar_height), material.hud_accent, 1)

        # 2. Status Indicators & Dynamic Finger Interaction Feedback
        hand_detected = len(poses) > 0
        latest_hit = self.renderer.latest_contact

        if hand_detected:
            primary_pose = poses[0]
            if latest_hit is not None:
                gesture_text = f"INTERACTION: [{latest_hit.finger_name.upper()} HIT! IMPULSE: {int(latest_hit.impulse_mag)}]"
                gesture_color = (120, 240, 255)  # Glowing electric cyan/gold for impact
            elif primary_pose.is_open:
                gesture_text = f"PALM: OPEN  [TILT P:{int(primary_pose.pitch_deg):+2d}° R:{int(primary_pose.roll_deg):+2d}°]"
                gesture_color = (200, 240, 200)  # Clean soft green
            else:
                gesture_text = "PALM: CLOSED [PHYSICS RETRACTED]"
                gesture_color = (180, 185, 195)  # Matte slate
            hand_text = f"HAND: {primary_pose.handedness.upper()}"
        else:
            hand_text = "HAND: NONE"
            gesture_text = "PALM: WAITING FOR HAND"
            gesture_color = (140, 140, 140)

        # Text rendering on Top Bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, hand_text, (20, 34), font, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, gesture_text, (180, 34), font, 0.60, gesture_color, 2, cv2.LINE_AA)

        # Camera & Material indicators
        cam_str = f"CAM: {self.current_cam_index}"
        cv2.putText(frame, cam_str, (w - 530, 34), font, 0.55, (180, 200, 225), 1, cv2.LINE_AA)

        mat_str = f"MAT: {material.name}"
        cv2.putText(frame, mat_str, (w - 430, 34), font, 0.55, material.hud_accent, 1, cv2.LINE_AA)

        # FPS indicator
        fps_str = f"FPS: {int(self.current_fps)}"
        cv2.putText(frame, fps_str, (w - 110, 34), font, 0.60, (220, 230, 220), 2, cv2.LINE_AA)

        # 3. Bottom Controls Info Bar (Blend only the bottom slice)
        bottom_bar_y = h - 36
        bottom_slice = frame[bottom_bar_y:h, :]
        bottom_tint = np.full_like(bottom_slice, (10, 12, 16))
        cv2.addWeighted(bottom_slice, 0.30, bottom_tint, 0.70, 0, bottom_slice)

        skel_status = "VISIBLE" if self.show_skeleton else "INVISIBLE"
        instructions = (
            f"[V] Cam: {self.current_cam_index}  |  "
            f"[S] Skeleton: {skel_status}  |  "
            f"Prismatic 6-DOF Organic Physics & Palm Emergence Active  |  "
            f"[Q/Esc] Exit"
        )
        cv2.putText(
            frame,
            instructions,
            (20, h - 12),
            font,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        """High-performance camera acquisition and rendering loop."""
        print("[HologramApp] Application running.")
        print("[HologramApp] Hold your hand in front of the camera:")
        print("  - OPEN your palm: 3 holographic cubes spawn and hover above your palm.")
        print("  - CLOSE your palm: cubes shrink and vanish smoothly.")
        print("  - Press 'V' to switch between cameras.")
        print("  - Press 'S' to toggle virtual skeleton overlay.")
        print("  - Press 'Q' or 'Esc' to exit.")

        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)

        last_frame_id = -1
        cached_poses: List[HandPose] = []

        try:
            while True:
                current_time = time.time()
                dt = current_time - self.prev_frame_time
                self.prev_frame_time = current_time
                if dt > 0:
                    self.current_fps = 0.92 * self.current_fps + 0.08 * (1.0 / dt)

                # Instantly retrieve freshest camera frame from background thread
                success, frame, frame_id = self.camera.read_latest()
                if not success or frame is None:
                    time.sleep(0.002)
                    continue

                # Mirror frame horizontally so gestures feel natural like looking in a mirror
                frame = cv2.flip(frame, 1)

                # If this is a newly arrived camera frame, update hand detection
                if frame_id != last_frame_id:
                    last_frame_id = frame_id
                    timestamp_ms = int(current_time * 1000)
                    cached_poses = self.tracker.process_frame(frame, timestamp_ms)

                poses = cached_poses
                hand_detected = len(poses) > 0
                is_open = poses[0].is_open if hand_detected else False

                # Update 3D cube physics and animations
                self.renderer.update(
                    hand_detected=hand_detected,
                    is_open=is_open,
                    pose=poses[0] if poses else None,
                )

                # Render skeleton and 3D hovering cubes
                for pose in poses:
                    if self.show_skeleton:
                        self.tracker.draw_skeleton(
                            frame, pose, color=self.current_material.bevel_color
                        )

                    # Render solid smooth grey cubes with lighting and bevel highlights
                    self.renderer.render(frame, pose, self.current_material)

                # Draw optimized HUD
                self._draw_hud(frame, poses)

                # Display frame
                cv2.imshow(config.WINDOW_TITLE, frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 27 = Esc
                    print("[HologramApp] Exit requested by user.")
                    break
                elif key in [ord("v"), ord("V")]:
                    self.switch_camera()
                elif key in [ord("s"), ord("S")]:
                    self.show_skeleton = not self.show_skeleton
                    status = "VISIBLE" if self.show_skeleton else "INVISIBLE"
                    print(f"[HologramApp] Skeleton overlay: {status}")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release camera, windows, and resources safely."""
        print("[HologramApp] Cleaning up resources...")
        if hasattr(self, "camera"):
            self.camera.release()
        if hasattr(self, "tracker"):
            self.tracker.close()
        cv2.destroyAllWindows()
        print("[HologramApp] Shutdown complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hand Gesture 3D Hologram")
    parser.add_argument(
        "--camera", "-c", type=int, default=config.CAMERA_INDEX,
        help="Camera device index (default: 0 for main webcam, 1 for secondary)"
    )
    args = parser.parse_args()

    try:
        app = HologramApplication(camera_index=args.camera)
        app.run()
    except Exception as exc:
        print(f"[HologramApp] Error: {exc}", file=sys.stderr)
        sys.exit(1)
