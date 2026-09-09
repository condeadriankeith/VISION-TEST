"""Hand detection, 3D virtual skeleton mapping, and gesture classification module.

Uses MediaPipe Tasks HandLandmarker to track 21 landmarks per hand in 3D,
computes palm orientation, and classifies Open vs Closed palm states with
temporal hysteresis filtering.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import os
import time
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np

import config

MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
LOCAL_MODEL_NAME: str = "hand_landmarker.task"


@dataclass
class LandmarkPoint:
    """3D point representing a single hand landmark."""
    x: float  # Normalized [0, 1] screen coordinates
    y: float  # Normalized [0, 1] screen coordinates
    z: float  # Relative depth
    px: int   # Pixel X coordinate
    py: int   # Pixel Y coordinate


@dataclass
class FingerTipCollider:
    """Identified fingertip with 3D position, smoothed velocity, and collider radius."""
    name: str              # "Thumb", "Index", "Middle", "Ring", "Pinky"
    tip_idx: int           # Landmark index (4, 8, 12, 16, 20)
    pos_3d: np.ndarray     # 3D coordinates [x, y, z] in camera/pixel space
    vel_3d: np.ndarray     # 3D velocity vector [vx, vy, vz] in px/sec
    radius: float          # Collision radius in pixels
    is_extended: bool      # Finger extension status


@dataclass
class HandPose:
    """Calculated hand pose, skeleton, and gesture state with 3D orientation."""
    handedness: str
    landmarks: List[LandmarkPoint]
    palm_center_px: Tuple[int, int]
    up_vector: Tuple[float, float]
    right_vector: Tuple[float, float]
    palm_scale: float
    is_open: bool
    open_confidence: float
    finger_states: List[bool]
    palm_normal_3d: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    palm_up_3d: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    palm_right_3d: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    palm_center_3d: Tuple[float, float, float] = (640.0, 360.0, 0.0)
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fingers: List[FingerTipCollider] = field(default_factory=list)


class HandTracker:
    """Encapsulates hand detection, skeleton tracking, and gesture recognition."""

    # Landmark indices for bones connections
    BONES: List[Tuple[int, int]] = [
        # Palm base
        (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index finger
        (5, 6), (6, 7), (7, 8),
        # Middle finger
        (9, 10), (10, 11), (11, 12),
        # Ring finger
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20),
    ]

    # Tip and PIP pairs for the 5 fingers (Thumb, Index, Middle, Ring, Pinky)
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [2, 6, 10, 14, 18]
    FINGER_MCPS = [1, 5, 9, 13, 17]
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the MediaPipe Hand Landmarker detector.

        Args:
            model_path: Optional path to the hand_landmarker.task model.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), LOCAL_MODEL_NAME)

        self._ensure_model_file(model_path)

        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Hysteresis smoothing buffers per hand (keyed by handedness)
        self._gesture_buffers: dict[str, deque[bool]] = {
            "Left": deque(maxlen=config.GESTURE_SMOOTHING_FRAMES),
            "Right": deque(maxlen=config.GESTURE_SMOOTHING_FRAMES),
        }

        # EMA temporal smoothing state (stores previous frame values per hand)
        self._prev_centers: dict[str, Tuple[float, float]] = {}
        self._prev_up_vectors: dict[str, Tuple[float, float]] = {}
        self._prev_scales: dict[str, float] = {}
        self._prev_normals: dict[str, Tuple[float, float, float]] = {}
        self._prev_centers_3d: dict[str, Tuple[float, float, float]] = {}

        # Fingertip kinematic velocity tracking state
        self._prev_finger_pos: dict[str, List[np.ndarray]] = {}
        self._prev_finger_vel: dict[str, List[np.ndarray]] = {}
        self._prev_finger_time: dict[str, float] = {}

    def _ensure_model_file(self, target_path: str) -> None:
        """Downloads the MediaPipe task model if not present locally."""
        if not os.path.exists(target_path):
            print(f"[HandTracker] Downloading model to {target_path}...")
            urllib.request.urlretrieve(MODEL_URL, target_path)
            print("[HandTracker] Model downloaded successfully.")

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> List[HandPose]:
        """Detect hands and extract virtual skeletons and gestures.

        Uses downsampled inference for high FPS and applies temporal smoothing
        to eliminate coordinate jitter.
        """
        height, width = frame_bgr.shape[:2]

        # Performance Optimization:
        # Resize to downscaled inference resolution (e.g. 640x360)
        # MediaPipe returns normalized [0, 1] coordinates, so precision is preserved
        inf_w = config.INFERENCE_WIDTH
        inf_h = config.INFERENCE_HEIGHT
        if width != inf_w or height != inf_h:
            small_bgr = cv2.resize(frame_bgr, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        poses: List[HandPose] = []

        if not result.hand_landmarks:
            # Clear smoothing state when hands leave frame
            self._prev_centers.clear()
            self._prev_up_vectors.clear()
            self._prev_scales.clear()
            self._prev_normals.clear()
            self._prev_centers_3d.clear()
            self._prev_finger_pos.clear()
            self._prev_finger_vel.clear()
            self._prev_finger_time.clear()
            return poses

        for i, raw_landmarks in enumerate(result.hand_landmarks):
            handedness = "Right"
            if i < len(result.handedness) and result.handedness[i]:
                handedness = result.handedness[i][0].category_name

            # Convert to LandmarkPoint list mapped to full output resolution
            points: List[LandmarkPoint] = []
            for lm in raw_landmarks:
                px = int(lm.x * width)
                py = int(lm.y * height)
                points.append(
                    LandmarkPoint(x=lm.x, y=lm.y, z=lm.z, px=px, py=py)
                )

            # 1. Compute Palm Center
            palm_indices = [0, 5, 9, 13, 17]
            raw_cx = sum(points[idx].px for idx in palm_indices) / len(palm_indices)
            raw_cy = sum(points[idx].py for idx in palm_indices) / len(palm_indices)

            # 2. Compute 2D & 3D Palm Orientation Vectors
            wrist = points[0]
            middle_mcp = points[9]
            index_mcp = points[5]
            pinky_mcp = points[17]

            dx = float(middle_mcp.px - wrist.px)
            dy = float(middle_mcp.py - wrist.py)
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                raw_up_x = dx / dist
                raw_up_y = dy / dist
            else:
                raw_up_x, raw_up_y = 0.0, -1.0

            # Compute true 3D Normal Vector from 3D Landmark vectors
            v_up_3d = np.array([
                middle_mcp.x - wrist.x,
                middle_mcp.y - wrist.y,
                (middle_mcp.z - wrist.z) * 1.5,
            ], dtype=np.float32)

            v_across_3d = np.array([
                index_mcp.x - pinky_mcp.x,
                index_mcp.y - pinky_mcp.y,
                (index_mcp.z - pinky_mcp.z) * 1.5,
            ], dtype=np.float32)

            # Cross-product yields palm normal (pointing outwards from palm face)
            if handedness == "Left":
                raw_normal = np.cross(v_up_3d, v_across_3d)
            else:
                raw_normal = np.cross(v_across_3d, v_up_3d)

            norm_len = float(np.linalg.norm(raw_normal))
            if norm_len > 1e-5:
                raw_normal = raw_normal / norm_len
            else:
                raw_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            # 3. Compute Palm Scale & Estimated Depth (Z)
            raw_scale = max(
                math.hypot(
                    float(index_mcp.px - pinky_mcp.px),
                    float(index_mcp.py - pinky_mcp.py),
                ),
                20.0,
            )
            # Baseline scale ~70px. Closer hand -> larger scale -> negative Z (closer to cam)
            dist_ratio = float(np.clip(raw_scale / 70.0, 0.5, 2.5))
            raw_cz = (1.0 - dist_ratio) * 180.0

            # 4. Temporal Smoothing (EMA filter removes sensor jitter)
            alpha = config.LANDMARK_SMOOTHING_ALPHA
            if handedness in self._prev_centers:
                prev_cx, prev_cy = self._prev_centers[handedness]
                cx = alpha * raw_cx + (1.0 - alpha) * prev_cx
                cy = alpha * raw_cy + (1.0 - alpha) * prev_cy

                prev_ux, prev_uy = self._prev_up_vectors[handedness]
                up_vx = alpha * raw_up_x + (1.0 - alpha) * prev_ux
                up_vy = alpha * raw_up_y + (1.0 - alpha) * prev_uy
                up_norm = math.hypot(up_vx, up_vy)
                if up_norm > 1e-4:
                    up_vx /= up_norm
                    up_vy /= up_norm

                prev_s = self._prev_scales[handedness]
                palm_scale = alpha * raw_scale + (1.0 - alpha) * prev_s

                prev_norm = np.array(self._prev_normals[handedness], dtype=np.float32)
                smoothed_normal = alpha * raw_normal + (1.0 - alpha) * prev_norm
                s_norm_len = float(np.linalg.norm(smoothed_normal))
                if s_norm_len > 1e-5:
                    smoothed_normal = smoothed_normal / s_norm_len
                else:
                    smoothed_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)

                prev_c3d = self._prev_centers_3d[handedness]
                cz = alpha * raw_cz + (1.0 - alpha) * prev_c3d[2]
            else:
                cx, cy = raw_cx, raw_cy
                up_vx, up_vy = raw_up_x, raw_up_y
                palm_scale = raw_scale
                smoothed_normal = raw_normal
                cz = raw_cz

            self._prev_centers[handedness] = (cx, cy)
            self._prev_up_vectors[handedness] = (up_vx, up_vy)
            self._prev_scales[handedness] = palm_scale
            self._prev_normals[handedness] = (
                float(smoothed_normal[0]),
                float(smoothed_normal[1]),
                float(smoothed_normal[2]),
            )
            self._prev_centers_3d[handedness] = (float(cx), float(cy), float(cz))

            # 3D Orthonormal Basis: Normal (N), Up (U), Right (R)
            norm_3d = smoothed_normal.copy()
            # 3D Up vector orthogonalized to normal
            raw_u = np.array([up_vx, up_vy, 0.0], dtype=np.float32)
            raw_u -= np.dot(raw_u, norm_3d) * norm_3d
            u_len = float(np.linalg.norm(raw_u))
            u_3d = raw_u / u_len if u_len > 1e-5 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
            # 3D Right vector: U x N
            r_3d = np.cross(u_3d, norm_3d)
            r_len = float(np.linalg.norm(r_3d))
            if r_len > 1e-5:
                r_3d = r_3d / r_len
            else:
                r_3d = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            # 2D Screen Rightward perpendicular vector
            right_vx = -up_vy
            right_vy = up_vx

            # Pitch and Roll angles in degrees
            pitch_deg = math.degrees(math.asin(float(np.clip(-norm_3d[1], -1.0, 1.0))))
            roll_deg = math.degrees(math.atan2(float(norm_3d[0]), float(-norm_3d[2])))

            # 5. Finger Extension Analysis
            finger_states: List[bool] = []
            wrist_px = (wrist.px, wrist.py)

            # Thumb extension
            thumb_tip_dist = math.hypot(points[4].px - points[2].px, points[4].py - points[2].py)
            thumb_base_dist = math.hypot(points[2].px - wrist.px, points[2].py - wrist.py)
            thumb_extended = thumb_tip_dist > (0.8 * thumb_base_dist)
            finger_states.append(thumb_extended)

            # 4 Fingers: Index, Middle, Ring, Pinky
            extended_count = 1 if thumb_extended else 0
            for tip_idx, pip_idx, mcp_idx in zip(
                self.FINGER_TIPS[1:], self.FINGER_PIPS[1:], self.FINGER_MCPS[1:]
            ):
                tip_dist = math.hypot(points[tip_idx].px - wrist_px[0], points[tip_idx].py - wrist_px[1])
                pip_dist = math.hypot(points[pip_idx].px - wrist_px[0], points[pip_idx].py - wrist_px[1])
                mcp_dist = math.hypot(points[mcp_idx].px - wrist_px[0], points[mcp_idx].py - wrist_px[1])

                is_extended = tip_dist > (pip_dist * 1.08) and tip_dist > (mcp_dist * 1.25)
                finger_states.append(is_extended)
                if is_extended:
                    extended_count += 1

            # 6. Gesture Decision with Temporal Hysteresis
            instant_open = extended_count >= config.OPEN_PALM_FINGER_THRESHOLD
            buffer = self._gesture_buffers.setdefault(
                handedness, deque(maxlen=config.GESTURE_SMOOTHING_FRAMES)
            )
            buffer.append(instant_open)

            open_ratio = sum(buffer) / len(buffer)
            is_open = open_ratio >= 0.6

            # 7. Fingertip Kinematics & Velocity Calculation (Thumb, Index, Middle, Ring, Pinky)
            now_sec = time.time()
            prev_time = self._prev_finger_time.get(handedness, now_sec - 0.033)
            dt_finger = float(np.clip(now_sec - prev_time, 0.005, 0.1))
            self._prev_finger_time[handedness] = now_sec

            prev_positions = self._prev_finger_pos.get(handedness, None)
            prev_velocities = self._prev_finger_vel.get(handedness, None)

            current_positions: List[np.ndarray] = []
            current_velocities: List[np.ndarray] = []
            finger_colliders: List[FingerTipCollider] = []

            for f_idx, tip_idx in enumerate(self.FINGER_TIPS):
                tip_pt = points[tip_idx]
                f_name = self.FINGER_NAMES[f_idx]
                is_ext = finger_states[f_idx]

                # Compute 3D position in screen/depth space
                fx = float(tip_pt.px)
                fy = float(tip_pt.py)
                fz = float(cz + (tip_pt.z - wrist.z) * width * 1.2)
                f_pos = np.array([fx, fy, fz], dtype=np.float32)
                current_positions.append(f_pos)

                # Velocity calculation with EMA filtering
                if prev_positions is not None and f_idx < len(prev_positions):
                    raw_vel = (f_pos - prev_positions[f_idx]) / dt_finger
                    if prev_velocities is not None and f_idx < len(prev_velocities):
                        f_vel = 0.55 * raw_vel + 0.45 * prev_velocities[f_idx]
                    else:
                        f_vel = raw_vel
                else:
                    f_vel = np.zeros(3, dtype=np.float32)
                current_velocities.append(f_vel)

                collider = FingerTipCollider(
                    name=f_name,
                    tip_idx=tip_idx,
                    pos_3d=f_pos,
                    vel_3d=f_vel,
                    radius=config.FINGER_COLLIDER_RADIUS,
                    is_extended=is_ext,
                )
                finger_colliders.append(collider)

            self._prev_finger_pos[handedness] = current_positions
            self._prev_finger_vel[handedness] = current_velocities

            pose = HandPose(
                handedness=handedness,
                landmarks=points,
                palm_center_px=(int(round(cx)), int(round(cy))),
                up_vector=(up_vx, up_vy),
                right_vector=(right_vx, right_vy),
                palm_scale=palm_scale,
                is_open=is_open,
                open_confidence=open_ratio,
                finger_states=finger_states,
                palm_normal_3d=(float(norm_3d[0]), float(norm_3d[1]), float(norm_3d[2])),
                palm_up_3d=(float(u_3d[0]), float(u_3d[1]), float(u_3d[2])),
                palm_right_3d=(float(r_3d[0]), float(r_3d[1]), float(r_3d[2])),
                palm_center_3d=(float(cx), float(cy), float(cz)),
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                fingers=finger_colliders,
            )
            poses.append(pose)

        return poses

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose: HandPose,
        color: Tuple[int, int, int] = (0, 255, 200),
    ) -> None:
        """Draws a crisp cyberpunk virtual skeleton directly onto the frame.

        Optimized with zero memory allocations or full-frame copies.
        """
        # 1. Draw bones directly
        for start_idx, end_idx in self.BONES:
            pt1 = (pose.landmarks[start_idx].px, pose.landmarks[start_idx].py)
            pt2 = (pose.landmarks[end_idx].px, pose.landmarks[end_idx].py)
            cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw joint nodes
        for pt in pose.landmarks:
            cv2.circle(frame, (pt.px, pt.py), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (pt.px, pt.py), 5, color, 1, lineType=cv2.LINE_AA)

        # 3. Highlight palm center anchor
        cx, cy = pose.palm_center_px
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
