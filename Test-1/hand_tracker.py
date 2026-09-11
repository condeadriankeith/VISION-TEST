"""Hand detection, 3D virtual skeleton mapping, and gesture classification module.

Uses MediaPipe Tasks HandLandmarker to track 21 landmarks per hand in 3D,
computes palm orientation, and classifies Open vs Closed palm states with
temporal hysteresis filtering.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import os
import threading
import time
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np

from spatial_math import BoneCapsuleCollider, OneEuroFilter
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
    pos_3d: np.ndarray     # 3D position [x, y, z] in camera coordinate space
    vel_3d: np.ndarray     # Dynamic 3D velocity vector (px/s)
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
    openness_ratio: float = 1.0
    palm_normal_3d: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    palm_up_3d: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    palm_right_3d: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    palm_center_3d: Tuple[float, float, float] = (640.0, 360.0, 0.0)
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fingers: List[FingerTipCollider] = field(default_factory=list)
    capsules: List[BoneCapsuleCollider] = field(default_factory=list)
    # Telekinesis Force Perception
    is_pinching: bool = False
    pinch_point_3d: Optional[Tuple[float, float, float]] = None
    pinch_dist_px: float = 999.0
    palm_thrust_speed: float = 0.0
    hand_velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_speed: float = 0.0


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

        # Continuous palm openness smoothing
        self._prev_openness: dict[str, float] = {}

        # 3D Up vector temporal smoothing
        self._prev_up_3d: dict[str, Tuple[float, float, float]] = {}

        # Hand/Palm 3D velocity and force thrust tracking
        self._prev_palm_pos: dict[str, np.ndarray] = {}
        self._prev_palm_vel: dict[str, np.ndarray] = {}
        self._prev_palm_time: dict[str, float] = {}

        # Pinch gesture hysteresis state
        self._prev_pinching: dict[str, bool] = {}

        # 1€ (One Euro) Adaptive Motion Filter banks per hand
        self._euro_landmarks: dict[str, List[OneEuroFilter]] = {}
        self._euro_palm_center: dict[str, OneEuroFilter] = {}
        self._euro_normal: dict[str, OneEuroFilter] = {}
        self._euro_up_3d: dict[str, OneEuroFilter] = {}
        self._euro_scale: dict[str, OneEuroFilter] = {}
        self._last_process_time: dict[str, float] = {}
        # VIDEO-mode timestamps must be strictly increasing; callers using
        # wall-clock ms can rarely repeat a value between fast frames.
        self._last_timestamp_ms: int = 0

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

        Uses downsampled inference for high FPS and applies 1€ adaptive filtering
        to eliminate coordinate jitter while preserving zero-latency response.
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
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
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
            self._prev_openness.clear()
            self._prev_up_3d.clear()
            self._prev_palm_pos.clear()
            self._prev_palm_vel.clear()
            self._prev_palm_time.clear()
            self._prev_pinching.clear()
            self._euro_landmarks.clear()
            self._euro_palm_center.clear()
            self._euro_normal.clear()
            self._euro_up_3d.clear()
            self._euro_scale.clear()
            self._last_process_time.clear()
            return poses

        now_sec = float(timestamp_ms * 0.001)

        for i, raw_landmarks in enumerate(result.hand_landmarks):
            handedness = "Right"
            if i < len(result.handedness) and result.handedness[i]:
                handedness = result.handedness[i][0].category_name

            # 1€ Filter initialization for this hand
            if handedness not in self._euro_landmarks:
                fc_min = getattr(config, "ONE_EURO_FC_MIN", 0.85)
                beta = getattr(config, "ONE_EURO_BETA", 0.045)
                d_cut = getattr(config, "ONE_EURO_D_CUTOFF", 1.0)
                self._euro_landmarks[handedness] = [
                    OneEuroFilter(fc_min=fc_min, beta=beta, d_cutoff=d_cut)
                    for _ in range(21)
                ]
                self._euro_palm_center[handedness] = OneEuroFilter(fc_min=fc_min, beta=beta, d_cutoff=d_cut)
                self._euro_normal[handedness] = OneEuroFilter(fc_min=fc_min, beta=beta, d_cutoff=d_cut)
                self._euro_up_3d[handedness] = OneEuroFilter(fc_min=fc_min, beta=beta, d_cutoff=d_cut)
                self._euro_scale[handedness] = OneEuroFilter(fc_min=fc_min, beta=beta, d_cutoff=d_cut)

            prev_proc_t = self._last_process_time.get(handedness, now_sec - 0.033)
            dt_euro = float(np.clip(now_sec - prev_proc_t, 0.001, 0.1))
            self._last_process_time[handedness] = now_sec

            # Filter raw 3D landmarks through 1€ Adaptive Low-Pass Filter bank
            points: List[LandmarkPoint] = []
            for lm_idx, lm in enumerate(raw_landmarks):
                raw_vec = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
                filt_vec = self._euro_landmarks[handedness][lm_idx].filter(raw_vec, dt_euro)
                px = int(round(filt_vec[0] * width))
                py = int(round(filt_vec[1] * height))
                points.append(
                    LandmarkPoint(x=float(filt_vec[0]), y=float(filt_vec[1]), z=float(filt_vec[2]), px=px, py=py)
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

            # Compute true 3D Normal Vector from aspect-corrected 3D Landmark vectors
            v_up_3d = np.array([
                (middle_mcp.x - wrist.x) * width,
                (middle_mcp.y - wrist.y) * height,
                (middle_mcp.z - wrist.z) * width,
            ], dtype=np.float32)

            v_across_3d = np.array([
                (index_mcp.x - pinky_mcp.x) * width,
                (index_mcp.y - pinky_mcp.y) * height,
                (index_mcp.z - pinky_mcp.z) * width,
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

            v_up_len = float(np.linalg.norm(v_up_3d))
            raw_up_3d = (v_up_3d / v_up_len) if v_up_len > 1e-5 else np.array([0.0, -1.0, 0.0], dtype=np.float32)

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

                prev_u3d = np.array(self._prev_up_3d.get(handedness, raw_up_3d), dtype=np.float32)
                smoothed_u3d = alpha * raw_up_3d + (1.0 - alpha) * prev_u3d
                s_u_len = float(np.linalg.norm(smoothed_u3d))
                smoothed_u3d = (smoothed_u3d / s_u_len) if s_u_len > 1e-5 else raw_up_3d

                prev_c3d = self._prev_centers_3d[handedness]
                cz = alpha * raw_cz + (1.0 - alpha) * prev_c3d[2]
            else:
                cx, cy = raw_cx, raw_cy
                up_vx, up_vy = raw_up_x, raw_up_y
                palm_scale = raw_scale
                smoothed_normal = raw_normal
                smoothed_u3d = raw_up_3d
                cz = raw_cz

            self._prev_centers[handedness] = (cx, cy)
            self._prev_up_vectors[handedness] = (up_vx, up_vy)
            self._prev_scales[handedness] = palm_scale
            self._prev_normals[handedness] = (
                float(smoothed_normal[0]),
                float(smoothed_normal[1]),
                float(smoothed_normal[2]),
            )
            self._prev_up_3d[handedness] = (
                float(smoothed_u3d[0]),
                float(smoothed_u3d[1]),
                float(smoothed_u3d[2]),
            )
            self._prev_centers_3d[handedness] = (float(cx), float(cy), float(cz))

            # 3D Orthonormal Basis: Normal (N), Up (U), Right (R)
            norm_3d = smoothed_normal.copy()
            # 3D Up vector: orthogonalize smoothed true 3D up vector against normal
            u_proj = smoothed_u3d - np.dot(smoothed_u3d, norm_3d) * norm_3d
            u_len = float(np.linalg.norm(u_proj))
            if u_len > 1e-4:
                u_3d = u_proj / u_len
            else:
                fallback = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                if abs(float(np.dot(fallback, norm_3d))) > 0.9:
                    fallback = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                u_proj = fallback - np.dot(fallback, norm_3d) * norm_3d
                u_3d = u_proj / max(float(np.linalg.norm(u_proj)), 1e-5)

            # 3D Right vector: R = U x N
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

            # 5. Finger Extension Analysis & Continuous Openness Ratio
            finger_states: List[bool] = []
            wrist_px = (wrist.px, wrist.py)

            # Thumb extension
            thumb_tip_dist = math.hypot(points[4].px - points[2].px, points[4].py - points[2].py)
            thumb_base_dist = math.hypot(points[2].px - wrist.px, points[2].py - wrist.py)
            thumb_extended = thumb_tip_dist > (0.8 * thumb_base_dist)
            finger_states.append(thumb_extended)
            thumb_ratio = float(np.clip((thumb_tip_dist / max(thumb_base_dist, 1e-4) - 0.45) / 0.55, 0.0, 1.0))

            # 4 Fingers: Index, Middle, Ring, Pinky
            extended_count = 1 if thumb_extended else 0
            finger_ratios = [thumb_ratio]
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
                cur_ratio = float(np.clip((tip_dist / max(mcp_dist, 1e-4) - 0.95) / 0.65, 0.0, 1.0))
                finger_ratios.append(cur_ratio)

            # Continuous openness ratio with EMA smoothing
            raw_openness = sum(finger_ratios) / len(finger_ratios)
            op_min = getattr(config, "CONTINUOUS_OPENNESS_MIN", 0.22)
            op_max = getattr(config, "CONTINUOUS_OPENNESS_MAX", 0.70)
            norm_openness = float(np.clip((raw_openness - op_min) / max(op_max - op_min, 1e-4), 0.0, 1.0))

            prev_op = self._prev_openness.get(handedness, norm_openness)
            smoothed_openness = 0.35 * norm_openness + 0.65 * prev_op
            self._prev_openness[handedness] = smoothed_openness

            # 6. Gesture Decision with Temporal Hysteresis
            instant_open = extended_count >= config.OPEN_PALM_FINGER_THRESHOLD
            buffer = self._gesture_buffers.setdefault(
                handedness, deque(maxlen=config.GESTURE_SMOOTHING_FRAMES)
            )
            buffer.append(instant_open)

            open_ratio = sum(buffer) / len(buffer)
            is_open = open_ratio >= 0.6

            # 7. Fingertip Kinematics & Velocity Calculation (Thumb, Index, Middle, Ring, Pinky)
            # Uses the monotonic VIDEO-mode timestamp clock (same basis as the 1€
            # filter), avoiding extra wall-clock syscalls per hand per frame.
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

            # 8. Optimized Pinch Detection with Hysteresis (Thumb Tip to Index Tip 3D distance)
            thumb_pos = current_positions[0]
            index_pos = current_positions[1]
            pinch_dist = float(np.linalg.norm(thumb_pos - index_pos))
            was_pinching = self._prev_pinching.get(handedness, False)
            pinch_thresh = (
                getattr(config, "PINCH_RELEASE_THRESHOLD_PX", 52.0)
                if was_pinching
                else getattr(config, "PINCH_THRESHOLD_PX", 36.0)
            )
            is_pinching = pinch_dist < pinch_thresh
            self._prev_pinching[handedness] = is_pinching
            pinch_mid = (thumb_pos + index_pos) * 0.5
            pinch_point_3d = (float(pinch_mid[0]), float(pinch_mid[1]), float(pinch_mid[2]))

            # 9. Palm 3D Velocity & Forward Thrust / Waving Speed (same timestamp clock)
            curr_palm_vec = np.array([cx, cy, cz], dtype=np.float32)
            prev_p_time = self._prev_palm_time.get(handedness, now_sec - 0.033)
            dt_palm = float(np.clip(now_sec - prev_p_time, 0.005, 0.1))
            self._prev_palm_time[handedness] = now_sec

            prev_p_pos = self._prev_palm_pos.get(handedness, None)
            prev_p_vel = self._prev_palm_vel.get(handedness, None)
            if prev_p_pos is not None:
                raw_p_vel = (curr_palm_vec - prev_p_pos) / dt_palm
                if prev_p_vel is not None:
                    p_vel = 0.60 * raw_p_vel + 0.40 * prev_p_vel
                else:
                    p_vel = raw_p_vel
            else:
                p_vel = np.zeros(3, dtype=np.float32)

            self._prev_palm_pos[handedness] = curr_palm_vec
            self._prev_palm_vel[handedness] = p_vel

            # Palm speed (px/s) drives tornado-wave detection; thrust speed is the
            # forward push component along the outward palm normal and drives
            # Force Push. Both default to 0 on the first sighting of a hand.
            palm_speed = float(np.linalg.norm(p_vel))
            thrust_speed = float(max(0.0, float(np.dot(p_vel, smoothed_normal))))

            # 10. Full-Hand Biomechanical Capsule Colliders (14 Phalanx Bones + Palm Plate)
            bone_capsules: List[BoneCapsuleCollider] = []
            if getattr(config, "FULL_HAND_CAPSULE_ENABLED", True):
                joints_3d: List[np.ndarray] = []
                for pt in points:
                    jx = float(pt.px)
                    jy = float(pt.py)
                    jz = float(cz + (pt.z - wrist.z) * width * 1.2)
                    joints_3d.append(np.array([jx, jy, jz], dtype=np.float32))

                base_radius = getattr(config, "CAPSULE_BONE_RADIUS", 15.0)
                for start_idx, end_idx in self.BONES:
                    p0 = joints_3d[start_idx]
                    p1 = joints_3d[end_idx]
                    is_distal = end_idx in self.FINGER_TIPS
                    is_palm = start_idx in palm_indices and end_idx in palm_indices
                    if is_palm:
                        r_bone = base_radius * 1.2
                    elif is_distal:
                        r_bone = base_radius * 0.85
                    else:
                        r_bone = base_radius

                    bone_capsules.append(
                        BoneCapsuleCollider(
                            name=f"{handedness}_{start_idx}_{end_idx}",
                            p0=p0,
                            p1=p1,
                            radius=r_bone,
                            velocity=p_vel.copy(),
                        )
                    )

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
                openness_ratio=smoothed_openness,
                palm_normal_3d=(float(norm_3d[0]), float(norm_3d[1]), float(norm_3d[2])),
                palm_up_3d=(float(u_3d[0]), float(u_3d[1]), float(u_3d[2])),
                palm_right_3d=(float(r_3d[0]), float(r_3d[1]), float(r_3d[2])),
                palm_center_3d=(float(cx), float(cy), float(cz)),
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                fingers=finger_colliders,
                capsules=bone_capsules,
                is_pinching=is_pinching,
                pinch_point_3d=pinch_point_3d if is_pinching else None,
                pinch_dist_px=pinch_dist,
                palm_thrust_speed=thrust_speed,
                hand_velocity_3d=(float(p_vel[0]), float(p_vel[1]), float(p_vel[2])),
                palm_speed=palm_speed,
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


class TrackingWorker:
    """Runs HandTracker inference on a background thread for realtime display.

    MediaPipe detection costs ~30-60ms on CPU. Called inline, it stalls the
    render loop so the video *and* skeleton lag behind the hand. This worker
    instead consumes the newest camera frame whenever it is free (dropping
    stale frames) and publishes the latest poses lock-free to the render
    thread, which never blocks and always draws the freshest camera image.
    The HandTracker instance is owned exclusively by this thread.
    """

    def __init__(self, tracker: "HandTracker", camera: object, mirror: bool = True) -> None:
        self._tracker = tracker
        self._camera = camera
        self._mirror = mirror
        self._lock = threading.Lock()
        self._poses: List[HandPose] = []
        self._frame_id: int = -1
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the background inference loop (daemon thread)."""
        self._stopped = False
        self._thread = threading.Thread(
            target=self._loop, name="TrackingWorker", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        last_id = -1
        while not self._stopped:
            ok, frame, frame_id = self._camera.read_latest()
            if not ok or frame is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            if self._mirror:
                frame = cv2.flip(frame, 1)
            # Monotonic clock: VIDEO mode rejects non-increasing timestamps.
            ts = int(time.monotonic() * 1000)
            try:
                poses = self._tracker.process_frame(frame, ts)
            except Exception:
                continue
            with self._lock:
                self._poses = poses
                self._frame_id = frame_id

    def get_latest(self) -> Tuple[List[HandPose], int]:
        """Return (poses, frame_id) without blocking. Poses may be one inference behind."""
        with self._lock:
            return list(self._poses), self._frame_id

    def set_camera(self, camera: object) -> None:
        """Hot-swap the video source (e.g. after a camera switch)."""
        with self._lock:
            self._camera = camera
            self._poses = []
            self._frame_id = -1

    def stop(self) -> None:
        """Signal shutdown and wait briefly for the thread to exit."""
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
