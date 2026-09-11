"""MediaPipe Tasks HandLandmarker wrapper and geometric gesture classifier.

Gesture mapping:
  PINCH     - Thumb tip and Index tip close together (pinch-out to spawn cube)
  OPEN_PALM - All four main fingers extended (spawn cloud particles)
  FIST      - All fingers folded (standby)
  NO_HAND   - No hand detected

Rotation of the cube is always driven by the hand's 3D pose whenever any hand is present.
"""

from dataclasses import dataclass
import os
from typing import List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np

from core.state_machine import GestureState

MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME: str = "hand_landmarker.task"


@dataclass
class LandmarkFrame:
    """Encapsulates hand detection output for a single frame."""
    landmarks_3d: Optional[np.ndarray]   # (21, 3) normalized coordinates [x, y, z]
    handedness: str                       # "Right" or "Left"
    classified_state: GestureState        # Instantaneous gesture state candidate
    finger_states: Tuple[bool, bool, bool, bool, bool]  # (thumb, index, middle, ring, pinky)
    pinch_distance: float                 # Normalized thumb-index tip distance


class GestureDetector:
    """Wraps MediaPipe HandLandmarker and classifies geometric postures."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        extension_ratio: float = 1.15,
        pinch_threshold: float = 0.065,
        num_hands: int = 1,
    ) -> None:
        """Initialize MediaPipe HandLandmarker.

        Args:
            model_path: Path to hand_landmarker.task. If None, resolves locally or downloads.
            extension_ratio: Tip-wrist to pip-wrist ratio for finger extension classification.
            pinch_threshold: Normalized distance below which thumb+index counts as a pinch.
            num_hands: Maximum hands to detect simultaneously.
        """
        self._extension_ratio: float = extension_ratio
        self._pinch_threshold: float = pinch_threshold
        resolved_path = self._resolve_model_path(model_path)

        base_options = BaseOptions(model_asset_path=resolved_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _resolve_model_path(provided_path: Optional[str]) -> str:
        """Ensure hand_landmarker.task exists, checking local folder, sibling folders, or downloading."""
        if provided_path and os.path.isfile(provided_path):
            return provided_path

        if os.path.isfile(MODEL_FILENAME):
            return MODEL_FILENAME

        candidates = [
            os.path.join(os.path.dirname(__file__), "..", MODEL_FILENAME),
            os.path.join(os.path.dirname(__file__), "..", "..", "Test-1", MODEL_FILENAME),
        ]
        for path in candidates:
            abs_p = os.path.abspath(path)
            if os.path.isfile(abs_p):
                return abs_p

        print(f"[INFO] Downloading {MODEL_FILENAME} from Google Cloud...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        return MODEL_FILENAME

    def process_frame(self, bgr_frame: np.ndarray) -> LandmarkFrame:
        """Process BGR video frame through MediaPipe and classify posture.

        Args:
            bgr_frame: Input color image from camera in BGR format.

        Returns:
            LandmarkFrame containing 3D landmarks, handedness, classified state, and pinch distance.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return LandmarkFrame(
                landmarks_3d=None,
                handedness="Right",
                classified_state=GestureState.NO_HAND,
                finger_states=(False, False, False, False, False),
                pinch_distance=1.0,
            )

        raw_landmarks = result.hand_landmarks[0]
        pts_3d = np.zeros((21, 3), dtype=np.float64)
        for i, lm in enumerate(raw_landmarks):
            pts_3d[i] = [lm.x, lm.y, lm.z]

        handedness = "Right"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        finger_states = self.classify_finger_extensions(pts_3d, self._extension_ratio)
        pinch_dist = self.compute_pinch_distance(pts_3d)

        # Index-tip-above-MCP guard: distinguishes a real pinch (index raised toward thumb)
        # from a closed fist (index tip curled below the knuckle line).
        # MediaPipe Y increases downward, so tip_y < mcp_y means the tip is raised.
        index_tip_above_mcp = bool(pts_3d[8][1] < pts_3d[5][1])

        state = self.classify_gesture(
            finger_states,
            pinch_dist,
            self._pinch_threshold,
            index_tip_above_mcp=index_tip_above_mcp,
        )

        return LandmarkFrame(
            landmarks_3d=pts_3d,
            handedness=handedness,
            classified_state=state,
            finger_states=finger_states,
            pinch_distance=pinch_dist,
        )

    @staticmethod
    def compute_pinch_distance(pts: np.ndarray) -> float:
        """Compute normalized 3D Euclidean distance between thumb tip (4) and index tip (8).

        Args:
            pts: (21, 3) normalized landmark array.

        Returns:
            Normalized thumb-to-index tip distance.
        """
        return float(np.linalg.norm(pts[4] - pts[8]))

    @staticmethod
    def classify_finger_extensions(
        pts: np.ndarray,
        extension_ratio: float = 1.15,
    ) -> Tuple[bool, bool, bool, bool, bool]:
        """Determine whether each of the 5 fingers is extended based on tip-pip-wrist geometry.

        Args:
            pts: (21, 3) array of landmarks.
            extension_ratio: Required ratio of tip-wrist distance to pip-wrist distance.

        Returns:
            Tuple of 5 booleans (thumb, index, middle, ring, pinky).
        """
        wrist = pts[0]
        finger_indices = [(8, 6), (12, 10), (16, 14), (20, 18)]

        extended_fingers: List[bool] = []
        for tip_idx, pip_idx in finger_indices:
            dist_tip = float(np.linalg.norm(pts[tip_idx] - wrist))
            dist_pip = float(np.linalg.norm(pts[pip_idx] - wrist))
            extended_fingers.append(dist_tip > dist_pip * extension_ratio)

        # Thumb: compare tip(4) to index_mcp(5) distance vs ip(3) to index_mcp(5)
        dist_thumb_tip = float(np.linalg.norm(pts[4] - pts[5]))
        dist_thumb_ip = float(np.linalg.norm(pts[3] - pts[5]))
        thumb_extended = dist_thumb_tip > dist_thumb_ip * 1.10

        return (thumb_extended, extended_fingers[0], extended_fingers[1], extended_fingers[2], extended_fingers[3])

    @staticmethod
    def classify_gesture(
        finger_states: Tuple[bool, bool, bool, bool, bool],
        pinch_distance: float,
        pinch_threshold: float,
        index_tip_above_mcp: bool = True,
    ) -> GestureState:
        """Map finger states and pinch distance to a GestureState.

        Priority order:
          1. OPEN_PALM  — all 4 main fingers extended
          2. PINCH      — thumb+index tips close AND index tip is raised above its MCP
          3. FIST       — all fingers folded

        The index-tip-above-MCP guard is the key discriminator:
          - Real pinch:  thumb and raised index meet — tip.y < mcp.y (above knuckle line)
          - Closed fist: thumb rests on curled index — tip.y > mcp.y (below knuckle line)

        MediaPipe Y-axis increases downward, so tip.y < mcp.y means the finger is raised.

        Args:
            finger_states: (thumb, index, middle, ring, pinky).
            pinch_distance: Normalized thumb-index tip distance.
            pinch_threshold: Distance below which a pinch is detected.
            index_tip_above_mcp: True when index tip Y < index MCP Y (tip raised above knuckle).

        Returns:
            Classified GestureState.
        """
        thumb, index, middle, ring, pinky = finger_states

        # Open palm: all four main fingers extended
        if index and middle and ring and pinky:
            return GestureState.OPEN_PALM

        # Partial open palm (3 fingers) → robust fallback
        main_count = sum([index, middle, ring, pinky])
        if main_count >= 3:
            return GestureState.OPEN_PALM

        # Pinch: thumb and index tips close together AND index is raised above knuckle.
        # This prevents a closed fist (where thumb presses on curled index) from
        # falsely triggering PINCH.
        if pinch_distance <= pinch_threshold and index_tip_above_mcp:
            return GestureState.PINCH

        # Fist: all main fingers folded
        if main_count == 0:
            return GestureState.FIST

        # 1-2 fingers extended, no open palm, no pinch → standby
        return GestureState.FIST
