"""Lightweight hand landmark extractor for the GPU particle system.

Returns 6 world-space attractor points per frame:
  - Index 0: Palm center (average of wrist + 5 MCP joints)
  - Index 1–5: Fingertips (thumb=4, index=8, middle=12, ring=16, pinky=20)

Coordinates are re-mapped from MediaPipe normalized [0,1] screen space into
a symmetric 3D world space centered at the origin with radius ~1.5 units.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np


MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME: str = "hand_landmarker.task"

# MediaPipe landmark indices
_TIP_INDICES = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
_MCP_INDICES = [1, 5, 9, 13, 17]   # MCP joints for palm center

# World-space scale: MediaPipe X/Y are [0,1], we map to [-1.5, 1.5]
_WORLD_SCALE = 3.0
_WORLD_OFFSET = 1.5
_DEPTH_SCALE  = 1.2  # MediaPipe Z is already relative; amplify for 3D effect


@dataclass
class HandState:
    """Attractor positions and interaction state for one frame."""
    attractors: np.ndarray      # (6, 3) float32 world-space positions
    hand_present: bool          # Whether hand is detected
    gesture_mode: float         # 0=attract(open), 1=repel(fist), 2=vortex(pinch)
    pinch_distance: float       # Raw normalized thumb-index distance


class HandTracker:
    """Wraps MediaPipe HandLandmarker and produces per-frame HandState."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        resolved = self._resolve_model(model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        # Smoothed attractor state — prevents jitter from frame-to-frame landmark noise
        self._smooth_attractors: np.ndarray = np.zeros((6, 3), dtype=np.float32)
        self._alpha: float = 0.28  # EMA smoothing factor
        self._last_timestamp_ms: int = 0

    @staticmethod
    def _resolve_model(provided: Optional[str]) -> str:
        if provided and os.path.isfile(provided):
            return provided
        candidates = [
            MODEL_FILENAME,
            os.path.join(os.path.dirname(__file__), "..", MODEL_FILENAME),
            os.path.join(os.path.dirname(__file__), "..", "..", "Test-1", MODEL_FILENAME),
        ]
        for path in candidates:
            abs_p = os.path.abspath(path)
            if os.path.isfile(abs_p):
                return abs_p
        print(f"[INFO] Downloading {MODEL_FILENAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        return MODEL_FILENAME

    def process(self, bgr_frame: np.ndarray) -> HandState:
        """Process a BGR camera frame and return attractor state.

        Args:
            bgr_frame: Camera frame in BGR format (H, W, 3).

        Returns:
            HandState with world-space attractor positions and gesture info.
        """
        h, w = bgr_frame.shape[:2]

        # Optimization: Downsample to 640x360 for high-FPS neural network inference.
        # MediaPipe returns normalized [0, 1] coordinates, preserving precision.
        if w != 640 or h != 360:
            small_bgr = cv2.resize(bgr_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic timestamp generation required by MediaPipe RunningMode.VIDEO
        now_ms = int(time.perf_counter() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms

        result = self._lm.detect_for_video(mp_img, now_ms)

        if not result.hand_landmarks:
            # Drift attractors slowly back to origin when no hand is present
            self._smooth_attractors *= 0.95
            return HandState(
                attractors=self._smooth_attractors.copy(),
                hand_present=False,
                gesture_mode=0.0,
                pinch_distance=1.0,
            )

        lms = result.hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

        # Palm center: mean of wrist + 5 MCPs
        palm_indices = [0] + _MCP_INDICES
        palm_center = pts[palm_indices].mean(axis=0)

        # Build raw attractor array: [palm, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        raw = np.vstack([palm_center, pts[_TIP_INDICES]])  # (6, 3)

        # Map from MediaPipe screen [0,1] to world [-1.5, +1.5]
        # X: flip horizontally (mirror) so it feels natural
        world = np.empty_like(raw)
        world[:, 0] = (1.0 - raw[:, 0]) * _WORLD_SCALE - _WORLD_OFFSET
        world[:, 1] = (1.0 - raw[:, 1]) * _WORLD_SCALE - _WORLD_OFFSET  # Y up
        world[:, 2] = raw[:, 2] * _DEPTH_SCALE

        # EMA smoothing
        self._smooth_attractors += self._alpha * (world - self._smooth_attractors)

        # ── Robust Gesture classification ─────────────────────────────────────
        # Pinch distance: thumb tip (4) to index tip (8)
        pinch_dist = float(np.linalg.norm(pts[4] - pts[8]))

        # Finger extension: distance of fingertips from wrist vs PIP joints
        wrist = pts[0]
        finger_tips = [pts[8], pts[12], pts[16], pts[20]]
        finger_pips = [pts[6], pts[10], pts[14], pts[18]]
        extended = sum(
            1 for tip, pip in zip(finger_tips, finger_pips)
            if float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist)) * 1.12
        )

        if extended >= 3:
            gesture_mode = 0.0  # Open palm → attract
        elif pinch_dist < 0.08:
            gesture_mode = 2.0  # Pinch → vortex
        else:
            gesture_mode = 1.0  # Fist/curled → repel

        return HandState(
            attractors=self._smooth_attractors.copy(),
            hand_present=True,
            gesture_mode=gesture_mode,
            pinch_distance=pinch_dist,
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._lm.close()
