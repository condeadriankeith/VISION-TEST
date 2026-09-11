"""Dual-hand MediaPipe tracker.

Roles (blueprint 1.1):
  Left hand  -> spatial anchor & orientation (palm centroid + R_hand).
  Right hand -> gesture trigger & state machine.

Assignment uses MediaPipe handedness when available, with a screen-X
fallback (x < 0.5 => Left). With a single hand, it drives both roles.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .gestures import (
    Gesture, classify_finger_extensions, classify_gesture,
    compute_pinch_distance,
)
from .pose_estimator import HandPoseEstimator, HandPose

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


@dataclass
class SingleHand:
    landmarks: np.ndarray       # (21,3) normalized
    handedness: str             # "Left" | "Right"
    fingers: Tuple[bool, bool, bool, bool, bool]
    gesture: Gesture
    pinch: float
    pose: HandPose


@dataclass
class DualHandFrame:
    hands: List[SingleHand] = field(default_factory=list)
    anchor: Optional[SingleHand] = None    # left / spatial role
    trigger: Optional[SingleHand] = None   # right / gesture role
    dual_fists: bool = False               # both hands FIST -> CLEAR

    @property
    def has_anchor(self) -> bool:
        return self.anchor is not None

    @property
    def trigger_gesture(self) -> Gesture:
        return self.trigger.gesture if self.trigger else Gesture.UNKNOWN


class DualHandTracker:
    def __init__(self, model_path: Optional[str] = None,
                 ext_ratio: float = 1.15, thumb_ratio: float = 1.10,
                 num_hands: int = 2) -> None:
        self._ext = ext_ratio
        self._thumb = thumb_ratio
        resolved = self._resolve_model(model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._estimators: Dict[str, HandPoseEstimator] = {}

    @staticmethod
    def _resolve_model(provided: Optional[str]) -> str:
        if provided and os.path.isfile(provided):
            return provided
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", MODEL_FILENAME),
            os.path.join(os.getcwd(), MODEL_FILENAME),
            MODEL_FILENAME,
            os.path.join(here, "..", "..", "Test-1", MODEL_FILENAME),
            os.path.join(here, "..", "..", "Test-2", MODEL_FILENAME),
        ]
        for c in candidates:
            if os.path.isfile(os.path.abspath(c)):
                return os.path.abspath(c)
        print(f"[INFO] Downloading {MODEL_FILENAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        return MODEL_FILENAME

    def _estimator(self, key: str) -> HandPoseEstimator:
        if key not in self._estimators:
            self._estimators[key] = HandPoseEstimator()
        return self._estimators[key]

    def process(self, bgr: np.ndarray) -> DualHandFrame:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect(mp_img)
        frame = DualHandFrame()
        if not res.hand_landmarks:
            for e in self._estimators.values():
                e.reset()
            return frame

        raw_hands = []
        n = len(res.hand_landmarks)
        for i in range(n):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[i]],
                           dtype=np.float64)
            label = "Right"
            try:
                if res.handedness and i < len(res.handedness) and res.handedness[i]:
                    label = res.handedness[i][0].category_name or "Right"
            except Exception:
                pass
            fingers = classify_finger_extensions(pts, self._ext, self._thumb)
            gesture = classify_gesture(fingers, pts)
            pinch = compute_pinch_distance(pts)
            # Stable estimator key: handedness + slot (handles two same-side hands).
            key = f"{label}_{i if n > 1 else 0}"
            pose = self._estimator(key).estimate(pts, label)
            raw_hands.append(SingleHand(pts, label, fingers, gesture, pinch, pose))

        frame.hands = raw_hands
        if len(raw_hands) == 1:
            frame.anchor = raw_hands[0]
            frame.trigger = raw_hands[0]
        else:
            # Prefer true Left/Right labels; fallback to screen-X split.
            lefts = [h for h in raw_hands if h.handedness == "Left"]
            rights = [h for h in raw_hands if h.handedness == "Right"]
            if lefts and rights:
                anchor, trigger = lefts[0], rights[0]
            else:
                by_x = sorted(raw_hands, key=lambda h: float(h.pose.palm_center_2d[0]))
                anchor, trigger = by_x[0], by_x[-1]
                if anchor is trigger and len(by_x) > 1:
                    trigger = by_x[1]
            frame.anchor = anchor
            frame.trigger = trigger
            g = [h.gesture for h in raw_hands]
            frame.dual_fists = len(g) >= 2 and all(x == Gesture.FIST for x in g[:2])
        return frame

    def release(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass
