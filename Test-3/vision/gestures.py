"""Geometric gesture classifier — exact video reference mapping.

Mapping (right / trigger hand):
  INDEX_ONLY (index extended only)          -> FLOWERS  (blooming lilies)
  FIST (all curled)                         -> DRAGON   (red dragon / phoenix)
  L (thumb + index at ~90 deg)              -> BUTTERFLY (blue morpho)
  PEACE (index + middle, ring/pinky folded) -> TREE     (bonsai / cosmic tree)
  OPEN_PALM (all 5 extended)                -> DISSOLVE (particle transition)
  DUAL_FISTS handled one level up (both hands closed -> CLEAR).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Tuple
import math
import numpy as np


class Gesture(Enum):
    UNKNOWN = auto()
    FIST = auto()        # -> DRAGON
    INDEX = auto()       # -> FLOWERS
    L_SHAPE = auto()     # -> BUTTERFLY
    PEACE = auto()       # -> TREE
    OPEN_PALM = auto()   # -> DISSOLVE


MODEL_FOR_GESTURE = {
    Gesture.INDEX: "flowers",
    Gesture.FIST: "dragon",
    Gesture.L_SHAPE: "butterfly",
    Gesture.PEACE: "tree",
}


def classify_finger_extensions(pts: np.ndarray, ext_ratio: float = 1.15,
                               thumb_ratio: float = 1.10) -> Tuple[bool, bool, bool, bool, bool]:
    """(thumb, index, middle, ring, pinky) via tip/pip-to-wrist distance ratios."""
    pts = np.asarray(pts, dtype=np.float64)
    wrist = pts[0]
    out = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        dt = float(np.linalg.norm(pts[tip] - wrist))
        dp = float(np.linalg.norm(pts[pip] - wrist))
        out.append(dt > dp * ext_ratio)
    dt = float(np.linalg.norm(pts[4] - pts[5]))
    dp = float(np.linalg.norm(pts[3] - pts[5]))
    thumb = dt > dp * thumb_ratio
    return (thumb, out[0], out[1], out[2], out[3])


def thumb_index_angle_deg(pts: np.ndarray) -> float:
    """Angle between thumb ray (2->4) and index ray (5->8), in degrees."""
    pts = np.asarray(pts, dtype=np.float64)
    a = pts[4] - pts[2]
    b = pts[8] - pts[5]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def compute_pinch_distance(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pts)[4] - np.asarray(pts)[8]))


def classify_gesture(finger_states: Tuple[bool, bool, bool, bool, bool],
                     pts: np.ndarray | None = None,
                     l_min: float = 60.0, l_max: float = 120.0) -> Gesture:
    """Map finger states (+ optional L angle) to a Gesture.

    Priority: OPEN_PALM > PEACE > L_SHAPE > INDEX > FIST > UNKNOWN.
    """
    thumb, index, middle, ring, pinky = finger_states
    main = (index, middle, ring, pinky)
    count = sum(1 for b in main if b)

    if count == 4:  # thumb may or may not be out; 4 mains extended is enough
        return Gesture.OPEN_PALM
    if count >= 3 and index and middle:
        return Gesture.OPEN_PALM  # robust fallback for wide hands
    # Peace: index + middle, ring + pinky folded (thumb ignored / usually folded).
    if index and middle and not ring and not pinky:
        return Gesture.PEACE
    # L: thumb + index only.
    if thumb and index and not middle and not ring and not pinky:
        if pts is not None:
            ang = thumb_index_angle_deg(np.asarray(pts))
            if l_min <= ang <= l_max:
                return Gesture.L_SHAPE
            # Slightly loose fallback: still an L if geometry is close.
            if 45.0 <= ang <= 135.0:
                return Gesture.L_SHAPE
            return Gesture.UNKNOWN
        return Gesture.L_SHAPE
    # Index pointing: index only.
    if index and not middle and not ring and not pinky and not thumb:
        return Gesture.INDEX
    # Fist: nothing extended (thumb resting counts as folded).
    if count == 0:
        return Gesture.FIST
    return Gesture.UNKNOWN
