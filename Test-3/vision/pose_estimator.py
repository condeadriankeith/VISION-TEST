"""3D hand pose estimator: smoothed anchor C_hand + rotation R_hand.

Blueprint 3.1:
  u = L9 - L0, v = L5 - L17, n = (u x v)/||u x v||
  Euler fallback (roll/pitch/yaw) + hand-frame rotation, blended by using
  the hand-frame rotation directly (orthonormalized), smoothed via EMA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from math3d.transforms import (
    euler_to_rotation_matrix, ema_vector, hand_frame_basis,
    rotation_from_hand_frame,
)


@dataclass
class HandPose:
    center_3d: np.ndarray      # smoothed anchor C_hand in normalized (x, y, z)
    rotation: np.ndarray       # (3,3) smoothed rotation matrix
    roll: float
    pitch: float
    yaw: float
    palm_center_2d: Tuple[float, float]
    normal: np.ndarray


class HandPoseEstimator:
    def __init__(self, pos_alpha: float = 0.30, rot_alpha: float = 0.22,
                 pitch_scale: float = 1.2, yaw_scale: float = 1.2) -> None:
        self._pos_alpha = pos_alpha
        self._rot_alpha = rot_alpha
        self._pitch_scale = pitch_scale
        self._yaw_scale = yaw_scale
        self._c: Optional[np.ndarray] = None
        self._r: Optional[np.ndarray] = None
        self._roll: Optional[float] = None
        self._pitch: Optional[float] = None
        self._yaw: Optional[float] = None

    def reset(self) -> None:
        self._c = None
        self._r = None
        self._roll = self._pitch = self._yaw = None

    def estimate(self, pts: np.ndarray, handedness: str = "Right") -> HandPose:
        pts = np.asarray(pts, dtype=np.float64)
        l0, l5, l9, l17 = pts[0], pts[5], pts[9], pts[17]

        u, v, n = hand_frame_basis(l0, l5, l9, l17, handedness)
        r_frame = rotation_from_hand_frame(u, v, n)

        # Euler angles from spatial vectors (Test-2 convention).
        raw_roll = math.atan2(float(u[1]), float(u[0])) + math.pi * 0.5
        planar_u = math.sqrt(float(u[0] ** 2 + u[1] ** 2))
        raw_pitch = math.atan2(float(u[2]), max(1e-6, planar_u)) * self._pitch_scale
        planar_v = math.sqrt(float(v[0] ** 2 + v[1] ** 2))
        raw_yaw = math.atan2(float(v[2]), max(1e-6, planar_v)) * self._yaw_scale

        raw_c = (pts[[0, 5, 9, 17]].mean(axis=0)).astype(np.float64)
        self._c = ema_vector(raw_c, self._c, self._pos_alpha)

        if self._r is None:
            self._r = r_frame
            self._roll, self._pitch, self._yaw = raw_roll, raw_pitch, raw_yaw
        else:
            # Slerp-lite: element-wise EMA + re-orthonormalize via SVD.
            blend = self._rot_alpha * r_frame + (1.0 - self._rot_alpha) * self._r
            try:
                U, _, Vt = np.linalg.svd(blend)
                r = U @ Vt
                if np.linalg.det(r) < 0:
                    U[:, -1] *= -1.0
                    r = U @ Vt
                self._r = r
            except np.linalg.LinAlgError:
                pass
            d = (raw_roll - self._roll + math.pi) % (2 * math.pi) - math.pi
            a = self._rot_alpha
            self._roll = float(self._roll + a * d)
            self._pitch = float(self._pitch + a * (raw_pitch - self._pitch))
            self._yaw = float(self._yaw + a * (raw_yaw - self._yaw))

        # Keep Euler matrix in sync is unnecessary; rotation _r is authoritative.
        _ = euler_to_rotation_matrix(self._roll, self._pitch, self._yaw)
        cx, cy = float(self._c[0]), float(self._c[1])
        return HandPose(center_3d=self._c.copy(), rotation=self._r.copy(),
                        roll=float(self._roll), pitch=float(self._pitch),
                        yaw=float(self._yaw), palm_center_2d=(cx, cy),
                        normal=n.copy())
