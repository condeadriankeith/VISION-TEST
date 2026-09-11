"""3D hand pose and spatial orientation estimator.

Computes longitudinal, transverse, and normal orientation vectors from anatomical hand landmarks
and derives smoothed 3D Euler angles (Roll, Pitch, Yaw) with temporal filtering.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from math3d.transforms import euler_to_rotation_matrix, exponential_moving_average


@dataclass
class HandOrientation:
    """Estimated 3D orientation and spatial pose parameters."""
    roll: float              # Rotation about X-axis in radians (phi)
    pitch: float             # Rotation about Y-axis in radians (theta)
    yaw: float               # Rotation about Z-axis in radians (psi)
    rotation_matrix: np.ndarray  # 3x3 orthonormal rotation matrix
    palm_center_2d: Tuple[float, float]  # Normalized [0, 1] palm center (x, y)
    normal_vector: np.ndarray            # 3D unit normal vector (nx, ny, nz)


class HandPoseEstimator:
    """Estimates and temporally filters 3D orientation angles from MediaPipe landmarks."""

    def __init__(
        self,
        smoothing_alpha: float = 0.22,
        pitch_scale: float = 1.2,
        yaw_scale: float = 1.2,
    ) -> None:
        """Initialize pose estimator.

        Args:
            smoothing_alpha: Weight for EMA temporal filter (0.0 < alpha <= 1.0).
            pitch_scale: Empirical scaling factor for pitch responsiveness (kappa_pitch).
            yaw_scale: Empirical scaling factor for yaw responsiveness (kappa_yaw).
        """
        self._smoothing_alpha: float = max(0.01, min(1.0, smoothing_alpha))
        self._pitch_scale: float = pitch_scale
        self._yaw_scale: float = yaw_scale

        # Internal smoothed state variables
        self._smoothed_roll: Optional[float] = None
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None
        self._smoothed_center_x: Optional[float] = None
        self._smoothed_center_y: Optional[float] = None

    def reset(self) -> None:
        """Reset temporal filter memory when tracking is lost."""
        self._smoothed_roll = None
        self._smoothed_pitch = None
        self._smoothed_yaw = None
        self._smoothed_center_x = None
        self._smoothed_center_y = None

    def estimate_orientation(
        self,
        landmarks_3d: np.ndarray,
        handedness: str = "Right",
    ) -> HandOrientation:
        """Derive 3D Euler angles and rotation matrix from 21 MediaPipe landmarks.

        Args:
            landmarks_3d: (21, 3) ndarray containing [x, y, z] for each landmark.
            handedness: "Right" or "Left" hand label.

        Returns:
            HandOrientation instance containing smoothed angles and rotation matrix.
        """
        # Landmark indices:
        # L0 = Wrist, L5 = Index MCP, L9 = Middle MCP, L17 = Pinky MCP
        l0 = landmarks_3d[0]
        l5 = landmarks_3d[5]
        l9 = landmarks_3d[9]
        l17 = landmarks_3d[17]

        # 1. Longitudinal vector: u = L9 - L0 (wrist to middle knuckle)
        u = l9 - l0
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-6:
            u = u / u_norm
        else:
            u = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        # 2. Transverse vector: v = L5 - L17 (pinky knuckle to index knuckle)
        v = l5 - l17
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-6:
            v = v / v_norm
        else:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Invert transverse vector if handedness is Left to ensure uniform palm normal
        if handedness == "Left":
            v = -v

        # 3. Normal vector: n = (u x v) / ||u x v||
        cross_uv = np.cross(u, v)
        cross_norm = np.linalg.norm(cross_uv)
        if cross_norm > 1e-6:
            n = cross_uv / cross_norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 4. Derive Euler angles from spatial vectors (Document Section 3.2):
        # Roll: phi = atan2(u_y, u_x) + pi/2
        # When hand points directly up (u_x = 0, u_y = -1), phi = atan2(-1, 0) + pi/2 = -pi/2 + pi/2 = 0
        raw_roll = math.atan2(float(u[1]), float(u[0])) + (math.pi * 0.5)

        # Pitch: theta = atan2(u_z, sqrt(u_x^2 + u_y^2)) * kappa_pitch
        planar_u = math.sqrt(float(u[0] ** 2 + u[1] ** 2))
        raw_pitch = math.atan2(float(u[2]), max(1e-6, planar_u)) * self._pitch_scale

        # Yaw: psi = atan2(v_z, sqrt(v_x^2 + v_y^2)) * kappa_yaw
        planar_v = math.sqrt(float(v[0] ** 2 + v[1] ** 2))
        raw_yaw = math.atan2(float(v[2]), max(1e-6, planar_v)) * self._yaw_scale

        # 5. Palm 2D Center calculation (mean of key anchors)
        raw_cx = float((l0[0] + l5[0] + l9[0] + l17[0]) * 0.25)
        raw_cy = float((l0[1] + l5[1] + l9[1] + l17[1]) * 0.25)

        # 6. Apply Temporal Smoothing (EMA)
        if self._smoothed_roll is None:
            self._smoothed_roll = raw_roll
            self._smoothed_pitch = raw_pitch
            self._smoothed_yaw = raw_yaw
            self._smoothed_center_x = raw_cx
            self._smoothed_center_y = raw_cy
        else:
            # Angle unwrapping for roll to prevent jitter across +-pi boundary
            diff_roll = (raw_roll - self._smoothed_roll + math.pi) % (2.0 * math.pi) - math.pi
            target_roll = self._smoothed_roll + diff_roll
            self._smoothed_roll = float(exponential_moving_average(
                target_roll, self._smoothed_roll, self._smoothing_alpha
            ))

            self._smoothed_pitch = float(exponential_moving_average(
                raw_pitch, self._smoothed_pitch, self._smoothing_alpha
            ))
            self._smoothed_yaw = float(exponential_moving_average(
                raw_yaw, self._smoothed_yaw, self._smoothing_alpha
            ))
            self._smoothed_center_x = float(exponential_moving_average(
                raw_cx, self._smoothed_center_x, self._smoothing_alpha
            ))
            self._smoothed_center_y = float(exponential_moving_average(
                raw_cy, self._smoothed_center_y, self._smoothing_alpha
            ))

        # 7. Compute rotation matrix
        rot_matrix = euler_to_rotation_matrix(
            self._smoothed_roll,
            self._smoothed_pitch,
            self._smoothed_yaw,
        )

        return HandOrientation(
            roll=self._smoothed_roll,
            pitch=self._smoothed_pitch,
            yaw=self._smoothed_yaw,
            rotation_matrix=rot_matrix,
            palm_center_2d=(self._smoothed_center_x, self._smoothed_center_y),
            normal_vector=n,
        )
