"""3D coordinate transformations, Euler rotations, and perspective projection.

Implements vectorized linear algebra routines for rotating 3D geometries and
projecting them onto a 2D camera viewport with strict depth guards.
"""

from typing import List, Tuple, Union
import numpy as np

# Standard 12 wireframe edge connectivity pairs for an 8-corner cube
CUBE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting lateral ribs
)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compute combined 3D rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll).

    Args:
        roll: Rotation around X-axis in radians (phi).
        pitch: Rotation around Y-axis in radians (theta).
        yaw: Rotation around Z-axis in radians (psi).

    Returns:
        3x3 orthonormal rotation matrix as float64 ndarray.
    """
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Rx(roll)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_r, -sin_r],
        [0.0, sin_r, cos_r],
    ], dtype=np.float64)

    # Ry(pitch)
    ry = np.array([
        [cos_p, 0.0, sin_p],
        [0.0, 1.0, 0.0],
        [-sin_p, 0.0, cos_p],
    ], dtype=np.float64)

    # Rz(yaw)
    rz = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Combined transformation: R = Rz * Ry * Rx
    return rz @ (ry @ rx)


def get_cube_vertices(size: float) -> np.ndarray:
    """Generate 8 3D vertices for a cube centered at origin (0, 0, 0).

    Args:
        size: Total edge length of the cube.

    Returns:
        (8, 3) ndarray of vertex coordinates [X, Y, Z].
    """
    half_s = size * 0.5
    return np.array([
        [-half_s, -half_s, -half_s],  # 0
        [ half_s, -half_s, -half_s],  # 1
        [ half_s,  half_s, -half_s],  # 2
        [-half_s,  half_s, -half_s],  # 3
        [-half_s, -half_s,  half_s],  # 4
        [ half_s, -half_s,  half_s],  # 5
        [ half_s,  half_s,  half_s],  # 6
        [-half_s,  half_s,  half_s],  # 7
    ], dtype=np.float64)


def project_points_3d(
    points_3d: np.ndarray,
    rot_matrix: np.ndarray,
    center_x: float,
    center_y: float,
    focal_length: float,
    camera_distance: float,
    near_clip_z: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized perspective projection of 3D points onto 2D screen viewport.

    P_rot = P * R^T (equivalent to R * P for column vectors)
    Z_eye = P_rot[:, 2] + camera_distance
    x_screen = center_x + (focal_length * P_rot[:, 0]) / max(Z_eye, near_clip_z)
    y_screen = center_y + (focal_length * P_rot[:, 1]) / max(Z_eye, near_clip_z)

    Args:
        points_3d: (N, 3) or (3,) ndarray of 3D positions in local model space.
        rot_matrix: (3, 3) orthonormal rotation matrix.
        center_x: Viewport principal point X on screen (pixels).
        center_y: Viewport principal point Y on screen (pixels).
        focal_length: Synthetic pinhole focal length (f).
        camera_distance: Eye distance along Z axis (d_cam).
        near_clip_z: Strict depth guard protecting against division by zero/negative depth.

    Returns:
        Tuple of:
            - screen_pts: (N, 2) projected 2D coordinates [x, y].
            - z_eye: (N,) eye-space depth values.
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    is_single = (pts.ndim == 1)
    if is_single:
        pts = pts.reshape(1, 3)

    # Vectorized rotation: (N, 3) @ (3, 3)
    # Since rot_matrix acts as R @ p, for row vectors pts @ R.T is mathematically equivalent
    rotated_pts = pts @ rot_matrix.T

    # Calculate eye-space depth
    z_eye_raw = rotated_pts[:, 2] + camera_distance

    # Strict singularity guard: clamp minimum depth to near_clip_z to prevent division by zero or inversion
    z_eye_clamped = np.maximum(z_eye_raw, near_clip_z)

    # Perspective division
    inv_z = focal_length / z_eye_clamped
    screen_x = center_x + rotated_pts[:, 0] * inv_z
    screen_y = center_y + rotated_pts[:, 1] * inv_z

    screen_pts = np.column_stack((screen_x, screen_y))

    if is_single:
        return screen_pts[0], float(z_eye_raw[0])

    return screen_pts, z_eye_raw


def exponential_moving_average(
    current_val: Union[float, np.ndarray],
    previous_val: Union[float, np.ndarray],
    alpha: float,
) -> Union[float, np.ndarray]:
    """Low-pass Exponential Moving Average (EMA) filter.

    hat{x}_t = alpha * x_t + (1 - alpha) * hat{x}_{t-1}

    Args:
        current_val: New raw observation at time t.
        previous_val: Prior smoothed estimate at time t-1.
        alpha: Smoothing factor in range (0.0, 1.0]. Higher values prioritize reactivity;
               lower values prioritize noise suppression.

    Returns:
        Smoothed value of same shape and type.
    """
    clamped_alpha = max(0.001, min(1.0, float(alpha)))
    return clamped_alpha * current_val + (1.0 - clamped_alpha) * previous_val
