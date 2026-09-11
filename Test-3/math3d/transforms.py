"""3D math: Euler rotations, hand-frame basis, perspective projection, EMA."""

from __future__ import annotations

from typing import Tuple
import numpy as np

# 12 edges of a unit bounding box (8 corners).
BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), float64 (3,3)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ (ry @ rx)


def hand_frame_basis(l0: np.ndarray, l5: np.ndarray,
                     l9: np.ndarray, l17: np.ndarray,
                     handedness: str = "Right"):
    """Compute orthonormal hand frame (u, v, n) per blueprint Section 3.1.

    u = L9 - L0 (longitudinal), v = L5 - L17 (transverse),
    n = (u x v) / ||u x v||.
    Returns orthonormalized (u_hat, v_hat, n_hat).
    """
    u = np.asarray(l9, dtype=np.float64) - np.asarray(l0, dtype=np.float64)
    v = np.asarray(l5, dtype=np.float64) - np.asarray(l17, dtype=np.float64)
    if np.linalg.norm(u) < 1e-9:
        u = np.array([0.0, -1.0, 0.0])
    if np.linalg.norm(v) < 1e-9:
        v = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    if handedness == "Left":
        v = -v
    n = np.cross(u, v)
    if np.linalg.norm(n) < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    # Gram-Schmidt re-orthogonalization for stability.
    v = v - np.dot(v, u) * u
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    return u, v, n


def rotation_from_hand_frame(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Build a rotation matrix whose columns are the hand frame axes.

    Maps local +X -> transverse (v), +Y -> longitudinal (-u, up), +Z -> normal (n).
    Falls back to identity if degenerate.
    """
    m = np.column_stack((v, -u, n))
    # Orthonormalize via SVD (nearest rotation, det=+1).
    try:
        U, _, Vt = np.linalg.svd(m)
        r = U @ Vt
        if np.linalg.det(r) < 0:
            U[:, -1] *= -1.0
            r = U @ Vt
        return r
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)


def get_box_vertices(size: float) -> np.ndarray:
    """8 corners of a cube centred at origin with edge length `size`."""
    h = size * 0.5
    return np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float64)


def transform_vertices(verts: np.ndarray, rot: np.ndarray,
                       scale: float, center: np.ndarray) -> np.ndarray:
    """Blueprint 3.1: v_world = R_hand @ (v_local * s) + C_hand."""
    v = np.asarray(verts, dtype=np.float64) * float(scale)
    return (v @ np.asarray(rot, dtype=np.float64).T) + np.asarray(center, dtype=np.float64)


def project_points(points_3d: np.ndarray, center_x: float, center_y: float,
                   focal: float, cam_dist: float,
                   near_clip: float = 0.05):
    """Perspective project world points to screen.

    Returns (screen_pts (N,2) float, z_eye (N,) float, scale (N,) focal/z).
    """
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    z_eye = pts[:, 2] + float(cam_dist)
    z_safe = np.maximum(z_eye, near_clip)
    inv = float(focal) / z_safe
    sx = float(center_x) + pts[:, 0] * inv
    sy = float(center_y) + pts[:, 1] * inv
    return np.column_stack((sx, sy)), z_eye, inv


def exponential_moving_average(cur, prev, alpha: float):
    a = max(0.001, min(1.0, float(alpha)))
    return a * cur + (1.0 - a) * prev


def ema_vector(cur: np.ndarray, prev: np.ndarray | None, alpha: float) -> np.ndarray:
    cur = np.asarray(cur, dtype=np.float64)
    if prev is None:
        return cur.copy()
    return np.asarray(exponential_moving_average(cur, np.asarray(prev, dtype=np.float64), alpha))
