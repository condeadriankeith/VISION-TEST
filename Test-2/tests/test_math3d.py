"""Unit tests for 3D spatial transformations and perspective projections."""

import unittest
import numpy as np

from math3d.transforms import (
    euler_to_rotation_matrix,
    project_points_3d,
    get_cube_vertices,
    exponential_moving_average,
    CUBE_EDGES,
)


class TestMath3D(unittest.TestCase):
    """Verifies geometric correctness and edge cases of math3d module."""

    def test_identity_rotation(self) -> None:
        """Zero Euler angles must yield the 3x3 identity matrix."""
        r = euler_to_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-7)

    def test_rotation_orthonormality(self) -> None:
        """Rotation matrices must be orthogonal with determinant equal to 1."""
        angles = [(0.3, -0.5, 1.2), (-1.0, 0.8, -0.4), (np.pi / 4, -np.pi / 6, np.pi / 3)]
        for roll, pitch, yaw in angles:
            r = euler_to_rotation_matrix(roll, pitch, yaw)
            # R @ R.T = I
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-7)
            # det(R) = +1
            det = float(np.linalg.det(r))
            self.assertAlmostEqual(det, 1.0, places=6)

    def test_cube_vertices_symmetry(self) -> None:
        """Cube vertices should form 8 points with symmetric extents."""
        size = 140.0
        verts = get_cube_vertices(size)
        self.assertEqual(verts.shape, (8, 3))
        self.assertAlmostEqual(float(np.min(verts)), -70.0)
        self.assertAlmostEqual(float(np.max(verts)), 70.0)
        self.assertEqual(len(CUBE_EDGES), 12)

    def test_origin_projection(self) -> None:
        """Origin (0,0,0) under identity rotation must project exactly to (center_x, center_y)."""
        r_ident = np.eye(3)
        origin = np.array([0.0, 0.0, 0.0])
        screen_pt, z_eye = project_points_3d(
            origin,
            r_ident,
            center_x=640.0,
            center_y=360.0,
            focal_length=500.0,
            camera_distance=500.0,
        )
        self.assertAlmostEqual(float(screen_pt[0]), 640.0)
        self.assertAlmostEqual(float(screen_pt[1]), 360.0)
        self.assertAlmostEqual(float(z_eye), 500.0)

    def test_vectorized_projection(self) -> None:
        """Batched projection should maintain shape (N, 2) and matching depths (N,)."""
        r = euler_to_rotation_matrix(0.1, 0.2, 0.3)
        verts = get_cube_vertices(100.0)
        screen_pts, z_depths = project_points_3d(
            verts,
            r,
            center_x=300.0,
            center_y=300.0,
            focal_length=550.0,
            camera_distance=500.0,
        )
        self.assertEqual(screen_pts.shape, (8, 2))
        self.assertEqual(z_depths.shape, (8,))
        # All depths should be positive and finite
        self.assertTrue(np.all(z_depths > 0))
        self.assertFalse(np.any(np.isnan(screen_pts)))

    def test_near_clip_singularity_guard(self) -> None:
        """Depth values <= 0 must be clamped without throwing ZeroDivisionError or generating NaN."""
        r_ident = np.eye(3)
        # Point placed behind camera distance such that Z_eye would be <= 0
        point_behind = np.array([[0.0, 0.0, -600.0], [50.0, -50.0, -500.0]])
        screen_pts, _ = project_points_3d(
            point_behind,
            r_ident,
            center_x=400.0,
            center_y=300.0,
            focal_length=500.0,
            camera_distance=500.0,
            near_clip_z=1.0,
        )
        self.assertEqual(screen_pts.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(screen_pts)))
        self.assertFalse(np.any(np.isinf(screen_pts)))

    def test_ema_smoothing(self) -> None:
        """EMA should smoothly interpolate between values."""
        val = exponential_moving_average(10.0, 0.0, alpha=0.2)
        self.assertAlmostEqual(float(val), 2.0)

        # Array version
        prev_arr = np.array([1.0, 2.0, 3.0])
        curr_arr = np.array([3.0, 4.0, 5.0])
        smoothed = exponential_moving_average(curr_arr, prev_arr, alpha=0.5)
        np.testing.assert_allclose(smoothed, np.array([2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
