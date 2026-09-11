"""Unit tests for particle simulation, curl flow field, and spatial voxel grid."""

import unittest
import numpy as np

from config.settings import ParticleConfig, Viewport3DConfig
from simulation.fields import compute_curl_flow_field
from simulation.particle_system import ParticleSystem
from math3d.spatial_grid import SpatialVoxelGrid


class TestSimulation(unittest.TestCase):
    """Verifies particle dynamics and spatial grid neighbor queries."""

    def test_curl_flow_field_dimensions(self) -> None:
        """Flow field must output matching (N, 3) matrix without NaN or Inf."""
        pts = np.random.uniform(-50.0, 50.0, size=(100, 3))
        flow = compute_curl_flow_field(pts, time_sec=1.5, curl_scale=0.024)
        self.assertEqual(flow.shape, (100, 3))
        self.assertFalse(np.any(np.isnan(flow)))
        self.assertFalse(np.any(np.isinf(flow)))

    def test_spatial_voxel_grid(self) -> None:
        """Voxel grid must detect connections within threshold and reject distant pairs."""
        grid = SpatialVoxelGrid(cell_size=30.0)
        # 3 points: p0 and p1 close (dist = 10), p2 far away (dist = 200)
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [200.0, 200.0, 200.0],
        ], dtype=np.float64)

        edges = grid.find_mesh_connections(pts, max_distance=15.0, max_neighbors=4)
        self.assertEqual(len(edges), 1)
        i, j, dist = edges[0]
        self.assertEqual((i, j), (0, 1))
        self.assertAlmostEqual(dist, 10.0, places=5)

    def test_particle_system_boundary_containment(self) -> None:
        """Particles must remain strictly within cube bounding volume over time."""
        p_cfg = ParticleConfig(max_particles=50)
        v_cfg = Viewport3DConfig(cube_size=100.0)
        ps = ParticleSystem(p_cfg, v_cfg)

        half_bound = (v_cfg.cube_size * 0.5) * p_cfg.bounding_ratio

        # Step simulation 120 times
        for step in range(120):
            ps.update(dt=0.016, elapsed_time=step * 0.016)

        # Confirm all particles are within bounded extents (with small numerical tolerance)
        max_coord = np.max(np.abs(ps.positions))
        self.assertLessEqual(max_coord, half_bound + 1e-4)


if __name__ == "__main__":
    unittest.main()
