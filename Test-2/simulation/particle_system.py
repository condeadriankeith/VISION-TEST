"""Vectorized 3D particle physics simulation and dynamic connection network.

Maintains contiguous position and velocity matrices, applies 3D trigonometric
flow vectors, enforces elastic cube boundary collisions, and queries spatial mesh links.
"""

from typing import List, Optional, Tuple
import numpy as np

from config.settings import ParticleConfig, Viewport3DConfig
from simulation.fields import compute_curl_flow_field
from math3d.spatial_grid import SpatialVoxelGrid


class ParticleSystem:
    """Vectorized particle physics engine confined within a 3D bounding volume."""

    def __init__(
        self,
        particle_config: ParticleConfig = ParticleConfig(),
        viewport_config: Viewport3DConfig = Viewport3DConfig(),
    ) -> None:
        """Initialize particle system buffers.

        Args:
            particle_config: Physical parameters (count, flow speed, damping, etc.).
            viewport_config: Geometric bounds of the container cube.
        """
        self._cfg: ParticleConfig = particle_config
        self._num_particles: int = particle_config.max_particles

        # Cube physical bounds: half-extent scaled by bounding ratio
        half_cube = (viewport_config.cube_size * 0.5)
        self._bound: float = half_cube * particle_config.bounding_ratio

        # Contiguous memory buffers for state
        self._positions: np.ndarray = np.zeros((self._num_particles, 3), dtype=np.float64)
        self._velocities: np.ndarray = np.zeros((self._num_particles, 3), dtype=np.float64)

        # Spatial grid partitioner for O(N) neighbor mesh queries
        self._grid: SpatialVoxelGrid = SpatialVoxelGrid(cell_size=particle_config.connection_distance)

        # Active mesh connectivity edges: list of (idx_a, idx_b, distance)
        self._mesh_edges: List[Tuple[int, int, float]] = []

        self.reset()

    def reset(self, seed: Optional[int] = 42) -> None:
        """Randomly seed particles uniformly distributed within the interior bounding box."""
        rng = np.random.default_rng(seed)
        b = self._bound * 0.90
        self._positions = rng.uniform(-b, b, size=(self._num_particles, 3))
        self._velocities = rng.uniform(-0.5, 0.5, size=(self._num_particles, 3))
        self._mesh_edges.clear()

    @property
    def positions(self) -> np.ndarray:
        """Contiguous (N, 3) array of particle 3D coordinates."""
        return self._positions

    @property
    def mesh_edges(self) -> List[Tuple[int, int, float]]:
        """List of active connection edges (idx_a, idx_b, distance)."""
        return self._mesh_edges

    def update(self, dt: float, elapsed_time: float) -> None:
        """Advance particle simulation by time step dt.

        Args:
            dt: Frame delta time in seconds.
            elapsed_time: Monotonic application running time in seconds.
        """
        # Clamp dt to prevent explosion on stalls
        safe_dt = min(0.05, max(0.001, dt))
        dt_scale = safe_dt * 60.0  # Normalize to 60 FPS baseline

        # 1. Evaluate procedural curl-noise flow vectors at current positions
        target_vel = compute_curl_flow_field(
            self._positions,
            elapsed_time,
            curl_scale=self._cfg.curl_scale,
        )

        # 2. Vectorized velocity integration with momentum damping
        blend = 1.0 - self._cfg.damping
        self._velocities = (
            self._velocities * self._cfg.damping
            + target_vel * (blend * self._cfg.flow_speed)
        )

        # 3. Vectorized position integration
        self._positions += self._velocities * dt_scale

        # 4. Vectorized boundary collision & reflection against cube boundaries [-bound, bound]
        bound = self._bound
        for axis in range(3):
            pos_axis = self._positions[:, axis]
            vel_axis = self._velocities[:, axis]

            # Positive wall collision
            pos_mask = pos_axis > bound
            if np.any(pos_mask):
                overshoot = pos_axis[pos_mask] - bound
                pos_axis[pos_mask] = bound - overshoot
                vel_axis[pos_mask] = -np.abs(vel_axis[pos_mask]) * 0.75

            # Negative wall collision
            neg_mask = pos_axis < -bound
            if np.any(neg_mask):
                overshoot = -bound - pos_axis[neg_mask]
                pos_axis[neg_mask] = -bound + overshoot
                vel_axis[neg_mask] = np.abs(vel_axis[neg_mask]) * 0.75

        # 5. Spatial voxel grid query for dynamic mesh edges
        self._mesh_edges = self._grid.find_mesh_connections(
            self._positions,
            max_distance=self._cfg.connection_distance,
            max_neighbors=self._cfg.max_neighbors_per_particle,
        )
