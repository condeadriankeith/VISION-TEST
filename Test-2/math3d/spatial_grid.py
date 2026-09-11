"""Spatial voxel grid partitioning for O(N) neighbor queries.

Divides 3D continuous space into discrete uniform voxel cells of size d_conn,
replacing O(N^2) brute-force pairwise distance checks with localized 3x3x3 cell queries.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


class SpatialVoxelGrid:
    """Uniform 3D spatial hash grid for fast proximity searches."""

    def __init__(self, cell_size: float = 52.0) -> None:
        """Initialize voxel grid.

        Args:
            cell_size: Edge length of each cubic voxel (set equal to connection_distance).
        """
        self._cell_size: float = max(1.0, float(cell_size))
        self._inv_cell_size: float = 1.0 / self._cell_size

    def set_cell_size(self, cell_size: float) -> None:
        """Update voxel cell dimension."""
        self._cell_size = max(1.0, float(cell_size))
        self._inv_cell_size = 1.0 / self._cell_size

    def find_mesh_connections(
        self,
        positions: np.ndarray,
        max_distance: float,
        max_neighbors: int = 6,
    ) -> List[Tuple[int, int, float]]:
        """Query adjacent 3x3x3 voxels to assemble undirected edges between nearby particles.

        Args:
            positions: (N, 3) ndarray of 3D particle positions.
            max_distance: Maximum Euclidean distance for an edge (d_conn).
            max_neighbors: Maximum permitted degree per particle node.

        Returns:
            List of (idx_a, idx_b, distance) tuples representing mesh edges.
        """
        num_particles = len(positions)
        if num_particles < 2:
            return []

        # 1. Bucket particles into discrete 3D integer coordinates: cell = floor(P / d_conn)
        cell_coords = np.floor(positions * self._inv_cell_size).astype(np.int32)
        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)

        for idx in range(num_particles):
            key = (int(cell_coords[idx, 0]), int(cell_coords[idx, 1]), int(cell_coords[idx, 2]))
            buckets[key].append(idx)

        # 2. Localized 3x3x3 query
        max_dist_sq = max_distance * max_distance
        edges: List[Tuple[int, int, float]] = []
        neighbor_counts = np.zeros(num_particles, dtype=np.int32)

        # Offsets for 27 neighboring cells (dx, dy, dz in {-1, 0, 1})
        # To avoid duplicate symmetric lookups, we evaluate cell pairs canonically
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        for cell_key, p_indices in buckets.items():
            cx, cy, cz = cell_key
            for i in p_indices:
                if neighbor_counts[i] >= max_neighbors:
                    continue

                pos_i = positions[i]

                # Check particles in all adjacent 27 voxels
                for dx, dy, dz in neighbor_offsets:
                    adj_key = (cx + dx, cy + dy, cz + dz)
                    adj_indices = buckets.get(adj_key)
                    if not adj_indices:
                        continue

                    for j in adj_indices:
                        # Strictly check j > i to guarantee each undirected edge is processed once
                        if j <= i:
                            continue

                        if neighbor_counts[j] >= max_neighbors:
                            continue

                        pos_j = positions[j]
                        dx_pos = pos_i[0] - pos_j[0]
                        dy_pos = pos_i[1] - pos_j[1]
                        dz_pos = pos_i[2] - pos_j[2]
                        dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos

                        if dist_sq <= max_dist_sq:
                            dist = float(np.sqrt(dist_sq))
                            edges.append((i, j, dist))
                            neighbor_counts[i] += 1
                            neighbor_counts[j] += 1

                            if neighbor_counts[i] >= max_neighbors:
                                break

                    if neighbor_counts[i] >= max_neighbors:
                        break

        return edges
