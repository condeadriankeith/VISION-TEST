"""Procedural particle mesh renderer with perspective node scaling and dynamic proximity edges.

Renders depth-attenuated particle nodes and Euclidean proximity-faded connection links
with anti-aliasing and smooth global state alpha crossfading.
"""

from typing import List, Tuple
import cv2
import numpy as np

from config.settings import ColorPalette, ParticleConfig, Viewport3DConfig
from math3d.transforms import project_points_3d


class MeshRenderer:
    """Renders 3D dynamic particle nodes and connecting mesh edges."""

    def __init__(
        self,
        particle_config: ParticleConfig = ParticleConfig(),
        viewport_config: Viewport3DConfig = Viewport3DConfig(),
        palette: ColorPalette = ColorPalette(),
    ) -> None:
        """Initialize rendering parameters and color palettes."""
        self._p_cfg: ParticleConfig = particle_config
        self._v_cfg: Viewport3DConfig = viewport_config
        self._palette: ColorPalette = palette
        self._conn_dist: float = particle_config.connection_distance
        self._mesh_canvas: Optional[np.ndarray] = None

    def render(
        self,
        target_frame: np.ndarray,
        positions_3d: np.ndarray,
        mesh_edges: List[Tuple[int, int, float]],
        rot_matrix: np.ndarray,
        center_x: float,
        center_y: float,
        alpha: float,
    ) -> None:
        """Render particle nodes and connecting mesh links onto target_frame.

        Args:
            target_frame: BGR destination canvas (H, W, 3) modified in-place.
            positions_3d: (N, 3) particle local coordinates.
            mesh_edges: List of (idx_a, idx_b, distance) connectivity links.
            rot_matrix: 3x3 model orientation matrix.
            center_x: Screen center X.
            center_y: Screen center Y.
            alpha: Global opacity factor [0.0, 1.0].
        """
        if alpha <= 0.01 or len(positions_3d) == 0:
            return

        h, w = target_frame.shape[:2]

        # 1. Project all particle 3D positions to 2D screen space
        screen_pts, z_depths = project_points_3d(
            positions_3d,
            rot_matrix,
            center_x=center_x,
            center_y=center_y,
            focal_length=self._v_cfg.focal_length,
            camera_distance=self._v_cfg.camera_distance,
            near_clip_z=self._v_cfg.near_clip_z,
        )

        # 2. Reuse scratch canvas for semi-transparent particle and mesh elements
        if self._mesh_canvas is None or self._mesh_canvas.shape != target_frame.shape:
            self._mesh_canvas = np.zeros_like(target_frame)
        else:
            self._mesh_canvas.fill(0)

        mesh_canvas = self._mesh_canvas

        # 3. Draw proximity-weighted mesh connection edges
        base_edge_color = np.array(self._palette.mesh_edge, dtype=np.float32)
        inv_conn_dist = 1.0 / self._conn_dist

        for idx_a, idx_b, dist in mesh_edges:
            # Alpha factor proportional to Euclidean proximity: (1.0 - d / d_conn)
            proximity = max(0.0, min(1.0, 1.0 - (dist * inv_conn_dist)))
            if proximity < 0.05:
                continue

            pt1 = (int(round(screen_pts[idx_a, 0])), int(round(screen_pts[idx_a, 1])))
            pt2 = (int(round(screen_pts[idx_b, 0])), int(round(screen_pts[idx_b, 1])))

            if -100 <= pt1[0] <= w + 100 and -100 <= pt1[1] <= h + 100:
                # Modulate edge color brightness based on proximity factor
                edge_col = tuple((base_edge_color * proximity).astype(int).tolist())
                cv2.line(mesh_canvas, pt1, pt2, edge_col, 1, lineType=cv2.LINE_AA)

        # 4. Draw depth-scaled particle nodes
        base_node_color = np.array(self._palette.particle_node, dtype=np.float32)
        base_radius = self._p_cfg.node_base_radius
        focal_len = self._v_cfg.focal_length
        cam_dist = self._v_cfg.camera_distance

        for i in range(len(positions_3d)):
            pt = (int(round(screen_pts[i, 0])), int(round(screen_pts[i, 1])))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                depth = z_depths[i]
                # Scale node radius inversely with camera distance: r = base_r * (f / Z_eye)
                depth_scale = focal_len / max(1.0, depth)
                radius = int(max(1, min(6, round(base_radius * depth_scale * 0.9))))

                # Depth cueing factor: closer particles are brighter [0.4 to 1.0]
                depth_brightness = max(0.4, min(1.0, (cam_dist * 1.2) / max(1.0, depth)))
                node_col = tuple((base_node_color * depth_brightness).astype(int).tolist())

                cv2.circle(mesh_canvas, pt, radius, node_col, -1, lineType=cv2.LINE_AA)

        # 5. Fast alpha composition directly into destination frame
        cv2.addWeighted(mesh_canvas, alpha, target_frame, 1.0, 0, dst=target_frame)
