"""Perspective-projected wireframe cube renderer.

Renders 12 anti-aliased cube edges with depth-sorted painter's layering and a
very subtle single-pixel glow pass. Thinner, crisper lines — no heavy bloom.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config.settings import ColorPalette, Viewport3DConfig
from math3d.transforms import CUBE_EDGES, get_cube_vertices, project_points_3d


class CubeRenderer:
    """Renders 3D wireframe cube with thin crisp lines and minimal depth glow."""

    def __init__(
        self,
        viewport_config: Viewport3DConfig = Viewport3DConfig(),
        palette: ColorPalette = ColorPalette(),
    ) -> None:
        """Initialize cube model vertices and style palette."""
        self._cfg: Viewport3DConfig = viewport_config
        self._palette: ColorPalette = palette

        # Pre-allocate static cube model vertices (8, 3)
        self._local_vertices: np.ndarray = get_cube_vertices(viewport_config.cube_size)
        self._glow_canvas: Optional[np.ndarray] = None
        self._core_canvas: Optional[np.ndarray] = None

    def render(
        self,
        target_frame: np.ndarray,
        rot_matrix: np.ndarray,
        center_x: float,
        center_y: float,
        alpha: float,
    ) -> None:
        """Draw wireframe cube with thin crisp edges and a subtle glow pass.

        Args:
            target_frame: BGR image frame (H, W, 3) modified in-place.
            rot_matrix: 3x3 orientation matrix.
            center_x: Screen center X in pixels (fixed left-side anchor).
            center_y: Screen center Y in pixels (fixed left-side anchor).
            alpha: Global opacity factor [0.0, 1.0] from state machine.
        """
        if alpha <= 0.01:
            return

        h, w = target_frame.shape[:2]

        # 1. Project 8 cube vertices to 2D screen space
        screen_pts, z_depths = project_points_3d(
            self._local_vertices,
            rot_matrix,
            center_x=center_x,
            center_y=center_y,
            focal_length=self._cfg.focal_length,
            camera_distance=self._cfg.camera_distance,
            near_clip_z=self._cfg.near_clip_z,
        )

        # 2. Depth sort 12 edges — furthest drawn first (painter's algorithm)
        edge_depth_list: List[Tuple[float, int, int]] = []
        for p1_idx, p2_idx in CUBE_EDGES:
            avg_depth = float((z_depths[p1_idx] + z_depths[p2_idx]) * 0.5)
            edge_depth_list.append((avg_depth, p1_idx, p2_idx))
        edge_depth_list.sort(key=lambda item: item[0], reverse=True)

        # 3. Reuse scratch canvases — zero heap churn
        if self._glow_canvas is None or self._glow_canvas.shape != target_frame.shape:
            self._glow_canvas = np.zeros_like(target_frame)
            self._core_canvas = np.zeros_like(target_frame)
        else:
            self._glow_canvas.fill(0)
            self._core_canvas.fill(0)

        # 4. Very subtle 2px glow pass — barely perceptible depth haze
        for _, p1_idx, p2_idx in edge_depth_list:
            pt1 = (int(round(screen_pts[p1_idx, 0])), int(round(screen_pts[p1_idx, 1])))
            pt2 = (int(round(screen_pts[p2_idx, 0])), int(round(screen_pts[p2_idx, 1])))
            if -200 <= pt1[0] <= w + 200 and -200 <= pt1[1] <= h + 200:
                cv2.line(self._glow_canvas, pt1, pt2, self._palette.cube_edge_glow, 2, lineType=cv2.LINE_AA)

        # Blend glow at very low opacity — 18% — so lines stay thin and clean
        cv2.addWeighted(self._glow_canvas, 0.18 * alpha, target_frame, 1.0, 0, dst=target_frame)

        # 5. Crisp 1px core edge lines — the primary wireframe
        for _, p1_idx, p2_idx in edge_depth_list:
            pt1 = (int(round(screen_pts[p1_idx, 0])), int(round(screen_pts[p1_idx, 1])))
            pt2 = (int(round(screen_pts[p2_idx, 0])), int(round(screen_pts[p2_idx, 1])))
            if -200 <= pt1[0] <= w + 200 and -200 <= pt1[1] <= h + 200:
                cv2.line(self._core_canvas, pt1, pt2, self._palette.cube_edge_core, 1, lineType=cv2.LINE_AA)

        # 6. Small 2px accent dots at each corner vertex
        for idx in range(8):
            corner_pt = (int(round(screen_pts[idx, 0])), int(round(screen_pts[idx, 1])))
            if 0 <= corner_pt[0] < w and 0 <= corner_pt[1] < h:
                cv2.circle(self._core_canvas, corner_pt, 2, self._palette.cube_corner, -1, lineType=cv2.LINE_AA)

        # Composite crisp core at full alpha
        cv2.addWeighted(self._core_canvas, alpha, target_frame, 1.0, 0, dst=target_frame)
