"""Unit tests for 3D cube, particle mesh, and 2D HUD renderers."""

import unittest
import numpy as np

from config.settings import SystemSettings
from core.state_machine import GestureState
from rendering.cube_renderer import CubeRenderer
from rendering.mesh_renderer import MeshRenderer
from rendering.renderer_2d import HUDOverlayRenderer


class TestRendering(unittest.TestCase):
    """Verifies render passes without throwing errors or mutating dimensions."""

    def setUp(self) -> None:
        self.settings = SystemSettings()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.rot_matrix = np.eye(3, dtype=np.float64)

    def test_cube_renderer_execution(self) -> None:
        """Cube renderer must draw without throwing and preserve frame shape."""
        renderer = CubeRenderer(self.settings.viewport, self.settings.colors)
        # Render with full alpha
        renderer.render(
            self.frame,
            self.rot_matrix,
            center_x=640.0,
            center_y=360.0,
            alpha=1.0,
        )
        self.assertEqual(self.frame.shape, (720, 1280, 3))
        # Non-zero pixels must exist (edges were drawn)
        self.assertTrue(np.any(self.frame > 0))

    def test_mesh_renderer_execution(self) -> None:
        """Mesh renderer must draw nodes and proximity links."""
        renderer = MeshRenderer(self.settings.particle, self.settings.viewport, self.settings.colors)
        pts = np.array([
            [-20.0, 0.0, 0.0],
            [ 20.0, 0.0, 0.0],
        ], dtype=np.float64)
        edges = [(0, 1, 40.0)]

        renderer.render(
            self.frame,
            pts,
            edges,
            self.rot_matrix,
            center_x=640.0,
            center_y=360.0,
            alpha=1.0,
        )
        self.assertEqual(self.frame.shape, (720, 1280, 3))
        self.assertTrue(np.any(self.frame > 0))

    def test_hud_overlay_execution(self) -> None:
        """HUD overlay must render pill in upper-right corner without throwing."""
        hud = HUDOverlayRenderer(self.settings.colors)
        hud.render(
            self.frame,
            fps=59.8,
            frame_time_ms=16.7,
            current_state=GestureState.OPEN_PALM,
        )
        self.assertEqual(self.frame.shape, (720, 1280, 3))
        self.assertTrue(np.any(self.frame > 0))


if __name__ == "__main__":
    unittest.main()
