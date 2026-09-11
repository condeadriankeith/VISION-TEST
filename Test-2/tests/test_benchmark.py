"""Headless throughput benchmark and memory footprint validation for Test-2."""

import time
import tracemalloc
import unittest
import numpy as np

from config.settings import SystemSettings
from core.state_machine import GestureState, GestureStateMachine
from math3d.transforms import euler_to_rotation_matrix
from rendering.cube_renderer import CubeRenderer
from rendering.mesh_renderer import MeshRenderer
from rendering.renderer_2d import HUDOverlayRenderer
from simulation.particle_system import ParticleSystem


class TestBenchmark(unittest.TestCase):
    """Measures sustained throughput (FPS) and ensures constant memory footprint."""

    def test_pipeline_throughput(self) -> None:
        """Measure sustained throughput across 200 full frames (Physics, Grid, Cube, Mesh, HUD)."""
        settings = SystemSettings()
        particle_sys = ParticleSystem(settings.particle, settings.viewport)
        cube_renderer = CubeRenderer(settings.viewport, settings.colors)
        mesh_renderer = MeshRenderer(settings.particle, settings.viewport, settings.colors)
        hud_renderer = HUDOverlayRenderer(settings.colors)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rot_matrix = euler_to_rotation_matrix(0.2, 0.1, -0.3)

        num_frames = 200
        t_start = time.perf_counter()
        for i in range(num_frames):
            dt = 0.016
            elapsed = i * 0.016

            particle_sys.update(dt, elapsed)
            cube_renderer.render(frame, rot_matrix, 640.0, 360.0, alpha=1.0)
            mesh_renderer.render(
                frame,
                particle_sys.positions,
                particle_sys.mesh_edges,
                rot_matrix,
                640.0,
                360.0,
                alpha=1.0,
            )
            hud_renderer.render(frame, fps=60.0, frame_time_ms=16.6, current_state=GestureState.OPEN_PALM)

        t_total = time.perf_counter() - t_start
        fps = num_frames / max(0.001, t_total)
        avg_frame_ms = (t_total / num_frames) * 1000.0

        print(f"\n[BENCHMARK] Sustained Throughput: {fps:.1f} FPS ({avg_frame_ms:.2f} ms/frame)")
        # 35 FPS threshold: cloud mode has 550 particles (2.5x original), with full physics + render passes.
        # Real interactive session is faster because MediaPipe inference dominates the loop timing.
        self.assertGreaterEqual(fps, 35.0, f"Expected >= 35 FPS, achieved {fps:.1f} FPS")

    def test_memory_stability(self) -> None:
        """Verify memory allocations remain bounded with near-zero growth."""
        settings = SystemSettings()
        particle_sys = ParticleSystem(settings.particle, settings.viewport)
        cube_renderer = CubeRenderer(settings.viewport, settings.colors)
        mesh_renderer = MeshRenderer(settings.particle, settings.viewport, settings.colors)
        hud_renderer = HUDOverlayRenderer(settings.colors)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rot_matrix = euler_to_rotation_matrix(0.2, 0.1, -0.3)

        # Warm up buffers
        for i in range(10):
            particle_sys.update(0.016, i * 0.016)
            cube_renderer.render(frame, rot_matrix, 640.0, 360.0, alpha=1.0)
            mesh_renderer.render(frame, particle_sys.positions, particle_sys.mesh_edges, rot_matrix, 640.0, 360.0, alpha=1.0)
            hud_renderer.render(frame, 60.0, 16.6, GestureState.OPEN_PALM)

        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()

        for i in range(50):
            dt = 0.016
            elapsed = i * 0.016
            particle_sys.update(dt, elapsed)
            cube_renderer.render(frame, rot_matrix, 640.0, 360.0, alpha=1.0)
            mesh_renderer.render(frame, particle_sys.positions, particle_sys.mesh_edges, rot_matrix, 640.0, 360.0, alpha=1.0)
            hud_renderer.render(frame, 60.0, 16.6, GestureState.OPEN_PALM)

        snap_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diffs = snap_end.compare_to(snap_start, "lineno")
        delta_kb = sum(stat.size_diff for stat in diffs) / 1024.0

        print(f"[BENCHMARK] Total RAM Delta over 50 frames: {delta_kb:.2f} KB")
        self.assertLessEqual(delta_kb, 500.0, "Unbounded memory growth detected!")


if __name__ == "__main__":
    unittest.main()
