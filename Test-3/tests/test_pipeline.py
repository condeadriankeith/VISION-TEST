"""Unit tests: gestures, transforms, particles, procedural models, state machine."""

import math
import unittest
import numpy as np


def _synth_hand(extended):
    """Build fake (21,3) landmarks: extended fingers reach y=-1, folded stay near palm."""
    pts = np.zeros((21, 3))
    pts[0] = [0, 0, 0]
    # finger chains: mcp, pip, tip triples along -y when extended
    chains = {8: (5, 6), 12: (9, 10), 16: (13, 14), 20: (17, 18)}
    mcps = {5: (-0.3, -0.3), 9: (0, -0.35), 13: (0.3, -0.3), 17: (0.5, -0.25)}
    for tip, (mcp, pip) in chains.items():
        mx, my = mcps[mcp]
        pts[mcp] = [mx, my, 0]
        pts[pip] = [mx, my - 0.2, 0]
        name = {8: 1, 12: 2, 16: 3, 20: 4}[tip]
        on = extended[name] if isinstance(extended, dict) else False
        pts[tip] = [mx, my - 0.6, 0] if on else [mx * 0.5, my + 0.05, 0]
    # thumb: ip=3, tip=4 relative to index mcp=5
    pts[2] = [-0.35, -0.15, 0]
    pts[3] = [-0.45, -0.2, 0]
    thumb_on = extended[0] if isinstance(extended, dict) else False
    pts[4] = [-0.75, -0.35, 0] if thumb_on else [-0.32, -0.22, 0]
    pts[5] = [-0.3, -0.3, 0]
    pts[1] = [-0.2, -0.1, 0]
    return pts


class TestGestures(unittest.TestCase):
    def test_mapping(self):
        from vision.gestures import classify_finger_extensions, classify_gesture, Gesture
        cases = [
            ({0: False, 1: True, 2: False, 3: False, 4: False}, Gesture.INDEX),
            ({0: False, 1: False, 2: False, 3: False, 4: False}, Gesture.FIST),
            ({0: False, 1: True, 2: True, 3: False, 4: False}, Gesture.PEACE),
            ({0: True, 1: True, 2: True, 3: True, 4: True}, Gesture.OPEN_PALM),
        ]
        for ext, want in cases:
            pts = _synth_hand(ext)
            fs = classify_finger_extensions(pts)
            self.assertEqual(classify_gesture(fs, pts), want, f"ext={ext} got states={fs}")
        # L-shape: thumb+index, check angle gate passes or is UNKNOWN (never misfires)
        pts = _synth_hand({0: True, 1: True, 2: False, 3: False, 4: False})
        fs = classify_finger_extensions(pts)
        g = classify_gesture(fs, pts)
        self.assertIn(g, (Gesture.L_SHAPE, Gesture.UNKNOWN))

    def test_thumb_index_angle(self):
        from vision.gestures import thumb_index_angle_deg
        pts = np.zeros((21, 3))
        pts[2] = [0, 0, 0]; pts[4] = [1, 0, 0]
        pts[5] = [0, 0, 0]; pts[8] = [0, 1, 0]
        self.assertAlmostEqual(thumb_index_angle_deg(pts), 90.0, places=4)


class TestTransforms(unittest.TestCase):
    def test_rotation_orthonormal(self):
        from math3d.transforms import euler_to_rotation_matrix
        r = euler_to_rotation_matrix(0.3, -0.2, 0.5)
        self.assertTrue(np.allclose(r @ r.T, np.eye(3), atol=1e-9))
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=9)

    def test_hand_frame(self):
        from math3d.transforms import hand_frame_basis
        l0 = np.array([0.5, 0.6, 0.0])
        l9 = np.array([0.5, 0.4, 0.0])
        l5 = np.array([0.4, 0.45, 0.0])
        l17 = np.array([0.6, 0.45, 0.0])
        u, v, n = hand_frame_basis(l0, l5, l9, l17)
        for x in (u, v, n):
            self.assertAlmostEqual(float(np.linalg.norm(x)), 1.0, places=9)
        self.assertAlmostEqual(float(np.dot(u, v)), 0.0, places=9)

    def test_project_guards_depth(self):
        from math3d.transforms import project_points
        scr, z, _ = project_points(np.array([[0, 0, -100.0]]), 640, 360, 700, 3.2)
        self.assertTrue(np.all(np.isfinite(scr)))

    def test_transform_pipeline(self):
        from math3d.transforms import transform_vertices
        v = transform_vertices(np.array([[1, 0, 0]]), np.eye(3), 2.0, np.array([5, 6, 7]))
        self.assertTrue(np.allclose(v[0], [7, 6, 7]))


class TestParticles(unittest.TestCase):
    def test_spawn_update_fade(self):
        from simulation.particles import ParticleDissolver
        d = ParticleDissolver(max_particles=200, dissolve_time=0.4)
        w = np.random.randn(40, 3) * 0.5
        self.assertGreater(d.spawn_from_vertices(w), 0)
        a0 = d.alphas()
        self.assertTrue(np.all(a0 > 0.9))
        for _ in range(120):
            d.update(1 / 60)
        self.assertFalse(d.active)


class TestModels(unittest.TestCase):
    def test_builders_valid(self):
        from models.procedural import BUILDERS
        for name, fn in BUILDERS.items():
            v, e, c = fn(0.5)
            self.assertGreater(v.shape[0], 8, name)
            self.assertGreater(e.shape[0], 4, name)
            self.assertEqual(v.shape[0], c.shape[0], name)
            self.assertTrue(np.all(e.min() >= 0) and np.all(e.max() < v.shape[0]), name)
            v2, _, _ = fn(1.5)
            self.assertEqual(v.shape, v2.shape, name)  # stable topology


class TestStateMachine(unittest.TestCase):
    def test_debounce_spawn_then_dissolve(self):
        from core.state_machine import GestureStateMachine, SceneState
        from vision.gestures import Gesture
        sm = GestureStateMachine(debounce_frames=3, dissolve_frames=2)
        ev = None
        for _ in range(3):
            ev = sm.update(Gesture.INDEX)
        self.assertIsNotNone(ev)
        self.assertEqual(sm.active_model, "flowers")
        ev = None
        for _ in range(3):
            ev = sm.update(Gesture.FIST)
        self.assertIsNotNone(ev)
        self.assertEqual(sm.state, SceneState.DISSOLVING)
        done = sm.finish_dissolve()
        self.assertEqual(sm.active_model, "dragon")

    def test_open_palm_dissolves(self):
        from core.state_machine import GestureStateMachine, SceneState
        from vision.gestures import Gesture
        sm = GestureStateMachine(debounce_frames=2, dissolve_frames=2)
        for _ in range(2):
            sm.update(Gesture.PEACE)
        self.assertEqual(sm.active_model, "tree")
        ev = None
        for _ in range(2):
            r = sm.update(Gesture.OPEN_PALM)
            ev = r if r is not None else ev
        self.assertIsNotNone(ev)
        self.assertEqual(sm.state, SceneState.DISSOLVING)

    def test_dual_fists_clears(self):
        from core.state_machine import GestureStateMachine, SceneState
        from vision.gestures import Gesture
        sm = GestureStateMachine(debounce_frames=2)
        for _ in range(2):
            sm.update(Gesture.L_SHAPE)
        ev = sm.update(Gesture.FIST, dual_fists=True)
        self.assertIsNotNone(ev)


if __name__ == "__main__":
    unittest.main()
