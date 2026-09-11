"""Unit tests for gesture classification rules, debounce state machine, and pose estimator."""

import unittest
import numpy as np

from core.state_machine import GestureState, GestureStateMachine
from vision.gesture_detector import GestureDetector
from vision.pose_estimator import HandPoseEstimator


def create_synthetic_hand(
    thumb: bool = False,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    pinch: bool = False,
) -> np.ndarray:
    """Construct synthetic (21, 3) landmarks simulating specific finger extensions.

    Wrist is at (0.5, 0.8, 0.0). Extended fingers reach upward to y=0.2.
    Folded fingers curl to y=0.7. Pinch moves thumb tip close to index tip.
    """
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[0] = [0.5, 0.8, 0.0]  # Wrist

    # MCPs
    pts[5] = [0.45, 0.55, 0.0]  # Index MCP
    pts[9] = [0.50, 0.53, 0.0]  # Middle MCP
    pts[13] = [0.55, 0.55, 0.0] # Ring MCP
    pts[17] = [0.60, 0.58, 0.0] # Pinky MCP

    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (Tip)
    pts[1] = [0.43, 0.72, 0.0]
    pts[2] = [0.38, 0.65, 0.0]
    pts[3] = [0.34, 0.58, 0.0]
    # For pinch, move thumb tip very close to index tip position (8)
    pts[4] = [0.45, 0.20, 0.0] if pinch else ([0.25, 0.48, 0.0] if thumb else [0.40, 0.60, 0.0])

    # Index: 6 (PIP), 7 (DIP), 8 (Tip)
    pts[6] = [0.45, 0.45, 0.0]
    pts[7] = [0.45, 0.35, 0.0]
    pts[8] = [0.45, 0.20, 0.0] if index else [0.45, 0.60, 0.0]

    # Middle: 10, 11, 12
    pts[10] = [0.50, 0.43, 0.0]
    pts[11] = [0.50, 0.33, 0.0]
    pts[12] = [0.50, 0.18, 0.0] if middle else [0.50, 0.60, 0.0]

    # Ring: 14, 15, 16
    pts[14] = [0.55, 0.45, 0.0]
    pts[15] = [0.55, 0.35, 0.0]
    pts[16] = [0.55, 0.20, 0.0] if ring else [0.55, 0.60, 0.0]

    # Pinky: 18, 19, 20
    pts[18] = [0.60, 0.50, 0.0]
    pts[19] = [0.60, 0.42, 0.0]
    pts[20] = [0.60, 0.25, 0.0] if pinky else [0.60, 0.62, 0.0]

    return pts


class TestGestureRules(unittest.TestCase):
    """Tests deterministic posture classification and debounce state machine."""

    def test_open_palm_classification(self) -> None:
        """All fingers extended must classify as OPEN_PALM."""
        landmarks = create_synthetic_hand(index=True, middle=True, ring=True, pinky=True)
        fingers = GestureDetector.classify_finger_extensions(landmarks)
        pinch_dist = GestureDetector.compute_pinch_distance(landmarks)
        index_tip_above_mcp = bool(landmarks[8][1] < landmarks[5][1])
        state = GestureDetector.classify_gesture(fingers, pinch_dist, pinch_threshold=0.065,
                                                 index_tip_above_mcp=index_tip_above_mcp)
        self.assertEqual(state, GestureState.OPEN_PALM)

    def test_pinch_classification(self) -> None:
        """Thumb and index tip close together with raised index must classify as PINCH."""
        landmarks = create_synthetic_hand(pinch=True, index=True)
        fingers = GestureDetector.classify_finger_extensions(landmarks)
        pinch_dist = GestureDetector.compute_pinch_distance(landmarks)
        self.assertLess(pinch_dist, 0.065, f"Expected pinch dist < 0.065, got {pinch_dist:.4f}")
        # index=True sets tip at y=0.20, mcp at y=0.55 → tip is raised above knuckle
        index_tip_above_mcp = bool(landmarks[8][1] < landmarks[5][1])
        self.assertTrue(index_tip_above_mcp, "Pinch test: index tip should be above MCP")
        state = GestureDetector.classify_gesture(fingers, pinch_dist, pinch_threshold=0.065,
                                                 index_tip_above_mcp=index_tip_above_mcp)
        self.assertEqual(state, GestureState.PINCH)

    def test_fist_classification(self) -> None:
        """All fingers folded: index tip is BELOW MCP (curled), so PINCH must NOT fire."""
        landmarks = create_synthetic_hand()  # All fingers folded; tip.y > mcp.y
        fingers = GestureDetector.classify_finger_extensions(landmarks)
        pinch_dist = GestureDetector.compute_pinch_distance(landmarks)
        # Fist: index tip (y=0.60) > index MCP (y=0.55) → tip is below knuckle → NOT raised
        index_tip_above_mcp = bool(landmarks[8][1] < landmarks[5][1])
        self.assertFalse(index_tip_above_mcp, "Fist test: index tip should be below MCP")
        state = GestureDetector.classify_gesture(fingers, pinch_dist, pinch_threshold=0.065,
                                                 index_tip_above_mcp=index_tip_above_mcp)
        self.assertEqual(state, GestureState.FIST)

    def test_state_machine_debouncing(self) -> None:
        """State machine must require exactly debounce_frames consecutive inputs to switch."""
        sm = GestureStateMachine(debounce_frames=4)
        self.assertEqual(sm.current_state, GestureState.NO_HAND)

        # 3 frames of OPEN_PALM — state must remain NO_HAND
        self.assertEqual(sm.update(GestureState.OPEN_PALM), GestureState.NO_HAND)
        self.assertEqual(sm.update(GestureState.OPEN_PALM), GestureState.NO_HAND)
        self.assertEqual(sm.update(GestureState.OPEN_PALM), GestureState.NO_HAND)
        # 4th frame triggers transition
        self.assertEqual(sm.update(GestureState.OPEN_PALM), GestureState.OPEN_PALM)
        self.assertEqual(sm.current_state, GestureState.OPEN_PALM)

        # A 1-frame FIST glitch must NOT switch state
        self.assertEqual(sm.update(GestureState.FIST), GestureState.OPEN_PALM)
        self.assertEqual(sm.update(GestureState.OPEN_PALM), GestureState.OPEN_PALM)

    def test_pinch_state_machine_transition(self) -> None:
        """PINCH must transition cleanly after debounce_frames."""
        sm = GestureStateMachine(debounce_frames=3)
        for _ in range(3):
            sm.update(GestureState.PINCH)
        self.assertEqual(sm.current_state, GestureState.PINCH)
        self.assertGreater(sm.cube_alpha, 0.0)
        self.assertEqual(sm.particle_alpha, 0.0)

    def test_pose_estimator_angles(self) -> None:
        """Pose estimator should produce valid rotation matrix and bounded angles."""
        estimator = HandPoseEstimator(smoothing_alpha=0.5)
        landmarks = create_synthetic_hand(index=True, middle=True, ring=True, pinky=True)
        orientation = estimator.estimate_orientation(landmarks, handedness="Right")
        self.assertIsNotNone(orientation.rotation_matrix)
        self.assertEqual(orientation.rotation_matrix.shape, (3, 3))
        self.assertAlmostEqual(float(np.linalg.norm(orientation.normal_vector)), 1.0, places=5)
        self.assertFalse(np.isnan(orientation.roll))
        self.assertFalse(np.isnan(orientation.pitch))
        self.assertFalse(np.isnan(orientation.yaw))


if __name__ == "__main__":
    unittest.main()
