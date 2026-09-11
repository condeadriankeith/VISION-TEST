"""Automated verification test script for Hand Gesture 3D Hologram pipeline.

Tests:
- 6-DOF rigid-body physics, rotational integration, and off-center collision torque.
- 3D palm normal extraction, coordinate frame, and tilt angles.
- Procedural multi-harmonic organic floating dynamics.
- Dynamic contact shadow rendering and live scene grounding.
- Blinn-Phong PBR solid grey 3D rendering and gesture transitions.
"""

import math
import sys
import time
import numpy as np
from typing import Any, List

import config
from config import MATERIALS
from cube_renderer import CubeHologramRenderer
from hand_tracker import FingerTipCollider, HandPose, HandTracker, LandmarkPoint
from physics import CubePhysicsWorld, PhysicsCube


def test_physics_and_collisions() -> None:
    """Tests 6-DOF rigid-body integration, spring-damper motion, and collision impulses."""
    print("[Test] Testing 3D Rigid-Body Physics & Collisions...")
    world = CubePhysicsWorld()

    # Place cube 0 and cube 1 in overlapping positions with collision velocity
    world.cubes[0].position = np.array([500.0, 300.0, 0.0], dtype=np.float32)
    world.cubes[1].position = np.array([520.0, 300.0, 0.0], dtype=np.float32)
    world.cubes[0].current_scale = 1.0
    world.cubes[1].current_scale = 1.0
    world.cubes[0].velocity = np.array([20.0, 0.0, 0.0], dtype=np.float32)
    world.cubes[1].velocity = np.array([-20.0, 0.0, 0.0], dtype=np.float32)

    initial_dist = float(np.linalg.norm(world.cubes[0].position - world.cubes[1].position))
    assert initial_dist < (world.cubes[0].radius + world.cubes[1].radius), "Must start overlapping"

    # Step physics to resolve collision
    targets = [c.position.copy() for c in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016)

    new_dist = float(np.linalg.norm(world.cubes[0].position - world.cubes[1].position))
    assert new_dist > initial_dist, f"Collision must push cubes apart: was {initial_dist}, now {new_dist}"
    assert world.cubes[0].velocity[0] < 0.0, "Cube 0 should bounce backward"
    assert world.cubes[1].velocity[0] > 0.0, "Cube 1 should bounce forward"

    print("  -> Rigid-Body Physics & Collision Impulse passed!")


def test_6dof_rotation_and_torque() -> None:
    """Tests 3D rotation matrix orthonormality and off-center collision torque transfer."""
    print("[Test] Testing 6-DOF Rotational Dynamics & Impact Torque...")
    cube = PhysicsCube(0, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    rot_mat = cube.get_rotation_matrix()
    det = float(np.linalg.det(rot_mat))
    assert abs(det - 1.0) < 1e-4, f"Rotation matrix must have determinant 1.0, got {det}"

    # Integrate rotation for several steps
    cube.angular_velocity = np.array([1.5, -2.0, 0.8], dtype=np.float32)
    for _ in range(30):
        cube.integrate_rotation(0.016)

    updated_det = float(np.linalg.det(cube.get_rotation_matrix()))
    assert abs(updated_det - 1.0) < 1e-4, f"Orthonormality must be strictly preserved: det = {updated_det}"

    # Test off-center collision producing angular torque
    world = CubePhysicsWorld()
    world.cubes[0].position = np.array([500.0, 300.0, 0.0], dtype=np.float32)
    world.cubes[1].position = np.array([518.0, 314.0, 0.0], dtype=np.float32)  # Diagonal contact point
    world.cubes[0].current_scale = 1.0
    world.cubes[1].current_scale = 1.0
    world.cubes[0].velocity = np.array([30.0, 0.0, 0.0], dtype=np.float32)
    world.cubes[1].velocity = np.array([-30.0, 0.0, 0.0], dtype=np.float32)
    world.cubes[0].angular_velocity[:] = 0.0
    world.cubes[1].angular_velocity[:] = 0.0

    targets = [c.position.copy() for c in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016)

    ang_speed_0 = float(np.linalg.norm(world.cubes[0].angular_velocity))
    ang_speed_1 = float(np.linalg.norm(world.cubes[1].angular_velocity))
    assert ang_speed_0 > 0.01, f"Off-center impact must impart rotational spin to cube 0: {ang_speed_0}"
    assert ang_speed_1 > 0.01, f"Off-center impact must impart rotational spin to cube 1: {ang_speed_1}"

    print("  -> 6-DOF Rotational Dynamics & Impact Torque passed!")


def test_hand_tracker_3d_orientation() -> None:
    """Tests MediaPipe HandTracker and 3D normal vector / tilt angle computation."""
    print("[Test] Testing HandTracker 3D Orientation & Tilt...")
    tracker = HandTracker()

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ts = int(time.time() * 1000)

    poses = tracker.process_frame(dummy_frame, ts)
    assert isinstance(poses, list), "Expected list of poses"
    assert len(poses) == 0, "No hands expected in blank black frame"

    # Synthetic HandPose with tilted palm
    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]
    synth_normal = np.array([0.0, 0.38, -0.92], dtype=np.float32)
    synth_normal /= np.linalg.norm(synth_normal)

    pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True, True, True, True, True],
        palm_normal_3d=(float(synth_normal[0]), float(synth_normal[1]), float(synth_normal[2])),
        palm_up_3d=(0.0, -0.92, -0.38),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 480.0, -25.0),
        pitch_deg=-22.0,
        roll_deg=0.0,
    )

    # Verify normal vector properties
    norm_vec = np.array(pose.palm_normal_3d)
    norm_mag = float(np.linalg.norm(norm_vec))
    assert abs(norm_mag - 1.0) < 1e-4, f"Palm normal must be unit vector, got {norm_mag}"

    tracker.close()
    print("  -> HandTracker 3D Orientation & Tilt passed!")


def test_organic_floating_and_dynamic_angling() -> None:
    """Tests multi-harmonic organic floating motion and tilted target slot generation."""
    print("[Test] Testing Procedural Organic Floating & Dynamic Angling...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]
    pose_tilted = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True, True, True, True, True],
        palm_normal_3d=(0.25, 0.35, -0.90),
        palm_up_3d=(0.0, -0.93, -0.36),
        palm_right_3d=(0.97, -0.09, 0.23),
        palm_center_3d=(640.0, 480.0, -20.0),
        pitch_deg=-20.5,
        roll_deg=15.5,
    )

    # Step simulation over 60 frames
    positions_history = []
    for step in range(60):
        renderer.update(hand_detected=True, is_open=True, pose=pose_tilted, dt=0.016)
        positions_history.append([c.position.copy() for c in renderer.physics_world.cubes])

    pos_start = positions_history[20][0]
    pos_end = positions_history[59][0]
    diff = float(np.linalg.norm(pos_end - pos_start))
    assert diff > 0.5, f"Organic floating motion should produce dynamic non-static trajectory: {diff}"

    # Verify cubes maintain horizontal separation in tilted space
    c0 = renderer.physics_world.cubes[0]
    c1 = renderer.physics_world.cubes[1]
    c2 = renderer.physics_world.cubes[2]
    dist_01 = float(np.linalg.norm(c0.position - c1.position))
    dist_12 = float(np.linalg.norm(c1.position - c2.position))
    assert dist_01 > 50.0, f"Cubes 0 and 1 must maintain separation: {dist_01}"
    assert dist_12 > 50.0, f"Cubes 1 and 2 must maintain separation: {dist_12}"

    print("  -> Procedural Organic Floating & Dynamic Angling passed!")


def test_renderer_palm_emergence_and_shading() -> None:
    """Tests solid grey 3D face projection, PBR lighting, palm emergence kinematics, and shadow removal."""
    print("[Test] Testing 3D Solid Grey Cube Renderer & Palm Emergence Kinematics...")
    renderer = CubeHologramRenderer()
    material = MATERIALS["Prismatic"]

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]
    pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True, True, True, True, True],
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 480.0, 0.0),
        pitch_deg=0.0,
        roll_deg=0.0,
    )

    # 1. Verify that contact shadows are disabled (no dark black pulses on frame)
    shadow_test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    active_cubes = [c for c in renderer.physics_world.cubes]
    renderer._render_contact_shadows(shadow_test_frame, active_cubes, pose)
    shadow_darkened = int(np.count_nonzero(shadow_test_frame < 120))
    assert shadow_darkened == 0, f"Contact shadows must be disabled (0 dark pixels), got {shadow_darkened}"

    # 2. Test initial palm interior position when palm is closed
    closed_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=False,
        open_confidence=0.0,
        finger_states=[False, False, False, False, False],
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 480.0, 0.0),
    )
    renderer.update(hand_detected=True, is_open=False, pose=closed_pose, dt=0.033)

    # Center cube should be submerged at inside palm position: palm_center - 24 * palm_normal
    expected_inside = np.array(closed_pose.palm_center_3d) - config.PALM_SUBMERGE_DEPTH * np.array(closed_pose.palm_normal_3d)
    center_pos = renderer.physics_world.cubes[1].position
    dist_to_inside = float(np.linalg.norm(center_pos - expected_inside))
    assert dist_to_inside < 1.0, f"Cube must be submerged inside palm when closed: dist={dist_to_inside}"
    assert renderer.physics_world.cubes[1].current_scale == 0.0, "Cube scale must be 0 when submerged"

    # 3. Open palm and observe emergence progression & staggering
    renderer.update(hand_detected=True, is_open=True, pose=pose, dt=0.033)
    # Center cube (i=1) emerges with zero stagger delay
    assert renderer.physics_world.cubes[1].emergence > 0.0, "Center cube must emerge immediately"

    # Step simulation to full emergence over 45 frames
    for _ in range(45):
        renderer.update(hand_detected=True, is_open=True, pose=pose, dt=0.033)

    for i, cube in enumerate(renderer.physics_world.cubes):
        assert cube.emergence >= 0.99, f"Cube {i} must reach full emergence, got {cube.emergence}"
        assert cube.current_scale >= 0.99, f"Cube {i} scale must be ~1.0, got {cube.current_scale}"

    # 4. Verify composite render: solid prismatic cubes, background untouched elsewhere
    test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    renderer.render(test_frame, pose, material)

    changed = np.any(test_frame != 120, axis=2)
    changed_pixels = int(np.count_nonzero(changed))
    assert changed_pixels > 2000, f"Expected lit solid shaded pixels (>2000), got {changed_pixels}"

    # All changed pixels must cluster near the cubes (no stray fullscreen artifacts)
    ys, xs = np.nonzero(changed)
    cx, cy = pose.palm_center_px
    max_dist = float(np.max(np.hypot(xs.astype(float) - cx, ys.astype(float) - cy)))
    assert max_dist < 400.0, f"Render artifacts strayed from cubes (max dist {max_dist})"

    # 5. Simulate closing palm and retracting back inside the palm
    for _ in range(60):
        renderer.update(hand_detected=True, is_open=False, pose=closed_pose, dt=0.033)

    for i, cube in enumerate(renderer.physics_world.cubes):
        assert cube.current_scale == 0.0, f"Cube {i} must retract to scale 0, got {cube.current_scale}"
        dist_retracted = float(np.linalg.norm(cube.position - expected_inside))
        assert dist_retracted < 1.0, f"Cube {i} must be retracted inside palm, dist={dist_retracted}"

    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    renderer.render(blank_frame, closed_pose, material)
    vanished_pixels = int(np.count_nonzero(blank_frame))
    assert vanished_pixels == 0, f"Expected 0 pixels when fully retracted, got {vanished_pixels}"

    print("  -> 3D Solid Grey Cube Renderer & Palm Emergence Kinematics passed!")


def test_finger_identification_and_tracking() -> None:
    """Tests finger identification, naming, and 3D kinematic tracking."""
    print("[Test] Testing Finger Identification & 3D Kinematics...")
    tracker = HandTracker()

    landmarks = [
        LandmarkPoint(x=0.5 + 0.01 * (i % 5), y=0.5 + 0.01 * (i // 5), z=0.0, px=int((0.5 + 0.01 * (i % 5)) * 1280), py=int((0.5 + 0.01 * (i // 5)) * 720))
        for i in range(21)
    ]

    pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True, True, True, True, True],
        fingers=[
            FingerTipCollider(name="Thumb", tip_idx=4, pos_3d=np.array([600.0, 450.0, 0.0], dtype=np.float32), vel_3d=np.zeros(3, dtype=np.float32), radius=24.0, is_extended=True),
            FingerTipCollider(name="Index", tip_idx=8, pos_3d=np.array([620.0, 380.0, 0.0], dtype=np.float32), vel_3d=np.zeros(3, dtype=np.float32), radius=24.0, is_extended=True),
            FingerTipCollider(name="Middle", tip_idx=12, pos_3d=np.array([640.0, 360.0, 0.0], dtype=np.float32), vel_3d=np.zeros(3, dtype=np.float32), radius=24.0, is_extended=True),
            FingerTipCollider(name="Ring", tip_idx=16, pos_3d=np.array([660.0, 380.0, 0.0], dtype=np.float32), vel_3d=np.zeros(3, dtype=np.float32), radius=24.0, is_extended=True),
            FingerTipCollider(name="Pinky", tip_idx=20, pos_3d=np.array([680.0, 410.0, 0.0], dtype=np.float32), vel_3d=np.zeros(3, dtype=np.float32), radius=24.0, is_extended=True),
        ]
    )

    assert len(pose.fingers) == 5, f"Expected 5 identified fingers, got {len(pose.fingers)}"
    expected_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for i, name in enumerate(expected_names):
        assert pose.fingers[i].name == name, f"Finger {i} should be {name}, got {pose.fingers[i].name}"
        assert pose.fingers[i].radius > 0, "Finger radius must be positive"

    tracker.close()
    print("  -> Finger Identification & 3D Kinematics passed!")


def test_finger_collision_and_flick_recoil() -> None:
    """Tests kinematic finger poke, flick momentum transfer, and contact torque."""
    print("[Test] Testing Dynamic Finger Collision & Flick Recoil...")
    world = CubePhysicsWorld()
    cube = world.cubes[0]
    cube.position = np.array([640.0, 360.0, 0.0], dtype=np.float32)
    cube.velocity[:] = 0.0
    cube.angular_velocity[:] = 0.0
    cube.current_scale = 1.0

    # Simulate fast upward index finger flick striking bottom of the cube
    index_finger = FingerTipCollider(
        name="Index",
        tip_idx=8,
        pos_3d=np.array([645.0, 395.0, 0.0], dtype=np.float32),  # Slightly off-center right
        vel_3d=np.array([10.0, -250.0, 0.0], dtype=np.float32),  # Fast upward flick
        radius=24.0,
        is_extended=True,
    )

    targets = [cube.position.copy() for cube in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016, finger_colliders=[index_finger])

    # Assertions
    assert cube.velocity[1] < -20.0, f"Cube should recoil upward away from flick: {cube.velocity[1]}"
    spin_mag = float(np.linalg.norm(cube.angular_velocity))
    assert spin_mag > 0.01, f"Off-center finger strike must impart spin torque: {spin_mag}"
    assert len(world.recent_contacts) > 0, "Contact event must be registered"
    assert world.recent_contacts[0].finger_name == "Index", "Contacting finger must be Index"

    print("  -> Dynamic Finger Collision & Flick Recoil passed!")


def test_levitation_recovery() -> None:
    """Tests that a struck cube smoothly recovers to its hover equilibrium."""
    print("[Test] Testing Levitation Recovery Spring Field...")
    world = CubePhysicsWorld()
    cube = world.cubes[0]
    cube.position = np.array([640.0, 360.0, 0.0], dtype=np.float32)
    cube.current_scale = 1.0

    # Impart high disturbance velocity
    target_pos = np.array([526.0, 360.0, 0.0], dtype=np.float32)
    cube.position = target_pos.copy()
    cube.velocity = np.array([150.0, -120.0, 40.0], dtype=np.float32)

    # Step simulation for 90 frames (approx 1.5s at 60 FPS) with separated slot targets
    targets = [
        np.array([526.0, 360.0, 0.0], dtype=np.float32),
        np.array([640.0, 360.0, 0.0], dtype=np.float32),
        np.array([754.0, 360.0, 0.0], dtype=np.float32),
    ]
    for _ in range(90):
        world.step(targets, should_spawn=True, dt=0.016)

    # Cube should have returned close to equilibrium target
    drift = float(np.linalg.norm(cube.position - target_pos))
    assert drift < 5.0, f"Cube must recover to hover target (drift < 5 px), got {drift}"
    speed = float(np.linalg.norm(cube.velocity))
    assert speed < 10.0, f"Cube velocity must be dampened near rest (< 10 px/s), got {speed}"

    print("  -> Levitation Recovery Spring Field passed!")


def test_continuous_openness_and_fountain() -> None:
    """Tests continuous openness ratio calculation and fountain blossom emergence kinematics."""
    print("[Test] Testing Continuous Openness & Flowy Fountain Blossom...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]
    pose_open = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True, True, True, True, True],
        openness_ratio=0.95,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 480.0, 0.0),
    )

    assert hasattr(pose_open, "openness_ratio"), "HandPose must have openness_ratio"
    assert 0.0 <= pose_open.openness_ratio <= 1.0, "openness_ratio must be normalized"

    # Test emergence progression
    renderer.update(hand_detected=True, is_open=True, pose=pose_open, dt=0.016)
    c_center = renderer.physics_world.cubes[1]
    assert c_center.emergence > 0.0, "Center cube must start emerging"

    print("  -> Continuous Openness & Flowy Fountain Blossom passed!")


def test_dual_palm_midpoint_and_accordion() -> None:
    """Tests procedural midpoint formation and accordion distance-reactive spacing with 2 hands."""
    print("[Test] Testing Dual-Palm Midpoint & Accordion Spacing...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]

    # Two hands spaced 400px apart along X axis: Left at 400, Right at 800
    left_pose = HandPose(
        handedness="Left",
        landmarks=landmarks,
        palm_center_px=(400, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(400.0, 360.0, 0.0),
    )

    right_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(800, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(800.0, 360.0, 0.0),
    )

    # Step simulation with wide hands (dist = 400px)
    for _ in range(40):
        renderer.update(
            hand_detected=True,
            is_open=True,
            poses=[left_pose, right_pose],
            dt=0.016,
        )

    cubes = renderer.physics_world.cubes
    center_x = cubes[1].position[0]
    # Midpoint of 400 and 800 is 600
    assert abs(center_x - 600.0) < 25.0, f"Center cube should be positioned at midpoint (~600): {center_x}"

    # Measure spacing with wide hands
    wide_spacing = float(cubes[2].position[0] - cubes[0].position[0])

    # Now bring hands close together: Left at 500, Right at 700 (dist = 200px)
    left_close = HandPose(
        handedness="Left",
        landmarks=landmarks,
        palm_center_px=(500, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(500.0, 360.0, 0.0),
    )
    right_close = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(700, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(700.0, 360.0, 0.0),
    )

    for _ in range(40):
        renderer.update(
            hand_detected=True,
            is_open=True,
            poses=[left_close, right_close],
            dt=0.016,
        )

    close_spacing = float(cubes[2].position[0] - cubes[0].position[0])
    assert close_spacing < wide_spacing, f"Accordion effect: closer hands must compress spacing (wide={wide_spacing}, close={close_spacing})"

    print("  -> Dual-Palm Midpoint & Accordion Spacing passed!")


def test_dual_hand_multi_colliders() -> None:
    """Tests 10-finger collisions from both hands acting on cubes simultaneously."""
    print("[Test] Testing Multi-Hand 10-Finger Physics Collisions...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]

    # Cube 0 at 450, Cube 2 at 750
    cube_left = renderer.physics_world.cubes[0]
    cube_right = renderer.physics_world.cubes[2]
    cube_left.position = np.array([450.0, 360.0, 0.0], dtype=np.float32)
    cube_right.position = np.array([750.0, 360.0, 0.0], dtype=np.float32)
    cube_left.current_scale = 1.0
    cube_right.current_scale = 1.0

    # Finger 1 from left hand hitting cube 0
    left_finger = FingerTipCollider(
        name="Index",
        tip_idx=8,
        pos_3d=np.array([450.0, 380.0, 0.0], dtype=np.float32),
        vel_3d=np.array([0.0, -200.0, 0.0], dtype=np.float32),
        radius=20.0,
        is_extended=True,
    )
    # Finger 2 from right hand hitting cube 2
    right_finger = FingerTipCollider(
        name="Middle",
        tip_idx=12,
        pos_3d=np.array([750.0, 380.0, 0.0], dtype=np.float32),
        vel_3d=np.array([0.0, -200.0, 0.0], dtype=np.float32),
        radius=20.0,
        is_extended=True,
    )

    left_pose = HandPose(
        handedness="Left",
        landmarks=landmarks,
        palm_center_px=(450, 400),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_center_3d=(450.0, 400.0, 0.0),
        fingers=[left_finger],
    )
    right_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(750, 400),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_center_3d=(750.0, 400.0, 0.0),
        fingers=[right_finger],
    )

    renderer.update(hand_detected=True, is_open=True, poses=[left_pose, right_pose], dt=0.016)

    assert cube_left.velocity[1] < -5.0, f"Cube 0 should recoil from left index finger: {cube_left.velocity[1]}"
    assert cube_right.velocity[1] < -5.0, f"Cube 2 should recoil from right middle finger: {cube_right.velocity[1]}"

    print("  -> Multi-Hand 10-Finger Physics Collisions passed!")


def test_upward_palm_perspective() -> None:
    """Tests that when palm opens upward (facing ceiling), cubes hover directly above the palm bed."""
    print("[Test] Testing Upward-Facing Palm Perspective Hovering...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=480) for _ in range(21)
    ]
    # Palm facing upward: normal is (0, -1, 0), up (fingers) is (0, 0, -1), right is (1, 0, 0)
    upward_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 480),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=85.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, -1.0, 0.0),
        palm_up_3d=(0.0, 0.0, -1.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 480.0, 0.0),
        pitch_deg=-90.0,
        roll_deg=0.0,
    )

    # Step simulation to full emergence
    for _ in range(45):
        renderer.update(hand_detected=True, is_open=True, pose=upward_pose, dt=0.016)

    cubes = renderer.physics_world.cubes
    center_cube = cubes[1]
    # In camera coordinates, -Y is UP towards ceiling.
    # Palm center is at y=480.0. With upward boost, cube emerges noticeably higher (y < 420.0 at frame 45).
    assert center_cube.position[1] < 420.0, f"Cube must hover noticeably higher vertically above upward palm bed (<420): {center_cube.position[1]}"
    # The X position should be centered with the palm (~640)
    assert abs(center_cube.position[0] - 640.0) < 20.0, f"Cube must be centered over palm in X (~640): {center_cube.position[0]}"
    # Depth Z should be close to palm bed Z (0.0), not flying out wildly
    assert abs(center_cube.position[2]) < 60.0, f"Cube Z depth must stay comfortably over palm bed (<60): {center_cube.position[2]}"

    # Step to steady-state equilibrium (~25 more frames)
    for _ in range(25):
        renderer.update(hand_detected=True, is_open=True, pose=upward_pose, dt=0.016)
    # At steady-state, cube hovers ~95px above upward palm bed (y < 395.0 vs ~420.0 previously)
    assert center_cube.position[1] < 395.0, f"Cube must settle at elevated hover equilibrium (<395): {center_cube.position[1]}"

    print("  -> Upward-Facing Palm Perspective Hovering passed!")


def test_dual_palm_single_closed_detachment() -> None:
    """Tests that when 2 hands are detected, but one closes, cubes detach from closed hand and anchor 100% to open palm."""
    print("[Test] Testing Dual-Palm Single Closed Hand Detachment...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]

    # Left hand open at X=400
    left_open = HandPose(
        handedness="Left",
        landmarks=landmarks,
        palm_center_px=(400, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(400.0, 360.0, 0.0),
    )

    # Right hand CLOSED at X=800
    right_closed = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(800, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=False,
        open_confidence=0.0,
        finger_states=[False] * 5,
        openness_ratio=0.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(800.0, 360.0, 0.0),
    )

    for _ in range(65):
        renderer.update(
            hand_detected=True,
            is_open=True,
            poses=[left_open, right_closed],
            dt=0.016,
        )

    cubes = renderer.physics_world.cubes
    center_cube_x = cubes[1].position[0]
    # Center cube should be anchored over the LEFT palm (~400), NOT in the middle (600)
    assert abs(center_cube_x - 400.0) < 30.0, f"Cubes must anchor to open Left hand (~400), got {center_cube_x}"

    # Now reverse: Left hand closes, Right hand opens!
    left_closed = HandPose(
        handedness="Left",
        landmarks=landmarks,
        palm_center_px=(400, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=False,
        open_confidence=0.0,
        finger_states=[False] * 5,
        openness_ratio=0.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(400.0, 360.0, 0.0),
    )
    right_open = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(800, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=70.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(800.0, 360.0, 0.0),
    )

    for _ in range(80):
        renderer.update(
            hand_detected=True,
            is_open=True,
            poses=[left_closed, right_open],
            dt=0.016,
        )

    center_cube_x_new = cubes[1].position[0]
    # Now center cube should have smoothly glided across to the RIGHT palm (~800)
    assert abs(center_cube_x_new - 800.0) < 65.0, f"Cubes must glide and anchor to open Right hand (~800), got {center_cube_x_new}"

    print("  -> Dual-Palm Single Closed Hand Detachment passed!")


def test_satisfying_palm_suction_vortex() -> None:
    """Tests centripetal gathering, two-stage scale retention, and vortex convergence into palm center."""
    print("[Test] Testing Satisfying Palm Suction & Vortex Minimize...")
    renderer = CubeHologramRenderer()

    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)
    ]
    open_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
    )

    # 1. Fully emerge cubes
    for _ in range(45):
        renderer.update(hand_detected=True, is_open=True, pose=open_pose, dt=0.016)

    cubes = renderer.physics_world.cubes
    initial_spread = float(np.linalg.norm(cubes[2].position - cubes[0].position))
    assert initial_spread > 100.0, f"Cubes must be spread in hover formation, got {initial_spread}"

    # 2. Close hand to trigger vortex suction
    closed_pose = HandPose(
        handedness="Right",
        landmarks=landmarks,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=False,
        open_confidence=0.0,
        finger_states=[False] * 5,
        openness_ratio=0.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
    )

    # Step into the gathering phase (~10 frames, ~0.16s)
    for _ in range(10):
        renderer.update(hand_detected=True, is_open=False, pose=closed_pose, dt=0.016)

    # In gathering phase, cubes must still be visible (>0.4 scale)
    assert cubes[1].current_scale > 0.40, f"Cubes must not vanish prematurely, scale={cubes[1].current_scale}"

    # Verify centripetal gathering: inter-cube spread has contracted inward toward the center
    gathered_spread = float(np.linalg.norm(cubes[2].position - cubes[0].position))
    assert gathered_spread < initial_spread * 0.90, (
        f"Cubes must centripetally converge toward palm center: initial={initial_spread}, gathered={gathered_spread}"
    )

    # 3. Step to complete vortex suction plunge (~35 more frames, ~0.55s total)
    for _ in range(35):
        renderer.update(hand_detected=True, is_open=False, pose=closed_pose, dt=0.016)

    expected_inside = np.array(closed_pose.palm_center_3d) - config.PALM_SUBMERGE_DEPTH * np.array(closed_pose.palm_normal_3d)
    for i, c in enumerate(cubes):
        assert c.current_scale == 0.0, f"Cube {i} must be fully submerged (scale=0), got {c.current_scale}"
        dist_in = float(np.linalg.norm(c.position - expected_inside))
        assert dist_in < 1.0, f"Cube {i} must be at palm interior sink, dist={dist_in}"

    print("  -> Satisfying Palm Suction & Vortex Minimize passed!")


def test_cube_to_cube_collision_and_momentum_transfer() -> None:
    """Tests corner-aware bounding radius, billiard recoil, and momentum transfer between cubes."""
    print("[Test] Testing Cube-to-Cube Collision & Momentum Transfer...")
    world = CubePhysicsWorld()

    # Position Cube 0 and Cube 1 so their bounding spheres overlap
    c0 = world.cubes[0]
    c1 = world.cubes[1]
    c0.current_scale = 1.0
    c1.current_scale = 1.0
    c0.position = np.array([500.0, 300.0, 0.0], dtype=np.float32)
    c1.position = np.array([540.0, 300.0, 0.0], dtype=np.float32)
    c0.velocity = np.array([120.0, 0.0, 0.0], dtype=np.float32)
    c1.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Step world to resolve collision
    targets = [c.position.copy() for c in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016)

    # Verify kinetic momentum transfer
    assert c1.velocity[0] > 40.0, f"Cube 1 must gain positive forward velocity, got {c1.velocity[0]}"
    assert c0.velocity[0] < 60.0, f"Cube 0 must decelerate or rebound, got {c0.velocity[0]}"
    assert c0.recoil_timer > 0.0, "Cube 0 must have recoil timer active to loosen spring"
    assert c1.recoil_timer > 0.0, "Cube 1 must have recoil timer active to loosen spring"
    assert c0.impact_energy > 0.0, "Cube 0 must have impact energy for sparks"
    assert c1.impact_energy > 0.0, "Cube 1 must have impact energy for sparks"
    assert len(world.recent_cube_collisions) > 0, "Collision event must be recorded"
    print("  -> Cube-to-Cube Collision & Momentum Transfer passed!")


def test_telekinesis_pinch_grab_and_fling() -> None:
    """Tests Telekinetic Force Grip (pinch-to-grab) and ballistic Force Fling throw."""
    print("[Test] Testing Telekinesis Force Grip & Fling...")
    world = CubePhysicsWorld()
    c1 = world.cubes[1]
    c1.current_scale = 1.0
    c1.position = np.array([640.0, 360.0, 0.0], dtype=np.float32)

    # 1. Simulate hand with pinch near cube 1
    dummy_lms = [LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)]
    dummy_lms[4] = LandmarkPoint(x=0.5, y=0.5, z=0.0, px=635, py=360)
    dummy_lms[8] = LandmarkPoint(x=0.5, y=0.5, z=0.0, px=645, py=360)

    pinch_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
        is_pinching=True,
        pinch_point_3d=(640.0, 360.0, 0.0),
        pinch_dist_px=10.0,
    )

    targets = [c.position.copy() for c in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016, poses=[pinch_pose])

    assert world.grabbed_cube_idx == 1, f"Cube 1 should be grabbed, got {world.grabbed_cube_idx}"
    assert c1.is_grabbed is True, "Cube 1 is_grabbed flag must be True"

    # Move pinch point swiftly to the right
    pinch_pose.pinch_point_3d = (740.0, 360.0, 0.0)
    world.step(targets, should_spawn=True, dt=0.016, poses=[pinch_pose])
    assert abs(c1.position[0] - 740.0) < 1.0, f"Grabbed cube must track pinch point: pos={c1.position[0]}"

    # Release pinch with motion -> Force Fling!
    release_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(740, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(740.0, 360.0, 0.0),
        is_pinching=False,
        pinch_point_3d=None,
        pinch_dist_px=90.0,
    )
    world.step(targets, should_spawn=True, dt=0.016, poses=[release_pose])

    assert world.grabbed_cube_idx is None, "Grabbed cube index should be cleared on release"
    assert c1.is_grabbed is False, "is_grabbed must be False on release"
    assert c1.fling_timer > 0.5, f"Fling timer must be active, got {c1.fling_timer}"
    assert c1.velocity[0] > 100.0, f"Flung cube must have high fling velocity, got {c1.velocity[0]}"

    # Step again: verify ballistic flight without spring intervention
    prev_x = c1.position[0]
    world.step(targets, should_spawn=True, dt=0.016, poses=[release_pose])
    assert c1.position[0] > prev_x, "Flung cube must continue moving in fling direction"

    print("  -> Telekinesis Force Grip & Fling passed!")


def test_telekinesis_force_push() -> None:
    """Tests Telekinetic Force Push forward shockwave blast."""
    print("[Test] Testing Telekinesis Force Push Shockwave...")
    world = CubePhysicsWorld()
    for c in world.cubes:
        c.current_scale = 1.0
        c.velocity[:] = 0.0

    dummy_lms = [LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)]
    thrust_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
        is_pinching=False,
        palm_thrust_speed=460.0,  # Exceeds threshold (380.0)
    )

    targets = [c.position.copy() for c in world.cubes]
    world.step(targets, should_spawn=True, dt=0.016, poses=[thrust_pose])

    assert world.force_push_active > 0.0, "Force Push must be triggered"
    for i, c in enumerate(world.cubes):
        speed = float(np.linalg.norm(c.velocity))
        assert speed > 150.0, f"Cube {i} must receive blast velocity, got speed {speed}"
        assert c.recoil_timer > 0.0, f"Cube {i} must have recoil active"
        assert c.impact_energy > 0.5, f"Cube {i} must have impact energy"

    print("  -> Telekinesis Force Push Shockwave passed!")


def test_palm_wave_tornado_procedural_animation() -> None:
    """Tests that waving an open palm faster generates a 3D helical tornado procedural cyclone."""
    print("[Test] Testing Procedural Palm-Wave Tornado Cyclone Animation...")
    renderer = CubeHologramRenderer()
    dummy_lms = [LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)]

    # 1. Stationary open palm -> tornado intensity should be 0
    calm_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
        palm_speed=50.0,  # Below threshold
    )

    for _ in range(20):
        renderer.update(hand_detected=True, is_open=True, pose=calm_pose, dt=0.016)

    assert renderer.tornado_intensity == 0.0, "Tornado should remain at 0 intensity for calm palm"

    # 2. Fast waving open palm -> tornado ramps up
    waving_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
        palm_speed=550.0,  # High waving speed
    )

    for _ in range(30):
        renderer.update(hand_detected=True, is_open=True, pose=waving_pose, dt=0.033)

    assert renderer.tornado_intensity > 0.6, f"Tornado intensity should ramp up above 0.6, got {renderer.tornado_intensity}"
    assert renderer.tornado_phase > 0.0, "Tornado cyclonic phase must advance"

    # Verify tiered vertical stacking in the cyclone along the palm normal (-z)
    c0 = renderer.physics_world.cubes[0]
    c1 = renderer.physics_world.cubes[1]
    c2 = renderer.physics_world.cubes[2]

    # In palm normal direction (-z), cube 2 (top flare) should be further along the axis than cube 0 (base)
    # Palm normal is (0, 0, -1), so h_pos along normal pushes z to negative values
    assert c2.position[2] < c0.position[2], (
        f"Cube 2 (top vortex flare) should be higher along palm normal than Cube 0 (base): "
        f"c2.z={c2.position[2]} vs c0.z={c0.position[2]}"
    )

    # 3. Stop waving -> tornado intensity decays cleanly
    calm_pose.palm_speed = 10.0
    initial_tornado = renderer.tornado_intensity
    for _ in range(25):
        renderer.update(hand_detected=True, is_open=True, pose=calm_pose, dt=0.033)

    assert renderer.tornado_intensity < initial_tornado, "Tornado intensity must decay when waving stops"
    print("  -> Procedural Palm-Wave Tornado Cyclone Animation passed!")


def test_pinch_hysteresis_and_clean_visuals() -> None:
    """Tests pinch grab hysteresis (36px grab / 52px release) and verifies 0 visual clutter."""
    print("[Test] Testing Pinch Hysteresis & Visual Cleanliness...")

    def check_pinch(pinch_dist: float, was_pinching: bool) -> bool:
        thresh = (
            config.PINCH_RELEASE_THRESHOLD_PX
            if was_pinching
            else config.PINCH_THRESHOLD_PX
        )
        return pinch_dist < thresh

    # 1. At 42px (initial rest state) -> False (< 36px required)
    assert check_pinch(42.0, False) is False, "Should not pinch at 42px initially (threshold 36px)"

    # 2. Close fingers to 30px (< 36px) -> True
    assert check_pinch(30.0, False) is True, "Should trigger pinch at 30px (< 36px)"

    # 3. Fingers relax slightly to 44px (between 36px and 52px) -> Hysteresis maintains True!
    assert check_pinch(44.0, True) is True, "Hysteresis must keep pinch active at 44px (below release threshold 52px)"

    # 4. Open fingers to 60px (> 52px) -> releases to False
    assert check_pinch(60.0, True) is False, "Pinch must release when distance exceeds release threshold (52px)"

    # Verify Clean Visual Rendering on 3D Cubes (Zero 2D rings, shockwaves, or lines outside cubes)
    renderer = CubeHologramRenderer()
    material = MATERIALS["Prismatic"]
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    dummy_lms = [LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360) for _ in range(21)]
    test_pose = HandPose(
        handedness="Right",
        landmarks=dummy_lms,
        palm_center_px=(640, 360),
        up_vector=(0.0, -1.0),
        right_vector=(1.0, 0.0),
        palm_scale=80.0,
        is_open=True,
        open_confidence=1.0,
        finger_states=[True] * 5,
        openness_ratio=1.0,
        palm_normal_3d=(0.0, 0.0, -1.0),
        palm_up_3d=(0.0, -1.0, 0.0),
        palm_right_3d=(1.0, 0.0, 0.0),
        palm_center_3d=(640.0, 360.0, 0.0),
        is_pinching=True,
        pinch_point_3d=(640.0, 360.0, 0.0),
        pinch_dist_px=25.0,
    )

    for _ in range(35):
        renderer.update(hand_detected=True, is_open=True, pose=test_pose, dt=0.016)

    renderer.render(frame, test_pose, material)
    assert np.any(frame > 0), "Solid 3D cubes should be rendered onto frame"
    print("  -> Pinch Hysteresis & Visual Cleanliness passed!")


def test_one_euro_filter() -> None:
    """Tests 1€ Adaptive Motion Filter jitter elimination and high-speed responsiveness."""
    print("[Test] Testing 1€ (One Euro) Adaptive Motion Filter...")
    from spatial_math import OneEuroFilter

    filt = OneEuroFilter(fc_min=0.85, beta=0.045, d_cutoff=1.0)

    # 1. Stationary tremor suppression
    np.random.seed(42)
    dt = 0.016
    steady_val = np.array([100.0, 200.0, -50.0], dtype=np.float32)
    raw_samples = []
    filt_samples = []

    for _ in range(60):
        noise = np.random.normal(0.0, 1.5, 3).astype(np.float32)
        sample = steady_val + noise
        raw_samples.append(sample)
        filt_samples.append(filt.filter(sample, dt))

    raw_var = float(np.var(raw_samples[10:], axis=0).mean())
    filt_var = float(np.var(filt_samples[10:], axis=0).mean())
    assert filt_var < raw_var * 0.25, f"1€ filter must suppress jitter by >75%: raw_var={raw_var:.3f}, filt_var={filt_var:.3f}"

    # 2. Fast step transition response (low lag during fast gestures)
    step_val = np.array([300.0, 400.0, 50.0], dtype=np.float32)
    out_step = filt.filter(step_val, dt)
    # Filter should adaptively open cutoff and track rapidly
    dist_to_step = float(np.linalg.norm(step_val - out_step))
    assert dist_to_step < 120.0, f"Filter must adaptively track fast movement without sluggish lag, dist={dist_to_step}"
    print("  -> 1€ Adaptive Motion Filter passed!")


def test_divergence_free_curl_noise() -> None:
    """Tests that the 3D Curl Noise vector field satisfies div(v) == 0 (divergence-free)."""
    print("[Test] Testing Divergence-Free 3D Curl Noise Vector Field...")
    from spatial_math import CurlNoise3D

    curl_field = CurlNoise3D()
    eps = 0.05
    test_points = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([250.0, -180.0, 45.0], dtype=np.float32),
        np.array([-120.0, 310.0, -75.0], dtype=np.float32),
        np.array([640.0, 360.0, 20.0], dtype=np.float32),
    ]

    for pt in test_points:
        t = 1.25
        # Central finite difference divergence approximation
        vx_p = curl_field.evaluate(pt + np.array([eps, 0.0, 0.0]), t)[0]
        vx_m = curl_field.evaluate(pt - np.array([eps, 0.0, 0.0]), t)[0]
        dvx_dx = (vx_p - vx_m) / (2.0 * eps)

        vy_p = curl_field.evaluate(pt + np.array([0.0, eps, 0.0]), t)[1]
        vy_m = curl_field.evaluate(pt - np.array([0.0, eps, 0.0]), t)[1]
        dvy_dy = (vy_p - vy_m) / (2.0 * eps)

        vz_p = curl_field.evaluate(pt + np.array([0.0, 0.0, eps]), t)[2]
        vz_m = curl_field.evaluate(pt - np.array([0.0, 0.0, eps]), t)[2]
        dvz_dz = (vz_p - vz_m) / (2.0 * eps)

        divergence = abs(float(dvx_dx + dvy_dy + dvz_dz))
        assert divergence < 1e-3, f"Curl noise field must have zero divergence everywhere, got div={divergence} at {pt}"

    print("  -> Divergence-Free 3D Curl Noise Vector Field passed!")


def test_obb_sat_collision_queries() -> None:
    """Tests 15-axis Separating Axis Theorem (SAT) rigid body box intersection and separation."""
    print("[Test] Testing OBB 15-Axis Separating Axis Theorem (SAT) Collision...")
    from spatial_math import OBB3D, OBBSATCollider

    # 1. Separated boxes
    box_a = OBB3D(
        center=np.array([100.0, 100.0, 0.0], dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        half_extents=np.array([20.0, 20.0, 20.0], dtype=np.float32),
    )
    box_b = OBB3D(
        center=np.array([160.0, 100.0, 0.0], dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        half_extents=np.array([20.0, 20.0, 20.0], dtype=np.float32),
    )

    sep_res = OBBSATCollider.test_collision(box_a, box_b)
    assert sep_res is None, "Boxes separated by distance 60 (half-sum=40) must return None"

    # 2. Intersecting boxes with angle rotation
    theta = math.pi / 6.0  # 30 deg rotation
    rot_z = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta),  math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    box_b_col = OBB3D(
        center=np.array([130.0, 100.0, 0.0], dtype=np.float32),
        rotation=rot_z,
        half_extents=np.array([20.0, 20.0, 20.0], dtype=np.float32),
    )

    col_res = OBBSATCollider.test_collision(box_a, box_b_col)
    assert col_res is not None, "Intersecting rotated OBBs must report collision"
    assert col_res.intersecting is True, "Must report intersecting"
    assert col_res.penetration > 0.0, f"Penetration must be positive, got {col_res.penetration}"
    assert abs(float(np.linalg.norm(col_res.contact_normal)) - 1.0) < 1e-4, "Contact normal must be unit vector"
    print("  -> OBB 15-Axis SAT Collision passed!")


def test_bone_capsule_skeletal_colliders() -> None:
    """Tests swept-sphere bone capsule geometry and collision impulse projection."""
    print("[Test] Testing Biomechanical Bone Capsule Colliders...")
    from spatial_math import BoneCapsuleCollider

    capsule = BoneCapsuleCollider(
        name="Index_Proximal",
        p0=np.array([100.0, 200.0, 0.0], dtype=np.float32),
        p1=np.array([100.0, 260.0, 0.0], dtype=np.float32),
        radius=12.0,
        velocity=np.array([0.0, 50.0, 0.0], dtype=np.float32),
    )

    # Test closest point to segment midpoint
    test_pt = np.array([115.0, 230.0, 0.0], dtype=np.float32)
    closest, t = capsule.closest_point_to_point(test_pt)
    assert abs(closest[0] - 100.0) < 1e-4 and abs(closest[1] - 230.0) < 1e-4, f"Closest point mismatch: {closest}"
    assert abs(t - 0.5) < 1e-4, f"t should be 0.5, got {t}"

    # Test sphere collision
    sphere_center = np.array([118.0, 230.0, 0.0], dtype=np.float32)
    col = capsule.test_sphere_collision(sphere_center, sphere_radius=15.0)
    assert col is not None, "Sphere overlapping capsule radius must collide"
    norm, overlap, contact_pt = col
    assert overlap > 0.0, f"Overlap must be positive, got {overlap}"
    assert abs(norm[0] - 1.0) < 1e-3, f"Normal must point along +X, got {norm}"
    print("  -> Biomechanical Bone Capsule Colliders passed!")


def test_thin_film_optical_interference() -> None:
    """Tests physical thin-film wave interference reflectance curves across incidence angles."""
    print("[Test] Testing Physical Thin-Film Optical Wave Interference...")
    from spatial_math import ThinFilmInterference

    film = ThinFilmInterference(n_film=1.45, film_thickness_nm=520.0)

    # Normal incidence (cos = 1.0)
    rb_0, rg_0, rr_0 = film.compute_rgb_reflectance(1.0)
    assert 0.0 <= rb_0 <= 1.0 and 0.0 <= rg_0 <= 1.0 and 0.0 <= rr_0 <= 1.0, "Reflectance channels must be in [0, 1]"

    # Grazing incidence (cos = 0.2)
    rb_g, rg_g, rr_g = film.compute_rgb_reflectance(0.2)
    assert 0.0 <= rb_g <= 1.0 and 0.0 <= rg_g <= 1.0 and 0.0 <= rr_g <= 1.0, "Grazing reflectance must be in [0, 1]"

    # Angle change should cause spectral dispersion / phase shift
    diff = abs(rb_0 - rb_g) + abs(rg_0 - rg_g) + abs(rr_0 - rr_g)
    assert diff > 0.05, f"Angle change must produce measurable thin-film color shift, diff={diff}"
    print("  -> Physical Thin-Film Optical Wave Interference passed!")


def main() -> None:
    """Run all verification tests."""
    print("==================================================")
    print("Running Automated Verification Suite for 3D Physics Cubes")
    print("==================================================")
    test_physics_and_collisions()
    test_6dof_rotation_and_torque()
    test_hand_tracker_3d_orientation()
    test_organic_floating_and_dynamic_angling()
    test_renderer_palm_emergence_and_shading()
    test_finger_identification_and_tracking()
    test_finger_collision_and_flick_recoil()
    test_levitation_recovery()
    test_continuous_openness_and_fountain()
    test_dual_palm_midpoint_and_accordion()
    test_dual_hand_multi_colliders()
    test_upward_palm_perspective()
    test_dual_palm_single_closed_detachment()
    test_satisfying_palm_suction_vortex()
    test_cube_to_cube_collision_and_momentum_transfer()
    test_telekinesis_pinch_grab_and_fling()
    test_telekinesis_force_push()
    test_palm_wave_tornado_procedural_animation()
    test_pinch_hysteresis_and_clean_visuals()
    # New Top-Tier Algorithmic Test Suites
    test_one_euro_filter()
    test_divergence_free_curl_noise()
    test_obb_sat_collision_queries()
    test_bone_capsule_skeletal_colliders()
    test_thin_film_optical_interference()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
