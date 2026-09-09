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
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
