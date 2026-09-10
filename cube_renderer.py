"""3D Solid Grey Cube Renderer with True Live Scene Grounding & 6-DOF Dynamics.

Renders true 3D solid grey smooth cubes governed by Newtonian rigid-body physics,
featuring:
- Dynamic soft contact shadows cast onto the live webcam scene & hand.
- Real-time environmental lighting sampling matching actual room conditions.
- PBR Blinn-Phong illumination with Fresnel rim reflection and inter-cube ambient occlusion.
- True camera pinhole perspective projection.
- Micro-beveled highlights on visible geometry edges.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

import config
from hand_tracker import HandPose
from physics import ContactEvent, CubePhysicsWorld, PhysicsCube


@dataclass
class RenderFace:
    """Represents a 3D solid face prepared for depth-sorted rendering."""
    avg_z: float
    poly_2d: np.ndarray
    color: Tuple[int, int, int]
    is_front: bool
    edges_2d: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    bevel_color: Tuple[int, int, int]


class CubeHologramRenderer:
    """Coordinates 6-DOF simulation and renders true 3D physically grounded cubes."""

    # Unit cube vertices in local coordinates
    UNIT_VERTICES: np.ndarray = np.array([
        [-1.0, -1.0, -1.0],  # 0: Back-Bottom-Left
        [ 1.0, -1.0, -1.0],  # 1: Back-Bottom-Right
        [ 1.0,  1.0, -1.0],  # 2: Back-Top-Right
        [-1.0,  1.0, -1.0],  # 3: Back-Top-Left
        [-1.0, -1.0,  1.0],  # 4: Front-Bottom-Left
        [ 1.0, -1.0,  1.0],  # 5: Front-Bottom-Right
        [ 1.0,  1.0,  1.0],  # 6: Front-Top-Right
        [-1.0,  1.0,  1.0],  # 7: Front-Top-Left
    ], dtype=np.float32)

    # 6 Quadrilateral faces (CCW winding)
    FACES: List[Tuple[int, int, int, int]] = [
        (4, 5, 6, 7),  # Front
        (1, 0, 3, 2),  # Back
        (0, 1, 5, 4),  # Bottom
        (7, 6, 2, 3),  # Top
        (5, 1, 2, 6),  # Right
        (0, 4, 7, 3),  # Left
    ]

    def __init__(self) -> None:
        """Initialize physics world, pre-compute light vectors, and setup environmental sampling."""
        self.physics_world: CubePhysicsWorld = CubePhysicsWorld()

        # Normalize lighting vectors
        k_dir = np.array(config.KEY_LIGHT_DIR, dtype=np.float32)
        self.key_light: np.ndarray = k_dir / np.linalg.norm(k_dir)

        f_dir = np.array(config.FILL_LIGHT_DIR, dtype=np.float32)
        self.fill_light: np.ndarray = f_dir / np.linalg.norm(f_dir)

        # Standard view direction (camera looking down -Z)
        self.view_dir: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Pre-compute Blinn-Phong half vector for key light
        h_vec = self.key_light + self.view_dir
        self.half_vec: np.ndarray = h_vec / np.linalg.norm(h_vec)

        # Real-time environmental lighting adaptation
        self.env_ambient_color: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.last_update_time: float = time.time()

        # Prismatic procedural rainbow clock
        self._prism_t0: float = time.time()

        # Motion tracking for hand velocity
        self.prev_palm_3d: Optional[np.ndarray] = None

        # Procedural Palm-Wave Tornado State
        self.tornado_intensity: float = 0.0
        self.tornado_phase: float = 0.0

    def update(
        self,
        hand_detected: bool,
        is_open: bool,
        pose: Optional[HandPose] = None,
        dt: Optional[float] = None,
        poses: Optional[List[HandPose]] = None,
    ) -> None:
        """Updates 6-DOF physics, multi-harmonic organic floating, dynamic angling, and dual-hand bridge.

        Args:
            hand_detected: Whether any hand is present.
            is_open: Whether palm gesture is OPEN.
            pose: Active HandPose data with 3D orientation (single-hand compatibility).
            dt: Optional explicit delta time (seconds).
            poses: Optional list of all tracked HandPose objects (for dual-hand mode).
        """
        now = time.time()
        if dt is None:
            dt = now - self.last_update_time
        self.last_update_time = now

        # Consolidate poses list
        if poses is None:
            active_poses = [pose] if pose is not None else []
        else:
            active_poses = [p for p in poses if p is not None]
            if not active_poses and pose is not None:
                active_poses = [pose]

        effective_detected = hand_detected or (len(active_poses) > 0)
        target_positions: List[np.ndarray] = []
        palm_norm = None
        palm_center = None
        palm_centers: List[np.ndarray] = []
        palm_right = None
        hand_velocity = None
        finger_colliders = []
        target_scale = 1.0

        if effective_detected and len(active_poses) >= 2:
            # -----------------------------------------------------------------
            # DUAL-PALM MODE: Procedural 3D Midpoint & Accordion Bridge
            # -----------------------------------------------------------------
            # Sort left-to-right across screen coordinates for spatial stability
            sorted_hands = sorted(active_poses[:2], key=lambda p: p.palm_center_3d[0])
            p_left, p_right = sorted_hands[0], sorted_hands[1]

            c_left = np.array(p_left.palm_center_3d, dtype=np.float32)
            c_right = np.array(p_right.palm_center_3d, dtype=np.float32)

            left_open = p_left.is_open
            right_open = p_right.is_open
            left_op = getattr(p_left, "openness_ratio", 1.0 if left_open else 0.0)
            right_op = getattr(p_right, "openness_ratio", 1.0 if right_open else 0.0)

            # Gather colliders from BOTH hands (all fingers remain physical)
            finger_colliders = p_left.fingers + p_right.fingers

            if left_open and right_open:
                # -------------------------------------------------------------
                # BOTH PALMS OPEN: Active Procedural Midpoint Bridge
                # -------------------------------------------------------------
                c_mid = (c_left + c_right) * 0.5
                palm_center = c_mid
                palm_centers = [c_left, c_right]

                # Inter-palm bridge unit vector and distance
                bridge_vec = c_right - c_left
                inter_dist = float(np.linalg.norm(bridge_vec))
                u_bridge = (bridge_vec / inter_dist) if inter_dist > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

                # Average palm normal
                n_left = np.array(p_left.palm_normal_3d, dtype=np.float32)
                n_right = np.array(p_right.palm_normal_3d, dtype=np.float32)
                n_avg = (n_left + n_right) * 0.5
                n_len = float(np.linalg.norm(n_avg))
                norm_3d = (n_avg / n_len) if n_len > 1e-4 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
                palm_norm = norm_3d

                # Longitudinal unit vector orthogonal to bridge and normal
                u_long = np.cross(u_bridge, norm_3d)
                u_long_len = float(np.linalg.norm(u_long))
                u_long = (u_long / u_long_len) if u_long_len > 1e-4 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                palm_right = u_bridge

                if self.prev_palm_3d is not None and dt > 1e-4:
                    hand_velocity = (c_mid - self.prev_palm_3d) / dt
                self.prev_palm_3d = c_mid.copy()

                # Dynamic accordion spacing
                avg_scale = (p_left.palm_scale + p_right.palm_scale) * 0.5
                dist_factor = float(np.clip(avg_scale / 70.0, 0.6, 2.2))
                ref_dist = getattr(config, "DUAL_HAND_REF_DISTANCE", 320.0)
                accordion_ratio = inter_dist / max(ref_dist, 1.0)
                min_spacing = getattr(config, "DUAL_HAND_ACCORDION_MIN_SPACING", 48.0) * dist_factor
                max_spacing = getattr(config, "DUAL_HAND_ACCORDION_MAX_SPACING", 165.0) * dist_factor
                base_spacing = float(np.clip(config.CUBE_HORIZONTAL_SPACING * accordion_ratio * dist_factor, min_spacing, max_spacing))

                should_spawn = True
                target_scale = max(left_op, right_op)
                bridge_elevation = getattr(config, "DUAL_HAND_ELEVATION_OFFSET", 40.0) * dist_factor

                for i in range(config.CUBE_COUNT):
                    phase = (2.0 * math.pi / config.CUBE_COUNT) * i
                    heave = math.sin(now * config.BOB_PRIMARY_FREQ + phase) * config.BOB_PRIMARY_AMP * dist_factor
                    flutter = math.sin(now * config.BOB_FLUTTER_FREQ + phase * 2.3) * config.BOB_FLUTTER_AMP * dist_factor
                    sway_x = math.cos(now * config.BOB_SWAY_FREQ_X + phase * 1.4) * config.BOB_SWAY_AMP_X * dist_factor
                    wander_z = math.sin(now * config.BOB_SWAY_FREQ_Z + phase * 0.8) * config.BOB_SWAY_AMP_Z * dist_factor
                    expansion = 1.0 + config.BREATHING_EXPANSION_RATIO * math.sin(now * config.BOB_PRIMARY_FREQ)

                    slot_offset = ((i - 1) * base_spacing * expansion) + sway_x
                    hover_lift = bridge_elevation * 0.82 + heave + flutter
                    long_offset = wander_z * 0.5

                    target = (
                        c_mid
                        + slot_offset * u_bridge
                        + long_offset * u_long
                        + hover_lift * norm_3d
                    )
                    target_positions.append(target)

            elif left_open and not right_open:
                # -------------------------------------------------------------
                # ONLY LEFT OPEN: Disconnect completely from right hand!
                # -------------------------------------------------------------
                primary_pose = p_left
                norm_3d = np.array(primary_pose.palm_normal_3d, dtype=np.float32)
                up_3d = np.array(primary_pose.palm_up_3d, dtype=np.float32)
                right_3d = np.array(primary_pose.palm_right_3d, dtype=np.float32)
                center_3d = c_left

                palm_norm = norm_3d
                palm_center = center_3d
                palm_centers = [center_3d]
                palm_right = right_3d

                if self.prev_palm_3d is not None and dt > 1e-4:
                    hand_velocity = (center_3d - self.prev_palm_3d) / dt
                self.prev_palm_3d = center_3d.copy()

                dist_factor = float(np.clip(primary_pose.palm_scale / 70.0, 0.6, 2.2))
                base_elevation = config.CUBE_ELEVATION_OFFSET * dist_factor
                base_spacing = config.CUBE_HORIZONTAL_SPACING * dist_factor
                should_spawn = True
                target_scale = left_op

                for i in range(config.CUBE_COUNT):
                    phase = (2.0 * math.pi / config.CUBE_COUNT) * i
                    heave = math.sin(now * config.BOB_PRIMARY_FREQ + phase) * config.BOB_PRIMARY_AMP * dist_factor
                    flutter = math.sin(now * config.BOB_FLUTTER_FREQ + phase * 2.3) * config.BOB_FLUTTER_AMP * dist_factor
                    sway_x = math.cos(now * config.BOB_SWAY_FREQ_X + phase * 1.4) * config.BOB_SWAY_AMP_X * dist_factor
                    wander_z = math.sin(now * config.BOB_SWAY_FREQ_Z + phase * 0.8) * config.BOB_SWAY_AMP_Z * dist_factor
                    expansion = 1.0 + config.BREATHING_EXPANSION_RATIO * math.sin(now * config.BOB_PRIMARY_FREQ)

                    slot_offset = ((i - 1) * base_spacing * expansion) + sway_x
                    forward_offset = 18.0 * dist_factor + wander_z * 0.4
                    hover_lift = base_elevation * 0.72 + heave + flutter

                    target = (
                        center_3d
                        + slot_offset * right_3d
                        + forward_offset * up_3d
                        + hover_lift * norm_3d
                    )
                    target_positions.append(target)

            elif right_open and not left_open:
                # -------------------------------------------------------------
                # ONLY RIGHT OPEN: Disconnect completely from left hand!
                # -------------------------------------------------------------
                primary_pose = p_right
                norm_3d = np.array(primary_pose.palm_normal_3d, dtype=np.float32)
                up_3d = np.array(primary_pose.palm_up_3d, dtype=np.float32)
                right_3d = np.array(primary_pose.palm_right_3d, dtype=np.float32)
                center_3d = c_right

                palm_norm = norm_3d
                palm_center = center_3d
                palm_centers = [center_3d]
                palm_right = right_3d

                if self.prev_palm_3d is not None and dt > 1e-4:
                    hand_velocity = (center_3d - self.prev_palm_3d) / dt
                self.prev_palm_3d = center_3d.copy()

                dist_factor = float(np.clip(primary_pose.palm_scale / 70.0, 0.6, 2.2))
                base_elevation = config.CUBE_ELEVATION_OFFSET * dist_factor
                base_spacing = config.CUBE_HORIZONTAL_SPACING * dist_factor
                should_spawn = True
                target_scale = right_op

                for i in range(config.CUBE_COUNT):
                    phase = (2.0 * math.pi / config.CUBE_COUNT) * i
                    heave = math.sin(now * config.BOB_PRIMARY_FREQ + phase) * config.BOB_PRIMARY_AMP * dist_factor
                    flutter = math.sin(now * config.BOB_FLUTTER_FREQ + phase * 2.3) * config.BOB_FLUTTER_AMP * dist_factor
                    sway_x = math.cos(now * config.BOB_SWAY_FREQ_X + phase * 1.4) * config.BOB_SWAY_AMP_X * dist_factor
                    wander_z = math.sin(now * config.BOB_SWAY_FREQ_Z + phase * 0.8) * config.BOB_SWAY_AMP_Z * dist_factor
                    expansion = 1.0 + config.BREATHING_EXPANSION_RATIO * math.sin(now * config.BOB_PRIMARY_FREQ)

                    slot_offset = ((i - 1) * base_spacing * expansion) + sway_x
                    forward_offset = 18.0 * dist_factor + wander_z * 0.4
                    hover_lift = base_elevation * 0.72 + heave + flutter

                    target = (
                        center_3d
                        + slot_offset * right_3d
                        + forward_offset * up_3d
                        + hover_lift * norm_3d
                    )
                    target_positions.append(target)

            else:
                # -------------------------------------------------------------
                # BOTH CLOSED: Funnel smoothly into respective closest palm
                # -------------------------------------------------------------
                c_mid = (c_left + c_right) * 0.5
                palm_center = c_mid
                palm_centers = [c_left, c_right]
                n_left = np.array(p_left.palm_normal_3d, dtype=np.float32)
                n_right = np.array(p_right.palm_normal_3d, dtype=np.float32)
                n_avg = (n_left + n_right) * 0.5
                n_len = float(np.linalg.norm(n_avg))
                norm_3d = (n_avg / n_len) if n_len > 1e-4 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
                palm_norm = norm_3d
                should_spawn = False
                target_scale = 0.0

                for i in range(config.CUBE_COUNT):
                    c_dst = c_left if i == 0 else (c_right if i == 2 else c_mid)
                    target_positions.append(c_dst)

        elif effective_detected and len(active_poses) == 1:
            # -----------------------------------------------------------------
            # SINGLE-PALM MODE: Hovering directly on/above tracked palm
            # -----------------------------------------------------------------
            primary_pose = active_poses[0]
            norm_3d = np.array(primary_pose.palm_normal_3d, dtype=np.float32)
            up_3d = np.array(primary_pose.palm_up_3d, dtype=np.float32)
            right_3d = np.array(primary_pose.palm_right_3d, dtype=np.float32)
            center_3d = np.array(primary_pose.palm_center_3d, dtype=np.float32)

            palm_norm = norm_3d
            palm_center = center_3d
            palm_centers = [center_3d]
            palm_right = right_3d

            if self.prev_palm_3d is not None and dt > 1e-4:
                hand_velocity = (center_3d - self.prev_palm_3d) / dt
            self.prev_palm_3d = center_3d.copy()

            dist_factor = float(np.clip(primary_pose.palm_scale / 70.0, 0.6, 2.2))
            base_elevation = config.CUBE_ELEVATION_OFFSET * dist_factor
            base_spacing = config.CUBE_HORIZONTAL_SPACING * dist_factor

            should_spawn = primary_pose.is_open
            target_scale = getattr(primary_pose, "openness_ratio", 1.0 if should_spawn else 0.0)

            # Procedural Multi-Harmonic Organic Floating Motion
            for i in range(config.CUBE_COUNT):
                phase = (2.0 * math.pi / config.CUBE_COUNT) * i
                heave = math.sin(now * config.BOB_PRIMARY_FREQ + phase) * config.BOB_PRIMARY_AMP * dist_factor
                flutter = math.sin(now * config.BOB_FLUTTER_FREQ + phase * 2.3) * config.BOB_FLUTTER_AMP * dist_factor
                sway_x = math.cos(now * config.BOB_SWAY_FREQ_X + phase * 1.4) * config.BOB_SWAY_AMP_X * dist_factor
                wander_z = math.sin(now * config.BOB_SWAY_FREQ_Z + phase * 0.8) * config.BOB_SWAY_AMP_Z * dist_factor
                expansion = 1.0 + config.BREATHING_EXPANSION_RATIO * math.sin(now * config.BOB_PRIMARY_FREQ)
                slot_offset = ((i - 1) * base_spacing * expansion) + sway_x

                # True normal-anchored hover elevation:
                # - hover_lift lifts outward along norm_3d (upwards when palm is facing up!)
                # - forward_offset centers the cubes comfortably over the palm bed along up_3d
                forward_offset = 18.0 * dist_factor + wander_z * 0.4
                hover_lift = base_elevation * 0.72 + heave + flutter

                target = (
                    center_3d
                    + slot_offset * right_3d
                    + forward_offset * up_3d
                    + hover_lift * norm_3d
                )
                target_positions.append(target)

            finger_colliders = primary_pose.fingers
        else:
            self.prev_palm_3d = None
            should_spawn = False
            for i in range(config.CUBE_COUNT):
                target_positions.append(
                    np.array([640.0 + (i - 1) * config.CUBE_HORIZONTAL_SPACING, 360.0, 0.0], dtype=np.float32)
                )

        # Procedural Palm-Wave Tornado Funnel Dynamics
        # Detect if user is opening their palm and waving it fast
        current_wave_speed = 0.0
        if effective_detected and any(getattr(p, "is_open", False) for p in active_poses):
            for p in active_poses:
                if getattr(p, "is_open", False):
                    p_spd = getattr(p, "palm_speed", 0.0)
                    if p_spd > current_wave_speed:
                        current_wave_speed = p_spd
            # Also factor in 3D center velocity if available
            if hand_velocity is not None:
                vel_mag = float(np.linalg.norm(hand_velocity))
                if vel_mag > current_wave_speed:
                    current_wave_speed = vel_mag

            min_w_spd = getattr(config, "TORNADO_WAVE_MIN_SPEED", 220.0)
            max_w_spd = getattr(config, "TORNADO_WAVE_MAX_SPEED", 650.0)
            if current_wave_speed > min_w_spd:
                target_intensity = float(np.clip(
                    (current_wave_speed - min_w_spd) / max(max_w_spd - min_w_spd, 1.0),
                    0.0,
                    1.0
                ))
            else:
                target_intensity = 0.0
        else:
            target_intensity = 0.0

        # Smooth attack / decay envelope for organic whirlwind ramp-up
        attack_rate = getattr(config, "TORNADO_ATTACK_RATE", 3.5)
        decay_rate = getattr(config, "TORNADO_DECAY_RATE", 2.2)
        if target_intensity > self.tornado_intensity:
            self.tornado_intensity = min(1.0, self.tornado_intensity + attack_rate * dt)
        else:
            self.tornado_intensity = max(0.0, self.tornado_intensity - decay_rate * dt)

        # Advance cyclonic phase
        if self.tornado_intensity > 0.001:
            spin_rate = getattr(config, "TORNADO_SPIN_RATE", 12.5) * (0.45 + 0.55 * self.tornado_intensity)
            self.tornado_phase = (self.tornado_phase + spin_rate * dt) % (math.pi * 200.0)

            # Reconfigure target positions into a 3D helical tornado cone
            if palm_center is not None and palm_norm is not None:
                p_scale = active_poses[0].palm_scale if active_poses else 70.0
                dist_factor = float(np.clip(p_scale / 70.0, 0.6, 2.2))

                funnel_axis = palm_norm
                if palm_right is not None:
                    u_axis1 = palm_right
                    u_axis2 = np.cross(funnel_axis, u_axis1)
                    u2_len = float(np.linalg.norm(u_axis2))
                    u_axis2 = (u_axis2 / u2_len) if u2_len > 1e-4 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                else:
                    ref_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    u_axis1 = np.cross(funnel_axis, ref_up)
                    u1_len = float(np.linalg.norm(u_axis1))
                    u_axis1 = (u_axis1 / u1_len) if u1_len > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    u_axis2 = np.cross(funnel_axis, u_axis1)

                base_elev = getattr(config, "CUBE_ELEVATION_OFFSET", 105.0) * dist_factor
                f_height = getattr(config, "TORNADO_FUNNEL_HEIGHT", 145.0) * dist_factor
                r_base = getattr(config, "TORNADO_BASE_RADIUS", 28.0) * dist_factor
                r_top = getattr(config, "TORNADO_TOP_RADIUS", 90.0) * dist_factor

                blend_t = self.tornado_intensity * self.tornado_intensity * (3.0 - 2.0 * self.tornado_intensity)

                for i in range(min(config.CUBE_COUNT, len(target_positions))):
                    u_tier = i / max(config.CUBE_COUNT - 1, 1)  # 0.0 bottom, 0.5 mid, 1.0 top
                    # Vertical position along funnel axis with undulating fluid wave
                    vert_wave = math.sin(self.tornado_phase * 0.75 + i * 2.2) * (14.0 * dist_factor * self.tornado_intensity)
                    h_pos = (base_elev * 0.5) + (u_tier * f_height) + vert_wave

                    # Expanding cone radius
                    tier_r = r_base + (r_top - r_base) * u_tier

                    # Orbital angle with vertical corkscrew phase twist
                    orbit_angle = self.tornado_phase + (i * 2.0 * math.pi / config.CUBE_COUNT) + (u_tier * 1.85)

                    pos_funnel = (
                        palm_center
                        + funnel_axis * h_pos
                        + (tier_r * math.cos(orbit_angle)) * u_axis1
                        + (tier_r * math.sin(orbit_angle)) * u_axis2
                    )

                    # Smoothly blend standard target into tornado vortex target
                    target_positions[i] = (1.0 - blend_t) * target_positions[i] + blend_t * pos_funnel

        # Step 6-DOF physics world
        self.physics_world.step(
            target_positions=target_positions,
            should_spawn=should_spawn,
            dt=dt,
            palm_normal=palm_norm,
            palm_center=palm_center,
            hand_velocity=hand_velocity,
            finger_colliders=finger_colliders,
            palm_centers=palm_centers,
            palm_right=palm_right,
            target_scale=target_scale,
            poses=active_poses,
            tornado_intensity=self.tornado_intensity,
        )

    @staticmethod
    def _hsv_to_bgr(h_deg: float, s: float, v: float) -> Tuple[int, int, int]:
        """Converts HSV to BGR ints without per-pixel OpenCV overhead."""
        h = (h_deg % 360.0) / 60.0
        c = v * s
        x = c * (1.0 - abs(h % 2.0 - 1.0))
        m = v - c
        hi = int(h) % 6
        if hi == 0:
            r, g, b = c, x, 0.0
        elif hi == 1:
            r, g, b = x, c, 0.0
        elif hi == 2:
            r, g, b = 0.0, c, x
        elif hi == 3:
            r, g, b = 0.0, x, c
        elif hi == 4:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        return (int((b + m) * 255.0), int((g + m) * 255.0), int((r + m) * 255.0))

    def _prismatic_face_colors(
        self,
        face_idx: int,
        cube_idx: int,
        norm: np.ndarray,
        illumination: float,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Procedural prismatic face + bevel colors for the single default material.

        Each of the 6 faces owns a hue lane drifting over time; per-cube offset
        keeps cubes distinct; slight normal-based shift adds view-dependent shimmer.
        Value rides on the existing illumination term so PBR grounding is preserved.
        """
        prism_t = time.time() - self._prism_t0
        face_hue = (
            face_idx * config.PRISM_FACE_STEP
            + cube_idx * config.PRISM_CUBE_STEP
            + prism_t * config.PRISM_HUE_SPEED
            + 12.0 * float(norm[2])
        ) % 360.0
        sat = float(np.clip(config.PRISM_SATURATION, 0.0, 1.0))
        val = min(1.0, 0.35 + 0.65 * illumination)
        face_bgr = self._hsv_to_bgr(face_hue, sat, val)
        bevel_bgr = self._hsv_to_bgr(face_hue, sat * 0.45, min(1.0, val + 0.22))
        return face_bgr, bevel_bgr

    def _update_environmental_lighting(self, frame: np.ndarray) -> None:
        """Samples real-time webcam frame ambient luminance and color temperature."""
        # Downsample sample grid for sub-millisecond execution
        sample_grid = frame[::16, ::16]
        mean_bgr = cv2.mean(sample_grid)[:3]
        norm_bgr = np.array(mean_bgr, dtype=np.float32) / 255.0

        # Exponential moving average adaptation
        rate = config.ENV_LIGHT_ADAPTATION_RATE
        self.env_ambient_color = (1.0 - rate) * self.env_ambient_color + rate * norm_bgr

    def _render_contact_shadows(
        self,
        frame: np.ndarray,
        active_cubes: List[PhysicsCube],
        pose: HandPose,
    ) -> None:
        """Renders dynamic soft contact shadows cast onto the live webcam scene & hand.

        Shadows dynamically scale in blur radius and opacity with cube height:
        close to palm -> dark, sharp contact shadow; high up -> soft, diffuse penumbra.
        """
        if not config.SHADOW_ENABLED or not active_cubes:
            return

        frame_h, frame_w = frame.shape[:2]
        dist_factor = float(np.clip(pose.palm_scale / 70.0, 0.6, 2.2))
        base_size = config.CUBE_SIZE * dist_factor
        palm_center = np.array(pose.palm_center_3d, dtype=np.float32)

        for cube in active_cubes:
            if cube.current_scale <= 0.05:
                continue

            # Calculate 3D distance to palm
            dist_to_palm = float(np.linalg.norm(cube.position - palm_center))
            alt_ratio = float(np.clip(dist_to_palm / (config.CUBE_ELEVATION_OFFSET * dist_factor * 1.5), 0.0, 1.0))

            # Dynamic shadow footprint parameters
            shadow_radius_x = int(base_size * cube.current_scale * (0.85 + 0.35 * alt_ratio))
            shadow_radius_y = int(shadow_radius_x * 0.65)  # Elliptical slant

            # Light projection displacement onto surface
            offset_x = int(-self.key_light[0] * dist_to_palm * config.SHADOW_OFFSET_FACTOR * 0.5)
            offset_y = int(-self.key_light[1] * dist_to_palm * config.SHADOW_OFFSET_FACTOR * 0.5)

            shadow_cx = int(round(cube.position[0] + offset_x))
            shadow_cy = int(round(cube.position[1] + offset_y))


            # Blur kernel and opacity modulated by altitude
            raw_blur = int(config.SHADOW_BLUR_MIN + (config.SHADOW_BLUR_MAX - config.SHADOW_BLUR_MIN) * alt_ratio)
            blur_k = raw_blur if (raw_blur % 2 == 1) else raw_blur + 1
            opacity = (
                config.SHADOW_MAX_OPACITY
                - (config.SHADOW_MAX_OPACITY - config.SHADOW_MIN_OPACITY) * alt_ratio
            ) * cube.current_scale

            # Local ROI bounds
            pad = blur_k + 4
            x1 = max(0, shadow_cx - shadow_radius_x - pad)
            y1 = max(0, shadow_cy - shadow_radius_y - pad)
            x2 = min(frame_w, shadow_cx + shadow_radius_x + pad)
            y2 = min(frame_h, shadow_cy + shadow_radius_y + pad)

            roi_w = x2 - x1
            roi_h = y2 - y1
            if roi_w <= 0 or roi_h <= 0:
                continue

            # Draw blurred shadow mask on ROI
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            local_center = (shadow_cx - x1, shadow_cy - y1)
            val = int(opacity * 255)
            cv2.ellipse(mask, local_center, (shadow_radius_x, shadow_radius_y), 0, 0, 360, val, -1)
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

            # Sub-millisecond vectorized darkening
            inv_mask = cv2.merge([255 - mask] * 3)
            frame_roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.multiply(frame_roi, inv_mask, scale=1.0 / 255.0)

    def render(
        self,
        frame: np.ndarray,
        pose: Optional[Any],
        material: config.GreyMaterial,
        poses: Optional[List[HandPose]] = None,
    ) -> None:
        """Projects and renders true 3D solid cubes with palm emergence aperture and PBR shading.

        Args:
            frame: Target BGR video frame to draw on.
            pose: Tracked HandPose from HandTracker, or None, or list of poses.
            material: Active GreyMaterial configuration.
            poses: Optional list of HandPose objects when multi-hand is present.
        """
        active_cubes = [c for c in self.physics_world.cubes if c.current_scale > 0.001]
        if not active_cubes:
            return

        # Canonicalize pose reference
        if isinstance(pose, list):
            active_poses = pose
            ref_pose = active_poses[0] if active_poses else None
        elif poses is not None:
            active_poses = poses
            ref_pose = active_poses[0] if active_poses else (pose if isinstance(pose, HandPose) else None)
        else:
            active_poses = [pose] if isinstance(pose, HandPose) else []
            ref_pose = pose if isinstance(pose, HandPose) else None

        # 1. Update real-time environmental lighting from the live frame
        self._update_environmental_lighting(frame)

        h, w = frame.shape[:2]
        if active_poses and len(active_poses) >= 2:
            avg_scale = (active_poses[0].palm_scale + active_poses[1].palm_scale) * 0.5
            dist_factor = float(np.clip(avg_scale / 70.0, 0.6, 2.2))
        elif ref_pose is not None:
            dist_factor = float(np.clip(ref_pose.palm_scale / 70.0, 0.6, 2.2))
        else:
            dist_factor = 1.0

        base_size = config.CUBE_SIZE * dist_factor

        focal_len = config.FOCAL_LENGTH
        cx_screen = w * 0.5
        cy_screen = h * 0.5
        z_baseline = config.CAMERA_BASELINE_DEPTH

        all_faces: List[RenderFace] = []

        # 3. Transform, Light, and Project each 3D Cube
        for cube_idx, cube in enumerate(active_cubes):
            effective_radius = base_size * cube.current_scale
            scaled_verts = self.UNIT_VERTICES.copy()

            # Dynamic Tidal Suction Stretching (Matter elongation toward the palm singularity)
            if cube.emergence < 0.95 and ref_pose is not None:
                c_sink = ref_pose.palm_center_3d
                if active_poses and len(active_poses) >= 2:
                    d0 = float(np.linalg.norm(cube.position - active_poses[0].palm_center_3d))
                    d1 = float(np.linalg.norm(cube.position - active_poses[1].palm_center_3d))
                    c_sink = active_poses[0].palm_center_3d if d0 <= d1 else active_poses[1].palm_center_3d

                to_sink = np.array(c_sink, dtype=np.float32) - cube.position
                dist_sink = float(np.linalg.norm(to_sink))
                if dist_sink > 1e-3:
                    pull_world = to_sink / dist_sink
                    rot_mat = cube.get_rotation_matrix()
                    pull_local = rot_mat.T @ pull_world
                    u_prog = float(1.0 - np.clip(cube.emergence, 0.0, 1.0))
                    stretch_gain = getattr(config, "SUCTION_STRETCH_FACTOR", 0.30)
                    stretch_mag = stretch_gain * u_prog * float(np.clip(140.0 / max(dist_sink, 25.0), 0.2, 1.4))

                    # Deform local vertices along pull direction
                    for v_i in range(8):
                        v = scaled_verts[v_i]
                        dot_p = float(np.dot(v, pull_local))
                        scaled_verts[v_i] += (pull_local * (dot_p * stretch_mag * 1.35) - (v - pull_local * dot_p) * (stretch_mag * 0.28))

            scaled_verts = scaled_verts * effective_radius
            rot_mat = cube.get_rotation_matrix()
            transformed_3d = scaled_verts @ rot_mat.T

            # 3D world coordinates of cube center
            center_x, center_y, center_z = cube.position

            # True Camera Pinhole Perspective Projection
            projected_2d = np.zeros((8, 2), dtype=np.int32)
            for v_idx in range(8):
                vx, vy, vz = transformed_3d[v_idx]
                world_x = center_x + vx
                world_y = center_y + vy
                world_z = center_z + vz

                cam_x = world_x - cx_screen
                cam_y = world_y - cy_screen
                cam_z = z_baseline + world_z

                # Pinhole projection equation
                px = int(round(cx_screen + focal_len * (cam_x / max(cam_z, 50.0))))
                py = int(round(cy_screen + focal_len * (cam_y / max(cam_z, 50.0))))
                projected_2d[v_idx] = [px, py]

            # Process 6 quadrilateral faces (each owns a prismatic hue lane)
            for face_idx, (v0, v1, v2, v3) in enumerate(self.FACES):
                p0 = transformed_3d[v0]
                p1 = transformed_3d[v1]
                p2 = transformed_3d[v2]
                p3 = transformed_3d[v3]

                avg_z = float(center_z + (p0[2] + p1[2] + p2[2] + p3[2]) * 0.25)

                # Face normal vector
                edge1 = p1 - p0
                edge2 = p3 - p0
                norm = np.cross(edge1, edge2)
                norm_len = float(np.linalg.norm(norm))
                if norm_len > 1e-6:
                    norm = norm / norm_len
                else:
                    norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                # Back-face culling check relative to view vector (0, 0, -1)
                is_front = norm[2] > -0.15

                # 1. Key Light diffuse
                diff_key = max(0.0, float(np.dot(norm, self.key_light)))

                # 2. Fill Light diffuse with soft ambient
                diff_fill = max(0.0, float(np.dot(norm, self.fill_light))) * 0.35

                # 3. Blinn-Phong specular highlight
                spec_dot = max(0.0, float(np.dot(norm, self.half_vec)))
                specular = (spec_dot ** config.SPECULAR_POWER) * config.SPECULAR_INTENSITY

                # 4. Fresnel rim reflection on grazing angles
                cos_theta_v = max(0.0, float(-np.dot(norm, self.view_dir)))
                fresnel = ((1.0 - cos_theta_v) ** config.FRESNEL_POWER) * config.FRESNEL_INTENSITY

                # 5. Inter-cube contact ambient occlusion
                ao_factor = 1.0
                for other_idx, other_cube in enumerate(active_cubes):
                    if other_idx == cube_idx:
                        continue
                    to_other = other_cube.position - cube.position
                    dist_to_other = float(np.linalg.norm(to_other))
                    proximity_thresh = base_size * 2.8
                    if 1e-4 < dist_to_other < proximity_thresh:
                        dir_to_other = to_other / dist_to_other
                        facing_alignment = max(0.0, float(np.dot(norm, dir_to_other)))
                        if facing_alignment > 0.4:
                            prox_ratio = 1.0 - (dist_to_other / proximity_thresh)
                            ao_factor *= max(0.60, 1.0 - 0.32 * facing_alignment * prox_ratio)

                # Environmental lighting tint
                env_lum = float(np.mean(self.env_ambient_color))
                ambient_base = config.AMBIENT_LIGHT_DEFAULT * (0.8 + 0.4 * env_lum)

                # Composite illumination
                illumination = np.clip(
                    (ambient_base + 0.65 * diff_key + diff_fill) * ao_factor,
                    0.12,
                    1.0,
                )

                # Prismatic procedural face color (value rides on illumination)
                (b_prism, g_prism, r_prism), (b_bev, g_bev, r_bev) = self._prismatic_face_colors(
                    face_idx, cube_idx, norm, illumination
                )

                # Add specular gleam + Fresnel rim reflection on top
                spec_val = int((specular + fresnel) * 255.0)
                b_final = min(255, b_prism + spec_val)
                g_final = min(255, g_prism + spec_val)
                r_final = min(255, r_prism + spec_val)

                # Bevel highlight derived from face hue without collision pulses
                b_bev_lit = b_bev
                g_bev_lit = g_bev
                r_bev_lit = r_bev

                poly = projected_2d[[v0, v1, v2, v3]]
                edges_2d = [
                    (tuple(projected_2d[v0]), tuple(projected_2d[v1])),
                    (tuple(projected_2d[v1]), tuple(projected_2d[v2])),
                    (tuple(projected_2d[v2]), tuple(projected_2d[v3])),
                    (tuple(projected_2d[v3]), tuple(projected_2d[v0])),
                ]

                all_faces.append(
                    RenderFace(
                        avg_z=avg_z,
                        poly_2d=poly,
                        color=(b_final, g_final, r_final),
                        is_front=is_front,
                        edges_2d=edges_2d,
                        bevel_color=(b_bev_lit, g_bev_lit, r_bev_lit),
                    )
                )

        if not all_faces:
            return

        # 4. Depth Sort Faces (Painter's Algorithm: Furthest first)
        all_faces.sort(key=lambda f: f.avg_z, reverse=True)

        # 5. Render Solid Opaque Faces and Micro-Bevels (Pristine 3D without 2D clutter)
        for face in all_faces:
            # Draw solid polygon face
            cv2.fillPoly(frame, [face.poly_2d], face.color, lineType=cv2.LINE_AA)

            # Draw micro-bevel highlight on visible front-facing edges
            if face.is_front:
                for pt1, pt2 in face.edges_2d:
                    cv2.line(
                        frame,
                        pt1,
                        pt2,
                        face.bevel_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

    @staticmethod
    def _project_point_3d(
        pt_3d: np.ndarray,
        cx_screen: float,
        cy_screen: float,
        focal_len: float,
        z_baseline: float,
    ) -> Tuple[int, int]:
        """Project a single 3D camera-space point to 2D screen coordinates."""
        cam_x = float(pt_3d[0]) - cx_screen
        cam_y = float(pt_3d[1]) - cy_screen
        cam_z = z_baseline + float(pt_3d[2])
        px = int(round(cx_screen + focal_len * (cam_x / max(cam_z, 50.0))))
        py = int(round(cy_screen + focal_len * (cam_y / max(cam_z, 50.0))))
        return (px, py)

    @property
    def latest_contact(self) -> Optional[ContactEvent]:
        """Returns the most recent physical contact event within the last 1.2s, if any."""
        now_t = time.time()
        for contact in reversed(self.physics_world.recent_contacts):
            if (now_t - contact.timestamp) < 1.2:
                return contact
        return None
