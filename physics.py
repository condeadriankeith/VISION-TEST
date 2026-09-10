"""Real-time 6-DOF rigid-body physics and procedural dynamics simulation.

Simulates true 3D Newtonian mechanics:
- 3D translation & rotation governed by moment-of-inertia tensors.
- Elastic cube-to-cube collisions with contact points, normal restitution, tangential friction, and impact torque.
- Dynamic angling: aerodynamic banking, gyroscopic leveling, and palm tilt conformance.
- Palm levitation cushion preventing hand penetration.
- Inertial lag responding to hand acceleration.
- Fluid aerodynamic drag and rotational damping.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, List, Optional, Tuple

import numpy as np

import config


@dataclass
class ContactEvent:
    """Represents a dynamic physical contact event between a finger and a cube."""
    finger_name: str
    cube_index: int
    contact_pos: np.ndarray      # 3D position of contact point in camera space
    contact_normal: np.ndarray   # Collision normal vector
    impulse_mag: float          # Magnitude of physical impulse applied
    timestamp: float            # Time of contact for decay rendering


@dataclass
class CubeCollisionEvent:
    """Represents an energetic physical collision between two cubes."""
    cube_a: int
    cube_b: int
    contact_pos: np.ndarray
    impulse_mag: float
    timestamp: float


class PhysicsCube:
    """6-DOF rigid body cube with 3D translation, 3D orientation, and contact torque."""

    def __init__(self, index: int, initial_pos: np.ndarray) -> None:
        """Initialize 6-DOF physics cube.

        Args:
            index: Unique identifier for this cube.
            initial_pos: Initial [x, y, z] 3D coordinates.
        """
        self.index: int = index

        # Linear dynamics (Newton's 2nd Law)
        self.position: np.ndarray = initial_pos.astype(np.float32).copy()
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        self.mass: float = config.CUBE_MASS

        # Rotational dynamics: 3D rotation matrix and angular velocity
        self.rotation_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.angular_velocity: np.ndarray = np.array(
            config.CUBE_ROTATION_RATES[index % len(config.CUBE_ROTATION_RATES)],
            dtype=np.float32,
        )
        # Base ambient spin target for organic tumbling
        self.ambient_spin: np.ndarray = self.angular_velocity.copy()

        # Euler angles kept synchronized for inspection
        self.angles: np.ndarray = np.array([index * 0.5, index * 0.9, index * 0.3], dtype=np.float32)
        self._init_rotation_from_angles()

        # Physical geometry & Moment of Inertia with corner-aware bounding radius
        radius_factor = getattr(config, "CUBE_COLLISION_RADIUS_FACTOR", 1.25)
        self.radius: float = config.CUBE_SIZE * radius_factor
        side_len = 2.0 * config.CUBE_SIZE
        self.inertia_scalar: float = (1.0 / 6.0) * self.mass * (side_len ** 2)
        self.inv_inertia: float = 1.0 / max(self.inertia_scalar, 1e-4)

        # Scale, emergence, and visibility state
        self.current_scale: float = 0.0
        self.target_scale: float = 0.0
        self.emergence: float = 0.0
        self.stagger_timer: float = 0.0
        self.retract_timer: float = 0.0
        # Vortex swirl direction for palm-suck (alternate for organic funnel)
        self.swirl_dir: float = 1.0 if (index % 2 == 0) else -1.0

        # Telekinesis & Kinetic Collision State
        self.is_grabbed: bool = False
        self.fling_timer: float = 0.0
        self.recoil_timer: float = 0.0
        self.impact_energy: float = 0.0
        self.last_impact_pos: Optional[np.ndarray] = None

    def _init_rotation_from_angles(self) -> None:
        """Initialize 3D rotation matrix from Euler angles."""
        ax, ay, az = self.angles
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)

        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.rotation_matrix = rz @ ry @ rx

    def update_scale(
        self,
        should_spawn: bool,
        dt: float,
        stagger_delay: float = 0.0,
        target_scale: float = 1.0,
    ) -> None:
        """Fast flowy spawn (ease-out-back pop) and vacuum suck collapse (ease-in dive)."""
        # Handle manual test overrides where scale was set directly
        if self.current_scale > 0.0 and self.emergence == 0.0 and should_spawn:
            self.emergence = self.current_scale

        spawn_speed = getattr(config, "SPAWN_SPEED", config.EMERGENCE_SPEED)
        collapse_speed = getattr(config, "COLLAPSE_SPEED", config.EMERGENCE_SPEED)

        if should_spawn:
            clamped_target = float(np.clip(target_scale, 0.25, 1.0))
            self.target_scale = clamped_target
            self.retract_timer = 0.0
            self.stagger_timer += dt
            if self.stagger_timer >= stagger_delay:
                if self.emergence < clamped_target:
                    self.emergence = min(clamped_target, self.emergence + spawn_speed * dt)
                elif self.emergence > clamped_target:
                    self.emergence = max(clamped_target, self.emergence - collapse_speed * 0.5 * dt)
        else:
            self.target_scale = 0.0
            self.stagger_timer = 0.0
            self.retract_timer += dt
            if self.retract_timer >= stagger_delay:
                self.emergence = max(0.0, self.emergence - collapse_speed * dt)

        t = float(np.clip(self.emergence, 0.0, 1.0))
        if should_spawn:
            # Ease-out-back pop: fast launch, soft overshoot settle (flowy)
            c1 = 1.70158
            c3 = c1 + 1.0
            u = t - 1.0
            back = 1.0 + c3 * (u ** 3) + c1 * (u ** 2)
            # Modulate with target scale
            self.current_scale = float(np.clip(back * self.target_scale, 0.0, 1.15 * self.target_scale))
            if t >= self.target_scale:
                self.current_scale = self.target_scale
        else:
            # Two-stage gravitational vacuum dive:
            # Stage 1 (t in [0.38, 1.0]): cubes remain substantial 3D bodies (55-100% scale)
            # while gathering and spiraling toward the center of the palm.
            # Stage 2 (t in [0.0, 0.38]): as cubes reach the palm depression,
            # scale plunges rapidly to 0 with cubic acceleration into the singularity.
            if t > 0.38:
                prog = (t - 0.38) / 0.62
                self.current_scale = float(0.55 + 0.45 * pow(prog, 0.85))
            else:
                prog = t / 0.38
                self.current_scale = float(0.55 * pow(prog, 2.2))
        if self.emergence <= 0.005:
            self.current_scale = 0.0

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns the current 3D orthonormal orientation matrix."""
        return self.rotation_matrix

    def integrate_rotation(self, dt: float) -> None:
        """Integrates 3D orientation using Rodrigues' formula and re-orthonormalizes."""
        theta_vec = self.angular_velocity * dt
        theta = float(np.linalg.norm(theta_vec))

        if theta > 1e-6:
            k = theta_vec / theta
            kx, ky, kz = k[0], k[1], k[2]
            cross_k = np.array([
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0]
            ], dtype=np.float32)
            delta_r = (
                np.eye(3, dtype=np.float32)
                + math.sin(theta) * cross_k
                + (1.0 - math.cos(theta)) * (cross_k @ cross_k)
            )
            self.rotation_matrix = delta_r @ self.rotation_matrix

            # SVD projection to strictly guarantee orthonormality without distortion
            u, _, vh = np.linalg.svd(self.rotation_matrix)
            self.rotation_matrix = (u @ vh).astype(np.float32)

        # Synchronize angles
        self.angles += self.angular_velocity * dt
        self.angles %= (2.0 * math.pi)


class CubePhysicsWorld:
    """Manages 6-DOF physics, organic floating dynamics, and collision response."""

    def __init__(self) -> None:
        """Initializes the physics world with 3 cubes and tuning parameters."""
        self.cubes: List[PhysicsCube] = [
            PhysicsCube(i, np.array([640.0 + (i - 1) * config.CUBE_HORIZONTAL_SPACING, 360.0, 0.0], dtype=np.float32))
            for i in range(config.CUBE_COUNT)
        ]

        # Physics tuning constants
        self.spring_k: float = config.SPRING_STIFFNESS
        self.damping_c: float = config.VELOCITY_DAMPING
        self.restitution: float = config.COLLISION_RESTITUTION
        self.friction: float = config.COLLISION_FRICTION

        # Tracking state for hand velocity/acceleration
        self.prev_palm_pos: Optional[np.ndarray] = None
        self.palm_velocity: np.ndarray = np.zeros(3, dtype=np.float32)

        # Dynamic physical contact events for visual feedback
        self.recent_contacts: List[ContactEvent] = []
        self.recent_cube_collisions: List[CubeCollisionEvent] = []

        # Telekinesis Force Suite Tracking State
        self.grabbed_cube_idx: Optional[int] = None
        self.last_pinch_state: bool = False
        self.last_pinch_point: Optional[np.ndarray] = None
        self.last_pinch_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_force_push_time: float = 0.0
        self.force_push_active: float = 0.0
        self.last_force_push_pos: Optional[np.ndarray] = None
        self.last_fling_speed: float = 0.0

    def step(
        self,
        target_positions: List[np.ndarray],
        should_spawn: bool,
        dt: float,
        palm_normal: Optional[np.ndarray] = None,
        palm_center: Optional[np.ndarray] = None,
        hand_velocity: Optional[np.ndarray] = None,
        finger_colliders: Optional[List[Any]] = None,
        palm_centers: Optional[List[np.ndarray]] = None,
        palm_right: Optional[np.ndarray] = None,
        target_scale: float = 1.0,
        poses: Optional[List[Any]] = None,
        tornado_intensity: float = 0.0,
    ) -> None:
        """Advances the 6-DOF physics simulation by time step dt.

        Args:
            target_positions: Target 3D coordinates for each cube.
            should_spawn: Whether palm is open (True) or closed (False).
            dt: Delta time in seconds since last frame.
            palm_normal: Optional 3D normal vector of the palm.
            palm_center: Optional 3D center position of the palm.
            hand_velocity: Optional 3D velocity of the hand for inertial transfer.
            finger_colliders: Optional list of identified fingertip colliders for physical interactions.
            palm_centers: Optional list of all detected palm centers (for dual-hand sinks).
            palm_right: Optional 3D right/lateral vector for blossom petal flare.
            target_scale: Target scale factor modulated by continuous openness.
            poses: Optional list of all detected HandPose instances for telekinetic gestures.
        """
        dt = float(np.clip(dt, 0.001, 0.05))

        if palm_normal is None:
            norm_3d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            norm_3d = palm_normal.astype(np.float32)

        # ---------------------------------------------------------------------
        # TELEKINESIS FORCE SUITE: Pinch-to-Grab, Force-Fling & Shockwave Blast
        # ---------------------------------------------------------------------
        pinching_hand = None
        if poses:
            for p in poses:
                if getattr(p, "is_pinching", False) and getattr(p, "pinch_point_3d", None) is not None:
                    pinching_hand = p
                    break

        is_pinching_now = pinching_hand is not None
        current_pinch_pt = (
            np.array(pinching_hand.pinch_point_3d, dtype=np.float32)
            if (is_pinching_now and pinching_hand is not None and pinching_hand.pinch_point_3d is not None)
            else None
        )

        # 1. Force Grip: Pinching grabs nearest cube and locks it to fingers
        if is_pinching_now and current_pinch_pt is not None:
            if self.grabbed_cube_idx is None:
                best_dist = 260.0
                best_idx = None
                for i, c in enumerate(self.cubes):
                    if c.current_scale > 0.08:
                        d = float(np.linalg.norm(c.position - current_pinch_pt))
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                if best_idx is not None:
                    self.grabbed_cube_idx = best_idx
                    gc = self.cubes[best_idx]
                    gc.is_grabbed = True
                    gc.fling_timer = 0.0
                    gc.recoil_timer = 0.0

            if self.grabbed_cube_idx is not None:
                gc = self.cubes[self.grabbed_cube_idx]
                if self.last_pinch_point is not None and dt > 1e-4:
                    p_vel = (current_pinch_pt - self.last_pinch_point) / dt
                    self.last_pinch_vel = 0.65 * p_vel + 0.35 * self.last_pinch_vel
                elif pinching_hand is not None:
                    self.last_pinch_vel = np.array(pinching_hand.hand_velocity_3d, dtype=np.float32)

                gc.position[:] = current_pinch_pt
                gc.velocity[:] = self.last_pinch_vel
                gc.angular_velocity *= (0.94 ** (dt * 60.0))

        # 2. Force Fling: Releasing pinch mid-motion throws cube with high velocity
        if (not is_pinching_now) and self.grabbed_cube_idx is not None:
            flung = self.cubes[self.grabbed_cube_idx]
            flung.is_grabbed = False
            fling_spd = float(np.linalg.norm(self.last_pinch_vel))
            if fling_spd > 80.0:
                mult = getattr(config, "FORCE_FLING_MULTIPLIER", 1.85)
                flung.velocity = self.last_pinch_vel * mult
                flung.fling_timer = getattr(config, "FORCE_FLING_FREE_TIME", 1.2)
                flung.angular_velocity = np.random.uniform(-10.0, 10.0, 3).astype(np.float32)
                self.last_fling_speed = fling_spd * mult
            self.grabbed_cube_idx = None

        self.last_pinch_state = is_pinching_now
        self.last_pinch_point = current_pinch_pt

        # 3. Force Push Shockwave: Open palm thrusting forward sends radial blast
        now_sec = time.time()
        self.force_push_active = max(0.0, self.force_push_active - dt)
        if poses:
            for p in poses:
                thrust_val = getattr(p, "palm_thrust_speed", 0.0)
                thrust_thresh = getattr(config, "FORCE_PUSH_SPEED_THRESH", 380.0)
                cooldown = getattr(config, "FORCE_PUSH_COOLDOWN", 0.4)
                if p.is_open and thrust_val >= thrust_thresh and (now_sec - self.last_force_push_time) > cooldown:
                    self.last_force_push_time = now_sec
                    self.force_push_active = 0.45
                    p_cnt = np.array(p.palm_center_3d, dtype=np.float32)
                    p_nrm = np.array(p.palm_normal_3d, dtype=np.float32)
                    self.last_force_push_pos = p_cnt.copy()

                    blast_impulse = getattr(config, "FORCE_PUSH_IMPULSE", 540.0)
                    for c in self.cubes:
                        if c.is_grabbed or c.current_scale <= 0.05:
                            continue
                        diff = c.position - p_cnt
                        d_c = max(float(np.linalg.norm(diff)), 15.0)
                        rad_dir = diff / d_c
                        b_dir = 0.60 * p_nrm + 0.40 * rad_dir
                        b_len = float(np.linalg.norm(b_dir))
                        b_dir = (b_dir / b_len) if b_len > 1e-4 else p_nrm
                        c.velocity += b_dir * blast_impulse
                        c.angular_velocity += np.random.uniform(-14.0, 14.0, 3).astype(np.float32)
                        c.recoil_timer = getattr(config, "CUBE_RECOIL_DURATION", 0.45) * 1.6
                        c.impact_energy = 1.0

        # Hand inertial lag with acceleration clamping to keep following crisp and responsive
        inertial_accel = np.zeros(3, dtype=np.float32)
        if hand_velocity is not None:
            delta_v = hand_velocity - self.palm_velocity
            lag_factor = getattr(config, "HAND_INERTIAL_LAG_FACTOR", 0.15)
            raw_accel = -(delta_v / dt) * lag_factor
            accel_mag = float(np.linalg.norm(raw_accel))
            max_accel = 320.0
            if accel_mag > max_accel:
                raw_accel = raw_accel * (max_accel / accel_mag)
            inertial_accel = raw_accel.astype(np.float32)
            self.palm_velocity = 0.65 * self.palm_velocity + 0.35 * hand_velocity

        # 1. Integrate Scale, Staggered Palm Emergence & Spring-Damper Dynamics
        for i, cube in enumerate(self.cubes):
            # Recoil and impact energy decay
            if cube.recoil_timer > 0.0:
                cube.recoil_timer = max(0.0, cube.recoil_timer - dt)
            if cube.impact_energy > 0.0:
                cube.impact_energy = max(0.0, cube.impact_energy - dt * 3.5)

            # A. If grabbed by Telekinesis Force Grip: lock scale & position, integrate rotation
            if cube.is_grabbed:
                cube.current_scale = target_scale
                cube.emergence = 1.0
                cube.integrate_rotation(dt)
                continue

            # B. If flying freely under Telekinesis Force Fling: ballistic motion with drag
            if cube.fling_timer > 0.0:
                cube.fling_timer = max(0.0, cube.fling_timer - dt)
                cube.current_scale = target_scale
                cube.emergence = 1.0
                speed = float(np.linalg.norm(cube.velocity))
                air_drag = -config.AIR_DRAG_QUADRATIC * speed * cube.velocity
                cube.velocity += (air_drag / cube.mass) * dt
                cube.position += cube.velocity * dt
                cube.angular_velocity *= (config.ANGULAR_DRAG ** (dt * 60.0))
                cube.integrate_rotation(dt)
                continue

            # Funnel stagger: spawn pops center-first, collapse sucks outers-first
            if should_spawn:
                delay = 0.0 if i == 1 else getattr(config, "SPAWN_STAGGER", config.EMERGENCE_STAGGER)
            else:
                delay = 0.0 if i != 1 else getattr(config, "COLLAPSE_STAGGER", config.EMERGENCE_STAGGER)
            cube.update_scale(should_spawn, dt, stagger_delay=delay, target_scale=target_scale)

            # Assign destination palm center for this cube
            c_palm = palm_center
            if palm_centers and len(palm_centers) >= 2:
                if i == 0:
                    c_palm = palm_centers[0]
                elif i == len(self.cubes) - 1:
                    c_palm = palm_centers[-1]
                else:
                    d0 = float(np.linalg.norm(cube.position - palm_centers[0]))
                    d1 = float(np.linalg.norm(cube.position - palm_centers[-1]))
                    c_palm = palm_centers[0] if d0 <= d1 else palm_centers[-1]

            # Submerged palm origin point (inside the hand)
            if c_palm is not None:
                p_inside = c_palm - config.PALM_SUBMERGE_DEPTH * norm_3d
            else:
                p_inside = target_positions[i] if i < len(target_positions) else cube.position

            # Fully submerged and retracted: lock to palm inside and rest
            if cube.current_scale <= 0.001:
                cube.position[:] = p_inside
                cube.velocity[:] = 0.0
                continue

            if should_spawn:
                # Dynamic emergence trajectory: smoothly transitions from inside palm to hover slot
                hover_slot = target_positions[i] if i < len(target_positions) else cube.position
                ease_factor = float(np.clip(cube.current_scale, 0.0, 1.12))
                # Clamp overshoot for target so pop is flowy, not wild
                ease_target = min(ease_factor, 1.08)
                target = p_inside + ease_target * (hover_slot - p_inside)

                # Flowy blossom fountain arc (active during emergence transition)
                if cube.emergence < 0.98:
                    u_up = float(np.clip(-norm_3d[1], 0.0, 1.0))
                    arc_envelope = math.sin(float(np.clip(cube.emergence, 0.0, 1.0)) * math.pi)
                    fountain_lift = getattr(config, "FOUNTAIN_ARC_HEIGHT", 42.0) * arc_envelope * (1.0 + 0.35 * u_up)
                    # Radial petal spread along lateral axis
                    spread_sign = -1.0 if i == 0 else (1.0 if i == (len(self.cubes) - 1) else 0.0)
                    fountain_spread = getattr(config, "FOUNTAIN_OUTWARD_SPREAD", 26.0) * arc_envelope * spread_sign
                    lat_axis = palm_right if palm_right is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    target = target + fountain_lift * norm_3d + fountain_spread * lat_axis
            else:
                # -----------------------------------------------------------------
                # VORTEX SUCTION FUNNEL: Inward Gathering + Accelerating Spiral Plunge
                # -----------------------------------------------------------------
                u = float(1.0 - np.clip(cube.emergence, 0.0, 1.0))  # Retraction progress [0.0 -> 1.0]
                sink_center = c_palm if c_palm is not None else (palm_center if palm_center is not None else p_inside)

                # Reference coordinate bases in palm plane
                lat_axis = palm_right if palm_right is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                fwd_axis = np.cross(norm_3d, lat_axis)
                fwd_len = float(np.linalg.norm(fwd_axis))
                fwd_axis = (fwd_axis / fwd_len) if fwd_len > 1e-4 else np.array([0.0, -1.0, 0.0], dtype=np.float32)

                # 1. Centripetal Radial Inflow: Horizontal spread contracts sharply inward
                r_inflow = float(1.0 - pow(u, 0.82))
                num_c = max(len(self.cubes), 1)
                slot_radial_dist = float(abs(i - (num_c - 1) * 0.5) * config.CUBE_HORIZONTAL_SPACING + 14.0)

                # 2. Dynamic Tangential Swirl (Conservation of Angular Momentum)
                base_phi = float((i - (num_c - 1) * 0.5) * (math.pi / 2.2))
                swirl_delta = float(cube.swirl_dir * 4.2 * pow(u, 1.5))
                phi = base_phi + swirl_delta

                vortex_x = float(math.cos(phi) * slot_radial_dist * r_inflow)
                vortex_y = float(math.sin(phi) * slot_radial_dist * r_inflow)
                r_plane = vortex_x * lat_axis + vortex_y * fwd_axis

                # 3. Steepening Vertical Plunge: Stays elevated while gathering, then dives
                h_hover = float(config.CUBE_ELEVATION_OFFSET * 0.72)
                h_lift = float(h_hover * pow(max(0.0, math.cos(u * 0.5 * math.pi)), 1.75))
                depth_sink = float(config.PALM_SUBMERGE_DEPTH * (u ** 2))

                target = sink_center + r_plane + (h_lift - depth_sink) * norm_3d

            # Hooke's spring attraction + velocity damping
            # Soften spring as vacuum takes over so collapse feels sucked, not yanked
            u_retract = float(0.0 if should_spawn else (1.0 - np.clip(cube.emergence, 0.0, 1.0)))
            suck_flow = float(pow(u_retract, 0.70))
            spring_gain = 1.0 - 0.35 * suck_flow

            # Recoil softening: when hit by another cube or force blast, loosen spring for billiard bounce
            if cube.recoil_timer > 0.0:
                r_dur = getattr(config, "CUBE_RECOIL_DURATION", 0.45)
                r_ratio = float(np.clip(cube.recoil_timer / r_dur, 0.0, 1.0))
                spring_gain *= max(0.12, 1.0 - r_ratio * 0.88)

            displacement = cube.position - target
            spring_force = -self.spring_k * spring_gain * displacement
            damping_gain = 0.35 if cube.recoil_timer > 0.0 else 1.0
            damping_force = -self.damping_c * damping_gain * cube.velocity

            # Quadratic aerodynamic air drag
            speed = float(np.linalg.norm(cube.velocity))
            air_drag = -config.AIR_DRAG_QUADRATIC * speed * cube.velocity

            # Palm cushion repulsion force (active ONLY when hovering to prevent palm clipping, DISABLED during suction)
            cushion_force = np.zeros(3, dtype=np.float32)
            cushion_ref = c_palm if c_palm is not None else palm_center
            if should_spawn and cushion_ref is not None and cube.emergence > 0.65:
                u_up = float(np.clip(-norm_3d[1], 0.0, 1.0))
                to_cube = cube.position - cushion_ref
                dist_along_normal = float(np.dot(to_cube, norm_3d))
                cushion_thresh = config.PALM_CUSHION_DISTANCE * cube.current_scale * (1.0 + 0.30 * u_up)
                if dist_along_normal < cushion_thresh:
                    penetration = cushion_thresh - dist_along_normal
                    cushion_weight = min(1.0, (cube.emergence - 0.65) / 0.35)
                    cushion_mag = config.PALM_CUSHION_STIFFNESS * ((penetration / cushion_thresh) ** 2) * cushion_weight
                    cushion_force = norm_3d * cushion_mag

            # Palm-center vacuum vortex: straight pull + tangential swirl = flowy spiral suck
            suck_force = np.zeros(3, dtype=np.float32)
            sink_palm = c_palm if c_palm is not None else palm_center
            if (not should_spawn) and sink_palm is not None and suck_flow > 1e-4:
                to_palm = sink_palm - cube.position
                dist_palm = float(np.linalg.norm(to_palm))
                if dist_palm > 1e-3:
                    pull_dir = (to_palm / dist_palm).astype(np.float32)
                    pull_strength = getattr(config, "SUCK_PULL_STRENGTH", 4800.0)
                    # Fade pull very close to center so cubes land softly inside palm
                    center_fade = float(np.clip(dist_palm / 80.0, 0.35, 1.0))
                    suck_force += pull_dir * (pull_strength * suck_flow * center_fade * cube.mass * 0.065)
                    # Tangential swirl around palm normal for spiral motion
                    tangent = np.cross(to_palm, norm_3d)
                    tan_len = float(np.linalg.norm(tangent))
                    if tan_len > 1e-3:
                        tangent = (tangent / tan_len).astype(np.float32)
                    else:
                        # Degenerate (directly above center): orbit in palm plane
                        tangent = np.cross(norm_3d, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                        tl = float(np.linalg.norm(tangent))
                        tangent = (tangent / tl).astype(np.float32) if tl > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    swirl_strength = getattr(config, "SUCK_SWIRL_STRENGTH", 2400.0)
                    dist_gain = float(np.clip(dist_palm / 120.0, 0.30, 1.0))
                    suck_force += tangent * (cube.swirl_dir * swirl_strength * suck_flow * dist_gain * cube.mass * 0.065)

            # Total force and linear acceleration
            net_force = spring_force + damping_force + air_drag + cushion_force + suck_force + (inertial_accel * cube.mass)
            accel = net_force / cube.mass
            cube.velocity += accel * dt
            # Clamp vacuum speed so it stays smooth, never teleporty
            suck_max = float(getattr(config, "SUCK_MAX_SPEED", 1600.0))
            spd = float(np.linalg.norm(cube.velocity))
            if (not should_spawn) and spd > suck_max:
                cube.velocity *= (suck_max / spd)
            cube.position += cube.velocity * dt

            # Dynamic Angling & Gyroscopic Torques
            # A. Aerodynamic banking torque: lean into velocity perpendicular to palm normal
            vel_horiz = cube.velocity - np.dot(cube.velocity, norm_3d) * norm_3d
            bank_torque = np.cross(vel_horiz, norm_3d) * config.AERODYNAMIC_BANKING_GAIN

            # B. Palm tilt alignment torque: align cube's normal with palm normal
            cube_up = cube.rotation_matrix[:, 1]
            tilt_axis = np.cross(cube_up, norm_3d)
            align_torque = tilt_axis * (config.GYROSCOPIC_RESTORE_K * config.PALM_TILT_WEIGHT)

            # C. Ambient spin restoring torque with angular damping
            # Relax leveling during vacuum so cubes tumble flowly into the palm
            level_gain = 1.0 - 0.7 * suck_flow
            align_torque = align_torque * level_gain
            spin_err = cube.ambient_spin - cube.angular_velocity
            damping_torque = spin_err * config.GYROSCOPIC_DAMPING

            net_torque = bank_torque + align_torque + damping_torque
            # Vortex spin-up: twirl around palm normal while being sucked in
            if suck_flow > 1e-4:
                net_torque = net_torque + norm_3d * (cube.swirl_dir * 48.0 * suck_flow)
                # Singularity pitch: tilt cube forward into the palm sink
                to_sink = (sink_palm - cube.position) if sink_palm is not None else -norm_3d
                dist_s = float(np.linalg.norm(to_sink))
                if dist_s > 1e-2:
                    p_dir = (to_sink / dist_s).astype(np.float32)
                    pitch_axis = np.cross(cube.rotation_matrix[:, 1], p_dir)
                    net_torque = net_torque + pitch_axis * (36.0 * suck_flow)

            # Cyclonic vorticity torque when palm-wave tornado is active
            if tornado_intensity > 0.02:
                net_torque = net_torque + norm_3d * (tornado_intensity * 36.0)

            angular_accel = net_torque * cube.inv_inertia
            cube.angular_velocity += angular_accel * dt
            cube.angular_velocity *= (config.ANGULAR_DRAG ** (dt * 60.0))

            # Integrate 3D orientation
            cube.integrate_rotation(dt)

        # 2. Pairwise Rigid-Body Collision Resolution with Contact Point & Torque
        num_cubes = len(self.cubes)
        restitution_coeff = getattr(config, "CUBE_COLLISION_RESTITUTION", 0.88)
        recoil_dur = getattr(config, "CUBE_RECOIL_DURATION", 0.45)
        now_col = time.time()
        self.recent_cube_collisions = [c for c in self.recent_cube_collisions if (now_col - c.timestamp) < 0.35]

        for i in range(num_cubes):
            for j in range(i + 1, num_cubes):
                c1 = self.cubes[i]
                c2 = self.cubes[j]

                if c1.current_scale <= 0.001 or c2.current_scale <= 0.001:
                    continue

                diff = c1.position - c2.position
                dist = float(np.linalg.norm(diff))
                effective_r1 = c1.radius * c1.current_scale
                effective_r2 = c2.radius * c2.current_scale
                min_dist = effective_r1 + effective_r2

                if dist < min_dist:
                    if dist < 1e-4:
                        norm = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        overlap = min_dist
                    else:
                        norm = (diff / dist).astype(np.float32)
                        overlap = min_dist - dist

                    # Positional separation: respect grabbed vs free rigid bodies
                    if not c1.is_grabbed and not c2.is_grabbed:
                        c1.position += norm * (overlap * 0.52)
                        c2.position -= norm * (overlap * 0.52)
                    elif c1.is_grabbed and not c2.is_grabbed:
                        c2.position -= norm * (overlap * 0.95)
                    elif not c1.is_grabbed and c2.is_grabbed:
                        c1.position += norm * (overlap * 0.95)

                    contact_pt = c1.position - norm * (effective_r1 - overlap * 0.5)
                    r1 = contact_pt - c1.position
                    r2 = contact_pt - c2.position

                    # Velocity at contact point including rotational spin (v + w x r)
                    u1 = c1.velocity + np.cross(c1.angular_velocity, r1)
                    u2 = c2.velocity + np.cross(c2.angular_velocity, r2)
                    rel_vel = u1 - u2
                    vel_along_norm = float(np.dot(rel_vel, norm))

                    # Loosen spring & register contact event
                    c1.recoil_timer = recoil_dur
                    c2.recoil_timer = recoil_dur
                    c1.last_impact_pos = contact_pt.copy()
                    c2.last_impact_pos = contact_pt.copy()

                    if vel_along_norm < 0.0 or overlap > 1.0:
                        inv_m1 = 0.0 if c1.is_grabbed else (1.0 / c1.mass)
                        inv_m2 = 0.0 if c2.is_grabbed else (1.0 / c2.mass)
                        inv_m = inv_m1 + inv_m2

                        if inv_m > 1e-5:
                            ang_term1 = np.cross(c1.inv_inertia * np.cross(r1, norm), r1) if not c1.is_grabbed else np.zeros(3, dtype=np.float32)
                            ang_term2 = np.cross(c2.inv_inertia * np.cross(r2, norm), r2) if not c2.is_grabbed else np.zeros(3, dtype=np.float32)
                            inv_eff_mass = inv_m + float(np.dot(norm, ang_term1 + ang_term2))

                            min_impulse = 35.0 if overlap > 1.0 else 0.0
                            impulse_n_mag = max(-(1.0 + restitution_coeff) * vel_along_norm / max(inv_eff_mass, 1e-5), min_impulse)
                            impulse_n = norm * impulse_n_mag

                            # Tangential friction impulse
                            v_tangent = rel_vel - vel_along_norm * norm
                            tangent_speed = float(np.linalg.norm(v_tangent))
                            if tangent_speed > 1e-4:
                                tangent_dir = v_tangent / tangent_speed
                                friction_mag = min(self.friction * impulse_n_mag, tangent_speed / inv_m)
                                impulse_t = -tangent_dir * friction_mag
                            else:
                                impulse_t = np.zeros(3, dtype=np.float32)

                            total_impulse = impulse_n + impulse_t

                            if not c1.is_grabbed:
                                c1.velocity += total_impulse * inv_m1
                                c1.angular_velocity += c1.inv_inertia * np.cross(r1, total_impulse)
                            if not c2.is_grabbed:
                                c2.velocity -= total_impulse * inv_m2
                                c2.angular_velocity -= c2.inv_inertia * np.cross(r2, total_impulse)

                            impact_intensity = min(1.0, impulse_n_mag / 150.0)
                            c1.impact_energy = max(c1.impact_energy, impact_intensity)
                            c2.impact_energy = max(c2.impact_energy, impact_intensity)

                            self.recent_cube_collisions.append(
                                CubeCollisionEvent(
                                    cube_a=c1.index,
                                    cube_b=c2.index,
                                    contact_pos=contact_pt.copy(),
                                    impulse_mag=impulse_n_mag,
                                    timestamp=now_col,
                                )
                            )

        # 3. Dynamic Finger-to-Cube 6-DOF Physical Collisions (Pokes, Flicks, Deflections)
        now_t = time.time()
        self.recent_contacts = [c for c in self.recent_contacts if (now_t - c.timestamp) < 0.5]

        if finger_colliders:
            for cube in self.cubes:
                if cube.current_scale <= 0.05:
                    continue

                effective_radius = cube.radius * cube.current_scale

                for finger in finger_colliders:
                    # Interact with extended fingers (or thumb)
                    if not finger.is_extended and finger.name != "Thumb":
                        continue

                    diff = cube.position - finger.pos_3d
                    dist = float(np.linalg.norm(diff))
                    min_dist = effective_radius + finger.radius

                    if dist < min_dist:
                        if dist < 1e-4:
                            norm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                            overlap = min_dist
                        else:
                            norm = (diff / dist).astype(np.float32)
                            overlap = min_dist - dist

                        # Positional separation: immediately push cube away from penetrating finger
                        cube.position += norm * (overlap * 0.88)

                        # Contact point on surface of cube
                        contact_pt = cube.position - norm * effective_radius
                        r_arm = contact_pt - cube.position

                        # Velocity at contact point including rotational spin
                        u_cube = cube.velocity + np.cross(cube.angular_velocity, r_arm)
                        v_rel = u_cube - finger.vel_3d
                        v_norm = float(np.dot(v_rel, norm))

                        # Apply impulse if moving into finger or sustained overlap push
                        if v_norm < 0.0 or overlap > 1.5:
                            e = config.FINGER_RESTITUTION
                            # Base restitution impulse
                            base_impulse = max(-(1.0 + e) * v_norm, 18.0) * cube.mass

                            # Flick speed boost: if finger is moving fast toward the cube, transfer momentum!
                            finger_speed_into_cube = float(np.dot(finger.vel_3d, norm))
                            if finger_speed_into_cube > config.FINGER_FLICK_MIN_SPEED:
                                flick_boost = finger_speed_into_cube * config.FINGER_FLICK_IMPULSE_SCALE * cube.mass
                                base_impulse += flick_boost

                            impulse_n = norm * base_impulse

                            # Tangential swipe friction: finger brushing along surface gives spin
                            v_tangent = v_rel - v_norm * norm
                            tan_speed = float(np.linalg.norm(v_tangent))
                            if tan_speed > 1e-3:
                                tan_dir = v_tangent / tan_speed
                                friction_mag = min(config.COLLISION_FRICTION * base_impulse, tan_speed * cube.mass)
                                impulse_t = -tan_dir * friction_mag
                            else:
                                impulse_t = np.zeros(3, dtype=np.float32)

                            total_impulse = impulse_n + impulse_t

                            # Linear velocity recoil
                            cube.velocity += total_impulse / cube.mass

                            # Off-center impact imparts angular spin (Torque = r x J)
                            torque = np.cross(r_arm, total_impulse) * config.FINGER_TORQUE_FACTOR
                            cube.angular_velocity += cube.inv_inertia * torque

                            # Loosen hover spring temporarily for tactile flick recoil and trigger impact energy
                            cube.recoil_timer = getattr(config, "CUBE_RECOIL_DURATION", 0.45)
                            cube.impact_energy = min(1.0, cube.impact_energy + float(np.linalg.norm(total_impulse)) / 160.0)

                            # Record physical contact event
                            self.recent_contacts.append(
                                ContactEvent(
                                    finger_name=finger.name,
                                    cube_index=cube.index,
                                    contact_pos=contact_pt.copy(),
                                    contact_normal=norm.copy(),
                                    impulse_mag=float(np.linalg.norm(total_impulse)),
                                    timestamp=now_t,
                                )
                            )
