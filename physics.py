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

        # Physical geometry & Moment of Inertia for a solid cube: I = 1/6 * m * (2*r)^2
        self.radius: float = config.CUBE_SIZE * 0.95
        side_len = 2.0 * self.radius
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
    ) -> None:
        """Fast flowy spawn (ease-out-back pop) and vacuum suck collapse (ease-in dive)."""
        # Handle manual test overrides where scale was set directly
        if self.current_scale > 0.0 and self.emergence == 0.0 and should_spawn:
            self.emergence = self.current_scale

        spawn_speed = getattr(config, "SPAWN_SPEED", config.EMERGENCE_SPEED)
        collapse_speed = getattr(config, "COLLAPSE_SPEED", config.EMERGENCE_SPEED)

        if should_spawn:
            self.target_scale = 1.0
            self.retract_timer = 0.0
            self.stagger_timer += dt
            if self.stagger_timer >= stagger_delay:
                self.emergence = min(1.0, self.emergence + spawn_speed * dt)
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
            # Clamp overshoot so physics radius stays sane
            self.current_scale = float(np.clip(back, 0.0, 1.12))
            if t >= 1.0:
                self.current_scale = 1.0
        else:
            # Ease-in vacuum dive: linger while spiraling, then accelerate into palm
            self.current_scale = float(pow(t, 1.45))
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

    def step(
        self,
        target_positions: List[np.ndarray],
        should_spawn: bool,
        dt: float,
        palm_normal: Optional[np.ndarray] = None,
        palm_center: Optional[np.ndarray] = None,
        hand_velocity: Optional[np.ndarray] = None,
        finger_colliders: Optional[List[Any]] = None,
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
        """
        dt = float(np.clip(dt, 0.001, 0.05))

        if palm_normal is None:
            norm_3d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            norm_3d = palm_normal.astype(np.float32)

        # Hand inertial lag
        inertial_accel = np.zeros(3, dtype=np.float32)
        if hand_velocity is not None:
            delta_v = hand_velocity - self.palm_velocity
            inertial_accel = -(delta_v / dt) * config.HAND_INERTIAL_LAG_FACTOR
            self.palm_velocity = hand_velocity.copy()

        # 1. Integrate Scale, Staggered Palm Emergence & Spring-Damper Dynamics
        for i, cube in enumerate(self.cubes):
            # Funnel stagger: spawn pops center-first, collapse sucks outers-first
            if should_spawn:
                delay = 0.0 if i == 1 else getattr(config, "SPAWN_STAGGER", config.EMERGENCE_STAGGER)
            else:
                delay = 0.0 if i != 1 else getattr(config, "COLLAPSE_STAGGER", config.EMERGENCE_STAGGER)
            cube.update_scale(should_spawn, dt, stagger_delay=delay)

            # Submerged palm origin point (inside the hand)
            if palm_center is not None:
                p_inside = palm_center - config.PALM_SUBMERGE_DEPTH * norm_3d
            else:
                p_inside = target_positions[i] if i < len(target_positions) else cube.position

            # Fully submerged and retracted: lock to palm inside and rest
            if cube.current_scale <= 0.001:
                cube.position[:] = p_inside
                cube.velocity[:] = 0.0
                continue

            # Dynamic emergence trajectory: smoothly transitions from inside palm to hover slot
            hover_slot = target_positions[i] if i < len(target_positions) else cube.position
            ease_factor = float(np.clip(cube.current_scale, 0.0, 1.12))
            # Clamp overshoot for target so pop is flowy, not wild
            ease_target = min(ease_factor, 1.08)
            target = p_inside + ease_target * (hover_slot - p_inside)

            # Hooke's spring attraction + velocity damping
            # Soften spring as vacuum takes over so collapse feels sucked, not yanked
            suck_t = float(0.0 if should_spawn else (1.0 - np.clip(cube.emergence, 0.0, 1.0)))
            suck_flow = suck_t * suck_t
            spring_gain = 1.0 - 0.55 * suck_flow
            displacement = cube.position - target
            spring_force = -self.spring_k * spring_gain * displacement
            damping_force = -self.damping_c * cube.velocity

            # Quadratic aerodynamic air drag
            speed = float(np.linalg.norm(cube.velocity))
            air_drag = -config.AIR_DRAG_QUADRATIC * speed * cube.velocity

            # Palm cushion repulsion force (active when hovering to prevent palm clipping)
            cushion_force = np.zeros(3, dtype=np.float32)
            if palm_center is not None and cube.emergence > 0.65:
                to_cube = cube.position - palm_center
                dist_along_normal = float(np.dot(to_cube, norm_3d))
                cushion_thresh = config.PALM_CUSHION_DISTANCE * cube.current_scale
                if dist_along_normal < cushion_thresh:
                    penetration = cushion_thresh - dist_along_normal
                    cushion_weight = min(1.0, (cube.emergence - 0.65) / 0.35)
                    cushion_mag = config.PALM_CUSHION_STIFFNESS * ((penetration / cushion_thresh) ** 2) * cushion_weight
                    cushion_force = norm_3d * cushion_mag

            # Palm-center vacuum vortex: straight pull + tangential swirl = flowy spiral suck
            suck_force = np.zeros(3, dtype=np.float32)
            if (not should_spawn) and palm_center is not None and suck_flow > 1e-4:
                to_palm = palm_center - cube.position
                dist_palm = float(np.linalg.norm(to_palm))
                if dist_palm > 1e-3:
                    pull_dir = (to_palm / dist_palm).astype(np.float32)
                    pull_strength = getattr(config, "SUCK_PULL_STRENGTH", 2600.0)
                    # Fade pull very close to center so cubes land softly inside palm
                    center_fade = float(np.clip(dist_palm / 90.0, 0.25, 1.0))
                    suck_force += pull_dir * (pull_strength * suck_flow * center_fade * cube.mass * 0.06)
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
                    swirl_strength = getattr(config, "SUCK_SWIRL_STRENGTH", 900.0)
                    dist_gain = float(np.clip(dist_palm / 130.0, 0.25, 1.0))
                    suck_force += tangent * (cube.swirl_dir * swirl_strength * suck_flow * dist_gain * cube.mass * 0.06)

            # Total force and linear acceleration
            net_force = spring_force + damping_force + air_drag + cushion_force + suck_force + (inertial_accel * cube.mass)
            accel = net_force / cube.mass
            cube.velocity += accel * dt
            # Clamp vacuum speed so it stays smooth, never teleporty
            suck_max = float(getattr(config, "SUCK_MAX_SPEED", 1400.0))
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
                net_torque = net_torque + norm_3d * (cube.swirl_dir * 22.0 * suck_flow)
            angular_accel = net_torque * cube.inv_inertia
            cube.angular_velocity += angular_accel * dt
            cube.angular_velocity *= (config.ANGULAR_DRAG ** (dt * 60.0))

            # Integrate 3D orientation
            cube.integrate_rotation(dt)

        # 2. Pairwise Rigid-Body Collision Resolution with Contact Point & Torque
        num_cubes = len(self.cubes)
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

                    # Positional separation
                    separation = norm * (overlap * 0.52)
                    c1.position += separation
                    c2.position -= separation

                    # Contact point between the two bounding spheres
                    contact_pt = c1.position - norm * (effective_r1 - overlap * 0.5)
                    r1 = contact_pt - c1.position
                    r2 = contact_pt - c2.position

                    # Velocity at contact point including rotational spin (v + w x r)
                    u1 = c1.velocity + np.cross(c1.angular_velocity, r1)
                    u2 = c2.velocity + np.cross(c2.angular_velocity, r2)
                    rel_vel = u1 - u2
                    vel_along_norm = float(np.dot(rel_vel, norm))

                    if vel_along_norm < 0.0:
                        # Effective mass along normal accounting for rotational inertia
                        inv_m = (1.0 / c1.mass) + (1.0 / c2.mass)
                        ang_term1 = np.cross(c1.inv_inertia * np.cross(r1, norm), r1)
                        ang_term2 = np.cross(c2.inv_inertia * np.cross(r2, norm), r2)
                        inv_eff_mass = inv_m + float(np.dot(norm, ang_term1 + ang_term2))

                        # Normal impulse
                        impulse_n_mag = -(1.0 + self.restitution) * vel_along_norm / max(inv_eff_mass, 1e-5)
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

                        # Apply linear impulse
                        c1.velocity += total_impulse / c1.mass
                        c2.velocity -= total_impulse / c2.mass

                        # Apply angular impulse (Torque = r x J)
                        c1.angular_velocity += c1.inv_inertia * np.cross(r1, total_impulse)
                        c2.angular_velocity -= c2.inv_inertia * np.cross(r2, total_impulse)

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
