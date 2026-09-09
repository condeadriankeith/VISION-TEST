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
from typing import List, Optional, Tuple

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

    def update(
        self,
        hand_detected: bool,
        is_open: bool,
        pose: Optional[HandPose] = None,
        dt: Optional[float] = None,
    ) -> None:
        """Updates 6-DOF physics, multi-harmonic organic floating, and dynamic angling.

        Args:
            hand_detected: Whether any hand is present.
            is_open: Whether palm gesture is OPEN.
            pose: Active HandPose data with 3D orientation.
            dt: Optional explicit delta time (seconds).
        """
        now = time.time()
        if dt is None:
            dt = now - self.last_update_time
        self.last_update_time = now

        should_spawn = hand_detected and is_open

        target_positions: List[np.ndarray] = []
        palm_norm = None
        palm_center = None
        hand_velocity = None

        if hand_detected and pose is not None:
            # Extract full 3D spatial basis from pose
            norm_3d = np.array(pose.palm_normal_3d, dtype=np.float32)
            up_3d = np.array(pose.palm_up_3d, dtype=np.float32)
            right_3d = np.array(pose.palm_right_3d, dtype=np.float32)
            center_3d = np.array(pose.palm_center_3d, dtype=np.float32)

            palm_norm = norm_3d
            palm_center = center_3d

            # Estimate hand velocity for inertial reaction
            if self.prev_palm_3d is not None and dt > 1e-4:
                hand_velocity = (center_3d - self.prev_palm_3d) / dt
            self.prev_palm_3d = center_3d.copy()

            # Distance scale factor
            dist_factor = float(np.clip(pose.palm_scale / 70.0, 0.6, 2.2))
            base_elevation = config.CUBE_ELEVATION_OFFSET * dist_factor
            base_spacing = config.CUBE_HORIZONTAL_SPACING * dist_factor

            # Procedural Multi-Harmonic Organic Floating Motion
            for i in range(config.CUBE_COUNT):
                phase = (2.0 * math.pi / config.CUBE_COUNT) * i

                # 1. Primary harmonic breathing heave
                heave = (
                    math.sin(now * config.BOB_PRIMARY_FREQ + phase)
                    * config.BOB_PRIMARY_AMP
                    * dist_factor
                )
                # 2. Subtle micro-buoyancy flutter
                flutter = (
                    math.sin(now * config.BOB_FLUTTER_FREQ + phase * 2.3)
                    * config.BOB_FLUTTER_AMP
                    * dist_factor
                )
                # 3. Horizontal Lissajous sway
                sway_x = (
                    math.cos(now * config.BOB_SWAY_FREQ_X + phase * 1.4)
                    * config.BOB_SWAY_AMP_X
                    * dist_factor
                )
                # 4. Depth wander
                wander_z = (
                    math.sin(now * config.BOB_SWAY_FREQ_Z + phase * 0.8)
                    * config.BOB_SWAY_AMP_Z
                    * dist_factor
                )
                # 5. Organic formation breathing expansion/contraction
                expansion = 1.0 + config.BREATHING_EXPANSION_RATIO * math.sin(now * config.BOB_PRIMARY_FREQ)
                slot_offset = ((i - 1) * base_spacing * expansion) + sway_x

                # Natural hover elevation: lifted along the palm's up vector (above the palm/fingers)
                # and cushioned outward along the palm normal (in front of the palm surface)
                elevation_up = base_elevation * 0.88 + heave + flutter
                elevation_out = base_elevation * 0.38 + wander_z

                # Target constructed directly in tilted 3D palm frame
                target = (
                    center_3d
                    + slot_offset * right_3d
                    + elevation_up * up_3d
                    + elevation_out * norm_3d
                )
                target_positions.append(target)
        else:
            self.prev_palm_3d = None
            # Idle formation when hand is absent
            for i in range(config.CUBE_COUNT):
                target_positions.append(
                    np.array([640.0 + (i - 1) * config.CUBE_HORIZONTAL_SPACING, 360.0, 0.0], dtype=np.float32)
                )

        # Step 6-DOF physics world
        self.physics_world.step(
            target_positions=target_positions,
            should_spawn=should_spawn,
            dt=dt,
            palm_normal=palm_norm,
            palm_center=palm_center,
            hand_velocity=hand_velocity,
            finger_colliders=pose.fingers if pose is not None else None,
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

    def _render_palm_portal(
        self,
        frame: np.ndarray,
        pose: HandPose,
        material: config.GreyMaterial,
        active_cubes: List[PhysicsCube],
    ) -> None:
        """Renders a sleek holographic aperture ring on the palm when cubes emerge or retract."""
        if not config.PORTAL_RING_ENABLED or not active_cubes:
            return

        # Check if any cube is actively in the emergence/retraction transition
        transition_cubes = [c for c in active_cubes if 0.01 < c.emergence < 0.98]
        if not transition_cubes:
            return

        # Average transition progress for aperture expansion/contraction
        avg_emergence = sum(c.emergence for c in transition_cubes) / len(transition_cubes)
        # Sine envelope: opens wide mid-transition, closes smoothly at 0.0 and 1.0
        portal_envelope = math.sin(avg_emergence * math.pi)
        if portal_envelope <= 0.02:
            return

        h, w = frame.shape[:2]
        cx_screen = w * 0.5
        cy_screen = h * 0.5
        focal_len = config.FOCAL_LENGTH
        z_baseline = config.CAMERA_BASELINE_DEPTH

        palm_center = pose.palm_center_3d
        cam_x = palm_center[0] - cx_screen
        cam_y = palm_center[1] - cy_screen
        cam_z = z_baseline + palm_center[2]

        px = int(round(cx_screen + focal_len * (cam_x / max(cam_z, 50.0))))
        py = int(round(cy_screen + focal_len * (cam_y / max(cam_z, 50.0))))

        if not (0 <= px < w and 0 <= py < h):
            return

        dist_factor = float(np.clip(pose.palm_scale / 70.0, 0.6, 2.2))
        portal_radius = int(config.PORTAL_MAX_RADIUS * dist_factor * portal_envelope)
        if portal_radius < 4:
            return

        b_acc, g_acc, r_acc = material.hud_accent
        ring_color = (
            int(b_acc * portal_envelope),
            int(g_acc * portal_envelope),
            int(r_acc * portal_envelope),
        )

        # Concentric aperture rings on palm surface
        cv2.circle(frame, (px, py), portal_radius, ring_color, 2, lineType=cv2.LINE_AA)
        if portal_radius > 10:
            inner_radius = int(portal_radius * 0.58)
            inner_color = (
                int(b_acc * portal_envelope * 0.7),
                int(g_acc * portal_envelope * 0.7),
                int(r_acc * portal_envelope * 0.7),
            )
            cv2.circle(frame, (px, py), inner_radius, inner_color, 1, lineType=cv2.LINE_AA)
            # Center emitter core dot
            cv2.circle(frame, (px, py), 3, ring_color, -1, lineType=cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        pose: HandPose,
        material: config.GreyMaterial,
    ) -> None:
        """Projects and renders true 3D solid cubes with palm emergence aperture and PBR shading.

        Args:
            frame: Target BGR video frame to draw on.
            pose: Tracked HandPose from HandTracker.
            material: Active GreyMaterial configuration.
        """
        active_cubes = [c for c in self.physics_world.cubes if c.current_scale > 0.001]
        if not active_cubes:
            return

        # 1. Update real-time environmental lighting from the live frame
        self._update_environmental_lighting(frame)

        # 2. Palm portal aperture ring (disabled to eliminate 2D ring overlays)
        if config.PORTAL_RING_ENABLED:
            self._render_palm_portal(frame, pose, material, active_cubes)

        h, w = frame.shape[:2]
        dist_factor = float(np.clip(pose.palm_scale / 70.0, 0.6, 2.2))
        base_size = config.CUBE_SIZE * dist_factor

        focal_len = config.FOCAL_LENGTH
        cx_screen = w * 0.5
        cy_screen = h * 0.5
        z_baseline = config.CAMERA_BASELINE_DEPTH

        all_faces: List[RenderFace] = []

        # 3. Transform, Light, and Project each 3D Cube
        for cube_idx, cube in enumerate(active_cubes):
            effective_radius = base_size * cube.current_scale
            scaled_verts = self.UNIT_VERTICES * effective_radius
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

                # Bevel highlight already derived from the face hue
                b_bev_lit, g_bev_lit, r_bev_lit = b_bev, g_bev, r_bev

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

        # 5. Render Solid Opaque Faces and Micro-Bevels
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

        # 6. Render Procedural Interaction FX (auras and ripples disabled for clean 3D scene)
        if config.FINGER_PROXIMITY_AURA:
            self._render_fingertip_auras(frame, pose, material, active_cubes)
        if getattr(config, "RIPPLE_ENABLED", False):
            self._render_contact_ripples(frame, material)

    @property
    def latest_contact(self) -> Optional[ContactEvent]:
        """Returns the most recent physical contact event within the last 1.2s, if any."""
        now_t = time.time()
        for contact in reversed(self.physics_world.recent_contacts):
            if (now_t - contact.timestamp) < 1.2:
                return contact
        return None

    def _render_fingertip_auras(
        self,
        frame: np.ndarray,
        pose: HandPose,
        material: config.GreyMaterial,
        active_cubes: List[PhysicsCube],
    ) -> None:
        """Renders subtle, interactive proximity auras at extended fingertips."""
        if not config.FINGER_PROXIMITY_AURA or not pose.fingers or not active_cubes:
            return

        for finger in pose.fingers:
            if not finger.is_extended and finger.name != "Thumb":
                continue

            # Check distance to nearest active cube
            min_dist = min(float(np.linalg.norm(finger.pos_3d - c.position)) for c in active_cubes)
            proximity = float(np.clip(1.0 - (min_dist / 160.0), 0.0, 1.0))

            fx = int(round(finger.pos_3d[0]))
            fy = int(round(finger.pos_3d[1]))

            # Subtle fingertip center dot
            cv2.circle(frame, (fx, fy), 3, material.hud_accent, -1, lineType=cv2.LINE_AA)

            # Proximity halo ring: expands and brightens when approaching a cube
            halo_r = int(10 + 6 * proximity)
            halo_alpha = 0.35 + 0.65 * proximity
            b, g, r = material.hud_accent
            halo_color = (
                min(255, int(b * halo_alpha)),
                min(255, int(g * halo_alpha)),
                min(255, int(r * halo_alpha)),
            )
            cv2.circle(frame, (fx, fy), halo_r, halo_color, 1, lineType=cv2.LINE_AA)

    def _render_contact_ripples(
        self,
        frame: np.ndarray,
        material: config.GreyMaterial,
    ) -> None:
        """Renders procedural expanding shockwave ripples from finger-cube contact points."""
        if not getattr(config, "RIPPLE_ENABLED", False) or not self.physics_world.recent_contacts:
            return

        now_t = time.time()
        h, w = frame.shape[:2]
        cx_screen = w * 0.5
        cy_screen = h * 0.5
        focal_len = config.FOCAL_LENGTH
        z_baseline = config.CAMERA_BASELINE_DEPTH

        for contact in self.physics_world.recent_contacts:
            age = now_t - contact.timestamp
            if age > config.RIPPLE_LIFETIME:
                continue

            life_ratio = age / config.RIPPLE_LIFETIME
            ripple_radius = int(8.0 + age * config.RIPPLE_EXPANSION_SPEED)
            alpha = max(0.0, 1.0 - life_ratio)

            # Project 3D contact point to 2D screen coordinates
            cam_x = contact.contact_pos[0] - cx_screen
            cam_y = contact.contact_pos[1] - cy_screen
            cam_z = z_baseline + contact.contact_pos[2]

            px = int(round(cx_screen + focal_len * (cam_x / max(cam_z, 50.0))))
            py = int(round(cy_screen + focal_len * (cam_y / max(cam_z, 50.0))))

            if 0 <= px < w and 0 <= py < h:
                # Tinted shockwave ring
                b, g, r = material.bevel_color
                ring_color = (int(b * alpha), int(g * alpha), int(r * alpha))
                cv2.circle(frame, (px, py), ripple_radius, ring_color, 2, lineType=cv2.LINE_AA)
                if ripple_radius > 8:
                    inner_color = (int(b * alpha * 0.6), int(g * alpha * 0.6), int(r * alpha * 0.6))
                    cv2.circle(frame, (px, py), ripple_radius - 6, inner_color, 1, lineType=cv2.LINE_AA)
