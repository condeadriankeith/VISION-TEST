"""Top-tier spatial mathematics, kinematics, and procedural physics utilities.

Provides:
- OneEuroFilter: 1€ adaptive motion filter (Casiez et al., CHI 2012).
- CurlNoise3D: Analytically exact divergence-free 3D vector field (curl of vector potential).
- OBB3D / OBBSATCollider: 3D Oriented Bounding Box with 15-axis Separating Axis Theorem (SAT).
- BoneCapsuleCollider: Swept-sphere capsule collider for biomechanical hand skeleton segments.
- ThinFilmInterference: Optical wave interference model for physical iridescence.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. ONE EURO (1€) ADAPTIVE MOTION FILTER
# =============================================================================

class LowPassFilter:
    """First-order low-pass filter with alpha parameterization."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha: float = alpha
        self.has_value: bool = False
        self.y: np.ndarray = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.has_value = False

    def filter(self, val: np.ndarray, alpha: float) -> np.ndarray:
        self.alpha = alpha
        if not self.has_value:
            self.y = np.array(val, dtype=np.float32).copy()
            self.has_value = True
            return self.y

        self.y = (alpha * val + (1.0 - alpha) * self.y).astype(np.float32)
        return self.y


class OneEuroFilter:
    """Velocity-adaptive low-pass filter for human motion tracking.

    Eliminates jitter when stationary while preserving responsiveness with zero
    lag during fast gestures. Reference: Casiez et al., CHI 2012.
    """

    def __init__(
        self,
        fc_min: float = 0.85,
        beta: float = 0.04,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize 1€ filter.

        Args:
            fc_min: Minimum cutoff frequency in Hz for jitter elimination at low speeds.
            beta: Speed coefficient scaling cutoff frequency with velocity.
            d_cutoff: Cutoff frequency for the derivative filter in Hz.
        """
        self.fc_min: float = fc_min
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff

        self.x_filt: LowPassFilter = LowPassFilter()
        self.dx_filt: LowPassFilter = LowPassFilter()
        self.prev_x: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets filter memory."""
        self.x_filt.reset()
        self.dx_filt.reset()
        self.prev_x = None

    @staticmethod
    def _compute_alpha(rate: float, cutoff: float) -> float:
        """Computes exponential smoothing factor alpha = 1 / (1 + tau / dt)."""
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        te = 1.0 / max(rate, 1e-4)
        return float(1.0 / (1.0 + tau / te))

    def filter(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Filters input signal x using time step dt.

        Args:
            x: Input array (1D, 2D, or 3D coordinate vector).
            dt: Elapsed time step in seconds.

        Returns:
            Smooth, jitter-free filtered array.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        dt_clamped = max(dt, 1e-4)
        rate = 1.0 / dt_clamped

        if self.prev_x is None:
            self.prev_x = x_arr.copy()
            self.x_filt.filter(x_arr, 1.0)
            self.dx_filt.filter(np.zeros_like(x_arr), 1.0)
            return x_arr.copy()

        # 1. Compute and filter signal derivative (velocity)
        raw_dx = (x_arr - self.prev_x) * rate
        alpha_d = self._compute_alpha(rate, self.d_cutoff)
        filtered_dx = self.dx_filt.filter(raw_dx, alpha_d)

        # 2. Dynamically adjust cutoff frequency based on velocity magnitude
        speed = float(np.linalg.norm(filtered_dx))
        cutoff = self.fc_min + self.beta * speed

        # 3. Filter position signal with velocity-adapted cutoff
        alpha = self._compute_alpha(rate, cutoff)
        filtered_x = self.x_filt.filter(x_arr, alpha)

        self.prev_x = x_arr.copy()
        return filtered_x


# =============================================================================
# 2. DIVERGENCE-FREE 3D CURL NOISE FIELD
# =============================================================================

class CurlNoise3D:
    """Exact analytical divergence-free 3D vector field.

    Computes v = curl(Psi) = (dPsi_z/dy - dPsi_y/dz, dPsi_x/dz - dPsi_z/dx, dPsi_y/dx - dPsi_x/dy).
    By vector calculus identity, div(curl(Psi)) == 0 strictly everywhere,
    guaranteeing natural, volume-preserving organic turbulence without clustering.
    """

    def __init__(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self.frequencies: np.ndarray = np.array([
            [0.0042, 0.0038, 0.0045],
            [0.0088, 0.0075, 0.0092],
            [0.0165, 0.0182, 0.0150],
            [0.0310, 0.0295, 0.0335],
        ], dtype=np.float32)

        self.amplitudes: np.ndarray = np.array([
            [1.00, 0.95, 1.05],
            [0.55, 0.60, 0.50],
            [0.28, 0.25, 0.32],
            [0.14, 0.16, 0.12],
        ], dtype=np.float32)

        self.time_speeds: np.ndarray = np.array([0.45, 0.95, 1.80, 3.20], dtype=np.float32)
        self.phases: np.ndarray = np.array([
            [0.15, 1.25, 2.45],
            [3.10, 0.85, 1.95],
            [2.20, 3.45, 0.65],
            [1.05, 2.70, 3.90],
        ], dtype=np.float32)

    def evaluate(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Evaluates divergence-free velocity vector at 3D position and time t.

        Args:
            pos: [x, y, z] 3D coordinates.
            t: Time in seconds.

        Returns:
            v_curl: 3D velocity vector with div(v) == 0.
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        dpsi_z_dy = 0.0
        dpsi_y_dz = 0.0
        dpsi_x_dz = 0.0
        dpsi_z_dx = 0.0
        dpsi_y_dx = 0.0
        dpsi_x_dy = 0.0

        for k in range(len(self.frequencies)):
            kx, ky, kz = self.frequencies[k]
            ax, ay, az = self.amplitudes[k]
            wt = float(self.time_speeds[k] * t)
            px, py, pz = self.phases[k]

            arg_x = float(ky * y + kz * z + wt + px)
            arg_y = float(kz * z + kx * x + wt + py)
            arg_z = float(kx * x + ky * y + wt + pz)

            cos_x = math.cos(arg_x)
            cos_y = math.cos(arg_y)
            cos_z = math.cos(arg_z)

            dpsi_x_dy += ax * ky * cos_x
            dpsi_x_dz += ax * kz * cos_x

            dpsi_y_dz += ay * kz * cos_y
            dpsi_y_dx += ay * kx * cos_y

            dpsi_z_dx += az * kx * cos_z
            dpsi_z_dy += az * ky * cos_z

        vx = dpsi_z_dy - dpsi_y_dz
        vy = dpsi_x_dz - dpsi_z_dx
        vz = dpsi_y_dx - dpsi_x_dy

        return np.array([vx, vy, vz], dtype=np.float32)


# =============================================================================
# 3. 3D ORIENTED BOUNDING BOX (OBB) & SEPARATING AXIS THEOREM (SAT)
# =============================================================================

@dataclass
class OBB3D:
    """3D Oriented Bounding Box defined by center, rotation matrix, and half-extents."""
    center: np.ndarray                     # [3] center in camera space
    rotation: np.ndarray                   # [3, 3] orthonormal basis vectors as columns
    half_extents: np.ndarray               # [3] half dimensions [hx, hy, hz]

    def get_corners(self) -> np.ndarray:
        """Returns the 8 3D world corners of the OBB."""
        signs = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        ], dtype=np.float32)
        local_pts = signs * self.half_extents
        return self.center + local_pts @ self.rotation.T


@dataclass
class SATCollisionResult:
    """Result of an OBB-to-OBB Separating Axis Theorem intersection test."""
    intersecting: bool
    contact_normal: np.ndarray             # Normal pointing from Box A to Box B
    penetration: float                     # Overlap depth
    contact_point: np.ndarray              # 3D contact point in camera space


class OBBSATCollider:
    """Performs 15-axis Separating Axis Theorem (SAT) rigid body collision queries."""

    @staticmethod
    def test_collision(box_a: OBB3D, box_b: OBB3D) -> Optional[SATCollisionResult]:
        """Tests whether two 3D OBBs intersect across all 15 potential separating axes.

        Axes tested:
        - 3 face normals of A: A.u0, A.u1, A.u2
        - 3 face normals of B: B.u0, B.u1, B.u2
        - 9 edge-cross products: A.ui x B.uj

        Returns:
            SATCollisionResult if colliding, None if a separating axis exists.
        """
        # Relative translation vector pointing from Box B to Box A
        t = box_a.center - box_b.center

        u_a = box_a.rotation
        u_b = box_b.rotation

        r_matrix = np.zeros((3, 3), dtype=np.float32)
        abs_r = np.zeros((3, 3), dtype=np.float32)
        eps = 1e-5

        for i in range(3):
            for j in range(3):
                r_matrix[i, j] = float(np.dot(u_a[:, i], u_b[:, j]))
                abs_r[i, j] = abs(r_matrix[i, j]) + eps

        ea = box_a.half_extents
        eb = box_b.half_extents

        t_a = np.array([float(np.dot(t, u_a[:, i])) for i in range(3)], dtype=np.float32)

        min_penetration = float("inf")
        best_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # 1. Test 3 Face Normals of Box A (A0, A1, A2)
        for i in range(3):
            ra = ea[i]
            rb = eb[0] * abs_r[i, 0] + eb[1] * abs_r[i, 1] + eb[2] * abs_r[i, 2]
            dist = abs(t_a[i])
            overlap = (ra + rb) - dist
            if overlap <= 0.0:
                return None  # Separating axis found!
            if overlap < min_penetration:
                min_penetration = overlap
                axis = u_a[:, i] * (1.0 if t_a[i] >= 0.0 else -1.0)
                best_axis = axis

        # 2. Test 3 Face Normals of Box B (B0, B1, B2)
        for j in range(3):
            ra = ea[0] * abs_r[0, j] + ea[1] * abs_r[1, j] + ea[2] * abs_r[2, j]
            rb = eb[j]
            dist = abs(t_a[0] * r_matrix[0, j] + t_a[1] * r_matrix[1, j] + t_a[2] * r_matrix[2, j])
            overlap = (ra + rb) - dist
            if overlap <= 0.0:
                return None  # Separating axis found!
            if overlap < min_penetration:
                min_penetration = overlap
                sign = 1.0 if float(np.dot(t, u_b[:, j])) >= 0.0 else -1.0
                best_axis = u_b[:, j] * sign

        # 3. Test 9 Edge-Cross Axes (A_i x B_j)
        for i in range(3):
            for j in range(3):
                if abs_r[i, j] > 0.999:
                    continue

                axis_cross = np.cross(u_a[:, i], u_b[:, j])
                axis_len = float(np.linalg.norm(axis_cross))
                if axis_len < 1e-4:
                    continue
                unit_axis = axis_cross / axis_len

                i1 = (i + 1) % 3
                i2 = (i + 2) % 3
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3

                ra = (ea[i1] * abs_r[i2, j] + ea[i2] * abs_r[i1, j]) / axis_len
                rb = (eb[j1] * abs_r[i, j2] + eb[j2] * abs_r[i, j1]) / axis_len
                dist = abs(float(np.dot(t, unit_axis)))
                overlap = (ra + rb) - dist
                if overlap <= 0.0:
                    return None  # Separating axis found!

        axis_len = float(np.linalg.norm(best_axis))
        if axis_len > 1e-5:
            norm = (best_axis / axis_len).astype(np.float32)
        else:
            diff = box_a.center - box_b.center
            d_len = float(np.linalg.norm(diff))
            norm = (diff / d_len).astype(np.float32) if d_len > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        contact_pt = box_a.center - norm * (box_a.half_extents[0] * 0.5)

        return SATCollisionResult(
            intersecting=True,
            contact_normal=norm,
            penetration=float(min_penetration),
            contact_point=contact_pt.astype(np.float32),
        )


# =============================================================================
# 4. SWEPT-SPHERE CAPSULE COLLIDER FOR COMPLETE HAND SKELETON
# =============================================================================

@dataclass
class BoneCapsuleCollider:
    """Biomechanical hand bone capsule defined by endpoints p0, p1, and radius."""
    name: str
    p0: np.ndarray                         # [3] Start joint in 3D camera space
    p1: np.ndarray                         # [3] End joint in 3D camera space
    radius: float                          # Cylinder/sphere collision radius in px
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def closest_point_to_point(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        """Finds closest point on this capsule segment to an arbitrary 3D point."""
        seg = self.p1 - self.p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-6:
            return self.p0.copy(), 0.0

        t = float(np.dot(point - self.p0, seg) / seg_len_sq)
        t_clamped = float(np.clip(t, 0.0, 1.0))
        closest_pt = self.p0 + t_clamped * seg
        return closest_pt.astype(np.float32), t_clamped

    def test_sphere_collision(
        self, sphere_center: np.ndarray, sphere_radius: float
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """Tests collision against a bounding sphere."""
        closest_pt, _ = self.closest_point_to_point(sphere_center)
        diff = sphere_center - closest_pt
        dist = float(np.linalg.norm(diff))
        min_dist = self.radius + sphere_radius

        if dist < min_dist:
            if dist < 1e-4:
                norm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                overlap = min_dist
            else:
                norm = (diff / dist).astype(np.float32)
                overlap = min_dist - dist

            contact_pt = closest_pt + norm * self.radius
            return norm, overlap, contact_pt

        return None


# =============================================================================
# 5. PHYSICAL THIN-FILM OPTICAL WAVE INTERFERENCE
# =============================================================================

class ThinFilmInterference:
    """Wave-optics thin-film interference model for physical iridescence."""

    WAVELENGTHS_NM: np.ndarray = np.array([460.0, 532.0, 650.0], dtype=np.float32)  # B, G, R

    def __init__(self, n_film: float = 1.45, film_thickness_nm: float = 520.0) -> None:
        self.n_film: float = n_film
        self.thickness_nm: float = film_thickness_nm

    def compute_rgb_reflectance(
        self,
        cos_theta_i: float,
        thickness_mod: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Computes B, G, R reflectance values [0.0, 1.0] for a given incidence angle cosine."""
        cos_i = float(np.clip(abs(cos_theta_i), 0.0, 1.0))
        sin_i = math.sqrt(max(0.0, 1.0 - cos_i * cos_i))

        sin_t = sin_i / max(self.n_film, 1.0)
        cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))

        d_eff = self.thickness_nm * thickness_mod

        f1 = ((1.0 - self.n_film) / (1.0 + self.n_film)) ** 2
        f2 = f1 * 0.75

        optical_path = 4.0 * math.pi * self.n_film * d_eff * cos_t

        refl_channels = []
        for wl in self.WAVELENGTHS_NM:
            delta = (optical_path / wl) + math.pi
            cos_delta = math.cos(delta)
            numerator = f1 + f2 + 2.0 * math.sqrt(f1 * f2) * cos_delta
            denominator = 1.0 + f1 * f2 + 2.0 * math.sqrt(f1 * f2) * cos_delta
            refl = max(0.0, min(1.0, numerator / max(denominator, 1e-4)))
            refl_channels.append(refl)

        return float(refl_channels[0]), float(refl_channels[1]), float(refl_channels[2])
