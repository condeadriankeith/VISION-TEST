"""Particle dissolution dynamics (blueprint 3.2, vectorized numpy).

P(t+dt) = P(t) + V(t) dt
V(t) = V_radial * exp(-gamma * age) + V_curl(P) * beta
V_curl = [sin(k Py + t), cos(k Px - t), sin(0.5 k (Px + Py))]
alpha(t) = max(0, 1 - age / T_dissolve)
"""

from __future__ import annotations

import numpy as np


class ParticleDissolver:
    def __init__(self, max_particles: int = 1200, dissolve_time: float = 1.1,
                 radial_speed: float = 1.6, curl_strength: float = 1.1,
                 curl_freq: float = 3.0, gamma: float = 1.6,
                 gravity: float = -0.25) -> None:
        self.max_n = int(max_particles)
        self.T = float(dissolve_time)
        self.radial_speed = float(radial_speed)
        self.beta = float(curl_strength)
        self.k = float(curl_freq)
        self.gamma = float(gamma)
        self.gravity = float(gravity)
        self.pos = np.zeros((0, 3), dtype=np.float64)
        self.vel_radial = np.zeros((0, 3), dtype=np.float64)
        self.age = np.zeros((0,), dtype=np.float64)
        self.life = np.zeros((0,), dtype=np.float64)
        self.color = np.zeros((0, 3), dtype=np.uint8)
        self._time = 0.0

    @property
    def active(self) -> bool:
        return self.pos.shape[0] > 0

    @property
    def count(self) -> int:
        return int(self.pos.shape[0])

    def clear(self) -> None:
        self.pos = np.zeros((0, 3), dtype=np.float64)
        self.vel_radial = np.zeros((0, 3), dtype=np.float64)
        self.age = np.zeros((0,), dtype=np.float64)
        self.life = np.zeros((0,), dtype=np.float64)
        self.color = np.zeros((0, 3), dtype=np.uint8)

    def spawn_from_vertices(self, world_verts: np.ndarray,
                            colors: np.ndarray | None = None,
                            upsample: int = 3) -> int:
        """Shatter mesh vertices into particles. Returns particle count."""
        w = np.asarray(world_verts, dtype=np.float64).reshape(-1, 3)
        if w.shape[0] == 0:
            return 0
        reps = np.repeat(w, upsample, axis=0)
        if reps.shape[0] > self.max_n:
            idx = np.random.choice(reps.shape[0], self.max_n, replace=False)
            reps = reps[idx]
        n = reps.shape[0]
        centroid = reps.mean(axis=0, keepdims=True)
        dirs = reps - centroid
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        dirs = dirs / norms
        jitter = np.random.randn(n, 3) * 0.35
        v_rad = (dirs + jitter) * self.radial_speed
        if colors is not None and len(colors) > 0:
            c = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            crep = np.repeat(c, upsample, axis=0)
            if crep.shape[0] > self.max_n:
                crep = crep[idx]
            cols = crep[:n]
        else:
            cols = np.tile(np.array([[150, 220, 255]], dtype=np.uint8), (n, 1))
        self.pos = reps.copy()
        self.vel_radial = v_rad
        self.age = np.zeros((n,), dtype=np.float64)
        self.life = np.full((n,), self.T * (0.7 + 0.6 * np.random.rand(n)))
        self.color = cols
        self._time = 0.0
        return n

    def update(self, dt: float) -> None:
        if self.pos.shape[0] == 0:
            return
        dt = float(max(0.0, min(dt, 0.05)))
        self._time += dt
        t = self._time
        self.age += dt
        decay = np.exp(-self.gamma * self.age)[:, None]
        px, py = self.pos[:, 0], self.pos[:, 1]
        curl = np.column_stack((
            np.sin(self.k * py + t),
            np.cos(self.k * px - t),
            np.sin(0.5 * self.k * (px + py)),
        )) * self.beta
        vel = self.vel_radial * decay + curl
        vel[:, 1] -= self.gravity * dt * -10.0 * 0.1  # gentle float
        self.pos = self.pos + vel * dt
        keep = self.age < self.life
        if not np.all(keep):
            self.pos = self.pos[keep]
            self.vel_radial = self.vel_radial[keep]
            self.age = self.age[keep]
            self.life = self.life[keep]
            self.color = self.color[keep]

    def alphas(self) -> np.ndarray:
        if self.pos.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.maximum(0.0, 1.0 - self.age / np.maximum(self.life, 1e-6))
