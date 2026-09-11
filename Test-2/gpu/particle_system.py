"""GPU particle system using OpenGL Transform Feedback for zero-CPU physics.

Architecture:
  - Two ping-pong VAO/VBO pairs: one for reading (current state), one for writing (next state)
  - Update pass: Transform Feedback vertex shader runs physics, writes to the write buffer
  - Render pass: standard vertex+fragment shader reads from the write buffer for display
  - Swap buffers each frame

Particle data per vertex (7 floats = 28 bytes):
  position  (vec3) — world-space XYZ
  velocity  (vec3) — XYZ velocity
  age       (float) — seconds alive (used for fade-in)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import moderngl
import numpy as np


_SHADER_DIR = Path(__file__).parent / "shaders"

# Per-particle stride: (pos.xyz, vel.xyz, age) = 7 floats × 4 bytes
_PARTICLE_FLOATS = 7
_STRIDE = _PARTICLE_FLOATS * 4


def _load_shader(name: str) -> str:
    return (_SHADER_DIR / name).read_text(encoding="utf-8")


class GPUParticleSystem:
    """Manages 50k particle sim via Transform Feedback on the GPU.

    Physics update and rendering are both vertex-shader driven.
    CPU only uploads 6 attractor positions per frame (~144 bytes).
    """

    def __init__(self, ctx: moderngl.Context, num_particles: int = 50_000) -> None:
        """Initialize ping-pong buffers and compile shaders.

        Args:
            ctx: Active moderngl context.
            num_particles: Total GPU particles (default 50,000).
        """
        self._ctx = ctx
        self._num = num_particles

        # ── Compile shaders ───────────────────────────────────────────────────
        # Physics update program (Transform Feedback — no fragment shader)
        self._update_prog = ctx.program(
            vertex_shader=_load_shader("update.vert"),
            varyings=["out_position", "out_velocity", "out_age"],
        )

        # Render program (standard vertex + fragment)
        self._render_prog = ctx.program(
            vertex_shader=_load_shader("particle.vert"),
            fragment_shader=_load_shader("particle.frag"),
        )

        # ── Initialize particle data ──────────────────────────────────────────
        rng = np.random.default_rng(42)
        # Positions: uniform sphere of radius 1.2
        phi   = rng.uniform(0, 2 * np.pi, num_particles).astype(np.float32)
        theta = np.arccos(rng.uniform(-1, 1, num_particles)).astype(np.float32)
        r     = rng.uniform(0.0, 1.2, num_particles).astype(np.float32) ** (1/3)
        px    = r * np.sin(theta) * np.cos(phi)
        py    = r * np.sin(theta) * np.sin(phi)
        pz    = r * np.cos(theta)
        # Velocities: small random
        vx, vy, vz = (rng.uniform(-0.05, 0.05, num_particles).astype(np.float32) for _ in range(3))
        # Ages: staggered so particles don't all fade in at once
        ages = rng.uniform(0, 2.0, num_particles).astype(np.float32)

        # Interleave into (N, 7) array → flat bytes
        initial = np.stack([px, py, pz, vx, vy, vz, ages], axis=1).astype(np.float32)
        initial_bytes = initial.tobytes()

        # ── Ping-pong buffers ─────────────────────────────────────────────────
        self._buf_a = ctx.buffer(initial_bytes)
        self._buf_b = ctx.buffer(reserve=len(initial_bytes))

        self._vao_update_a = self._make_update_vao(self._buf_a)
        self._vao_update_b = self._make_update_vao(self._buf_b)
        self._vao_render_a = self._make_render_vao(self._buf_a)
        self._vao_render_b = self._make_render_vao(self._buf_b)

        # Current read/write pair (flip each frame)
        self._read_is_a: bool = True

        # Default attractor positions (sphere surface, distributed)
        self._null_attractors = np.zeros((6, 3), dtype=np.float32)

    # ── VAO construction ──────────────────────────────────────────────────────

    def _make_update_vao(self, src: moderngl.Buffer) -> moderngl.VertexArray:
        """Build a VAO for the Transform Feedback update pass (reads from src)."""
        return self._ctx.vertex_array(
            self._update_prog,
            [(src, "3f 3f 1f", "in_position", "in_velocity", "in_age")],
        )

    def _make_render_vao(self, src: moderngl.Buffer) -> moderngl.VertexArray:
        """Build a VAO for the render pass."""
        return self._ctx.vertex_array(
            self._render_prog,
            [(src, "3f 3f 1f", "in_position", "in_velocity", "in_age")],
        )

    # ── Per-frame API ─────────────────────────────────────────────────────────

    def update(
        self,
        dt: float,
        elapsed: float,
        attractors: np.ndarray,   # (6, 3) float32 world-space positions
        hand_present: bool,
        gesture_mode: float,      # 0=attract, 1=repel, 2=vortex
        bounding_radius: float = 1.5,
    ) -> None:
        """Run one GPU physics step via Transform Feedback.

        Args:
            dt: Frame delta time in seconds.
            elapsed: Total elapsed time in seconds.
            attractors: (6, 3) array of attractor world positions.
            hand_present: Whether a hand is currently tracked.
            gesture_mode: 0=attract, 1=repel/fist, 2=vortex/pinch.
            bounding_radius: Soft bounding sphere radius; particles wrap on exit.
        """
        prog = self._update_prog

        # Upload uniforms
        prog["dt"].value = float(dt)
        prog["time"].value = float(elapsed)
        prog["attractor_active"].value = 1.0 if hand_present else 0.0
        prog["gesture_mode"].value = float(gesture_mode)
        prog["bounding_radius"].value = float(bounding_radius)

        # Upload attractor positions — moderngl vec3[6] expects flat tuple of 18 floats
        flat = attractors.flatten().astype(np.float32)
        prog["attractors"].write(flat.tobytes())

        # Select write destination buffer based on current ping-pong state
        src_vao = self._vao_update_a if self._read_is_a else self._vao_update_b
        dst_buf = self._buf_b if self._read_is_a else self._buf_a
        src_vao.transform(dst_buf, moderngl.POINTS, vertices=self._num)

    def render(
        self,
        mvp: np.ndarray,           # (4, 4) float32 model-view-projection matrix
        attractors: np.ndarray,    # (6, 3) float32 world-space positions
        hand_present: bool,
        viewport_h: float,
    ) -> None:
        """Render particle point sprites to the active framebuffer.

        Args:
            mvp: 4×4 MVP matrix (column-major).
            attractors: (6, 3) world-space attractor positions.
            hand_present: Whether a hand is tracked.
            viewport_h: Viewport pixel height for gl_PointSize scaling.
        """
        prog = self._render_prog
        prog["u_mvp"].write(mvp.astype(np.float32).tobytes())
        prog["attractor_active"].value = 1.0 if hand_present else 0.0
        prog["viewport_h"].value = float(viewport_h)

        flat = attractors.flatten().astype(np.float32)
        prog["attractors"].write(flat.tobytes())

        # Render from the buffer that was WRITTEN last frame
        vao = self._vao_render_b if self._read_is_a else self._vao_render_a
        vao.render(moderngl.POINTS, vertices=self._num)

        # Swap ping-pong
        self._read_is_a = not self._read_is_a

    def release(self) -> None:
        """Free all GPU resources."""
        for res in [self._buf_a, self._buf_b,
                    self._vao_update_a, self._vao_update_b,
                    self._vao_render_a, self._vao_render_b,
                    self._update_prog, self._render_prog]:
            res.release()
