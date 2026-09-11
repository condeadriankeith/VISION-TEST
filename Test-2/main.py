"""Galaxy Vortex Hand-Particle System — Main Application.

Architecture:
  - pygame + OpenGL window (DOUBLEBUF | OPENGL) — hardware-accelerated display.
  - Camera feed captured in a background thread; uploaded to GPU texture.
  - MediaPipe hand tracking executed on the main thread each frame.
  - GPU Transform Feedback updates 50,000 particles entirely on the GPU.
  - Additive blending creates natural bright nebula glow over the live camera.

Gesture Controls:
  Open palm      -> particles attracted toward fingertips
  Pinch          -> tight orbital vortex spin
  Fist           -> shockwave repulsion burst

Keyboard Controls:
  [Q / ESC]      -> quit
  [F]            -> toggle fullscreen
  [C]            -> switch camera index
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import moderngl
import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE, QUIT,
    KEYDOWN, K_ESCAPE, K_q, K_f, K_c,
)

from gpu.particle_system import GPUParticleSystem
from vision.hand_tracker import HandState, HandTracker


# ── Configuration ──────────────────────────────────────────────────────────────

NUM_PARTICLES: int = 50_000
DEFAULT_CAM_INDEX: int = 0
WIN_W: int         = 1280
WIN_H: int         = 720
TARGET_FPS: int    = 120
FOV_DEG: float     = 60.0
Z_NEAR: float      = 0.1
Z_FAR: float       = 20.0


# ── Math helpers ───────────────────────────────────────────────────────────────

def perspective_matrix(fov_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    """Return a column-major 4x4 OpenGL perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (z_far + z_near) / (z_near - z_far)
    mat[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    mat[3, 2] = -1.0
    return mat


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a column-major 4x4 OpenGL look-at view matrix."""
    f = center - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = r
    mat[1, :3] = u
    mat[2, :3] = -f
    mat[0, 3]  = -np.dot(r, eye)
    mat[1, 3]  = -np.dot(u, eye)
    mat[2, 3]  =  np.dot(f, eye)
    return mat


# ── Background camera renderer (default unmodified camera feed) ───────────────

_BG_VERT = """
#version 430
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_BG_FRAG = """
#version 430
uniform sampler2D u_texture;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    // Pure, default live camera feed with 100% natural colors and brightness
    fragColor = vec4(texture(u_texture, v_uv).rgb, 1.0);
}
"""

_QUAD_DATA = np.array([
    # x      y     u    v
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 0.0,
     1.0,  1.0,  1.0, 0.0,
], dtype=np.float32)


class BackgroundRenderer:
    """Renders the standard default camera feed without filters or darkening."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._width = width
        self._height = height
        self._prog = ctx.program(vertex_shader=_BG_VERT, fragment_shader=_BG_FRAG)
        buf = ctx.buffer(_QUAD_DATA.tobytes())
        self._vao = ctx.vertex_array(self._prog, [(buf, "2f 2f", "in_vert", "in_uv")])
        self._tex = ctx.texture((width, height), 3)
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._prog["u_texture"].value = 0
        self._has_frame = False

    def upload(self, bgr_frame: np.ndarray) -> None:
        """Convert BGR -> RGB, flip horizontally, dynamically reallocate if size changed."""
        h, w = bgr_frame.shape[:2]
        if w != self._width or h != self._height:
            self._tex.release()
            self._width, self._height = w, h
            self._tex = self._ctx.texture((w, h), 3)
            self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 1)
        self._tex.write(rgb.tobytes())
        self._has_frame = True

    def render(self) -> None:
        if not self._has_frame:
            return
        self._tex.use(0)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        self._vao.release()
        self._tex.release()
        self._prog.release()


# ── HUD Overlay renderer ───────────────────────────────────────────────────────

_HUD_VERT = """
#version 430
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_HUD_FRAG = """
#version 430
uniform sampler2D u_hud_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_hud_tex, v_uv);
}
"""


class HUDOverlay:
    """Renders 2D text HUD directly to the OpenGL context using an RGBA texture."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._width = width
        self._height = height
        self._prog = ctx.program(vertex_shader=_HUD_VERT, fragment_shader=_HUD_FRAG)
        buf = ctx.buffer(_QUAD_DATA.tobytes())
        self._vao = ctx.vertex_array(self._prog, [(buf, "2f 2f", "in_vert", "in_uv")])
        self._tex = ctx.texture((width, height), 4)
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._prog["u_hud_tex"].value = 0
        self._font_title = pygame.font.SysFont("monospace", 15, bold=True)
        self._font_body  = pygame.font.SysFont("monospace", 13, bold=False)

    def resize(self, width: int, height: int) -> None:
        if width != self._width or height != self._height:
            self._tex.release()
            self._width, self._height = width, height
            self._tex = self._ctx.texture((width, height), 4)
            self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def render(self, lines: list[str]) -> None:
        surface = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        # Semi-transparent background card
        card_w = min(400, self._width - 32)
        card_h = len(lines) * 20 + 22
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((10, 15, 30, 160))
        pygame.draw.rect(card_surf, (70, 130, 210, 120), (0, 0, card_w, card_h), width=1, border_radius=6)
        surface.blit(card_surf, (16, 16))

        y = 24
        for idx, line in enumerate(lines):
            if not line:
                y += 8
                continue
            font = self._font_title if idx == 0 else self._font_body
            color = (255, 230, 130) if idx == 0 else ((180, 225, 255) if ":" in line else (150, 170, 200))
            shadow = font.render(line, True, (0, 0, 0))
            text   = font.render(line, True, color)
            surface.blit(shadow, (27, y + 1))
            surface.blit(text,   (26, y))
            y += 20

        # Upload surface RGBA to OpenGL texture (unflipped so row 0 stays at top-left)
        raw = pygame.image.tostring(surface, "RGBA", False)
        self._tex.write(raw)
        self._tex.use(0)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        self._vao.release()
        self._tex.release()
        self._prog.release()


# ── Threaded camera capture ────────────────────────────────────────────────────

class CameraThread:
    """Grabs camera frames in a daemon thread with dynamic index switching."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._open_capture(self._index)

    def _open_capture(self, index: int) -> bool:
        """Attempts to open a camera by index using DirectShow."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

        print(f"[Camera] Connecting to camera index {index} (DirectShow)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Camera] Camera {index} failed to open.")
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Test warmup frames
        ret = False
        test_frame = None
        for _ in range(3):
            ret, test_frame = cap.read()

        if ret and test_frame is not None:
            self._cap = cap
            self._index = index
            h, w = test_frame.shape[:2]
            print(f"[Camera] Connected: Index {index} ({w}x{h}, live video ready)")
            return True

        print(f"[Camera] Camera {index} opened but could not read frames.")
        cap.release()
        return False

    def switch_camera(self, target_index: Optional[int] = None) -> int:
        """Switch to a specified or next alternate camera index."""
        next_idx = target_index if target_index is not None else (1 if self._index == 0 else 0)
        with self._lock:
            success = self._open_capture(next_idx)
            if not success and target_index is None:
                self._open_capture(self._index)
        return self._index

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._running:
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def index(self) -> int:
        return self._index

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._cap is not None:
            self._cap.release()


# ── Main application ──────────────────────────────────────────────────────────

class GalaxyApp:
    """Orchestrates camera, hand tracking, GPU particles, and OpenGL rendering."""

    def __init__(self) -> None:
        # ── pygame window ─────────────────────────────────────────────────────
        pygame.init()
        pygame.font.init()
        self._win_w = WIN_W
        self._win_h = WIN_H
        self._fullscreen = False
        self._surface = pygame.display.set_mode(
            (WIN_W, WIN_H), DOUBLEBUF | OPENGL | RESIZABLE
        )
        pygame.display.set_caption("Galaxy Vortex — Hand Particle System")

        # ── moderngl context ──────────────────────────────────────────────────
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ── Camera + tracker ──────────────────────────────────────────────────
        self._camera = CameraThread(DEFAULT_CAM_INDEX, WIN_W, WIN_H)
        self._tracker = HandTracker()
        self._hand_state = HandState(
            attractors=np.zeros((6, 3), dtype=np.float32),
            hand_present=False,
            gesture_mode=0.0,
            pinch_distance=1.0,
        )

        # ── Renderers ─────────────────────────────────────────────────────────
        self._bg = BackgroundRenderer(self._ctx, WIN_W, WIN_H)
        self._particles = GPUParticleSystem(self._ctx, NUM_PARTICLES)
        self._hud = HUDOverlay(self._ctx, WIN_W, WIN_H)

        # ── Timing ────────────────────────────────────────────────────────────
        self._clock = pygame.time.Clock()
        self._start = time.perf_counter()
        self._fps = 60.0

    def _mvp(self) -> np.ndarray:
        aspect = self._win_w / max(self._win_h, 1)
        eye    = np.array([0.0, 0.0, 3.2], dtype=np.float32)
        center = np.zeros(3, dtype=np.float32)
        up     = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        proj   = perspective_matrix(FOV_DEG, aspect, Z_NEAR, Z_FAR)
        view   = look_at(eye, center, up)
        return proj @ view

    def _gesture_label(self) -> str:
        hs = self._hand_state
        if not hs.hand_present:
            return "Searching for hand..."
        mode = hs.gesture_mode
        if mode < 0.5:
            return "ATTRACT  (Open Palm)"
        if mode > 1.5:
            return "VORTEX   (Pinch)"
        return "REPEL    (Fist)"

    def _draw_hud(self) -> None:
        """Render HUD text lines using the OpenGL overlay quad."""
        lines = [
            "GALAXY VORTEX SYSTEM",
            f"FPS:       {self._fps:.0f}",
            f"Particles: {NUM_PARTICLES:,}",
            f"Gesture:   {self._gesture_label()}",
            f"Camera:    Index {self._camera.index} (Default Feed)",
            "",
            "[C] Switch Cam   [F] Fullscreen   [Q] Quit",
        ]
        self._hud.render(lines)

    def run(self) -> None:
        """Main render loop."""
        self._camera.start()
        prev_time = time.perf_counter()

        print("=" * 60)
        print("  GALAXY VORTEX — Hand Particle System")
        print("=" * 60)
        print("  Open palm   ->  attract particles to fingertips")
        print("  Pinch       ->  tight vortex orbital spin")
        print("  Fist        ->  shockwave repulsion burst")
        print("  [C]         ->  switch camera index")
        print("  [Q / ESC]   ->  quit      [F] -> fullscreen")
        print("=" * 60)

        running = True
        while running:
            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        running = False
                    elif event.key == K_f:
                        self._toggle_fullscreen()
                    elif event.key == K_c:
                        idx = self._camera.switch_camera()
                        print(f"[Controls] Switched to Camera {idx}")
                elif event.type == pygame.VIDEORESIZE:
                    self._win_w, self._win_h = event.w, event.h
                    self._ctx.viewport = (0, 0, self._win_w, self._win_h)
                    self._hud.resize(self._win_w, self._win_h)

            # ── Delta time ────────────────────────────────────────────────────
            now = time.perf_counter()
            dt = min(now - prev_time, 0.05)
            prev_time = now
            elapsed = now - self._start

            # ── Camera frame + hand tracking ──────────────────────────────────
            frame = self._camera.read()
            if frame is not None:
                self._hand_state = self._tracker.process(frame)
                self._bg.upload(frame)

            hs = self._hand_state

            # ── GL clear ──────────────────────────────────────────────────────
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)

            # ── Background live camera (default clean camera feed) ────────────
            self._ctx.disable(moderngl.BLEND)
            self._bg.render()

            # ── GPU physics update ────────────────────────────────────────────
            self._particles.update(
                dt=dt,
                elapsed=elapsed,
                attractors=hs.attractors,
                hand_present=hs.hand_present,
                gesture_mode=hs.gesture_mode,
            )

            # ── Particle render (additive blend over camera) ──────────────────
            self._ctx.enable(moderngl.BLEND)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._particles.render(
                mvp=self._mvp(),
                attractors=hs.attractors,
                hand_present=hs.hand_present,
                viewport_h=float(self._win_h),
            )

            # ── HUD text overlay ──────────────────────────────────────────────
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._draw_hud()

            # ── Present ───────────────────────────────────────────────────────
            pygame.display.flip()
            self._clock.tick(TARGET_FPS)
            instant = self._clock.get_fps()
            self._fps = 0.92 * self._fps + 0.08 * max(instant, 1.0)

        self._shutdown()

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        flags = DOUBLEBUF | OPENGL
        if self._fullscreen:
            info = pygame.display.Info()
            self._win_w, self._win_h = info.current_w, info.current_h
            self._surface = pygame.display.set_mode(
                (self._win_w, self._win_h), flags | pygame.FULLSCREEN
            )
        else:
            self._win_w, self._win_h = WIN_W, WIN_H
            self._surface = pygame.display.set_mode(
                (self._win_w, self._win_h), flags | RESIZABLE
            )
        self._ctx.viewport = (0, 0, self._win_w, self._win_h)
        self._hud.resize(self._win_w, self._win_h)

    def _shutdown(self) -> None:
        """Release all resources cleanly."""
        self._camera.stop()
        self._tracker.release()
        self._particles.release()
        self._bg.release()
        self._hud.release()
        pygame.quit()


def main() -> None:
    """Application entry point."""
    app = GalaxyApp()
    app.run()


if __name__ == "__main__":
    main()
