"""Composited AR renderer: bounding box, procedural mesh, particles, HUD.

Pipeline per frame:
  world = R_hand @ (local * s) + C_world,  C_world = (0, 0, z_depth)
  screen = project(world) with principal point = anchor pixel.
Edges depth-sorted (far first), double-pass glow box, alpha-faded particles.
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from math3d.transforms import BOX_EDGES, get_box_vertices, transform_vertices, project_points


def _depth_shade(color, z: float, z_min: float, z_max: float, dim: float = 0.45):
    span = max(1e-6, z_max - z_min)
    k = 1.0 - dim * (z - z_min) / span  # near=1.0, far=1-dim
    c = np.asarray(color, dtype=np.float64) * k
    return tuple(int(v) for v in np.clip(c, 0, 255))


class ARRenderer:
    def __init__(self, settings) -> None:
        self.s = settings
        vp = settings.viewport
        self.model_scale = float(vp.model_scale)
        if self.model_scale > 10.0:  # back-compat if configured in pixels
            self.model_scale = self.model_scale / 400.0
        self.box_size = float(vp.box_size)
        self.focal = float(vp.focal_length)
        self.cam_dist = float(vp.camera_distance)
        self.near = float(vp.near_clip_z)
        self._box_local = get_box_vertices(self.box_size)

    # ── projection helpers ──────────────────────────────────────────────
    def anchor_pixel(self, cx_norm: float, cy_norm: float, w: int, h: int):
        return float(cx_norm * w), float(cy_norm * h)

    def depth_from_z(self, z_norm: float) -> float:
        # MediaPipe z ~ [-0.2, 0.1]; push into eye-space offset.
        return float(np.clip(z_norm * 2.0, -0.8, 0.8))

    def project_world(self, world: np.ndarray, ax: float, ay: float):
        return project_points(world, ax, ay, self.focal, self.cam_dist, self.near)

    # ── primitives ──────────────────────────────────────────────────────
    def draw_box(self, frame: np.ndarray, world_box: np.ndarray,
                 ax: float, ay: float) -> None:
        scr, z_eye, _ = self.project_world(world_box, ax, ay)
        order = sorted(range(len(BOX_EDGES)),
                       key=lambda e: (z_eye[BOX_EDGES[e][0]] + z_eye[BOX_EDGES[e][1]]) * 0.5,
                       reverse=True)
        glow, core = self.s.colors.box_glow, self.s.colors.box_core
        for e in order:
            i, j = BOX_EDGES[e]
            p1 = (int(scr[i, 0]), int(scr[i, 1]))
            p2 = (int(scr[j, 0]), int(scr[j, 1]))
            cv2.line(frame, p1, p2, glow, 7, cv2.LINE_AA)
        for e in order:
            i, j = BOX_EDGES[e]
            p1 = (int(scr[i, 0]), int(scr[i, 1]))
            p2 = (int(scr[j, 0]), int(scr[j, 1]))
            cv2.line(frame, p1, p2, core, 2, cv2.LINE_AA)
        for k in range(8):
            c = (int(scr[k, 0]), int(scr[k, 1]))
            cv2.circle(frame, c, 7, glow, -1, cv2.LINE_AA)
            cv2.circle(frame, c, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_mesh(self, frame: np.ndarray, world: np.ndarray,
                  edges: np.ndarray, colors: np.ndarray,
                  ax: float, ay: float) -> None:
        if world.shape[0] == 0 or edges.shape[0] == 0:
            return
        scr, z_eye, _ = self.project_world(world, ax, ay)
        z_min, z_max = float(z_eye.min()), float(z_eye.max())
        h, w = frame.shape[:2]
        oob = (scr[:, 0] < -200) | (scr[:, 0] > w + 200) | (scr[:, 1] < -200) | (scr[:, 1] > h + 200)
        depths = (z_eye[edges[:, 0]] + z_eye[edges[:, 1]]) * 0.5
        order = np.argsort(-depths)  # far first
        cols = np.asarray(colors, dtype=np.float64)
        for idx in order:
            i, j = int(edges[idx, 0]), int(edges[idx, 1])
            if oob[i] and oob[j]:
                continue
            p1 = (int(scr[i, 0]), int(scr[i, 1]))
            p2 = (int(scr[j, 0]), int(scr[j, 1]))
            base = (cols[i] + cols[j]) * 0.5
            zm = float(depths[idx])
            shaded = _depth_shade(base, zm, z_min, z_max)
            # glow pass + core pass
            cv2.line(frame, p1, p2, shaded, 5, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
        # nodes
        for k in range(world.shape[0]):
            if oob[k]:
                continue
            c = (int(scr[k, 0]), int(scr[k, 1]))
            shaded = _depth_shade(cols[k], float(z_eye[k]), z_min, z_max, dim=0.3)
            cv2.circle(frame, c, 3, shaded, -1, cv2.LINE_AA)

    def draw_particles(self, frame: np.ndarray, pos: np.ndarray,
                       colors: np.ndarray, alphas: np.ndarray,
                       ax: float, ay: float) -> None:
        if pos.shape[0] == 0:
            return
        scr, z_eye, inv = self.project_world(pos, ax, ay)
        h, w = frame.shape[:2]
        pr = self.s.particles.point_radius
        cols = np.asarray(colors, dtype=np.float64)
        for k in range(pos.shape[0]):
            a = float(alphas[k])
            if a <= 0.01:
                continue
            x, y = int(scr[k, 0]), int(scr[k, 1])
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            scale = float(np.clip(inv[k] / (self.focal / self.cam_dist), 0.4, 2.2))
            r = max(1, int(pr * scale * (0.5 + 0.5 * a)))
            base = cols[k] * a
            b, g, rr = int(base[0]), int(base[1]), int(base[2])
            cv2.circle(frame, (x, y), r + 2, (b // 3, g // 3, rr // 3), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, (b, g, rr), -1, cv2.LINE_AA)

    # ── HUD ─────────────────────────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray, lines, accent_idx=()) -> None:
        x, y, pad = 14, 14, 8
        fs, th = 0.55, 1
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for t in lines]
        bw, bh = max(widths) + pad * 2, len(lines) * 20 + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), self.s.colors.hud_bg, -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), self.s.colors.hud_border, 1, cv2.LINE_AA)
        for i, t in enumerate(lines):
            col = self.s.colors.hud_accent if i in accent_idx else self.s.colors.hud_text
            cv2.putText(frame, t, (x + pad, y + pad + 14 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(frame, t, (x + pad, y + pad + 14 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)

    def draw_control_bar(self, frame: np.ndarray, active_model: Optional[str],
                         trigger_name: str, dissolving: bool) -> None:
        h, w = frame.shape[:2]
        cells = [
            ("INDEX", "Lilies", "flowers"),
            ("FIST", "Dragon", "dragon"),
            ("L", "Morpho", "butterfly"),
            ("PEACE", "Bonsai", "tree"),
            ("PALM", "Dissolve", None),
            ("2xFIST", "Clear", None),
        ]
        bw, bh, gap = 150, 44, 8
        total = len(cells) * bw + (len(cells) - 1) * gap
        x0 = (w - total) // 2
        y0 = h - bh - 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 8), (x0 + total + 10, y0 + bh + 8), (12, 14, 18), -1)
        cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)
        for i, (key, label, mid) in enumerate(cells):
            x = x0 + i * (bw + gap)
            is_active = (mid is not None and mid == active_model)
            if mid is None and dissolving and key == "PALM":
                is_active = True
            bg = (70, 60, 30) if is_active else (30, 32, 36)
            border = self.s.colors.hud_accent if is_active else self.s.colors.hud_border
            cv2.rectangle(frame, (x, y0), (x + bw, y0 + bh), bg, -1)
            cv2.rectangle(frame, (x, y0), (x + bw, y0 + bh), border, 1, cv2.LINE_AA)
            cv2.putText(frame, key, (x + 8, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 8, y0 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.s.colors.hud_accent if is_active else (240, 240, 240), 1, cv2.LINE_AA)
        hint = f"Trigger: {trigger_name}"
        cv2.putText(frame, hint, (x0, y0 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 220), 1, cv2.LINE_AA)
