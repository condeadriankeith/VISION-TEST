"""Minimalist 2D HUD overlay renderer.

Draws an uncluttered translucent pill in the upper-right corner displaying
real-time FPS, frame latency, and current state indicator without visual clutter.
"""

from typing import Tuple
import cv2
import numpy as np

from config.settings import ColorPalette
from core.state_machine import GestureState


class HUDOverlayRenderer:
    """Renders a sleek, unobtrusive glassmorphic HUD status pill in the upper-right corner."""

    def __init__(self, palette: ColorPalette = ColorPalette()) -> None:
        """Initialize HUD styling palette."""
        self._palette: ColorPalette = palette

    def render(
        self,
        target_frame: np.ndarray,
        fps: float,
        frame_time_ms: float,
        current_state: GestureState,
    ) -> None:
        """Draw minimalist upper-right HUD pill directly onto target_frame.

        Args:
            target_frame: BGR destination canvas (H, W, 3).
            fps: Instantaneous or smoothed frames per second.
            frame_time_ms: Frame processing time in milliseconds.
            current_state: Current active gesture state.
        """
        h, w = target_frame.shape[:2]

        # Pill dimensions and positioning in upper-right margin
        pill_w = 200
        pill_h = 36
        pill_x = w - pill_w - 24
        pill_y = 20

        # Extract ROI for glassmorphism blending
        if pill_x < 0 or pill_y < 0 or pill_x + pill_w > w or pill_y + pill_h > h:
            return

        roi = target_frame[pill_y:pill_y + pill_h, pill_x:pill_x + pill_w]
        pill_overlay = np.full_like(roi, self._palette.hud_bg)

        # 70% glassmorphism blend for background
        cv2.addWeighted(pill_overlay, 0.75, roi, 0.25, 0, dst=roi)

        # Subtle 1px rounded pill border
        cv2.rectangle(
            target_frame,
            (pill_x, pill_y),
            (pill_x + pill_w, pill_y + pill_h),
            self._palette.hud_border,
            1,
            lineType=cv2.LINE_AA,
        )

        # State indicator text and dot color
        state_label, dot_color = self._get_state_visuals(current_state)

        # Small status indicator dot
        dot_center = (pill_x + 16, pill_y + (pill_h // 2))
        cv2.circle(target_frame, dot_center, 4, dot_color, -1, lineType=cv2.LINE_AA)

        # State badge text
        cv2.putText(
            target_frame,
            state_label,
            (pill_x + 28, pill_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self._palette.hud_text_primary,
            1,
            lineType=cv2.LINE_AA,
        )

        # Performance readout: FPS and latency
        fps_text = f"{fps:4.1f} FPS"
        cv2.putText(
            target_frame,
            fps_text,
            (pill_x + 100, pill_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self._palette.hud_text_accent,
            1,
            lineType=cv2.LINE_AA,
        )

    def _get_state_visuals(self, state: GestureState) -> Tuple[str, Tuple[int, int, int]]:
        """Return concise label and dot color corresponding to state."""
        if state == GestureState.OPEN_PALM:
            return "CLOUD", (240, 220, 80)     # Neon Cyan
        elif state == GestureState.PINCH:
            return "CUBE",  (80, 220, 160)     # Emerald Green
        elif state == GestureState.FIST:
            return "IDLE",  (60, 140, 240)     # Warm Amber
        else:
            return "SCAN",  (120, 120, 120)    # Neutral Gray
