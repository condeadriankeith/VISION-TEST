"""Debounced gesture state machine with smooth visual fade transitions.

States:
  NO_HAND    - No hand detected; everything hidden.
  FIST       - Standby / Idle; everything fades out.
  PINCH      - Thumb+Index pinch detected; cube is active.
  OPEN_PALM  - Open palm; cube + full particle cloud active.

Cube position is always fixed; only rotation changes when a hand is present.
"""

from enum import Enum, auto
from typing import Optional


class GestureState(Enum):
    """Deterministic hand gesture interaction states."""
    NO_HAND = auto()    # No hand in frame
    FIST = auto()       # Standby: cube and particles fade out
    PINCH = auto()      # Thumb-Index pinch: wireframe cube active, particles inactive
    OPEN_PALM = auto()  # Open palm: cube + full particle cloud active


class GestureStateMachine:
    """Manages state transitions with debounce filtering and visual alpha smoothing."""

    def __init__(self, debounce_frames: int = 4, fade_speed: float = 0.12) -> None:
        """Initialize state machine.

        Args:
            debounce_frames: Consecutive identical observations required to trigger transition.
            fade_speed: Linear rate per frame [0.0, 1.0] for alpha crossfading.
        """
        self._debounce_frames: int = max(1, debounce_frames)
        self._fade_speed: float = max(0.01, min(1.0, fade_speed))

        self._current_state: GestureState = GestureState.NO_HAND
        self._candidate_state: Optional[GestureState] = None
        self._candidate_counter: int = 0

        # Alpha levels for rendering crossfades [0.0 = invisible, 1.0 = fully visible]
        self._cube_alpha: float = 0.0
        self._particle_alpha: float = 0.0

    @property
    def current_state(self) -> GestureState:
        """Return the current stable gesture state."""
        return self._current_state

    @property
    def cube_alpha(self) -> float:
        """Smoothed rendering alpha for wireframe cube [0.0, 1.0]."""
        return self._cube_alpha

    @property
    def particle_alpha(self) -> float:
        """Smoothed rendering alpha for particle mesh [0.0, 1.0]."""
        return self._particle_alpha

    def update(self, observed_candidate: GestureState) -> GestureState:
        """Update state machine with newly observed instantaneous gesture.

        Args:
            observed_candidate: Instantaneous gesture classification from current frame.

        Returns:
            Stable current state after debouncing.
        """
        if observed_candidate == self._current_state:
            self._candidate_state = None
            self._candidate_counter = 0
        else:
            if observed_candidate == self._candidate_state:
                self._candidate_counter += 1
                if self._candidate_counter >= self._debounce_frames:
                    self._current_state = observed_candidate
                    self._candidate_state = None
                    self._candidate_counter = 0
            else:
                self._candidate_state = observed_candidate
                self._candidate_counter = 1

        self._update_alphas()
        return self._current_state

    def _update_alphas(self) -> None:
        """Gradually transition cube and particle alpha towards target state values."""
        # Cube is visible in PINCH and OPEN_PALM
        target_cube = 1.0 if self._current_state in (GestureState.PINCH, GestureState.OPEN_PALM) else 0.0
        # Particles are only visible in OPEN_PALM
        target_particles = 1.0 if self._current_state == GestureState.OPEN_PALM else 0.0

        if self._cube_alpha < target_cube:
            self._cube_alpha = min(1.0, self._cube_alpha + self._fade_speed)
        elif self._cube_alpha > target_cube:
            self._cube_alpha = max(0.0, self._cube_alpha - self._fade_speed)

        if self._particle_alpha < target_particles:
            self._particle_alpha = min(1.0, self._particle_alpha + self._fade_speed)
        elif self._particle_alpha > target_particles:
            self._particle_alpha = max(0.0, self._particle_alpha - self._fade_speed)

    def is_cube_active(self) -> bool:
        """Check if cube has non-zero rendering alpha."""
        return self._cube_alpha > 0.01

    def is_particle_active(self) -> bool:
        """Check if particle system has non-zero rendering alpha."""
        return self._particle_alpha > 0.01
