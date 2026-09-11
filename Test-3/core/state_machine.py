"""Debounced gesture state machine + transition controller.

States: EMPTY, FLOWERS, DRAGON, BUTTERFLY, TREE, DISSOLVING.
  - Model gestures (INDEX/FIST/L/PEACE) debounce N frames -> switch asset.
    Switching while a model is active first runs a particle dissolve
    (spawn from current world mesh), then the new model forms.
  - OPEN_PALM -> dissolve current model (transition), ends in EMPTY.
  - DUAL_FISTS (both hands FIST) -> immediate CLEAR to EMPTY.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional
import numpy as np

from vision.gestures import Gesture, MODEL_FOR_GESTURE


class SceneState(Enum):
    EMPTY = auto()
    FLOWERS = auto()
    DRAGON = auto()
    BUTTERFLY = auto()
    TREE = auto()
    DISSOLVING = auto()


_GESTURE_TO_STATE = {
    Gesture.INDEX: SceneState.FLOWERS,
    Gesture.FIST: SceneState.DRAGON,
    Gesture.L_SHAPE: SceneState.BUTTERFLY,
    Gesture.PEACE: SceneState.TREE,
}

_STATE_TO_MODEL = {
    SceneState.FLOWERS: "flowers",
    SceneState.DRAGON: "dragon",
    SceneState.BUTTERFLY: "butterfly",
    SceneState.TREE: "tree",
}

MODEL_LABEL = {
    "flowers": "Blooming Lilies",
    "dragon": "Red Dragon",
    "butterfly": "Blue Morpho",
    "tree": "Cosmic Bonsai",
}


@dataclass
class TransitionEvent:
    kind: str  # "spawn" | "dissolve_start" | "dissolve_done" | "clear"
    model: Optional[str] = None
    dissolve_world_verts: Optional[np.ndarray] = None
    dissolve_colors: Optional[np.ndarray] = None


class GestureStateMachine:
    def __init__(self, debounce_frames: int = 5, dissolve_frames: int = 3) -> None:
        self._debounce = max(1, int(debounce_frames))
        self._dissolve_need = max(1, int(dissolve_frames))
        self._buf: Deque[Gesture] = deque(maxlen=self._debounce)
        self._open_run = 0
        self.state: SceneState = SceneState.EMPTY
        self.active_model: Optional[str] = None
        self._pending_model: Optional[str] = None
        self._dissolving_from: Optional[str] = None

    @property
    def pending(self) -> Optional[str]:
        return self._pending_model

    def begin_dissolve(self, next_model: Optional[str]) -> None:
        """Enter DISSOLVING; renderer spawns particles, then `finish_dissolve`."""
        self._dissolving_from = self.active_model
        self._pending_model = next_model
        self.state = SceneState.DISSOLVING

    def finish_dissolve(self) -> TransitionEvent:
        """Call when particle system is exhausted (or immediately for clear)."""
        nxt = self._pending_model
        self._pending_model = None
        self._dissolving_from = None
        if nxt is None:
            self.state = SceneState.EMPTY
            self.active_model = None
            return TransitionEvent("dissolve_done", None)
        self.active_model = nxt
        self.state = {v: k for k, v in _STATE_TO_MODEL.items()}[nxt]
        return TransitionEvent("spawn", nxt)

    def update(self, gesture: Gesture, dual_fists: bool = False) -> Optional[TransitionEvent]:
        # Highest priority: dual fists -> clear everything now.
        if dual_fists:
            self._buf.clear()
            self._open_run = 0
            if self.state == SceneState.DISSOLVING or self.active_model is None:
                self._pending_model = None
                self._dissolving_from = None
                self.state = SceneState.EMPTY
                self.active_model = None
                return TransitionEvent("clear")
            self._pending_model = None
            from_model = self.active_model
            self.state = SceneState.DISSOLVING
            # main loop spawns dissolve then immediately finishes for a snappy clear.
            return TransitionEvent("dissolve_start", None)

        if self.state == SceneState.DISSOLVING:
            return None  # wait for particles; main loop calls finish_dissolve()

        # Track open-palm run for dissolve trigger.
        if gesture == Gesture.OPEN_PALM:
            self._open_run += 1
        else:
            self._open_run = 0

        self._buf.append(gesture)
        if len(self._buf) < self._debounce:
            return None
        # Debounce: all buffered frames must agree and be actionable.
        first = self._buf[0]
        if any(g != first for g in self._buf):
            return None

        if first == Gesture.OPEN_PALM:
            if self._open_run >= self._dissolve_need and self.active_model is not None:
                self._buf.clear()
                self._open_run = 0
                self.begin_dissolve(None)
                return TransitionEvent("dissolve_start", None)
            return None

        if first in _GESTURE_TO_STATE:
            want_model = MODEL_FOR_GESTURE[first]
            if want_model == self.active_model:
                return None
            self._buf.clear()
            if self.active_model is not None:
                # Transition through particles.
                self.begin_dissolve(want_model)
                return TransitionEvent("dissolve_start", want_model)
            self.active_model = want_model
            self.state = _GESTURE_TO_STATE[first]
            return TransitionEvent("spawn", want_model)
        return None

    def force_clear(self) -> TransitionEvent:
        self._buf.clear()
        self._open_run = 0
        self._pending_model = None
        self.state = SceneState.EMPTY
        self.active_model = None
        return TransitionEvent("clear")

    def model_id(self) -> Optional[str]:
        return self._pending_model if self.state == SceneState.DISSOLVING else self.active_model
