"""Thread-safe non-blocking camera grabber (DSHOW backend, Windows)."""

from __future__ import annotations

import threading
import time
from typing import Optional
import cv2
import numpy as np


class CameraGrabber:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._w = width
        self._h = height
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._open(index)

    def _open(self, index: int) -> bool:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frm = False, None
        for _ in range(3):
            ok, frm = cap.read()
        if ok and frm is not None:
            self._cap = cap
            self._index = index
            return True
        cap.release()
        return False

    @property
    def index(self) -> int:
        return self._index

    def switch(self) -> int:
        nxt = 1 if self._index == 0 else 0
        with self._lock:
            if not self._open(nxt):
                self._open(self._index)
        return self._index

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            if self._cap is not None and self._cap.isOpened():
                ok, frm = self._cap.read()
                if ok and frm is not None:
                    with self._lock:
                        self._frame = frm
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._cap is not None:
            self._cap.release()
