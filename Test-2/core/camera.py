"""Threaded, non-blocking webcam video grabber.

Decouples high-frequency camera I/O from computation/rendering threads using a
thread-safe single-frame buffer, frame-dropping, and device fallback.
"""

import sys
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import DisplayConfig


class ThreadedCamera:
    """Non-blocking background video capture reader."""

    def __init__(
        self,
        display_config: DisplayConfig = DisplayConfig(),
        preferred_device_id: int = 0,
    ) -> None:
        """Initialize camera parameters.

        Args:
            display_config: Desired resolution and target frame rate.
            preferred_device_id: Camera index (usually 0, fallback to 1).
        """
        self._cfg: DisplayConfig = display_config
        self._device_id: int = preferred_device_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._has_new_frame: bool = False
        self._frame_count: int = 0

    def start(self) -> "ThreadedCamera":
        """Open camera hardware and launch capture worker thread.

        Returns:
            Self instance for chaining.
        """
        self._init_capture()

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, name="CameraCaptureThread", daemon=True)
        self._thread.start()

        # Wait briefly for first frame to arrive
        for _ in range(50):
            if self._latest_frame is not None:
                break
            time.sleep(0.02)

        return self

    def _init_capture(self) -> None:
        """Attempt to open primary camera with DSHOW backend, falling back to alternative device ID."""
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

        candidates = [self._device_id, 1 if self._device_id == 0 else 0]
        opened = False

        for dev_id in candidates:
            cap = cv2.VideoCapture(dev_id, backend)
            if cap.isOpened():
                # Configure resolution and target frame rate
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.cam_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.cam_height)
                cap.set(cv2.CAP_PROP_FPS, self._cfg.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal driver buffer to minimize latency

                # Test grab
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self._cap = cap
                    self._device_id = dev_id
                    opened = True
                    print(f"[INFO] Camera opened successfully on device {dev_id} ({self._cfg.cam_width}x{self._cfg.cam_height} @ {self._cfg.target_fps}fps)")
                    break
                else:
                    cap.release()

        if not opened:
            print("[WARN] No physical webcam detected. Generating synthetic test frame feed.")
            self._cap = None

    def _capture_worker(self) -> None:
        """Continuous background thread worker grabbing frames."""
        synthetic_t = 0.0
        while self._running:
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    # Mirror frame horizontally for natural interactive interaction
                    mirrored = cv2.flip(frame, 1)
                    with self._lock:
                        self._latest_frame = mirrored
                        self._has_new_frame = True
                        self._frame_count += 1
                else:
                    time.sleep(0.005)
            else:
                # Synthetic dark canvas fallback if no camera hardware is attached
                time.sleep(1.0 / max(1, self._cfg.target_fps))
                synthetic_t += 0.016
                h, w = self._cfg.cam_height, self._cfg.cam_width
                synth_frame = np.full((h, w, 3), (12, 14, 16), dtype=np.uint8)
                with self._lock:
                    self._latest_frame = synth_frame
                    self._has_new_frame = True
                    self._frame_count += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest captured frame in a thread-safe manner.

        Returns:
            Tuple of (success, BGR ndarray copy or reference).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            frame = self._latest_frame.copy()
            self._has_new_frame = False
            return True, frame

    def is_running(self) -> bool:
        """Check whether camera thread is active."""
        return self._running

    def stop(self) -> None:
        """Terminate capture thread and release hardware resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
