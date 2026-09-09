"""High-performance threaded camera stream manager.

Runs camera frame acquisition on a dedicated background thread with minimal buffering
(CAP_PROP_BUFFERSIZE=1) to eliminate blocking I/O latency and input lag.
"""

import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


class ThreadedCamera:
    """Non-blocking threaded video capture stream."""

    def __init__(self, capture: cv2.VideoCapture, index: int) -> None:
        """Initialize threaded camera.

        Args:
            capture: An already opened cv2.VideoCapture instance.
            index: The device index of this camera.
        """
        self.cap: cv2.VideoCapture = capture
        self.index: int = index
        self.stopped: bool = False
        self.lock: threading.Lock = threading.Lock()

        # Frame cache
        self.has_frame: bool = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0

        # Prime with initial frame
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.has_frame = True
            self.latest_frame = frame
            self.frame_id = 1

        # Launch background acquisition thread
        self.thread: threading.Thread = threading.Thread(
            target=self._capture_worker,
            name=f"CameraThread-{index}",
            daemon=True,
        )
        self.thread.start()

    def _capture_worker(self) -> None:
        """Dedicated background loop constantly reading the latest hardware frame."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame
                self.has_frame = True
                self.frame_id += 1

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Instantly returns the latest available frame without blocking.

        Returns:
            Tuple of (success, frame, frame_id).
        """
        with self.lock:
            if not self.has_frame or self.latest_frame is None:
                return False, None, 0
            # Return copy of the frame to prevent race conditions during write
            return True, self.latest_frame.copy(), self.frame_id

    def release(self) -> None:
        """Stops the worker thread and releases the hardware camera device."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=0.8)
        if self.cap.isOpened():
            self.cap.release()


class CameraManager:
    """Manages active camera detection, selection, and live switching."""

    @staticmethod
    def open_camera(target_index: int) -> ThreadedCamera:
        """Explicitly opens a specific camera index using DirectShow."""
        print(f"[CameraManager] Opening camera index {target_index} via DirectShow...")
        for b_name, backend in [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]:
            cap = cv2.VideoCapture(target_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                for _ in range(3):
                    ret, _ = cap.read()
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"[CameraManager] Successfully connected to Camera {target_index} ({w}x{h}).")
                    return ThreadedCamera(cap, target_index)
                cap.release()
        raise RuntimeError(f"Could not open camera index {target_index}.")

    @staticmethod
    def open_best_camera(preferred_index: Optional[int] = None) -> ThreadedCamera:
        """Scans connected devices and opens the best camera with a live, visible video feed.

        Prioritizes Camera 0 (main webcam) and preferred index.
        """
        candidate_indices = [0, 1, 2]
        if preferred_index is not None and preferred_index in candidate_indices:
            candidate_indices.remove(preferred_index)
            candidate_indices.insert(0, preferred_index)

        print("[CameraManager] Scanning for active camera streams...")

        best_cap: Optional[cv2.VideoCapture] = None
        best_score: float = -1.0
        best_idx: int = candidate_indices[0]

        for cam_idx in candidate_indices:
            for b_name, backend in [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]:
                cap = cv2.VideoCapture(cam_idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Set zero-latency buffer size and resolution
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

                # Warmup frames
                test_frame = None
                for _ in range(4):
                    ret, test_frame = cap.read()
                    if not ret:
                        break

                if test_frame is not None and test_frame.size > 0:
                    mean_val = float(np.mean(test_frame))
                    std_val = float(np.std(test_frame))
                    h, w = test_frame.shape[:2]

                    score = std_val if mean_val > 4.0 else 0.01

                    print(
                        f"[CameraManager] Found Camera {cam_idx} via {b_name} "
                        f"({w}x{h}, brightness={mean_val:.1f}, contrast={std_val:.1f})"
                    )

                    # If this is the preferred index and it has real imagery, select immediately
                    if preferred_index is not None and cam_idx == preferred_index and score > 5.0:
                        if best_cap is not None:
                            best_cap.release()
                        print(f"[CameraManager] Selected preferred Camera {cam_idx}.")
                        return ThreadedCamera(cap, cam_idx)

                    if score > best_score:
                        if best_cap is not None:
                            best_cap.release()
                        best_cap = cap
                        best_score = score
                        best_idx = cam_idx
                        if score > 5.0:
                            break
                    else:
                        cap.release()
                else:
                    cap.release()

            if best_cap is not None and best_score > 5.0:
                break

        if best_cap is not None:
            print(f"[CameraManager] Active camera selected: Index {best_idx}.")
            return ThreadedCamera(best_cap, best_idx)

        raise RuntimeError(
            "Failed to acquire video stream from any connected camera.\n"
            "Please check Windows Settings -> Privacy & security -> Camera and ensure permissions are granted."
        )
