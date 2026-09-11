# Hand-Controlled 3D Particle Mesh System (Test-2)

A modular, high-performance interactive vision and physics system. Tracks 3D hand orientation via MediaPipe Tasks, classifies gestures with temporal debouncing, and renders a 3D perspective-projected wireframe cube and procedural particle mesh in real time.

---

## Architecture Overview

```
Test-2/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized dataclasses (Display, Viewport, Particles, Gestures)
├── core/
│   ├── __init__.py
│   ├── camera.py            # Thread-safe non-blocking camera grabber with DSHOW backend
│   └── state_machine.py     # Debounced deterministic gesture state machine
├── vision/
│   ├── __init__.py
│   ├── gesture_detector.py  # MediaPipe HandLandmarker + geometric finger classifier
│   └── pose_estimator.py    # 3D Euler/Quaternion orientation & spatial vector solver
├── math3d/
│   ├── __init__.py
│   ├── transforms.py        # Vectorized Euler rotation, perspective projection, EMA smoothing
│   └── spatial_grid.py      # O(N) 3D uniform spatial voxel grid for neighbor queries
├── simulation/
│   ├── __init__.py
│   ├── fields.py            # Procedural trigonometric harmonic curl flow field
│   └── particle_system.py   # Vectorized particle physics with boundary bouncing
├── rendering/
│   ├── __init__.py
│   ├── cube_renderer.py     # Anti-aliased wireframe cube with soft glow pass
│   ├── mesh_renderer.py     # Depth-attenuated particle nodes & proximity-blended links
│   └── renderer_2d.py       # Minimalist upper-right glassmorphic HUD pill
├── tests/
│   ├── test_math3d.py       # Geometric, projection, and depth-guard unit tests
│   ├── test_gesture_rules.py# Synthetic landmark classification & debounce tests
│   ├── test_simulation.py   # Vectorized flow fields and spatial grid tests
│   └── test_rendering.py    # Render pipeline pass validation
├── main.py                  # High-performance application loop and orchestration
├── run.bat                  # Convenient Windows launch script
└── requirements.txt
```

---

## Interaction Gestures

| Gesture | Finger Pose | Interaction State | Visual Output |
| :--- | :--- | :--- | :--- |
| **FIST** | All fingers curled / folded | `FIST` (Standby / Idle) | Standby mode. Graphics smoothly fade out. |
| **THUMB+INDEX** | Thumb & Index extended, other fingers folded | `THUMB_INDEX` (Active Cube) | 3D Wireframe Cube tracks palm orientation and hover position. |
| **OPEN PALM** | All 5 fingers extended | `OPEN_PALM` (Full Simulation) | 3D Wireframe Cube active with dynamic procedural particle mesh flowing inside. |

---

## Key Controls

- **`ESC` / `Q`**: Clean application exit
- **`F`**: Toggle Fullscreen mode
- **`R`**: Reset / reseed particle distribution

---

## Running the Application

```powershell
# From the Test-2 folder:
python main.py

# Or double-click:
run.bat
```

## Running the Test Suite

```powershell
python -m unittest discover tests
```
