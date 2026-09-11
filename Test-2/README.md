# Test-2 — Modular Hand-Controlled 3D Particle Mesh System

> **Milestone 2** · A complete architectural overhaul of Test-1. Fully decomposed into independent packages, upgraded to GPU-accelerated particle rendering via ModernGL/GLSL, and backed by a comprehensive unit test suite.

---

## What It Does

Tracks 3D hand orientation via MediaPipe Tasks, classifies gestures with temporal debouncing through a deterministic state machine, and renders a perspective-projected 3D wireframe cube with a dynamic procedural particle mesh flowing inside — all in real time.

---

## Architecture

```
Test-2/
├── config/
│   ├── __init__.py
│   └── settings.py           # Centralized dataclasses (Display, Viewport, Particles, Gestures)
├── core/
│   ├── __init__.py
│   ├── camera.py             # Thread-safe non-blocking camera grabber (DSHOW backend)
│   └── state_machine.py      # Debounced deterministic gesture state machine
├── vision/
│   ├── __init__.py
│   ├── gesture_detector.py   # MediaPipe HandLandmarker + geometric finger classifier
│   ├── hand_tracker.py       # Hand landmark pipeline wrapper
│   └── pose_estimator.py     # 3D Euler/Quaternion orientation & spatial vector solver
├── math3d/
│   ├── __init__.py
│   ├── transforms.py         # Vectorized Euler rotation, perspective projection, EMA smoothing
│   └── spatial_grid.py       # O(N) 3D uniform spatial voxel grid for neighbor queries
├── simulation/
│   ├── __init__.py
│   ├── fields.py             # Procedural trigonometric harmonic curl flow field
│   └── particle_system.py    # Vectorized particle physics with boundary bouncing
├── rendering/
│   ├── __init__.py
│   ├── cube_renderer.py      # Anti-aliased wireframe cube with soft glow pass
│   ├── mesh_renderer.py      # Depth-attenuated particle nodes & proximity-blended links
│   └── renderer_2d.py        # Glassmorphic HUD pill overlay
├── gpu/
│   ├── __init__.py
│   ├── particle_system.py    # GPU particle system (ModernGL transform feedback)
│   └── shaders/
│       ├── particle.vert     # Particle vertex shader
│       ├── particle.frag     # Particle fragment shader
│       └── update.vert       # Transform feedback update shader
├── tests/
│   ├── __init__.py
│   ├── test_math3d.py        # Geometric, projection, and depth-guard unit tests
│   ├── test_gesture_rules.py # Synthetic landmark classification & debounce tests
│   ├── test_simulation.py    # Vectorized flow fields and spatial grid tests
│   ├── test_rendering.py     # Render pipeline pass validation
│   └── test_benchmark.py     # Performance benchmarks
├── main.py                   # High-performance application loop and orchestration
├── run.bat                   # Windows launch script
└── requirements.txt
```

---

## Key Improvements Over Test-1

| Area | Test-1 | Test-2 |
| :--- | :--- | :--- |
| Architecture | Flat files, mixed concerns | Full package decomposition (6 modules) |
| Particle rendering | CPU NumPy | GPU via ModernGL + GLSL transform feedback |
| Neighbor queries | O(N²) brute force | O(N) uniform voxel spatial grid |
| Gesture handling | Simple threshold | Debounced state machine with hysteresis |
| Pose estimation | Basic 2D | Full 3D Euler/Quaternion + EMA smoothing |
| Testing | Single script | Comprehensive unittest suite (5 modules) |

---

## Gesture Interaction

| Gesture | Finger Pose | State | Visual Output |
| :--- | :--- | :--- | :--- |
| **Fist** | All fingers curled | `FIST` (Standby) | Graphics fade out |
| **Thumb + Index** | Thumb & index extended | `THUMB_INDEX` (Cube) | 3D wireframe cube tracks palm orientation |
| **Open Palm** | All 5 fingers extended | `OPEN_PALM` (Full Sim) | Cube + dynamic particle mesh flowing inside |

---

## Controls

| Key | Action |
| :---: | :--- |
| **`ESC` / `Q`** | Exit |
| **`F`** | Toggle fullscreen |
| **`R`** | Reset / reseed particle distribution |

---

## Running

```powershell
# From the Test-2 folder:
pip install -r requirements.txt
python main.py

# Or double-click:
run.bat
```

---

## Running the Test Suite

```powershell
python -m unittest discover tests
```

---

## Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | ≥ 4.9.0 | Camera capture & 2D compositing |
| `mediapipe` | ≥ 1.0.0 | Hand landmark detection (Tasks API) |
| `numpy` | ≥ 1.26.0 | Vectorized physics & math |
| `moderngl` | ≥ 5.8.0 | GPU particle rendering (OpenGL) |
| `pygame` | ≥ 2.5.0 | Window management & GPU context |

---

> ← [Test-1](../Test-1/README.md) &nbsp;|&nbsp; [Back to root](../README.md) &nbsp;|&nbsp; Next → [Test-3](../Test-3/README.md)
