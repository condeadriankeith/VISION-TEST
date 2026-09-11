# Test-1 — Hand Gesture 3D Hologram

> **Milestone 1** · The original prototype. Turns your webcam into an interactive holographic display using MediaPipe hand tracking and a pure-NumPy software 3D renderer.

---

## What It Does

Detects your hand in real time, classifies open vs. closed palm gestures, and spawns **three animated 3D cubes** that float above your palm. The cubes physically respond to hand motion, collide with each other, and smoothly vanish when you close your fist.

---

## Features

- **21-point 3D Hand Skeleton** — MediaPipe HandLandmarker tracks anatomical landmarks in full 3D. Optional glowing skeleton overlay (`S` key).
- **Blinn-Phong Shaded Cubes** — Software-rasterized solid grey cubes with directional key light, soft ambient fill, and specular highlights. Smooth beveled edges give a manufactured, tangible look.
- **Spring-Damper Hovering** — Cubes are physically tethered to procedural slots above your palm. Whip or tilt your hand and they physically lag behind with mass and momentum.
- **Pairwise Elastic Collisions** — Cubes push apart with restitution impulse and tangential spin torque so they never clip or overlap.
- **Smooth Spawn / Vanish** — Open palm: cubes push outward and settle into formation. Closed fist: cubes retract inward and disappear.
- **4 Switchable Materials** — Studio Matte Grey, Gunmetal Slate, Platinum Ceramic, Obsidian Graphite (`C` key).

---

## File Structure

```
Test-1/
├── main.py             # Application loop, camera capture, HUD, keyboard controls
├── hand_tracker.py     # MediaPipe HandLandmarker wrapper, gesture classifier
├── cube_renderer.py    # 3D geometry, perspective projection, depth sorting, shading
├── config.py           # Physics constants, color themes, display settings
├── physics.py          # Spring-damper, collision resolution, rigid body dynamics
├── camera.py           # Camera capture and device switching
├── spatial_math.py     # 3D vector / rotation utility functions
├── test_pipeline.py    # Automated verification test suite
├── requirements.txt    # Python dependencies
└── run.bat             # Windows one-click launcher
```

---

## Installation & Setup

**Requirements:** Python 3.10+ · A working webcam

```powershell
pip install -r requirements.txt
```

> The `hand_landmarker.task` MediaPipe model file must be present in this folder.
> If missing, download it from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

---

## Running

```powershell
# Option 1 — terminal
python main.py

# Option 2 — Windows launcher
run.bat
```

---

## Controls

| Key | Action |
| :---: | :--- |
| **Open Palm** | Spawns and expands the 3 hovering 3D cubes |
| **Closed Fist** | Smoothly shrinks and vanishes the cubes |
| **`V`** | Switch camera (cycles between detected devices) |
| **`S`** | Toggle virtual skeleton overlay |
| **`C`** | Cycle material themes |
| **`Q` / `Esc`** | Exit |

---

## Automated Tests

```powershell
python test_pipeline.py
```

---

## Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | ≥ 4.9.0 | Camera capture & image rendering |
| `mediapipe` | ≥ 0.10.14 | Hand landmark detection |
| `numpy` | ≥ 1.26.0 | 3D math, projection, physics |

---

> ← [Back to root](../README.md) &nbsp;|&nbsp; Next milestone → [Test-2](../Test-2/README.md)
