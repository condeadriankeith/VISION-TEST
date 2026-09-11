# VISION-TEST

> A personal learning repository documenting an iterative journey through **computer vision**, **hand tracking**, **3D rendering**, and **physics simulation** using Python, OpenCV, and MediaPipe.

Each folder represents a distinct milestone — from a monolithic prototype to a fully modular, GPU-accelerated system and finally a dual-hand AR pipeline.

---

## 📁 Project Structure

```
VISION-TEST/
├── Test-1/     ← Prototype: Monolithic hand-tracking + 3D cube hologram
├── Test-2/     ← Modular refactor: GPU particles, spatial grids, full test suite
└── Test-3/     ← AR pipeline: Dual-hand control, procedural models, particle dissolve
```

---

## 🧪 Milestone Overview

### [Test-1](./Test-1/) — Hand Gesture 3D Hologram
The first working prototype. A single-file-per-concern application that detects hand gestures via MediaPipe and renders three Blinn-Phong shaded 3D cubes hovering above your palm using pure NumPy math. Includes spring-damper physics, pairwise elastic collisions, and smooth spawn/vanish transitions.

**Stack:** `opencv-python` · `mediapipe` · `numpy`

---

### [Test-2](./Test-2/) — Modular Particle Mesh System
A complete architectural overhaul. Fully decomposed into `vision/`, `rendering/`, `gpu/`, `simulation/`, `math3d/`, and `core/` packages. Adds a GPU-accelerated particle system via ModernGL/GLSL transform feedback, an O(N) voxel spatial grid, and a comprehensive unit test suite.

**Stack:** `opencv-python` · `mediapipe` · `numpy` · `moderngl` · `pygame`

---

### [Test-3](./Test-3/) — AR Dual-Hand 3D Object System
The most advanced iteration. Implements a dual-hand AR pipeline where the **left hand** anchors a 3D bounding box in space and the **right hand** triggers gesture-based object switching. Procedural 3D models (flowers, dragon, butterfly, bonsai tree) dissolve into curl-turbulence particle bursts on command.

**Stack:** `opencv-python` · `mediapipe` · `numpy`

---

## 🚀 Quick Start

Each test is self-contained. Navigate into the folder you want and run:

```powershell
pip install -r requirements.txt
python main.py
# or double-click run.bat
```

> **Note:** `Test-1/hand_landmarker.task` is the MediaPipe model file. Test-2 and Test-3 will reuse it automatically if found in a sibling folder.

---

## 🛠️ Prerequisites

- Python **3.10+** (3.12 recommended)
- A working **webcam**
- Windows (all `run.bat` launchers are Windows-targeted; `main.py` runs cross-platform)

---

## 📚 Concepts Explored

| Concept | Where |
| :--- | :--- |
| MediaPipe HandLandmarker (21-point 3D skeleton) | All tests |
| Perspective projection & depth sorting | Test-1, Test-2, Test-3 |
| Blinn-Phong shading (software rasterizer) | Test-1 |
| Spring-damper physics & elastic collisions | Test-1 |
| Temporal gesture debouncing (state machine) | Test-2, Test-3 |
| GPU particle system (GLSL transform feedback) | Test-2 |
| O(N) uniform voxel spatial grid | Test-2 |
| Quaternion / Euler pose estimation | Test-2, Test-3 |
| Procedural 3D mesh generation | Test-3 |
| Dual-hand role assignment & AR compositing | Test-3 |
| Curl-turbulence particle dissolution | Test-3 |

---

## 🔗 Repository

**GitHub:** [github.com/condeadriankeith/VISION-TEST](https://github.com/condeadriankeith/VISION-TEST)
