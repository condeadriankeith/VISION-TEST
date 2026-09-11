# Test-3 — AR Dual-Hand 3D Object System

> **Milestone 3** · The most advanced iteration. A dual-hand AR pipeline where your left hand anchors a 3D bounding box in world space and your right hand triggers gesture-based object switching, with procedural meshes that dissolve into curl-turbulence particle bursts.

---

## What It Does

Inspired by the *"I Control 3D Objects With My Hands"* AR pipeline. Detects two hands simultaneously, assigns spatial and trigger roles, and renders procedural 3D objects (flowers, dragon, butterfly, bonsai tree) anchored to the hand-tracked bounding box in the camera frame — using zero external 3D assets.

---

## Dual-Hand Role System

| Hand | Role | Mechanics |
| :--- | :--- | :--- |
| **Left** | Spatial anchor & orientation | Palm centroid `C_hand` + hand-frame rotation `R_hand` drives the wireframe bounding box |
| **Right** | Gesture trigger & state machine | Finger-extension archetypes switch models, dissolve, or clear the scene |
| **Single hand** | Fallback mode | One visible hand drives **both** anchor and trigger simultaneously |

> Hand assignment uses MediaPipe handedness with a screen-X (`x < W/2` = left) fallback.

---

## Gesture Mapping

| Gesture | Finger Pose | Result |
| :--- | :--- | :--- |
| **Index point** | Index finger only | 🌸 Blooming Lilies — petals bloom and orbit |
| **Fist** | All fingers curled | 🐉 Red Dragon / Phoenix — procedural flapping wings |
| **`L` shape** | Thumb + index at ~90° | 🦋 Blue Morpho Butterfly — articulated dual wings |
| **Peace** | Index + middle extended | 🌿 Cosmic Bonsai Tree — branching trunk + canopy |
| **Open palm** | All 5 fingers extended | 💥 Particle dissolve — active object shatters into curl-turbulence particles |
| **Dual fists** | Both hands closed | ✖️ Scene clear — bounding box + objects collapse |

---

## Architecture

```
Test-3/
├── config/
│   ├── __init__.py
│   └── settings.py       # Constants (maps 1:1 to TouchDesigner CHOP ranges)
├── core/
│   ├── __init__.py
│   ├── camera.py         # Threaded camera capture with device switching
│   └── state_machine.py  # Debounced dual-hand gesture state machine
├── vision/
│   ├── __init__.py
│   ├── hand_tracker.py   # Dual-hand MediaPipe pipeline + handedness assignment
│   ├── gestures.py       # Geometric finger-extension classifier
│   └── pose_estimator.py # u=L9-L0, v=L5-L17, n=u×v/‖u×v‖ hand-frame solver
├── math3d/
│   ├── __init__.py
│   └── transforms.py     # v_world = R_hand·(v_local·s) + C_hand, EMA, projection
├── models/
│   ├── __init__.py
│   └── procedural.py     # Flowers, dragon/phoenix, butterfly, bonsai tree generators
├── simulation/
│   ├── __init__.py
│   └── particles.py      # P+=V·dt, V=V_radial·e^(-γt)+V_curl·β, α=max(0,1-t/T)
├── rendering/
│   ├── __init__.py
│   └── renderer.py       # Depth-sorted edges, glowing double-edge box + 8 nodes, HUD
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py  # Integration test for the full dual-hand pipeline
├── main.py               # Application loop and orchestration
├── run.bat               # Windows launch script
└── requirements.txt
```

---

## Key Advances Over Test-2

| Area | Test-2 | Test-3 |
| :--- | :--- | :--- |
| Hand count | Single hand | Dual-hand with role assignment |
| 3D objects | Cube + particle mesh | 4 procedural organic models |
| Dissolution | N/A | Curl-turbulence particle burst |
| Anchor system | Hand position only | Full 6-DOF bounding box (R_hand + C_hand) |
| Model assets | None | Fully procedural (zero external files) |

---

## Pipeline Flow

```
Live Camera (720p, mirrored)
  → MediaPipe Hands (×2)
  → Left anchor / Right trigger assignment
  → Bounding-Box Transform + Transition Controller
  → 3D Object Render
  → AR Composite
```

---

## Running

```powershell
# From the Test-3 folder:
pip install -r requirements.txt
python main.py

# Or double-click:
run.bat
```

> **Model file:** On first run, `hand_landmarker.task` is downloaded automatically.
> It will also reuse `../Test-1/hand_landmarker.task` or `../Test-2/hand_landmarker.task` if present.

---

## Controls

| Key | Action |
| :---: | :--- |
| **`ESC` / `Q`** | Exit |
| **`F`** | Toggle fullscreen |
| **`C`** | Switch camera device |
| **`R` / `X`** | Clear scene |

---

## Running Tests

```powershell
python -m unittest discover tests
```

---

## Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | ≥ 4.9.0 | Camera capture & AR compositing |
| `mediapipe` | ≥ 0.10.0 | Dual-hand landmark detection |
| `numpy` | ≥ 1.26.0 | 3D transforms, simulation, mesh gen |

---

## TouchDesigner Port

To recreate inside TouchDesigner:
1. `Video Device In TOP` → `MediaPipe CHOP` (or Script CHOP running `mediapipe.hands`)
2. Geometry COMP `tx/ty/tz/rx/ry/rz` from CHOP channels
3. `Switch SOP` on gesture IDs for FBX/OBJ models
4. `Point Transform SOP` + noise for the dissolve burst

> `config/settings.py` constants map 1:1 to TouchDesigner CHOP ranges.

---

> ← [Test-2](../Test-2/README.md) &nbsp;|&nbsp; [Back to root](../README.md)
