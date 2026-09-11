# AR Hand-Controlled 3D Object System (Test-3)

Standalone Python replica of the *"I Control 3D Objects With My HANDS!"* AR pipeline:
dual-hand gesture interaction, hand-anchored 3D objects, and particle dissolution —
**zero external assets** (procedural meshes, OpenCV + MediaPipe + NumPy only).

## Dual-Hand Roles

| Hand | Role | Mechanics |
| :--- | :--- | :--- |
| Left | Spatial anchor & orientation | Palm centroid `C_hand` + hand-frame rotation `R_hand` drives the wireframe bounding box |
| Right | Gesture trigger & state machine | Finger-extension archetypes switch models / dissolve / clear |
| Single hand | Fallback | One visible hand drives **both** anchor and trigger |

Assignment uses MediaPipe handedness with a screen-X (`x < W/2` = left) fallback.

## Gesture Mapping (exact video reference)

| Trigger gesture | Finger pose | Result |
| :--- | :--- | :--- |
| Index pointing | Index only | Blooming Lilies / Flowers (petals bloom + orbit) |
| Fist | All curled | Red Dragon / Phoenix (flapping wings) |
| `L` | Thumb + index at ~90° | Blue Morpho Butterfly (articulated dual wings) |
| Peace | Index + middle | Cosmic Bonsai Tree (branching trunk + canopy) |
| Open palm | All 5 extended | **Particle dissolve** — active object shatters into curl-turbulence particles |
| Dual fists | Both hands closed | **Scene clear** — box + objects collapse |

## Architecture

```
Live Camera (720p, mirrored) -> MediaPipe Hands (x2) -> Left anchor / Right trigger
 -> Bounding-Box Transform + Transition Controller -> 3D Render -> Composited AR
```

* `vision/` — dual-hand tracker, geometric classifier, pose estimator (`u=L9-L0`, `v=L5-L17`, `n=u×v/‖u×v‖`)
* `math3d/` — `v_world = R_hand·(v_local·s) + C_hand`, perspective projection, EMA
* `models/` — procedural flowers / dragon / butterfly / tree
* `simulation/` — `P+=V·dt`, `V=V_radial·e^(-γt)+V_curl·β`, `α=max(0,1-t/T)`
* `rendering/` — depth-sorted edges, glowing double-edge box + 8 nodes, HUD + control bar
* `core/` — threaded camera, debounced state machine

## Run

```powershell
# From the Test-3 folder:
pip install -r requirements.txt
python main.py
# Or double-click:
run.bat
```

First run downloads `hand_landmarker.task` automatically (also reuses
`../Test-1/hand_landmarker.task` or `../Test-2/hand_landmarker.task` if present).

Keys: `ESC/Q` quit · `F` fullscreen · `C` switch camera · `R/X` clear scene.

## Tests

```powershell
python -m unittest discover tests
```

## TouchDesigner Port

To recreate inside TouchDesigner: `Video Device In TOP` → `MediaPipe CHOP` (or Script CHOP
running `mediapipe.hands`) → Geometry COMP `tx/ty/tz/rx/ry/rz` from CHOP channels →
`Switch SOP` on gesture IDs for FBX/OBJ models → `Point Transform SOP` + noise for the
dissolve burst. This repo's `config/settings.py` constants map 1:1 to those CHOP ranges.
