# Hand Gesture 3D Hologram Application

A complete, high-performance Python application that turns your webcam into an interactive holographic display using **MediaPipe** and **OpenCV**. The application maps an invisible 3D virtual hand skeleton, detects open vs. closed palm gestures, and spawns three animated 3D hovering cubes above your palm that smoothly scale down and vanish when you close your fist.

---

## Key Features

- **Real-Time Hand Detection & Virtual Skeleton**: Tracks 21 anatomical landmarks per hand in 3D. The skeleton operates as an invisible virtual tracker, with an optional toggle (`S` key) to reveal a glowing skeleton overlay.
- **True 3D Solid Grey Rendered Cubes**:
  - **Blinn-Phong Lighting**: Key directional light + soft ambient fill light + crisp specular highlights.
  - **Smooth Beveled Edges**: Front-facing edges are highlighted with subtle micro-bevels for a tangible manufactured appearance.
  - **No Palm Tethers**: Floating freely without connecting lines or clutter.
- **Newtonian Rigid-Body Physics & Collisions**:
  - **Pairwise Elastic Collisions**: When cubes come into contact, they push apart with restitution impulse and tangential spin torque so they never clip or overlap.
  - **Spring-Damper Hovering**: Cubes are gently tethered by physical spring-dampers to procedural slots above your palm, swaying realistically with hand momentum.
  - **Procedural Inertia**: If you whip or tilt your hand, cubes physically lag behind with mass and momentum.
- **Smooth Spawn & Vanish Transitions**:
  - **Open Palm**: Cubes physically push outward from your palm and settle into formation.
  - **Closed Palm (Fist)**: Cubes smoothly retract inward and vanish completely.
- **4 Switchable Grey Materials**:
  - Studio Matte Grey, Gunmetal Slate, Platinum Ceramic, Obsidian Graphite (switch via `C` key).

---

## File Structure

```
VISION-TEST/
├── main.py              # Application loop, camera capture, HUD, and keyboard controls
├── hand_tracker.py       # MediaPipe HandLandmarker wrapper, virtual skeleton, gesture classifier
├── cube_renderer.py      # 3D geometry math, perspective projection, depth sorting, shading
├── config.py             # Configuration parameters, physics constants, color themes
├── test_pipeline.py      # Automated verification test suite
├── requirements.txt      # Python dependencies (opencv-python, mediapipe, numpy)
└── hand_landmarker.task  # Downloaded MediaPipe hand landmarker model
```

---

## Installation & Setup

1. **Activate your Python 3.10+ / 3.12 environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the application**:
   - Double-click [`run.bat`](file:///c:/Users/conde/Downloads/VISION-TEST/run.bat) (Windows 1-click launcher)
   - Or run from terminal:
     ```bash
     python main.py
     ```

---

## Interactive Controls

| Key | Action |
| :---: | :--- |
| **Open Palm** | Spawns and expands the 3 hovering 3D cubes above your palm. |
| **Close Palm / Fist** | Smoothly shrinks and vanishes the cubes. |
| **`V`** | Switch Camera (cycles between available camera devices live). |
| **`S`** | Toggle Virtual Skeleton visibility (Invisible $\leftrightarrow$ Glowing Cyberpunk Skeleton). |
| **`C`** | Cycle Color Themes (Cyber Cyan $\rightarrow$ Neon Violet $\rightarrow$ Matrix Emerald $\rightarrow$ Solar Amber). |
| **`Q` / `Esc`** | Cleanly exit the program and release the camera. |

---

## Automated Verification

To run the automated verification test suite:
```bash
python test_pipeline.py
```
