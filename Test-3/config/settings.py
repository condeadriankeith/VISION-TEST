"""Centralized configuration for Test-3 AR Hand-Controlled 3D Object System."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "AR Hand-Controlled 3D Objects - Test 3"
    cam_width: int = 1280
    cam_height: int = 720
    target_fps: int = 60
    mirror: bool = True


@dataclass(frozen=True)
class Viewport3DConfig:
    # World scale applied to local model before rotation (s in blueprint).
    model_scale: float = 0.62
    # Bounding-box edge length in world units before projection scaling.
    box_size: float = 1.35
    focal_length: float = 700.0
    camera_distance: float = 3.2
    near_clip_z: float = 0.05
    # Anchor smoothing + depth mapping.
    anchor_alpha: float = 0.30
    depth_scale: float = 1.0


@dataclass(frozen=True)
class GestureThresholds:
    finger_extension_ratio: float = 1.15
    thumb_extension_ratio: float = 1.10
    pinch_threshold: float = 0.065
    # L-gesture: angle (deg) between thumb and index direction vectors.
    l_angle_min: float = 60.0
    l_angle_max: float = 120.0
    debounce_frames: int = 5
    # Frames an OPEN_PALM must persist to arm a dissolve.
    dissolve_trigger_frames: int = 3
    smoothing_alpha_rot: float = 0.22
    pitch_scale: float = 1.2
    yaw_scale: float = 1.2


@dataclass(frozen=True)
class ParticleConfig:
    max_particles: int = 1200
    dissolve_time: float = 1.1       # T_dissolve (seconds)
    radial_speed: float = 1.6        # |V_radial| scale
    curl_strength: float = 1.1       # beta
    curl_freq: float = 3.0           # k
    damping_gamma: float = 1.6       # gamma (exp decay)
    gravity: float = -0.25           # slight float upward
    point_radius: int = 2


@dataclass(frozen=True)
class ColorPalette:
    # Bounding box — cyan/white double edge.
    box_glow: Tuple[int, int, int] = (255, 220, 60)
    box_core: Tuple[int, int, int] = (255, 255, 255)
    box_node: Tuple[int, int, int] = (255, 255, 255)
    # Models (BGR).
    flower_petal: Tuple[int, int, int] = (180, 120, 255)
    flower_center: Tuple[int, int, int] = (80, 220, 255)
    flower_leaf: Tuple[int, int, int] = (120, 200, 80)
    dragon_core: Tuple[int, int, int] = (60, 60, 255)
    dragon_glow: Tuple[int, int, int] = (60, 140, 255)
    dragon_wing: Tuple[int, int, int] = (80, 80, 255)
    butterfly_wing: Tuple[int, int, int] = (255, 120, 40)
    butterfly_body: Tuple[int, int, int] = (230, 230, 230)
    tree_trunk: Tuple[int, int, int] = (70, 130, 200)
    tree_leaf: Tuple[int, int, int] = (120, 220, 130)
    particle: Tuple[int, int, int] = (255, 220, 150)
    hud_bg: Tuple[int, int, int] = (24, 26, 30)
    hud_border: Tuple[int, int, int] = (80, 90, 100)
    hud_text: Tuple[int, int, int] = (240, 240, 240)
    hud_accent: Tuple[int, int, int] = (120, 220, 255)


@dataclass(frozen=True)
class SystemSettings:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    viewport: Viewport3DConfig = field(default_factory=Viewport3DConfig)
    gestures: GestureThresholds = field(default_factory=GestureThresholds)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    colors: ColorPalette = field(default_factory=ColorPalette)
