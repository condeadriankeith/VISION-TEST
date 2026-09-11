"""Centralized configuration dataclasses and runtime constants.

Defines typed, immutable configuration schemas for display settings,
3D viewport geometry, particle physics parameters, and gesture classification thresholds.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DisplayConfig:
    """Display window and video capture specifications."""
    window_name: str = "3D Particle Mesh - Hand Tracking"
    cam_width: int = 1280
    cam_height: int = 720
    target_fps: int = 60
    enable_fullscreen: bool = False


@dataclass(frozen=True)
class Viewport3DConfig:
    """3D perspective projection and cube geometric properties.

    Cube is anchored to the left side of the viewport. Position is fixed; only rotation changes.
    """
    # Fixed left-side anchor: ~27% from left edge, centered vertically
    cube_screen_x: int = 340
    cube_screen_y: int = 360
    # Larger cube occupying most of the left-side viewport area
    cube_size: float = 280.0
    focal_length: float = 650.0
    camera_distance: float = 700.0
    near_clip_z: float = 1.0


@dataclass(frozen=True)
class ParticleConfig:
    """Vectorized particle physics and dynamic network mesh settings.

    Cloud-formation style: high particle density, short connection range, slow damping.
    """
    max_particles: int = 550
    # Tighter connection distance creates cloud-cluster topology
    connection_distance: float = 38.0
    max_neighbors_per_particle: int = 5
    flow_speed: float = 0.8
    bounding_ratio: float = 0.86
    # High damping creates slow-drifting cloud movement
    damping: float = 0.94
    curl_scale: float = 0.018
    node_base_radius: float = 1.8


@dataclass(frozen=True)
class GestureThresholds:
    """Geometric thresholds and debounce settings for gesture state machine."""
    finger_extension_ratio: float = 1.15
    # Pinch: thumb tip to index tip distance threshold (normalized units)
    pinch_threshold: float = 0.065
    smoothing_alpha_rot: float = 0.18
    smoothing_alpha_pos: float = 0.25
    debounce_frames: int = 4
    pitch_scale: float = 1.2
    yaw_scale: float = 1.2


@dataclass(frozen=True)
class ColorPalette:
    """Minimalist, curated futuristic BGR color palette for OpenCV rendering."""
    # Wireframe Cube — clean, thin, bright lines
    cube_edge_core: Tuple[int, int, int] = (240, 230, 110)       # Bright cold-white/teal (BGR)
    cube_edge_glow: Tuple[int, int, int] = (160, 120, 30)        # Very subtle warm glow
    cube_corner: Tuple[int, int, int] = (255, 255, 255)          # Accent White

    # Particles & Cloud Mesh — cool cyan/blue tones
    particle_node: Tuple[int, int, int] = (255, 210, 80)         # Bright cool cyan nodes
    mesh_edge: Tuple[int, int, int] = (190, 140, 40)             # Dim translucent cloud links

    # UI Overlay
    hud_bg: Tuple[int, int, int] = (20, 22, 24)                  # Dark Glassmorphism Pill Background
    hud_border: Tuple[int, int, int] = (70, 75, 80)              # Subtle border
    hud_text_primary: Tuple[int, int, int] = (245, 245, 245)     # Crisp Monospace Off-White
    hud_text_accent: Tuple[int, int, int] = (230, 200, 80)       # Cyan Accent for FPS


@dataclass(frozen=True)
class SystemSettings:
    """Master aggregate configuration container."""
    display: DisplayConfig = DisplayConfig()
    viewport: Viewport3DConfig = Viewport3DConfig()
    particle: ParticleConfig = ParticleConfig()
    gestures: GestureThresholds = GestureThresholds()
    colors: ColorPalette = ColorPalette()
