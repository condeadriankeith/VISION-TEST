"""Application configuration settings for Hand Gesture 3D Hologram.

Defines camera properties, detection thresholds, rendering palettes,
and 3D cube physics/animation parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Camera Configuration
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
WINDOW_TITLE: str = "Hand Hologram - 3D Hovering Cubes"

# Optimization Configuration
# Downsample frame for neural network inference while rendering at full 1280x720
INFERENCE_WIDTH: int = 640
INFERENCE_HEIGHT: int = 360
# Exponential moving average filter factor (0 = frozen, 1 = raw jittery)
LANDMARK_SMOOTHING_ALPHA: float = 0.48

# MediaPipe Hand Tracking Configuration
MAX_NUM_HANDS: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.6

# Gesture Recognition Configuration
# Fingers considered for open palm: Index, Middle, Ring, Pinky (+ Thumb)
OPEN_PALM_FINGER_THRESHOLD: int = 4   # At least 4 extended fingers = Open
CLOSED_PALM_FINGER_THRESHOLD: int = 1 # At most 1 extended finger = Closed
GESTURE_SMOOTHING_FRAMES: int = 5     # Temporal smoothing buffer length

# 3D Camera Projection Configuration
FOCAL_LENGTH: float = 950.0            # Camera focal length in pixels
CAMERA_BASELINE_DEPTH: float = 850.0   # Reference camera distance in screen units

# 3D Lighting & PBR Shading Configuration
KEY_LIGHT_DIR: Tuple[float, float, float] = (0.45, -0.80, -0.55)    # Primary directional key light
FILL_LIGHT_DIR: Tuple[float, float, float] = (-0.40, 0.60, -0.40)   # Soft ambient fill
SPECULAR_POWER: float = 32.0                                        # Blinn-Phong shininess
SPECULAR_INTENSITY: float = 0.75                                    # Highlight strength
FRESNEL_POWER: float = 2.5                                          # Rim sheen grazing exponent
FRESNEL_INTENSITY: float = 0.38                                     # Rim sheen strength
AMBIENT_LIGHT_DEFAULT: float = 0.22                                 # Fallback ambient level
ENV_LIGHT_ADAPTATION_RATE: float = 0.15                             # Smooth adaptation to webcam ambient

# Contact Shadow Configuration (Disabled to eliminate dark pulsing artifacts)
SHADOW_ENABLED: bool = False
SHADOW_MAX_OPACITY: float = 0.0
SHADOW_MIN_OPACITY: float = 0.0
SHADOW_BLUR_MIN: int = 9
SHADOW_BLUR_MAX: int = 35
SHADOW_OFFSET_FACTOR: float = 0.25

# Palm Emergence & Retraction Kinematics
EMERGENCE_SPEED: float = 3.6           # Legacy base speed (kept for compat)
SPAWN_SPEED: float = 7.0               # Fast flowy pop-out rate (units/sec) — full spawn ~0.15s
COLLAPSE_SPEED: float = 4.6            # Vacuum suck-in rate (units/sec) — full collapse ~0.22s
EMERGENCE_STAGGER: float = 0.10        # Legacy stagger (kept for compat)
SPAWN_STAGGER: float = 0.055           # Snappy stagger between center and outer cubes on spawn
COLLAPSE_STAGGER: float = 0.075        # Funnel stagger on collapse (outers dive first, center last)
PALM_SUBMERGE_DEPTH: float = 18.0      # Submerged depth inside palm where cubes emerge from (px)
# Palm-center vacuum suck / vortex flow (collapse only)
SUCK_PULL_STRENGTH: float = 2600.0     # Acceleration pulling cubes toward palm center (px/s^2)
SUCK_SWIRL_STRENGTH: float = 900.0     # Tangential vortex force for flowy spiral (px/s^2)
SUCK_MAX_SPEED: float = 1400.0         # Clamp on suck-induced speed so it stays smooth, not teleporty
PORTAL_RING_ENABLED: bool = False      # Disabled to eliminate 2D ring/ripple overlays on palm
PORTAL_MAX_RADIUS: float = 20.0        # Radius of palm emergence aperture ring (px)

# 3D Cube Geometry & Spatial Layout — 2-inch cubes relative to palm width.
# Palm width (index-to-pinky knuckles) ~= 3.5in ~= palm_scale px, so a 2in cube
# side = 0.571 * palm_scale. Rendered side = 2 * CUBE_SIZE * dist_factor
# with dist_factor = palm_scale / 70, thus CUBE_SIZE = 20 gives true 2-inch cubes
# at any camera distance.
CUBE_COUNT: int = 3
CUBE_SIZE: float = 30.0                # Half-extent -> 60px side ~= 3 inches at ref distance
CUBE_HORIZONTAL_SPACING: float = 74.0  # Center-to-center (60px cube + ~14px gap)
CUBE_ELEVATION_OFFSET: float = 105.0   # Hover height above palm, scales with distance
MIN_VISIBLE_SCALE: float = 0.02        # Below this threshold, cube is considered vanished

# Procedural Organic Floating Configuration (Multi-Octave Harmonics)
# Amplitudes scaled for 2-inch cubes (~0.45x) so motion stays proportional
BOB_PRIMARY_FREQ: float = 1.65         # Primary harmonic breathing rate (rad/sec)
BOB_PRIMARY_AMP: float = 7.0           # Primary vertical heave amplitude
BOB_SWAY_FREQ_X: float = 2.10          # Organic horizontal sway frequency (rad/sec)
BOB_SWAY_AMP_X: float = 5.0            # Horizontal sway amplitude
BOB_SWAY_FREQ_Z: float = 2.75          # Depth wander frequency (rad/sec)
BOB_SWAY_AMP_Z: float = 6.0            # Depth wander amplitude
BOB_FLUTTER_FREQ: float = 4.80         # Subtle micro-buoyancy turbulence (rad/sec)
BOB_FLUTTER_AMP: float = 1.8           # Micro-buoyancy turbulence amplitude
BREATHING_EXPANSION_RATIO: float = 0.08# Formation breathing radial pulse ratio

# Dynamic Angling & 6-DOF Physical Kinetics
PALM_TILT_WEIGHT: float = 0.88         # Alignment stiffness towards palm normal vector
AERODYNAMIC_BANKING_GAIN: float = 0.008# Lean angle into velocity direction
GYROSCOPIC_RESTORE_K: float = 18.0     # Angular spring stiffness pulling back to resting orientation
GYROSCOPIC_DAMPING: float = 4.2        # Angular velocity damping factor
CUBE_MASS: float = 1.0                 # Rigid body mass
SPRING_STIFFNESS: float = 48.0         # Spring stiffness pulling to target hover slot
VELOCITY_DAMPING: float = 6.2          # Linear velocity damping factor
PALM_CUSHION_STIFFNESS: float = 140.0  # Levitation cushion repulsion from palm surface
PALM_CUSHION_DISTANCE: float = 40.0    # Distance threshold where palm cushion activates
COLLISION_RESTITUTION: float = 0.72    # Elastic collision restitution (bounciness)
COLLISION_FRICTION: float = 0.35       # Tangential friction generating rotational spin
AIR_DRAG_QUADRATIC: float = 0.0016     # Aerodynamic quadratic air drag
ANGULAR_DRAG: float = 0.975            # Multiplicative rotational damping per tick
HAND_INERTIAL_LAG_FACTOR: float = 0.40 # Inertia lag response to hand acceleration
# Finger Collision & Procedural Interaction Configuration
FINGER_COLLIDER_RADIUS: float = 20.0   # Effective collision radius for each fingertip (px)
FINGER_RESTITUTION: float = 0.80       # Elastic bounciness when striking a cube
FINGER_FLICK_IMPULSE_SCALE: float = 1.45# Multiplier converting fast finger flicks to impulse
FINGER_FLICK_MIN_SPEED: float = 75.0   # Minimum finger speed (px/sec) to trigger flick boost
FINGER_TORQUE_FACTOR: float = 1.60     # Rotational spin multiplier on finger contact
RIPPLE_ENABLED: bool = False           # Disabled to eliminate 2D black ripples on contact
RIPPLE_EXPANSION_SPEED: float = 260.0  # Expansion speed for contact shockwave ring (px/sec)
RIPPLE_LIFETIME: float = 0.38          # Duration of contact ripple effect (seconds)
FINGER_PROXIMITY_AURA: bool = False    # Hide dots and proximity halo rings on fingertips

# Default Angular velocities for organic tumbling (Pitch, Yaw, Roll in rad/sec)
CUBE_ROTATION_RATES: List[Tuple[float, float, float]] = [
    (0.45, 0.75, 0.30),   # Left cube
    (-0.60, 0.50, 0.55),  # Center cube
    (0.35, -0.80, 0.45),  # Right cube
]


# Prismatic procedural animation tuning
PRISM_HUE_SPEED: float = 22.0        # Rainbow drift rate (degrees/sec)
PRISM_SATURATION: float = 0.68       # Color intensity (0 = grey, 1 = full rainbow)
PRISM_FACE_STEP: float = 60.0        # Hue spacing between the 6 faces (degrees)
PRISM_CUBE_STEP: float = 25.0        # Hue offset per cube so cubes differ (degrees)


@dataclass(frozen=True)
class GreyMaterial:
    """Material definition for the single default prismatic cube material."""
    name: str
    base_color: Tuple[int, int, int]    # BGR fallback diffuse tone (unused by prism shader)
    bevel_color: Tuple[int, int, int]   # BGR edge highlight / skeleton color
    shadow_color: Tuple[int, int, int]  # BGR ambient shadow occlusion
    hud_accent: Tuple[int, int, int]    # BGR for UI elements


# Single default material — prismatic procedural faces are computed per-face
# in cube_renderer from the PRISM_* tunables above.
MATERIALS: Dict[str, GreyMaterial] = {
    "Prismatic": GreyMaterial(
        name="Prismatic",
        base_color=(175, 178, 184),     # Fallback neutral grey
        bevel_color=(220, 224, 230),    # Crisp edge highlight
        shadow_color=(75, 78, 85),      # Soft ambient occlusion
        hud_accent=(255, 220, 180),     # Bright prism accent for HUD
    ),
}
