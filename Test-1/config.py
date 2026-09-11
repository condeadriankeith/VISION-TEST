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
# Exponential moving average filter factor (0 = frozen, 1 = raw jittery).
# Raised for realtime response: the 1 Euro filter already removes rest jitter,
# so stacking a heavy EMA here only added skeleton lag.
LANDMARK_SMOOTHING_ALPHA: float = 0.62
# Requested capture pixel format / frame rate (falls back gracefully).
CAMERA_FOURCC: str = "MJPG"  # MJPEG keeps 720p at ~30fps; YUYV often drops to ~10fps
CAMERA_FPS: int = 30

# MediaPipe Hand Tracking Configuration
MAX_NUM_HANDS: int = 2
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.6

# Gesture Recognition Configuration
# Fingers considered for open palm: Index, Middle, Ring, Pinky (+ Thumb)
OPEN_PALM_FINGER_THRESHOLD: int = 4   # At least 4 extended fingers = Open
CLOSED_PALM_FINGER_THRESHOLD: int = 1 # At most 1 extended finger = Closed
GESTURE_SMOOTHING_FRAMES: int = 5     # Temporal smoothing buffer length
CONTINUOUS_OPENNESS_MIN: float = 0.22 # Below this normalized ratio = fully closed fist
CONTINUOUS_OPENNESS_MAX: float = 0.70 # Above this normalized ratio = fully open palm

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

# Palm Emergence & Retraction Kinematics (Flowy Blossom & Vortex Funnel)
EMERGENCE_SPEED: float = 3.6           # Legacy base speed (kept for compat)
SPAWN_SPEED: float = 7.0               # Fast flowy pop-out rate (units/sec) — full spawn ~0.15s
COLLAPSE_SPEED: float = 2.4            # Satisfying suction rate (units/sec) — full vacuum suck ~0.42s
EMERGENCE_STAGGER: float = 0.10        # Legacy stagger (kept for compat)
SPAWN_STAGGER: float = 0.055           # Snappy stagger between center and outer cubes on spawn
COLLAPSE_STAGGER: float = 0.040        # Funnel stagger on collapse (outers sweep in first, framing center)
PALM_SUBMERGE_DEPTH: float = 18.0      # Submerged depth inside palm where cubes emerge from (px)

# Fountain Blossom & Vortex Suction Dynamics
FOUNTAIN_ARC_HEIGHT: float = 42.0      # Curved peak elevation during blossom fountain emergence (px)
FOUNTAIN_OUTWARD_SPREAD: float = 26.0  # Radial flower-petal splay as cubes fountain out (px)
SUCK_PULL_STRENGTH: float = 4800.0     # Gravitational acceleration pulling cubes toward palm center (px/s^2)
SUCK_SWIRL_STRENGTH: float = 2400.0    # Tangential vortex force for accelerated whirlpool spiral (px/s^2)
SUCK_MAX_SPEED: float = 1600.0         # Clamp on suck-induced speed so it stays smooth, not teleporty
SUCTION_STRETCH_FACTOR: float = 0.30   # Tidal spaghettification stretch along pull vector as cubes dive in
PORTAL_RING_ENABLED: bool = False      # Disabled to eliminate 2D ring/ripple overlays on palm
PORTAL_MAX_RADIUS: float = 20.0        # Radius of palm emergence aperture ring (px)

# Dual-Palm Procedural Midpoint Configuration
DUAL_HAND_ACCORDION_MIN_SPACING: float = 48.0   # Minimum cube spacing when hands are close (px)
DUAL_HAND_ACCORDION_MAX_SPACING: float = 165.0  # Maximum cube spacing when hands are far apart (px)
DUAL_HAND_REF_DISTANCE: float = 320.0           # Reference inter-palm distance for 1:1 accordion scale (px)
DUAL_HAND_ELEVATION_OFFSET: float = 40.0        # Hover lift perpendicular to inter-palm bridge axis (px)
DUAL_HAND_SMOOTHING_ALPHA: float = 0.35         # Temporal smoothing for dual-hand midpoint coordinate frame

# 3D Cube Geometry & Spatial Layout — 2-inch cubes relative to palm width.
# Palm width (index-to-pinky knuckles) ~= 3.5in ~= palm_scale px, so a 2in cube
# side = 0.571 * palm_scale. Rendered side = 2 * CUBE_SIZE * dist_factor
# with dist_factor = palm_scale / 70, thus CUBE_SIZE = 20 gives true 2-inch cubes
# at any camera distance.
CUBE_COUNT: int = 3
CUBE_SIZE: float = 30.0                # Half-extent -> 60px side ~= 3 inches at ref distance
CUBE_HORIZONTAL_SPACING: float = 74.0  # Center-to-center (60px cube + ~14px gap)
CUBE_ELEVATION_OFFSET: float = 105.0   # Hover height above palm, scales with distance
UPWARD_ELEVATION_BOOST: float = 38.0  # Additional elevation lift when palm opens/faces skyward (px)
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

# Inter-Cube Physical Collision Tunables
CUBE_COLLISION_RADIUS_FACTOR: float = 1.25  # Corner-aware bounding sphere (37.5px radius for 30px half-extent)
CUBE_COLLISION_RESTITUTION: float = 0.88    # High-elasticity rebound between colliding cubes
CUBE_RECOIL_DURATION: float = 0.45          # Duration to loosen hover spring after collision for billiard recoil (s)
CUBE_SPARK_ENABLED: bool = False            # Disabled to keep 3D scene clean without visual noise

# Telekinesis Force Mechanics Tunables
PINCH_THRESHOLD_PX: float = 36.0            # Thumb-to-index distance threshold to trigger Force Grip (px)
PINCH_RELEASE_THRESHOLD_PX: float = 52.0    # Hysteresis release distance threshold to prevent jitter
FORCE_FLING_MULTIPLIER: float = 1.85        # Fling impulse velocity multiplier on release
FORCE_FLING_FREE_TIME: float = 1.2          # Ballistic free-flight duration before spring recall (seconds)
FORCE_PUSH_SPEED_THRESH: float = 380.0      # Forward palm thrust velocity (px/s) to trigger Force Push
FORCE_PUSH_IMPULSE: float = 540.0           # Shockwave blast impulse knocking cubes away
FORCE_PUSH_COOLDOWN: float = 0.4            # Minimum interval between force pushes (seconds)

# Palm-Wave Tornado Procedural Cyclonic Funnel Tunables
TORNADO_WAVE_MIN_SPEED: float = 220.0       # Minimum hand waving speed (px/sec) to spin up cyclone
TORNADO_WAVE_MAX_SPEED: float = 650.0       # Waving speed at full cyclone intensity
TORNADO_ATTACK_RATE: float = 3.5            # Intensity ramp-up speed per second
TORNADO_DECAY_RATE: float = 2.2             # Intensity cool-down speed per second
TORNADO_SPIN_RATE: float = 12.5             # Base orbital cyclone spin rate (rad/sec)
TORNADO_FUNNEL_HEIGHT: float = 145.0        # Vertical elevation extent of tornado (px)
TORNADO_BASE_RADIUS: float = 28.0           # Tight base spiral radius (px)
TORNADO_TOP_RADIUS: float = 90.0            # Flaring top vortex radius (px)
TORNADO_VERTICAL_WAVE_FREQ: float = 4.2     # Heave frequency up/down tornado funnel

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

# =============================================================================
# Advanced Procedural & Top-Tier Algorithmic Tunables
# =============================================================================

# 1. 1€ (One Euro) Adaptive Motion Filter
# Retuned for realtime: higher rest cutoff halves skeleton lag (~190ms -> ~110ms
# group delay) while beta opens the cutoff fast during flicks, preserving
# zero-lag feel. Rest jitter is still suppressed ~4x vs raw (see test suite).
ONE_EURO_FC_MIN: float = 1.4           # Minimum cutoff frequency (Hz) for stationary stillness
ONE_EURO_BETA: float = 0.09            # Velocity scaling factor for zero-lag flicks
ONE_EURO_D_CUTOFF: float = 1.0         # Derivative cutoff frequency (Hz)

# 2. Symplectic Integrator Sub-Stepping
PHYSICS_SUBSTEPS: int = 4              # Number of sub-steps per frame tick for numerical stability
PHYSICS_FIXED_DT: float = 1.0 / 120.0  # Target fixed sub-step delta time (120 Hz)
INTEGRATOR_TYPE: str = "symplectic"    # "symplectic" (semi-implicit) or "verlet"

# 3. Divergence-Free 3D Curl Noise Field (Volume-Preserving Turbulence)
CURL_NOISE_ENABLED: bool = True        # Enable organic fluid turbulence around palm
CURL_NOISE_STRENGTH: float = 16.0      # Maximum fluid acceleration (px/s^2)
CURL_NOISE_TEMPORAL_SPEED: float = 0.85# Evolution speed of turbulence vector field

# 4. 3D OBB Separating Axis Theorem (SAT) Rigid-Body Collisions
OBB_SAT_COLLISION_ENABLED: bool = True # Enable true 15-axis 3D box collision detection
OBB_SAT_RESTITUTION: float = 0.88      # High-elasticity rebound on box impact
OBB_SAT_FRICTION: float = 0.35         # Tangential surface friction

# 5. Full-Hand Biomechanical Capsule Colliders
FULL_HAND_CAPSULE_ENABLED: bool = True # Enable swept-sphere capsules along all finger phalanges
CAPSULE_BONE_RADIUS: float = 15.0      # Collision thickness radius for finger bones (px)

# 6. Physical Thin-Film Optical Wave Interference
THIN_FILM_ENABLED: bool = True         # Wave-optics thin film iridescence on cube faces
THIN_FILM_IOR: float = 1.45            # Refractive index of thin dielectric coating
THIN_FILM_THICKNESS_NM: float = 520.0  # Base physical film thickness in nanometers
THIN_FILM_BLEND: float = 0.55          # Blend factor with base prismatic lighting

