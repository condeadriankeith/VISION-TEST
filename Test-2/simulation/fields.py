"""Procedural 3D curl and turbulent flow vector fields.

Implements coupled trigonometric harmonic vector fields providing incompressible,
divergence-free fluid motion characteristics for particle simulations.
"""

import numpy as np


def compute_curl_flow_field(
    positions: np.ndarray,
    time_sec: float,
    curl_scale: float = 0.024,
) -> np.ndarray:
    """Compute 3D target flow velocities for an array of particle positions.

    Mathematical formulation:
        V_{x, target} = sin(k * Y + t) * 1.8 + cos(0.5 * k * Z) * 1.2 - 0.015 * Y
        V_{y, target} = cos(k * X - t) * 1.8 + sin(0.5 * k * Z) * 1.2 + 0.015 * X
        V_{z, target} = sin(0.5 * k * (X + Y) + t) * 1.5

    Args:
        positions: (N, 3) ndarray of particle coordinates [X, Y, Z].
        time_sec: Current simulation timestamp in seconds (t).
        curl_scale: Spatial frequency scalar (k).

    Returns:
        (N, 3) ndarray of target velocities [Vx, Vy, Vz].
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    k = curl_scale
    t = time_sec

    # Vectorized coupled trigonometric harmonic flow equations
    vx = np.sin(k * y + t) * 1.8 + np.cos(0.5 * k * z) * 1.2 - 0.015 * y
    vy = np.cos(k * x - t) * 1.8 + np.sin(0.5 * k * z) * 1.2 + 0.015 * x
    vz = np.sin(0.5 * k * (x + y) + t) * 1.5

    return np.column_stack((vx, vy, vz))
