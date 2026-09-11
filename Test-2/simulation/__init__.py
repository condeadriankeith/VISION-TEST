"""Simulation package: procedural flow fields and vectorized particle physics."""
from simulation.fields import compute_curl_flow_field
from simulation.particle_system import ParticleSystem

__all__ = [
    "compute_curl_flow_field",
    "ParticleSystem",
]
