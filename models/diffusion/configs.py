"""Configuration objects for diffusion rollouts."""

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    dt: float
    n_steps: int
    thermal_bc: str = "dirichlet"
    record_trajectory: bool = False

