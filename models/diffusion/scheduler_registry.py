"""Noise-schedule registry for diffusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass
class NoiseSchedule:
    betas: torch.Tensor

    def __post_init__(self) -> None:
        if self.betas.dim() != 1:
            raise ValueError("betas must be 1-D increasing sequence.")
        if torch.any(self.betas <= 0):
            raise ValueError("betas must be > 0.")
        if torch.any(self.betas >= 1):
            raise ValueError("betas must be < 1.")
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.log_snr = torch.log(self.alpha_bar.clamp_min(1e-12)) - torch.log(
            (1 - self.alpha_bar).clamp_min(1e-12)
        )

    @property
    def timesteps(self) -> int:
        return self.betas.numel()


def _linear_betas(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    t = steps / timesteps
    return torch.cos((t + s) / (1 + s) * torch.pi / 2).pow(2).clamp_min(1e-4)


def LinearNoiseSchedule(timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> NoiseSchedule:
    betas = _linear_betas(timesteps, beta_start, beta_end)
    return NoiseSchedule(betas)


def CosineNoiseSchedule(timesteps: int = 1000, s: float = 0.008) -> NoiseSchedule:
    alpha_bar = _cosine_alpha_bar(timesteps, s=s)
    alphas = alpha_bar[1:] / alpha_bar[:-1]
    betas = (1 - alphas).clamp(min=1e-5, max=0.999)
    return NoiseSchedule(betas)


def LogSNRLaplaceSchedule(timesteps: int = 1000, loc: float = 2.0, scale: float = 1.0) -> NoiseSchedule:
    t = torch.linspace(0.0, 1.0, timesteps, dtype=torch.float32)
    centered = t - 0.5
    logsnr = loc - torch.abs(centered) * (2 * scale)
    alpha_bar = torch.sigmoid(logsnr)
    alphas = alpha_bar.clone()
    alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
    betas = (1 - alphas).clamp(min=1e-5, max=0.999)
    return NoiseSchedule(betas)


def LearnedNoiseSchedule(beta_series: Sequence[float]) -> NoiseSchedule:
    betas = torch.as_tensor(beta_series, dtype=torch.float32)
    if betas.dim() != 1:
        raise ValueError("beta_series must be 1-D.")
    return NoiseSchedule(betas)


_SCHEDULE_REGISTRY: Dict[str, callable] = {
    "linear": LinearNoiseSchedule,
    "cosine": CosineNoiseSchedule,
    "logsnr_laplace": LogSNRLaplaceSchedule,
    "learned": LearnedNoiseSchedule,
}


def get_noise_schedule(name: str, **kwargs) -> NoiseSchedule:
    key = str(name).strip().lower()
    if key not in _SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown noise_schedule '{name}'. Registered: {sorted(_SCHEDULE_REGISTRY)}")
    return _SCHEDULE_REGISTRY[key](**kwargs)
