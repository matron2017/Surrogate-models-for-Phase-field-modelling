"""Registry for timestep samplers that operate on NoiseSchedule instances."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .scheduler_registry import NoiseSchedule


class UniformTimestepSampler:
    def __init__(self, schedule: NoiseSchedule, device: Optional[torch.device] = None):
        self.schedule = schedule
        self.device = device

    def sample(self, batch_size: int) -> torch.Tensor:
        dev = self.device or torch.device("cpu")
        return torch.randint(0, self.schedule.timesteps, (batch_size,), device=dev)


class LogSNRImportanceSampler:
    def __init__(self, schedule: NoiseSchedule, temperature: float = 1.0, device: Optional[torch.device] = None):
        self.schedule = schedule
        self.temperature = max(float(temperature), 1e-6)
        weights = torch.softmax(schedule.log_snr / self.temperature, dim=0)
        self.weights = weights / weights.sum()
        self.device = device

    def sample(self, batch_size: int) -> torch.Tensor:
        dev = self.device or torch.device("cpu")
        return torch.multinomial(self.weights, num_samples=batch_size, replacement=True).to(dev)


class AdaptiveRegionSampler:
    """Placeholder that defaults to uniform sampling until region-aware logic is added."""

    def __init__(self, schedule: NoiseSchedule, **kwargs):
        self.inner = UniformTimestepSampler(schedule, device=kwargs.get("device"))

    def sample(self, batch_size: int) -> torch.Tensor:
        return self.inner.sample(batch_size)


_SAMPLER_REGISTRY: Dict[str, type] = {
    "uniform": UniformTimestepSampler,
    "logsnr_importance": LogSNRImportanceSampler,
    "adaptive_region": AdaptiveRegionSampler,
}


def get_timestep_sampler(name: str, schedule: NoiseSchedule, **kwargs):
    key = str(name).strip().lower()
    if key not in _SAMPLER_REGISTRY:
        raise ValueError(f"Unknown timestep_sampler '{name}'. Registered: {sorted(_SAMPLER_REGISTRY)}")
    cls = _SAMPLER_REGISTRY[key]
    return cls(schedule=schedule, **kwargs)
