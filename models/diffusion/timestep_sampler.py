"""Registry for timestep samplers that operate on NoiseSchedule instances."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .scheduler_registry import NoiseSchedule


class UniformTimestepSampler:
    def __init__(
        self,
        schedule: NoiseSchedule,
        device: Optional[torch.device] = None,
        t_min: int = 0,
        t_max: Optional[int] = None,
        include_terminal_unidb: bool = False,
    ):
        self.schedule = schedule
        self.schedule_kind = str(getattr(schedule, "kind", "")).lower()
        self.device = device
        self.t_min = int(t_min)
        self.t_max = int(schedule.timesteps if t_max is None else t_max)
        self.include_terminal_unidb = bool(include_terminal_unidb)
        if self.t_min < 0:
            raise ValueError(f"t_min must be >= 0 (got {self.t_min})")
        max_valid_t = int(schedule.timesteps)
        if self.t_max > max_valid_t:
            raise ValueError(f"t_max must be <= schedule.timesteps={max_valid_t} (got {self.t_max})")

        # Sampling backend uses high-exclusive bounds.
        # - VP/VE schedules index arrays of length `timesteps` with valid t in [0, timesteps-1],
        #   so we sample in [t_min, t_max) and users should set t_max accordingly.
        # - UniDB training commonly samples [1, T-1] to avoid endpoint degeneracy.
        #   Use include_terminal_unidb=True to opt into [t_min, t_max] behavior.
        if self.schedule_kind == "unidb" and self.include_terminal_unidb:
            self._rand_high = self.t_max + 1
        else:
            self._rand_high = self.t_max
        if self._rand_high <= self.t_min:
            cmp = ">=" if (self.schedule_kind == "unidb" and self.include_terminal_unidb) else ">"
            raise ValueError(
                f"t_max must be {cmp} t_min for schedule kind '{self.schedule_kind or 'default'}' "
                f"(got t_min={self.t_min}, t_max={self.t_max})"
            )

    def sample(self, batch_size: int) -> torch.Tensor:
        dev = self.device or torch.device("cpu")
        return torch.randint(self.t_min, self._rand_high, (batch_size,), device=dev)


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
        self.inner = UniformTimestepSampler(
            schedule,
            device=kwargs.get("device"),
            t_min=int(kwargs.get("t_min", 0)),
            t_max=kwargs.get("t_max", None),
            include_terminal_unidb=bool(kwargs.get("include_terminal_unidb", False)),
        )

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
