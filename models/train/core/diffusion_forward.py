"""
Diffusion-family forward-process helpers shared by train and validation loops.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from models.train.core.utils import _q_bridge_sample, _q_sample, _q_unidb_sample


def _sample_diffusion_noisy_pair(
    y: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    noise_schedule_obj,
    use_chlast: bool,
    device: torch.device,
) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    sched_kind = getattr(noise_schedule_obj, "kind", "")
    if sched_kind == "bridge":
        x_noisy, eps = _q_bridge_sample(y, x, t, noise_schedule_obj)
    elif sched_kind == "unidb":
        x_noisy, eps = _q_unidb_sample(y, x, t, noise_schedule_obj)
    else:
        x_noisy, eps = _q_sample(y, t, noise_schedule_obj)

    log_snr_t = None
    if hasattr(noise_schedule_obj, "log_snr"):
        log_snr_series = getattr(noise_schedule_obj, "log_snr")
        if torch.is_tensor(log_snr_series):
            log_snr_t = log_snr_series.to(device=device, dtype=torch.float32)[t]

    if use_chlast:
        x_noisy = x_noisy.contiguous(memory_format=torch.channels_last)

    return sched_kind, x_noisy, eps, log_snr_t


def _build_diffusion_model_input(
    x_noisy: torch.Tensor,
    source: torch.Tensor,
    noise_schedule_obj,
    sched_kind: str,
) -> torch.Tensor:
    if sched_kind == "unidb":
        input_mode = str(getattr(noise_schedule_obj, "input_mode", "delta_source_concat")).lower()
        if input_mode == "delta_source_concat":
            # Match UniDB/DBFM backbone conditioning: model sees (x_t - source, source).
            return torch.cat([x_noisy - source, source], dim=1)
        if input_mode == "raw_source_concat":
            return torch.cat([x_noisy, source], dim=1)
        raise ValueError(
            f"Unknown UniDB input_mode '{input_mode}'. "
            "Use one of {'delta_source_concat','raw_source_concat'}."
        )
    return torch.cat([x_noisy, source], dim=1)
