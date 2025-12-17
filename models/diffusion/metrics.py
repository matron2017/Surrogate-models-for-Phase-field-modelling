"""Diffusion-specific evaluation helpers."""

import torch


def mass_conservation(pred: torch.Tensor, target: torch.Tensor, channel: int | None = None) -> torch.Tensor:
    """
    Compare total mass/energy per sample.
    If channel is set, computes on that channel only.
    """
    if channel is not None:
        pred = pred[:, channel : channel + 1]
        target = target[:, channel : channel + 1]
    return (pred.sum(dim=(-2, -1)) - target.sum(dim=(-2, -1))).abs().mean()


def energy_like(pred: torch.Tensor) -> torch.Tensor:
    # Simple energy proxy (L2 norm over space).
    return (pred**2).sum(dim=(-2, -1)).mean()
