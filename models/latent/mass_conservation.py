from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _normalize_channels(channels: Sequence[int] | str | int | None, num_channels: int) -> list[int]:
    if channels is None:
        return []
    if isinstance(channels, str):
        if channels.lower() == "all":
            return list(range(num_channels))
        raise ValueError(f"Unsupported channels spec: {channels!r}")
    if isinstance(channels, int):
        channels = [channels]
    out: list[int] = []
    for ch in channels:
        ch = int(ch)
        if ch < 0:
            ch = num_channels + ch
        if ch < 0 or ch >= num_channels:
            raise ValueError(f"Channel index {ch} out of range for {num_channels} channels.")
        if ch not in out:
            out.append(ch)
    return out


def apply_mean_conservation(
    x_pred: torch.Tensor,
    x_ref: torch.Tensor,
    channels: Sequence[int] | str | int | None,
    dims: Iterable[int] | None = None,
) -> torch.Tensor:
    """
    Shift selected channels of x_pred so their spatial mean matches x_ref.

    This mirrors the mean-shift adjustment used in PCFM's RegionConservationLaw,
    but uses x_ref to set the target mean per sample.
    """
    if x_pred.dim() < 3:
        return x_pred
    num_channels = x_pred.size(1)
    idx = _normalize_channels(channels, num_channels)
    if not idx:
        return x_pred
    if dims is None:
        dims = tuple(range(2, x_pred.dim()))
    dims = tuple(int(d) for d in dims)
    if not dims:
        return x_pred

    ref_mean = x_ref.mean(dim=dims, keepdim=True)
    pred_mean = x_pred.mean(dim=dims, keepdim=True)
    diff = ref_mean - pred_mean

    x_out = x_pred.clone()
    for ch in idx:
        x_out[:, ch, ...] = x_out[:, ch, ...] + diff[:, ch, ...]
    return x_out
