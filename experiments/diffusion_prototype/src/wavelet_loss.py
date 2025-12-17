"""Wavelet-based auxiliary loss used by both the VAE and diffusion trainer."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:  # optional dependency already present in physics_ml
    from pytorch_wavelets import DWTForward
except Exception:  # pragma: no cover - fallback when package missing
    DWTForward = None  # type: ignore[assignment]


def _dwt_components(x: torch.Tensor) -> torch.Tensor:
    if DWTForward is None:
        return x
    wavelet = DWTForward(J=1, wave='haar').to(x.device)
    yl, yh = wavelet(x)
    comp = [yl]
    if isinstance(hh := yh, (tuple, list)):
        comp.extend(hh)
    else:
        comp.append(hh)
    return torch.cat([c.reshape(x.shape[0], -1) for c in comp], dim=-1)


def wavelet_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    pred_feat = _dwt_components(pred)
    target_feat = _dwt_components(target)
    loss = F.l1_loss(pred_feat, target_feat, reduction=reduction)
    return loss
