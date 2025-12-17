"""Loss registry for rapid solidification trainers."""

from __future__ import annotations

from typing import Dict, Any

import torch.nn.functional as F

try:
    from models.train.core.wavelet_weight import wavelet_importance_weighted_mse
except Exception:
    wavelet_importance_weighted_mse = None


def build_surrogate_loss(loss_cfg: Dict[str, Any]):
    """Returns a callable(pred, target, weight=None) computing the base surrogate loss."""
    weight_wavelet = float(loss_cfg.get("weight_wavelet_loss", 0.0))

    def _loss(pred, target, weight=None):
        if weight is not None:
            return (weight * (pred - target).pow(2)).mean()
        if weight_wavelet > 0.0:
            if wavelet_importance_weighted_mse is None:
                raise ImportError("pytorch_wavelets required for wavelet-weighted loss.")
            return wavelet_importance_weighted_mse(pred, target, theta=loss_cfg.get("theta", 0.8))
        return F.mse_loss(pred, target)

    return _loss


def build_diffusion_loss(loss_cfg: Dict[str, Any]):
    weight_wavelet = float(loss_cfg.get("weight_wavelet_loss", 0.0))

    def _loss(pred_eps, true_eps, **kwargs):
        base = F.mse_loss(pred_eps, true_eps)
        if weight_wavelet > 0.0 and wavelet_importance_weighted_mse is not None:
            ref = kwargs.get("target", true_eps)
            weighted = wavelet_importance_weighted_mse(pred_eps, true_eps, y_ref=ref)
            return (1.0 - weight_wavelet) * base + weight_wavelet * weighted
        return base

    return _loss
