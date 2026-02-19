"""
Channel-wise metric accumulators used by train/validation loops.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.train.core.utils import _allreduce_sum_tensor


def _init_channel_stats(num_channels: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "sum": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "sumsq": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "se_sum": torch.zeros(num_channels, dtype=torch.float64, device=device),
        "count": torch.zeros(1, dtype=torch.float64, device=device),
    }


def _update_channel_stats(stats: Dict[str, torch.Tensor], pred: torch.Tensor, target: torch.Tensor) -> None:
    target_flat = target.detach().float().reshape(target.shape[0], target.shape[1], -1)
    pred_flat = pred.detach().float().reshape(pred.shape[0], pred.shape[1], -1)
    stats["sum"] += target_flat.sum(dim=(0, 2), dtype=torch.float64)
    stats["sumsq"] += (target_flat * target_flat).sum(dim=(0, 2), dtype=torch.float64)
    diff = pred_flat - target_flat
    stats["se_sum"] += (diff * diff).sum(dim=(0, 2), dtype=torch.float64)
    stats["count"] += float(target_flat.shape[0] * target_flat.shape[2])


def _compute_vrmse_from_stats(
    stats: Dict[str, torch.Tensor],
    vrmse_eps: float,
    ref_var: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    sum_c = _allreduce_sum_tensor(stats["sum"])
    sumsq_c = _allreduce_sum_tensor(stats["sumsq"])
    se_sum_c = _allreduce_sum_tensor(stats["se_sum"])
    count_t = _allreduce_sum_tensor(stats["count"])
    count = float(count_t.item())
    if count <= 0 or sum_c.numel() == 0:
        return None
    mean_c = sum_c / count
    var_c = sumsq_c / count - mean_c * mean_c
    var_c = torch.clamp(var_c, min=0.0)
    if ref_var is not None:
        ref_var_t = ref_var
        if not torch.is_tensor(ref_var_t):
            ref_var_t = torch.as_tensor(ref_var_t, device=sum_c.device, dtype=torch.float64)
        else:
            ref_var_t = ref_var_t.to(device=sum_c.device, dtype=torch.float64)
        if ref_var_t.numel() == var_c.numel():
            var_c = ref_var_t.view_as(var_c)
    mse_c = torch.clamp(se_sum_c / count, min=0.0)
    eps_safe = max(float(vrmse_eps), 0.0)
    vmse_c = mse_c / (var_c + eps_safe).clamp_min(1e-12)
    vrmse_c = torch.sqrt(vmse_c)
    return {
        "mse_per_channel": mse_c.detach().cpu().tolist(),
        "rmse_per_channel": torch.sqrt(mse_c).detach().cpu().tolist(),
        "var_per_channel": var_c.detach().cpu().tolist(),
        "mean_per_channel": mean_c.detach().cpu().tolist(),
        "vmse_per_channel": vmse_c.detach().cpu().tolist(),
        "vrmse_per_channel": vrmse_c.detach().cpu().tolist(),
        "vmse": float(vmse_c.mean().item()),
        "vrmse": float(vrmse_c.mean().item()),
    }
