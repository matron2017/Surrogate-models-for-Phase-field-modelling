"""Task wrapper that marries a backbone with a diffusion operator."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.diffusion.configs import DiffusionConfig
from models.diffusion.metrics import mass_conservation
from models.diffusion.operators import DiffusionOperator


class DiffusionTask(nn.Module):
    def __init__(self, backbone: nn.Module, diff_cfg: DiffusionConfig):
        super().__init__()
        self.backbone = backbone
        self.operator = DiffusionOperator(diff_cfg)

    def forward(self, init_state: torch.Tensor, cond_vec: Optional[torch.Tensor]):
        return self.operator.rollout_model(self.backbone, init_state, cond_vec)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mass_error": mass_conservation(pred, target),
            "mass_error_concentration": mass_conservation(pred, target, channel=1),
        }
