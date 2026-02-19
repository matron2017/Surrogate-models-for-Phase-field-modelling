from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.latent.autoencoder_lola import LoLAAutoencoder2D
from models.latent.mass_conservation import apply_mean_conservation


class LatentAELoLAModel(nn.Module):
    """
    Wrapper to train the LoLA-style autoencoder with the existing surrogate loop.
    Forward returns reconstruction in pixel space.
    """

    def __init__(
        self,
        channel_masking: Optional[Dict[str, Any]] = None,
        mass_conservation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        cfg = channel_masking or {}
        self.channel_masking_enabled = bool(cfg.get("enabled", False))
        self.channel_masking_choices = cfg.get("choices")
        self.channel_masking_min = int(cfg.get("min_channels", 1))
        self.channel_masking_max = cfg.get("max_channels")
        mc_cfg = mass_conservation or {}
        self.mass_conservation_enabled = bool(mc_cfg.get("enabled", False))
        self.mass_conservation_channels = mc_cfg.get(
            "channels",
            [1] if self.mass_conservation_enabled else None,
        )
        self.mass_conservation_dims = mc_cfg.get("dims")
        self.autoencoder = LoLAAutoencoder2D(**kwargs)
        self.latent_channels = self.autoencoder.latent_channels

    def _apply_channel_mask(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[int]]:
        if not self.channel_masking_enabled or not self.training:
            return z, None
        total = z.shape[1]
        if self.channel_masking_choices:
            valid = [int(c) for c in self.channel_masking_choices if 1 <= int(c) <= total]
            if not valid:
                return z, None
            idx = torch.randint(0, len(valid), (1,), device=z.device)
            keep = valid[int(idx.item())]
        else:
            min_c = max(1, min(self.channel_masking_min, total))
            max_c = total if self.channel_masking_max is None else int(self.channel_masking_max)
            max_c = max(min_c, min(max_c, total))
            keep = int(torch.randint(min_c, max_c + 1, (1,), device=z.device).item())
        mask = z.new_zeros((1, total, 1, 1))
        mask[:, :keep, ...] = 1
        return z * mask, keep

    def forward(self, x, cond=None):
        _ = cond
        z = self.autoencoder.encode(x)
        z, _ = self._apply_channel_mask(z)
        x_rec = self.autoencoder.decode(z)
        if self.mass_conservation_enabled:
            x_rec = apply_mean_conservation(
                x_rec,
                x,
                channels=self.mass_conservation_channels,
                dims=self.mass_conservation_dims,
            )
        return x_rec
