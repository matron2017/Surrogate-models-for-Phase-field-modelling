from __future__ import annotations

import torch
import torch.nn as nn

from models.latent.autoencoder_v2 import LatentAutoencoderV2
from models.latent.mass_conservation import apply_mean_conservation


class LatentAEModelV2(nn.Module):
    """
    Wrapper to train the higher-capacity latent autoencoder with the existing surrogate loop.
    Forward ignores conditioning and returns reconstruction in pixel space.
    """

    def __init__(self, mass_conservation: dict | None = None, **kwargs):
        super().__init__()
        self.autoencoder = LatentAutoencoderV2(**kwargs)
        self.latent_channels = self.autoencoder.latent_channels
        mc_cfg = mass_conservation or {}
        self.mass_conservation_enabled = bool(mc_cfg.get("enabled", False))
        self.mass_conservation_channels = mc_cfg.get(
            "channels",
            [1] if self.mass_conservation_enabled else None,
        )
        self.mass_conservation_dims = mc_cfg.get("dims")

    def forward(self, x, cond=None):
        _ = cond
        _, x_rec = self.autoencoder(x)
        if self.mass_conservation_enabled:
            x_rec = apply_mean_conservation(
                x_rec,
                x,
                channels=self.mass_conservation_channels,
                dims=self.mass_conservation_dims,
            )
        return x_rec
