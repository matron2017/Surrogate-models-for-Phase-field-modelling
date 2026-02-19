from __future__ import annotations

import torch
import torch.nn as nn

from models.latent.autoencoder import LatentAutoencoder2D
from models.latent.mass_conservation import apply_mean_conservation


class LatentAEModel(nn.Module):
    """
    Wrapper to train the latent autoencoder with the existing surrogate loop.
    Forward ignores conditioning and returns reconstruction in pixel space.
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 64,
        downsample_factor: int = 16,
        saturate_bound: float = 5.0,
        hidden_channels: int | None = None,
        mass_conservation: dict | None = None,
    ):
        super().__init__()
        self.autoencoder = LatentAutoencoder2D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
            saturate_bound=saturate_bound,
            hidden_channels=hidden_channels,
        )
        mc_cfg = mass_conservation or {}
        self.mass_conservation_enabled = bool(mc_cfg.get("enabled", False))
        self.mass_conservation_channels = mc_cfg.get(
            "channels",
            [1] if self.mass_conservation_enabled else None,
        )
        self.mass_conservation_dims = mc_cfg.get("dims")
        # Expose for checkpoint/loader consistency
        self.latent_channels = latent_channels

    def forward(self, x, cond=None):
        _ = cond  # unused
        _, x_rec = self.autoencoder(x)
        if self.mass_conservation_enabled:
            x_rec = apply_mean_conservation(
                x_rec,
                x,
                channels=self.mass_conservation_channels,
                dims=self.mass_conservation_dims,
            )
        return x_rec
