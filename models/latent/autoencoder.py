"""
Lightweight latent autoencoder for physics fields.

Design goals (aligned with the LoLa/DC-AE style in the paper discussion):
- Deterministic encoder/decoder with space-to-depth downsampling by a fixed factor r.
- Channel bottleneck sets the compression (latent_channels); no KL term.
- Saturating nonlinearity on the latent to bound activations.
- Near-identity initialization on the shallow convs so an untrained model reconstructs
  reasonably (useful for quick smoke tests before dedicated AE training).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _saturate(z: torch.Tensor, bound: float = 5.0) -> torch.Tensor:
    if bound <= 0:
        return z
    b2 = bound * bound
    return z / torch.sqrt(1.0 + (z * z) / b2)


def _init_identity_conv(conv: nn.Conv2d):
    # Initialize to (approx) identity on channels, zeros elsewhere.
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        in_ch, out_ch = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        center = k // 2
        for c in range(min(in_ch, out_ch)):
            conv.weight[c, c, center, center] = 1.0


class SpaceToDepth2d(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(x, self.factor)


class DepthToSpace2d(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x, self.factor)


@dataclass
class LatentAEConfig:
    in_channels: int
    latent_channels: int
    downsample_factor: int = 16
    saturate_bound: float = 5.0
    hidden_channels: int | None = None


class LatentAutoencoder2D(nn.Module):
    """
    Deterministic convolutional autoencoder with fixed spatial downsample factor.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        downsample_factor: int = 16,
        hidden_channels: int | None = None,
        saturate_bound: float = 5.0,
    ):
        super().__init__()
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >=1")
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)
        hid = hidden_channels or max(in_channels, latent_channels)

        self.enc_norm = nn.GroupNorm(1, in_channels)
        self.enc_pre = nn.Conv2d(in_channels, hid, kernel_size=3, padding=1)
        _init_identity_conv(self.enc_pre)
        self.s2d = SpaceToDepth2d(self.downsample_factor)
        self.enc_proj = nn.Conv2d(hid * (self.downsample_factor**2), latent_channels, kernel_size=1)

        self.dec_proj = nn.Conv2d(latent_channels, hid * (self.downsample_factor**2), kernel_size=1)
        self.d2s = DepthToSpace2d(self.downsample_factor)
        self.dec_post = nn.Conv2d(hid, in_channels, kernel_size=3, padding=1)
        _init_identity_conv(self.dec_post)
        self.saturate_bound = float(saturate_bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_norm(x)
        h = self.enc_pre(h)
        h = self.s2d(h)
        h = self.enc_proj(h)
        return _saturate(h, self.saturate_bound)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_proj(z)
        h = self.d2s(h)
        h = self.dec_post(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec

