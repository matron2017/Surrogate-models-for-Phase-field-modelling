"""
LoLA DCAE wrapper built from vendored LoLA components.

This replaces the previous hand-rolled LoLA-style AE with the LoLA DCAE
encoder/decoder blocks and keeps the same forward signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from models.latent.lola_ref_dcae import DCDecoder, DCEncoder


def _saturate(z: torch.Tensor, bound: float = 5.0, kind: str = "softclip2") -> torch.Tensor:
    if bound <= 0:
        return z
    if kind in (None, "none"):
        return z
    if kind == "softclip":
        return z / (1.0 + torch.abs(z) / bound)
    if kind == "softclip2":
        return z * torch.rsqrt(1.0 + (z / bound).square())
    if kind == "tanh":
        return torch.tanh(z / bound) * bound
    if kind == "arcsinh":
        return torch.arcsinh(z)
    if kind == "rmsnorm":
        return z * torch.rsqrt(torch.mean(z.square(), dim=1, keepdim=True) + 1e-5)
    raise ValueError(f"Unknown saturation '{kind}'")


@dataclass
class LoLAEConfig:
    in_channels: int
    latent_channels: int = 64
    hid_channels: Sequence[int] = (64, 128, 256, 512, 768, 1024)
    hid_blocks: Sequence[int] = (3, 3, 3, 3, 3, 3)
    kernel_size: Union[int, Sequence[int]] = 3
    stride: Union[int, Sequence[int]] = 2
    stage_strides: Sequence[int] | None = None
    pixel_shuffle: bool = True
    norm: str = "layer"
    attention_heads: Optional[Dict[int, int]] = None
    ffn_factor: int = 1
    patch_size: Union[int, Sequence[int]] = 1
    periodic: bool | str = False
    dropout: float = 0.05
    checkpointing: bool = False
    identity_init: bool = True
    saturate_bound: float = 5.0
    saturate_kind: str = "softclip2"
    latent_noise: float = 0.0


class LoLAAutoencoder2D(nn.Module):
    """LoLA DCAE autoencoder wrapper (2D)."""

    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 64,
        hid_channels: Sequence[int] = (64, 128, 256, 512, 768, 1024),
        hid_blocks: Sequence[int] = (3, 3, 3, 3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        stage_strides: Sequence[int] | None = None,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: Dict[int, int] | None = None,
        ffn_factor: int = 1,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool | str = False,
        dropout: float = 0.05,
        checkpointing: bool = False,
        identity_init: bool = True,
        saturate_bound: float = 5.0,
        saturate_kind: str = "softclip2",
        latent_noise: float = 0.0,
    ):
        super().__init__()
        if len(hid_channels) != len(hid_blocks):
            raise ValueError("hid_channels and hid_blocks must have the same length.")

        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.hid_channels = tuple(int(c) for c in hid_channels)
        self.hid_blocks = tuple(int(b) for b in hid_blocks)
        self.dropout = float(dropout)
        self.saturate_bound = float(saturate_bound)
        self.saturate_kind = str(saturate_kind)
        self.latent_noise = float(latent_noise)

        attn = attention_heads if attention_heads is not None else {}

        self.encoder = DCEncoder(
            in_channels=self.in_channels,
            out_channels=self.latent_channels,
            hid_channels=self.hid_channels,
            hid_blocks=self.hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            stage_strides=stage_strides,
            pixel_shuffle=pixel_shuffle,
            norm=norm,
            attention_heads=attn,
            ffn_factor=ffn_factor,
            spatial=2,
            patch_size=patch_size,
            periodic=periodic,
            dropout=self.dropout,
            checkpointing=checkpointing,
            identity_init=identity_init,
        )

        self.decoder = DCDecoder(
            in_channels=self.latent_channels,
            out_channels=self.in_channels,
            hid_channels=self.hid_channels,
            hid_blocks=self.hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            stage_strides=stage_strides,
            pixel_shuffle=pixel_shuffle,
            norm=norm,
            attention_heads=attn,
            ffn_factor=ffn_factor,
            spatial=2,
            patch_size=patch_size,
            periodic=periodic,
            dropout=self.dropout,
            checkpointing=checkpointing,
            identity_init=identity_init,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = _saturate(z, self.saturate_bound, self.saturate_kind)
        return z

    def decode(self, z: torch.Tensor, noisy: bool = True) -> torch.Tensor:
        if noisy and self.latent_noise > 0:
            z = z + self.latent_noise * torch.randn_like(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec
