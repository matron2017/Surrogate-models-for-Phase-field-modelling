"""
Higher-capacity latent autoencoder for phase-field fields.

Design: ResNet-style encoder/decoder with optional self-attention at coarse scales.
This stays deterministic (no KL) so it can drop into the existing latent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _saturate(z: torch.Tensor, bound: float = 0.0) -> torch.Tensor:
    if bound <= 0:
        return z
    b2 = bound * bound
    return z / torch.sqrt(1.0 + (z * z) / b2)


def _group_norm(channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        b, c, h_sz, w_sz = q.shape
        q = q.reshape(b, c, h_sz * w_sz).permute(0, 2, 1)
        k = k.reshape(b, c, h_sz * w_sz)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        v = v.reshape(b, c, h_sz * w_sz).permute(0, 2, 1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h_sz, w_sz)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


@dataclass
class LatentAEV2Config:
    in_channels: int
    z_channels: int = 64
    base_channels: int = 256
    ch_mult: Sequence[int] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Sequence[int] = (16, 8)
    dropout: float = 0.0
    saturate_bound: float = 0.0


class LatentAutoencoderV2(nn.Module):
    """
    Deterministic ResNet-style autoencoder with optional attention at coarse scales.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int = 64,
        base_channels: int = 256,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16, 8),
        dropout: float = 0.0,
        saturate_bound: float = 0.0,
        latent_channels: int | None = None,
    ):
        super().__init__()
        if latent_channels is not None:
            z_channels = int(latent_channels)
        self.latent_channels = int(z_channels)
        self.base_channels = int(base_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_resolutions = {int(r) for r in attn_resolutions} if attn_resolutions else set()
        self.saturate_bound = float(saturate_bound)

        ch_mult = list(ch_mult)
        self.downsample_factor = 2 ** max(len(ch_mult) - 1, 0)

        self.conv_in = nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = self.base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = self.base_channels * int(mult)
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            self.down_attn.append(AttnBlock(in_ch))
            if i != len(ch_mult) - 1:
                self.downsamples.append(Downsample(in_ch))
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(in_ch, in_ch, dropout)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, dropout)

        self.conv_z = nn.Conv2d(in_ch, self.latent_channels, kernel_size=3, padding=1)
        self.conv_z_in = nn.Conv2d(self.latent_channels, in_ch, kernel_size=3, padding=1)

        self.mid_dec_block1 = ResBlock(in_ch, in_ch, dropout)
        self.mid_dec_attn = AttnBlock(in_ch)
        self.mid_dec_block2 = ResBlock(in_ch, in_ch, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        rev_mult = list(reversed(ch_mult))
        for i, mult in enumerate(rev_mult):
            out_ch = self.base_channels * int(mult)
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attn.append(AttnBlock(in_ch))
            if i != len(rev_mult) - 1:
                self.upsamples.append(Upsample(in_ch))
            else:
                self.upsamples.append(nn.Identity())

        self.norm_out = _group_norm(in_ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def _maybe_attend(self, h: torch.Tensor, attn: nn.Module) -> torch.Tensor:
        if self.attn_resolutions and h.shape[-2] in self.attn_resolutions and h.shape[-1] in self.attn_resolutions:
            return attn(h)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for blocks, attn, down in zip(self.down_blocks, self.down_attn, self.downsamples):
            for block in blocks:
                h = block(h)
            h = self._maybe_attend(h, attn)
            h = down(h)
        h = self.mid_block1(h)
        h = self._maybe_attend(h, self.mid_attn)
        h = self.mid_block2(h)
        z = self.conv_z(h)
        return _saturate(z, self.saturate_bound)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_z_in(z)
        h = self.mid_dec_block1(h)
        h = self._maybe_attend(h, self.mid_dec_attn)
        h = self.mid_dec_block2(h)
        for blocks, attn, up in zip(self.up_blocks, self.up_attn, self.upsamples):
            for block in blocks:
                h = block(h)
            h = self._maybe_attend(h, attn)
            h = up(h)
        h = self.conv_out(self.act(self.norm_out(h)))
        return h

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec
