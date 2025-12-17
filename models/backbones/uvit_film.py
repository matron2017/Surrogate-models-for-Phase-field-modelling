"""
U-ViT style UNet with FiLM time/conditioning and attention bottleneck.

Structure:
- Time embedding (sinusoidal + MLP) and conditioning MLP fused for FiLM.
- Encoder/decoder ConvBlocks with FiLM modulation on activations.
- Single MHA at bottleneck (flattened tokens) with simple coordinate positional enc.
- Strided conv downsampling, transpose-conv upsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


def _group_norm(c: int) -> nn.GroupNorm:
    groups = max(1, math.gcd(c, 32))
    return nn.GroupNorm(groups, c)


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard 1D sinusoidal embedding for time scalars.
    t: [B] or [B,1]
    returns [B, dim]
    """
    if t.dim() == 2 and t.shape[1] == 1:
        t = t.view(-1)
    device = t.device
    half_dim = dim // 2
    freq = torch.exp(torch.arange(half_dim, device=device) * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
    angles = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    """Generates per-channel scale/shift from a context vector."""

    def __init__(self, in_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * channels)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(ctx).chunk(2, dim=1)  # [B,C] each
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class ConvBlock(nn.Module):
    """Conv → Norm → SiLU → FiLM → Conv → Norm → SiLU with residual."""

    def __init__(self, in_ch: int, out_ch: int, film_dim: int, padding: str = "zeros", dropout: float = 0.0):
        super().__init__()
        pad_mode = "circular" if padding == "circular" else "zeros"
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=pad_mode)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=pad_mode)
        self.norm2 = _group_norm(out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(film_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.proj_skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.film(h, ctx)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return self.proj_skip(x) + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, film_dim: int, padding: str = "zeros", dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, padding_mode=("circular" if padding == "circular" else "zeros"))
        self.block = ConvBlock(in_ch, out_ch, film_dim, padding=padding, dropout=dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.block(x, ctx)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, film_dim: int, padding: str = "zeros", dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch + out_ch, out_ch, film_dim, padding=padding, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, ctx)


class BottleneckAttention(nn.Module):
    """Flattened MHA at bottleneck with simple coordinate positional enc."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        pool: int = 1,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.heads = heads
        self.pool = max(1, int(pool))
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.mha = nn.MultiheadAttention(channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.pos_proj = nn.Linear(2, channels)

    def _compute_pool(self, h: int, w: int) -> int:
        """
        Chooses a pooling factor that keeps token count under max_tokens if set.
        """
        pool = self.pool
        if self.max_tokens is None:
            return pool

        tokens = (h // pool) * (w // pool)
        if tokens <= self.max_tokens:
            return pool

        scale = math.ceil(math.sqrt(tokens / float(self.max_tokens)))
        return pool * max(1, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_full, w_full = x.shape

        pool = min(self._compute_pool(h_full, w_full), h_full, w_full)
        if pool > 1:
            x_attn = F.avg_pool2d(x, kernel_size=pool, stride=pool)
        else:
            x_attn = x

        b, c, h, w = x_attn.shape
        x_flat = x_attn.view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        # coordinate grid in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).view(h * w, 2)  # [HW,2]
        pos = self.pos_proj(coords).unsqueeze(0)  # [1, HW, C]
        q = k = self.norm(x_flat + pos)
        attn_out, _ = self.mha(q, k, x_flat)
        out = x_flat + self.dropout(attn_out)
        out = out.permute(0, 2, 1).view(b, c, h, w)

        if pool > 1 and (h != h_full or w != w_full):
            out = F.interpolate(out, size=(h_full, w_full), mode="bilinear", align_corners=False)
        return out


@dataclass
class UVitMetaData(ModelMetaData):
    name: str = "UVit_FiLM_Velocity"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UVitFiLMVelocity(Module):
    """
    U-ViT velocity backbone with FiLM time/conditioning and attention bottleneck.

    Expects inputs:
      x: [B, C_in, H, W] current state
      cond: [B, C_cond] conditioning vector (e.g., physical scalars + time appended)
    Returns velocity field [B, C_out, H, W].
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        cond_dim: int = 3,  # e.g., 2 scalars + time
        channels: Sequence[int] | None = None,
        heads: int = 8,
        film_dim: int = 256,
        padding: str = "zeros",
        dropout: float = 0.0,
        attn_pool: int = 1,
        attn_max_tokens: Optional[int] = None,
    ):
        super().__init__(meta=UVitMetaData())
        channels = list(channels) if channels is not None else [64, 128, 256]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.film_dim = film_dim
        self.padding = padding

        self.time_mlp = nn.Sequential(
            nn.Linear(film_dim, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim - 1, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        ) if cond_dim > 1 else None

        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1, padding_mode=("circular" if padding == "circular" else "zeros"))

        enc_blocks = []
        downs = []
        for idx, ch in enumerate(channels):
            enc_blocks.append(ConvBlock(ch, ch, film_dim, padding=padding, dropout=dropout))
            if idx < len(channels) - 1:
                downs.append(DownBlock(ch, channels[idx + 1], film_dim, padding=padding, dropout=dropout))
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        self.attn = BottleneckAttention(
            channels[-1],
            heads=heads,
            dropout=dropout,
            pool=attn_pool,
            max_tokens=attn_max_tokens,
        )

        ups = []
        for i in reversed(range(len(channels) - 1)):
            ups.append(UpBlock(channels[i + 1], channels[i], film_dim, padding=padding, dropout=dropout))
        self.ups = nn.ModuleList(ups)

        self.out_norm = _group_norm(channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1, padding_mode=("circular" if padding == "circular" else "zeros"))

    def _build_ctx(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: [B, cond_dim] where the last entry is time.
        Returns FiLM context [B, film_dim].
        """
        if cond.dim() != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(f"Expected cond shape [B,{self.cond_dim}], got {cond.shape}")
        t = cond[:, -1]
        t_emb = self.time_mlp(_sinusoidal_embedding(t, self.film_dim))
        if self.cond_mlp is not None:
            c_emb = self.cond_mlp(cond[:, :-1])
            ctx = t_emb + c_emb
        else:
            ctx = t_emb
        return ctx

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        cond: [B, cond_dim] (includes time as last entry)
        """
        ctx = self._build_ctx(cond)
        skips: List[torch.Tensor] = []

        h = self.in_conv(x)
        # encoder blocks + downsample
        for idx, block in enumerate(self.enc_blocks):
            h = block(h, ctx)
            if idx < len(self.downs):
                skips.append(h)
                h = self.downs[idx](h, ctx)

        # bottleneck attention
        h = self.attn(h)

        # decoder
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            h = up(h, skip, ctx)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


__all__ = ["UVitFiLMVelocity"]
