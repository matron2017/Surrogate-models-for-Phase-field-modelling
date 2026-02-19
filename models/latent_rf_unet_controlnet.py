"""
Latent-space rectified-flow U-Net with ControlNet-style thermal injection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


def _group_norm(ch: int) -> nn.GroupNorm:
    groups = max(1, math.gcd(ch, 32))
    return nn.GroupNorm(groups, ch)


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
    if half_dim < 1:
        raise ValueError(f"Time embedding dim must be >= 2 (got {dim}).")
    freq = torch.exp(torch.arange(half_dim, device=device) * (-math.log(10000.0) / max(half_dim - 1, 1)))
    angles = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by an MLP."""

    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = int(out_dim or dim)
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(t.float(), self.dim)
        return self.mlp(t_emb)


class FiLM(nn.Module):
    """Feature-wise affine modulation from time embedding."""

    def __init__(self, t_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class FiLMResBlock2d(nn.Module):
    """Residual conv block with GroupNorm + FiLM time modulation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        dropout: float = 0.1,
        use_film: bool = True,
        film_position: str = "post_act",
    ):
        super().__init__()
        film_position = str(film_position or "post_act").lower()
        if film_position not in {"post_act", "pre_act"}:
            raise ValueError("film_position must be 'post_act' or 'pre_act'.")
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.film = FiLM(t_dim, out_ch) if use_film else None
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.film_position = film_position

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        if self.film_position == "pre_act":
            if self.film is not None:
                h = self.film(h, t_emb)
            h = self.act(h)
        else:
            h = self.act(h)
            if self.film is not None:
                h = self.film(h, t_emb)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return self.skip(x) + h


class Downsample2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class SelfAttention2d(nn.Module):
    """Self-attention at a single resolution (bottleneck only)."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.0, use_pos_emb: bool = True):
        super().__init__()
        self.norm = _group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pos_proj = nn.Linear(2, channels)
        self.cached_coords: Optional[torch.Tensor] = None
        self.cached_hw: Optional[tuple[int, int]] = None
        self.use_pos_emb = bool(use_pos_emb)

    def _coords(self, h: int, w: int, device, dtype) -> torch.Tensor:
        if self.cached_coords is not None and self.cached_hw == (h, w) and self.cached_coords.device == device and self.cached_coords.dtype == dtype:
            return self.cached_coords
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).view(h * w, 2)
        self.cached_coords = coords
        self.cached_hw = (h, w)
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)
        tokens = h_in.view(b, c, h * w).permute(0, 2, 1)
        if self.use_pos_emb:
            coords = self._coords(h, w, x.device, x.dtype)
            pos = self.pos_proj(coords).unsqueeze(0)
            qkv = tokens + pos
        else:
            qkv = tokens
        attn_out, _ = self.attn(qkv, qkv, qkv)
        out = tokens + self.dropout(attn_out)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + out


class ZeroConv2d(nn.Module):
    """1x1 conv initialized to zero weights/bias."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ThermalControlEncoder2d(nn.Module):
    """Thermal branch producing a multi-scale feature pyramid."""

    def __init__(self, theta_in_ch: int = 1, channels: Sequence[int] | None = None):
        super().__init__()
        channels = list(channels) if channels is not None else [128, 256, 256, 256]
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = theta_in_ch
        for idx, ch in enumerate(channels):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            if idx < len(channels) - 1:
                self.downs.append(Downsample2d(ch, channels[idx + 1]))
                in_ch = channels[idx + 1]
            else:
                in_ch = ch

    def forward(self, theta: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        h = theta
        for idx, block in enumerate(self.blocks):
            h = block(h)
            feats.append(h)
            if idx < len(self.downs):
                h = self.downs[idx](h)
        return feats


@dataclass
class LatentRFMetaData(ModelMetaData):
    name: str = "LatentRFUNetControlNet"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class LatentRFUNetControlNet(Module):
    """
    Latent-space rectified flow U-Net with ControlNet-style thermal injection.

    Forward signature:
      forward(z_t, t, z_n, theta_n=None) -> v
    """

    def __init__(
        self,
        Cz: int = 32,
        channels: Sequence[int] | None = None,
        blocks_per_level: int = 4,
        dropout: float = 0.1,
        use_attn_bottleneck: bool = True,
        attn_heads: int = 8,
        time_dim: int = 256,
        theta_in_ch: int = 1,
        use_theta: bool = True,
        theta_inject: str = "per_level",
        use_film: bool = True,
        film_position: str = "post_act",
        cache_theta: bool = False,
        cache_theta_detach: bool = True,
    ):
        super().__init__(meta=LatentRFMetaData())
        channels = list(channels) if channels is not None else [128, 256, 256, 256]
        if len(channels) != 4:
            raise ValueError("channels must have 4 entries for 256→128→64→32 pyramid.")
        if blocks_per_level < 1:
            raise ValueError("blocks_per_level must be >= 1.")
        theta_inject = str(theta_inject or "per_level").lower()
        if theta_inject not in {"per_level", "per_block"}:
            raise ValueError("theta_inject must be 'per_level' or 'per_block'.")

        self.Cz = int(Cz)
        self.channels = channels
        self.blocks_per_level = int(blocks_per_level)
        self.use_theta = bool(use_theta)
        self.theta_inject = theta_inject
        self.cache_theta = bool(cache_theta)
        self.cache_theta_detach = bool(cache_theta_detach)
        self._theta_cache: Optional[tuple[tuple[int, torch.Size, torch.device, torch.dtype], List[torch.Tensor]]] = None

        self.time_embed = TimeEmbedding(time_dim, time_dim)

        in_ch = 2 * self.Cz
        self.stem = nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for idx, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for b in range(self.blocks_per_level):
                blocks.append(
                    FiLMResBlock2d(
                        ch,
                        ch,
                        t_dim=time_dim,
                        dropout=dropout,
                        use_film=use_film,
                        film_position=film_position,
                    )
                )
            self.enc_blocks.append(blocks)
            if idx < len(channels) - 1:
                self.downs.append(Downsample2d(ch, channels[idx + 1]))

        self.attn = SelfAttention2d(channels[-1], heads=attn_heads, dropout=dropout) if use_attn_bottleneck else None

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx in reversed(range(len(channels) - 1)):
            self.ups.append(Upsample2d(channels[idx + 1], channels[idx]))
            blocks = nn.ModuleList()
            in_ch = channels[idx] * 2
            blocks.append(
                FiLMResBlock2d(
                    in_ch,
                    channels[idx],
                    t_dim=time_dim,
                    dropout=dropout,
                    use_film=use_film,
                    film_position=film_position,
                )
            )
            for _ in range(self.blocks_per_level - 1):
                blocks.append(
                    FiLMResBlock2d(
                        channels[idx],
                        channels[idx],
                        t_dim=time_dim,
                        dropout=dropout,
                        use_film=use_film,
                        film_position=film_position,
                    )
                )
            self.dec_blocks.append(blocks)

        self.out_norm = _group_norm(channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], self.Cz, kernel_size=3, padding=1)

        if self.use_theta:
            self.theta_encoder = ThermalControlEncoder2d(theta_in_ch=theta_in_ch, channels=channels)
            self.theta_zero_convs = nn.ModuleList([ZeroConv2d(ch, ch) for ch in channels])
        else:
            self.theta_encoder = None
            self.theta_zero_convs = None

    def clear_theta_cache(self) -> None:
        self._theta_cache = None

    def _theta_cache_key(self, theta_n: torch.Tensor) -> tuple[int, torch.Size, torch.device, torch.dtype]:
        return (theta_n.data_ptr(), theta_n.shape, theta_n.device, theta_n.dtype)

    def _encode_theta(self, theta_n: torch.Tensor) -> List[torch.Tensor]:
        feats = self.theta_encoder(theta_n)
        if self.cache_theta_detach:
            feats = [f.detach() for f in feats]
        return feats

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_n: torch.Tensor,
        theta_n: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_t.dim() != 4 or z_n.dim() != 4:
            raise ValueError("z_t and z_n must be 4D tensors [B,C,H,W].")
        if z_t.shape != z_n.shape:
            raise ValueError(f"z_t shape {tuple(z_t.shape)} must match z_n shape {tuple(z_n.shape)}.")
        t_emb = self.time_embed(t)

        x_in = torch.cat([z_t, z_n], dim=1)
        h = self.stem(x_in)

        theta_feats = None
        if self.use_theta and theta_n is not None:
            if self.cache_theta:
                key = self._theta_cache_key(theta_n)
                cached = self._theta_cache
                if cached is None or cached[0] != key:
                    theta_feats = self._encode_theta(theta_n)
                    self._theta_cache = (key, theta_feats)
                else:
                    theta_feats = cached[1]
            else:
                theta_feats = self.theta_encoder(theta_n)

        skips: List[torch.Tensor] = []
        for idx, blocks in enumerate(self.enc_blocks):
            if theta_feats is not None and self.theta_inject == "per_level":
                h = h + self.theta_zero_convs[idx](theta_feats[idx])
            for block in blocks:
                if theta_feats is not None and self.theta_inject == "per_block":
                    h = h + self.theta_zero_convs[idx](theta_feats[idx])
                h = block(h, t_emb)
            if idx < len(self.downs):
                skips.append(h)
                h = self.downs[idx](h)

        if self.attn is not None:
            h = self.attn(h)

        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in self.dec_blocks[idx]:
                h = block(h, t_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


__all__ = [
    "TimeEmbedding",
    "FiLM",
    "FiLMResBlock2d",
    "Downsample2d",
    "Upsample2d",
    "SelfAttention2d",
    "ZeroConv2d",
    "ThermalControlEncoder2d",
    "LatentRFUNetControlNet",
]
