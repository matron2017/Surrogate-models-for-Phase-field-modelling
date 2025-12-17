"""
Canonical FiLM-conditioned UNet2D with bottleneck self-attention and skip FiLM gating.

Design goals (aligned to the ReFlow description):
- Field conditioning (noise + initial + BCs) enters via channel-wise concatenation upstream.
- Time/process conditioning enters every ConvBlock through FiLM (GroupNorm + SiLU + FiLM).
- Strided-conv downsamples; transposed-conv upsamples; configurable padding (zeros/circular).
- Multi-head self-attention only at the bottleneck with positional embedding.
- Optional FiLM modulation of skip paths driven by process scalars (not time).

Shape conventions
- Inputs:  x ∈ ℝ[B, C_in, H, W] (should already include concatenated conditioning fields)
- Cond:    cond_vec ∈ ℝ[B, physical_cond_dim (+1 if τ is appended)]; physical scalars are
           (t_phys_norm, g_therm_norm), and generative time τ can be passed either as the
           extra entry or as a separate argument.
- Output:  y ∈ ℝ[B, C_out, H, W]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from models.conditioning.mixed_padding import MixedBCConv2d


# ---- helpers -----------------------------------------------------------------
def _group_norm(ch: int) -> nn.GroupNorm:
    # Use gcd(C, 32) groups to stay divisible for arbitrary widths.
    groups = max(1, math.gcd(ch, 32))
    return nn.GroupNorm(groups, ch)


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard 1D sinusoidal embedding.
    t: [B] or [B, 1] -> returns [B, dim]
    """
    if t.dim() == 2 and t.shape[1] == 1:
        t = t.view(-1)
    device = t.device
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / max(half - 1, 1)))
    angles = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    """Feature-wise affine modulation with zero-init for identity start."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.linear(cond).chunk(2, dim=1)  # [B, C] each
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class ConvBlock(nn.Module):
    """Conv → GN → (FiLM) → SiLU → Dropout → Conv → GN → SiLU with residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        film_mode: str = "additive",  # "additive" per ReFlow A.1, or "affine" for (1+γ)·x+β
        dropout: float = 0.1,
        padding: str = "zeros",
    ):
        super().__init__()
        film_mode = str(film_mode or "additive").lower()
        if film_mode not in {"additive", "affine"}:
            raise ValueError(f"film_mode must be 'additive' or 'affine' (got {film_mode})")
        pad_mode = "circular" if padding == "circular" else "zeros"
        use_mixed = padding == "mixed"
        self.conv1 = MixedBCConv2d(in_ch, out_ch, kernel_size=3) if use_mixed else nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, padding_mode=pad_mode
        )
        self.norm1 = _group_norm(out_ch)
        self.conv2 = MixedBCConv2d(out_ch, out_ch, kernel_size=3) if use_mixed else nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, padding_mode=pad_mode
        )
        self.norm2 = _group_norm(out_ch)
        self.act = nn.SiLU()
        if film_mode == "additive":
            self.film = nn.Linear(cond_dim, out_ch)
        else:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * out_ch),
            )
        self.film_mode = film_mode
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act(self.norm1(h))
        if self.film_mode == "additive":
            bias = self.film(cond).unsqueeze(-1).unsqueeze(-1)
            h = h + bias
        else:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        return self.skip_proj(x) + h


class BottleneckAttention(nn.Module):
    """Multi-head self-attention at the bottleneck with simple coordinate positional enc."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = _group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pos_proj = nn.Linear(2, channels)
        self.cached_coords: Tuple[int, int, torch.Tensor] | None = None

    def _coords(self, h: int, w: int, device, dtype) -> torch.Tensor:
        if self.cached_coords is not None:
            ch, cw, grid = self.cached_coords
            if ch == h and cw == w and grid.device == device and grid.dtype == dtype:
                return grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).view(h * w, 2)  # [HW, 2]
        self.cached_coords = (h, w, grid)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        coords = self._coords(h, w, x.device, x.dtype)
        pos = self.pos_proj(coords).unsqueeze(0)  # [1, HW, C]

        qkv = tokens + pos
        attn_out, _ = self.attn(qkv, qkv, qkv)
        tokens = tokens + self.dropout(attn_out)
        out = tokens.permute(0, 2, 1).view(b, c, h, w)
        return x + out


# ---- main backbone ----------------------------------------------------------
@dataclass
class UNetAttnMetaData(ModelMetaData):
    name: str = "UNet_FiLM_Attn"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UNetFiLMAttn(Module):
    """
    FiLM-conditioned UNet with bottleneck attention and skip FiLM gating.

    Defaults follow the ~100M design:
      channels = [48, 96, 192, 256, 384, 512, 512]
      num_blocks = [2, 2, 2, 2, 2, 2, 2]
      single attention block at bottleneck (heads=8).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        cond_dim: int = 2,            # physical scalars only (e.g., t_phys, g_therm)
        channels: Sequence[int] | None = None,
        num_blocks: Sequence[int] | None = None,
        bottleneck_blocks: Tuple[int, int] = (2, 2),  # (pre-attn, post-attn)
        attn_heads: int = 8,
        film_dim: int = 512,
        time_emb_dim: int = 128,
        dropout: float = 0.1,
        padding: str = "mixed",      # "mixed" (y periodic, x mixed) | "zeros" | "circular"
        use_time: bool = True,
        skip_film: bool = True,
        film_mode: str = "additive",  # ReFlow-style additive bias; set "affine" for (1+γ)·x+β
    ):
        super().__init__(meta=UNetAttnMetaData())
        channels = list(channels) if channels is not None else [48, 96, 192, 256, 384, 512, 512]
        num_blocks = list(num_blocks) if num_blocks is not None else [2] * len(channels)
        if len(num_blocks) != len(channels):
            raise ValueError("num_blocks length must match channels length")
        if len(channels) < 2:
            raise ValueError("Need at least 2 levels for UNet.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.use_time = use_time
        self.film_dim = film_dim
        self.padding = padding
        self.pad_mode = "circular" if self.padding == "circular" else "zeros"
        self.use_mixed = self.padding == "mixed"
        self.skip_film_enabled = bool(skip_film)
        self.film_mode = film_mode

        scalar_dim = cond_dim
        if scalar_dim <= 0:
            raise ValueError("cond_dim must be positive (physical scalars only).")

        # Conditioning embeddings
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim if scalar_dim > 0 else 1, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        )
        if use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, film_dim),
                nn.SiLU(),
                nn.Linear(film_dim, film_dim),
            )
        else:
            self.time_mlp = None
        self.cond_ctx_dim = film_dim * (2 if use_time else 1)

        # Encoder
        if self.use_mixed:
            self.in_conv = MixedBCConv2d(in_channels, channels[0], kernel_size=3)
        else:
            self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, padding_mode=self.pad_mode)

        enc_levels: List[nn.ModuleList] = []
        downs: List[nn.Conv2d] = []
        for idx, (ch, blocks) in enumerate(zip(channels, num_blocks)):
            prev = channels[idx] if idx > 0 else channels[0]
            level_blocks = nn.ModuleList()
            level_blocks.append(ConvBlock(prev, ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding))
            for _ in range(1, blocks):
                level_blocks.append(ConvBlock(ch, ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding))
            enc_levels.append(level_blocks)
            if idx < len(channels) - 1:
                if self.use_mixed:
                    downs.append(MixedBCConv2d(ch, channels[idx + 1], kernel_size=3, stride=2))
                else:
                    downs.append(
                        nn.Conv2d(ch, channels[idx + 1], kernel_size=3, stride=2, padding=1, padding_mode=self.pad_mode)
                    )
        self.enc_blocks = nn.ModuleList(enc_levels)
        self.downs = nn.ModuleList(downs)

        # Bottleneck
        bot_ch = channels[-1]
        pre_blocks, post_blocks = bottleneck_blocks
        self.bot_pre = nn.ModuleList(
            [ConvBlock(bot_ch, bot_ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding) for _ in range(pre_blocks)]
        )
        self.bot_attn = BottleneckAttention(bot_ch, heads=attn_heads, dropout=dropout)
        self.bot_post = nn.ModuleList(
            [ConvBlock(bot_ch, bot_ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding) for _ in range(post_blocks)]
        )

        # Decoder (skip bottleneck level)
        up_convs: List[nn.ConvTranspose2d] = []
        dec_levels: List[nn.ModuleList] = []
        for idx in range(len(channels) - 1, 0, -1):
            in_ch = channels[idx]
            skip_ch = channels[idx - 1]
            up_convs.append(nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2))
            blocks = nn.ModuleList()
            blocks.append(ConvBlock(skip_ch * 2, skip_ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding))
            for _ in range(1, num_blocks[idx - 1]):
                blocks.append(ConvBlock(skip_ch, skip_ch, self.cond_ctx_dim, film_mode=film_mode, dropout=dropout, padding=padding))
            dec_levels.append(blocks)
        self.up_convs = nn.ModuleList(up_convs)
        self.dec_blocks = nn.ModuleList(dec_levels)

        # Skip FiLMs (scalars only)
        self.skip_films = nn.ModuleList(
            [nn.Linear(film_dim, 2 * ch) for ch in channels[:-1]]
        ) if self.skip_film_enabled else None
        if self.skip_films is not None:
            for lin in self.skip_films:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        # Output head
        self.out_norm = _group_norm(channels[0])
        self.out_act = nn.SiLU()
        if self.use_mixed:
            self.out_conv = MixedBCConv2d(channels[0], out_channels, kernel_size=3)
        else:
            self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, padding_mode=self.pad_mode)

    # ---- conditioning utils -------------------------------------------------
    def _split_cond(
        self, cond_vec: Optional[torch.Tensor], timestep: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cond_vec is None:
            raise ValueError("cond_vec is required for FiLM conditioning.")
        if cond_vec.dim() != 2:
            raise ValueError(f"cond_vec must be 2D [B, C], got {tuple(cond_vec.shape)}.")
        cond_vec = cond_vec.float()
        scalar_dim = self.cond_dim
        if cond_vec.shape[1] < scalar_dim:
            raise ValueError(f"cond_vec has {cond_vec.shape[1]} dims but needs at least {scalar_dim} scalars.")
        scalars = cond_vec[:, :scalar_dim]
        t = None
        if self.use_time:
            if timestep is not None:
                t = timestep.view(timestep.shape[0], -1).float()
            elif cond_vec.shape[1] >= scalar_dim + 1:
                t = cond_vec[:, scalar_dim:scalar_dim + 1]
            else:
                raise ValueError("Generative timestep is required (pass as arg or append to cond_vec).")
        return scalars, t

    def _cond_embeddings(
        self, scalars: torch.Tensor, t: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scalar_emb = self.scalar_mlp(scalars)
        if self.use_time:
            if t is not None:
                time_emb = self.time_mlp(_sinusoidal_embedding(t.view(-1), self.time_mlp[0].in_features))
            else:
                time_emb = torch.zeros_like(scalar_emb)
            cond_ctx = torch.cat([time_emb, scalar_emb], dim=1)
        else:
            cond_ctx = scalar_emb
        return cond_ctx, scalar_emb

    def _apply_skip_film(self, skip: torch.Tensor, gamma_beta: torch.Tensor) -> torch.Tensor:
        g, b = gamma_beta.chunk(2, dim=1)
        g = g[:, :, None, None]
        b = b[:, :, None, None]
        return skip * (1 + g) + b

    def _split_args(
        self, cond_vec: Optional[torch.Tensor], args: Tuple[torch.Tensor, ...]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Supports:
          - forward(x, cond)
          - forward(x, cond, t)
          - forward(x, t, cond)
          - forward(x, t, cond, region_info=...)    # extra kwargs ignored upstream
        """
        if len(args) == 0:
            return cond_vec, None
        if len(args) == 1:
            extra = args[0]
            if cond_vec is not None and cond_vec.dim() <= 1 and extra.dim() == 2:
                return extra, cond_vec
            return cond_vec, extra
        if len(args) == 2:
            a, b = args
            # Heuristic: 2D tensor with more than 1 channel → cond_vec; 1D/1ch → timestep.
            def _is_time(tensor: torch.Tensor) -> bool:
                return tensor.dim() <= 1 or (tensor.dim() == 2 and tensor.shape[1] == 1)

            if _is_time(a) and not _is_time(b):
                return b, a
            if _is_time(b) and not _is_time(a):
                return a, b
            # Fallback: treat first as cond, second as time.
            return a, b
        raise TypeError(f"UNetFiLMAttn.forward expected at most 3 positional args, got {2 + len(args)}")

    # ---- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, cond_vec: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        # region_info or other aux kwargs are ignored; kept for trainer API compatibility.
        kwargs.pop("region_info", None)
        cond_vec, timestep = self._split_args(cond_vec, args)

        scalars, t = self._split_cond(cond_vec, timestep)
        film_ctx, scalar_ctx = self._cond_embeddings(scalars, t)

        # Precompute skip FiLMs (exclude bottleneck level)
        skip_gb = None
        if self.skip_films is not None:
            skip_gb = [lin(scalar_ctx) for lin in self.skip_films]

        # Encoder
        x = self.in_conv(x)
        skips: List[torch.Tensor] = []
        for idx, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                x = block(x, film_ctx)
            if skip_gb is not None and idx < len(skip_gb):
                skip = self._apply_skip_film(x, skip_gb[idx])
            else:
                skip = x
            if idx < len(self.enc_blocks) - 1:
                skips.append(skip)
            if idx < len(self.downs):
                x = self.downs[idx](x)

        # Bottleneck
        for block in self.bot_pre:
            x = block(x, film_ctx)
        x = self.bot_attn(x)
        for block in self.bot_post:
            x = block(x, film_ctx)

        # Decoder (use skips except bottleneck)
        for up, dec_blocks, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in dec_blocks:
                x = block(x, film_ctx)

        x = self.out_act(self.out_norm(x))
        return self.out_conv(x)
