"""Thermal-field conditioned U-ViT surrogate (no scalar conditioning path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

from models.backbones.uvit_film import (
    BottleneckAttention,
    ConvBlock,
    DownBlock,
    ThermalPyramidMapper,
    UpBlock,
    _sinusoidal_embedding,
    _group_norm,
)
from physicsnemo.models.afno.afno import AFNO


@dataclass
class UVitThermalMetaData(ModelMetaData):
    name: str = "UVit_Thermal_Surrogate"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UVitThermalSurrogate(Module):
    """
    U-ViT backbone for latent next-step prediction with thermal-field-only conditioning.

    Expected inputs:
      - x: [B, C_in, H, W]       latent state z_n
      - theta: [B, C_theta, H0, W0] thermal field (full-res or latent-res)
    Output:
      - [B, C_out, H, W]         predicted latent z_{n+1}
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        channels: Sequence[int] | None = None,
        heads: int = 8,
        film_dim: int = 256,
        padding: str = "zeros",
        dropout: float = 0.0,
        attn_pool: int = 1,
        attn_max_tokens: Optional[int] = None,
        theta_in_ch: int = 1,
        theta_mode: str = "film",
        theta_hidden: int = 64,
        afno_depth: int = 0,
        afno_mlp_ratio: float = 4.0,
        afno_num_blocks: int = 16,
        afno_patch_size: int | Sequence[int] = 1,
        afno_inp_shape: Optional[Sequence[int]] = None,
        afno_stage: Optional[int] = None,
        afno_sparsity_threshold: float = 0.01,
        afno_hard_thresholding_fraction: float = 1.0,
        use_bottleneck_attn: bool = True,
    ):
        super().__init__(meta=UVitThermalMetaData())
        channels = list(channels) if channels is not None else [128, 256, 384]
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.film_dim = int(film_dim)
        self.theta_in_ch = int(theta_in_ch)
        self.padding = str(padding)
        self.theta_mode = str(theta_mode).lower()
        if self.theta_mode not in {"film", "add"}:
            raise ValueError("theta_mode must be one of {'film', 'add'}.")
        self.afno_stage = None if afno_stage is None else int(afno_stage)
        self.afno_inp_shape = None if afno_inp_shape is None else tuple(int(v) for v in afno_inp_shape)

        pad_mode = "circular" if self.padding == "circular" else "zeros"
        self.in_conv = nn.Conv2d(self.in_channels, channels[0], 3, padding=1, padding_mode=pad_mode)

        enc_blocks = []
        downs = []
        for idx, ch in enumerate(channels):
            enc_blocks.append(ConvBlock(ch, ch, self.film_dim, padding=self.padding, dropout=dropout))
            if idx < len(channels) - 1:
                downs.append(DownBlock(ch, channels[idx + 1], self.film_dim, padding=self.padding, dropout=dropout))
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(downs)

        self.attn = (
            BottleneckAttention(
                channels[-1],
                heads=heads,
                dropout=dropout,
                pool=attn_pool,
                max_tokens=attn_max_tokens,
            )
            if bool(use_bottleneck_attn)
            else nn.Identity()
        )

        self.afno = None
        if int(afno_depth) > 0:
            if self.afno_inp_shape is None:
                raise ValueError("afno_inp_shape must be set when afno_depth > 0.")
            if isinstance(afno_patch_size, int):
                afno_patch_size = (afno_patch_size, afno_patch_size)
            elif len(afno_patch_size) != 2:
                raise ValueError("afno_patch_size must be an int or length-2 sequence.")
            if self.afno_stage is None:
                afno_channels = channels[-1]
            else:
                if self.afno_stage < 0 or self.afno_stage >= len(channels):
                    raise ValueError(f"afno_stage {self.afno_stage} is out of range for {len(channels)} stages.")
                afno_channels = channels[self.afno_stage]
            self.afno = AFNO(
                inp_shape=list(self.afno_inp_shape),
                in_channels=afno_channels,
                out_channels=afno_channels,
                patch_size=list(afno_patch_size),
                embed_dim=afno_channels,
                depth=int(afno_depth),
                mlp_ratio=float(afno_mlp_ratio),
                drop_rate=float(dropout),
                num_blocks=int(afno_num_blocks),
                sparsity_threshold=float(afno_sparsity_threshold),
                hard_thresholding_fraction=float(afno_hard_thresholding_fraction),
            )

        ups = []
        for i in reversed(range(len(channels) - 1)):
            ups.append(UpBlock(channels[i + 1], channels[i], self.film_dim, padding=self.padding, dropout=dropout))
        self.ups = nn.ModuleList(ups)

        self.out_norm = _group_norm(channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], self.out_channels, 3, padding=1, padding_mode=pad_mode)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.film_dim, self.film_dim),
            nn.SiLU(),
            nn.Linear(self.film_dim, self.film_dim),
        )
        self.theta_mapper = ThermalPyramidMapper(theta_in_ch, channels, hidden=theta_hidden)
        self.theta_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels[0], self.film_dim),
            nn.SiLU(),
            nn.Linear(self.film_dim, self.film_dim),
        )

        def _make_mod(ch: int) -> nn.Module:
            out_ch = 2 * ch if self.theta_mode == "film" else ch
            return nn.Conv2d(ch, out_ch, kernel_size=1)

        self.theta_enc_mod = nn.ModuleList([_make_mod(ch) for ch in channels])
        self.theta_dec_mod = nn.ModuleList([_make_mod(channels[i]) for i in reversed(range(len(channels) - 1))])

    def _apply_theta_mod(self, x: torch.Tensor, theta_feat: torch.Tensor, mod: nn.Module) -> torch.Tensor:
        if theta_feat.shape[-2:] != x.shape[-2:]:
            theta_feat = nn.functional.interpolate(theta_feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.theta_mode == "film":
            gamma, beta = mod(theta_feat).chunk(2, dim=1)
            return x * (1.0 + gamma) + beta
        return x + mod(theta_feat)

    def _apply_afno(self, x: torch.Tensor) -> torch.Tensor:
        if self.afno is None:
            return x
        if self.afno_inp_shape is not None and x.shape[-2:] != self.afno_inp_shape:
            raise ValueError(
                "AFNO input spatial size mismatch. "
                f"Expected {self.afno_inp_shape}, got {tuple(x.shape[-2:])}."
            )
        return self.afno(x)

    @staticmethod
    def _is_time(tensor: torch.Tensor) -> bool:
        return tensor.dim() <= 1 or (tensor.dim() == 2 and tensor.shape[1] == 1)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *args,
        theta: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        timestep: Optional[torch.Tensor] = None
        kwargs.pop("region_info", None)
        if "theta" in kwargs:
            theta = kwargs.pop("theta")
        if cond is not None and isinstance(cond, torch.Tensor) and self._is_time(cond):
            timestep = cond

        extra_state: Optional[torch.Tensor] = None
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                continue
            if self._is_time(arg):
                if timestep is None:
                    timestep = arg
                continue
            if arg.dim() == 4:
                if theta is None and arg.shape[1] == self.theta_in_ch:
                    theta = arg
                    continue
                if extra_state is None:
                    extra_state = arg
                elif theta is None:
                    theta = arg

        if theta is None:
            raise ValueError("UVitThermalSurrogate requires a thermal field via theta=...")

        if (
            extra_state is not None
            and extra_state.shape[0] == x.shape[0]
            and extra_state.shape[-2:] == x.shape[-2:]
            and x.shape[1] + extra_state.shape[1] == self.in_channels
        ):
            x = torch.cat([x, extra_state], dim=1)

        theta_feats = self.theta_mapper(theta, target_hw=x.shape[-2:])
        ctx = self.theta_ctx(theta_feats[0])
        if timestep is not None:
            t = timestep.view(timestep.shape[0], -1)[:, -1]
            ctx = ctx + self.time_mlp(_sinusoidal_embedding(t.float(), self.film_dim))
        theta_dec_feats = tuple(theta_feats[-2::-1])

        skips: List[torch.Tensor] = []
        h = self.in_conv(x)
        h = self._apply_theta_mod(h, theta_feats[0], self.theta_enc_mod[0])

        for idx, block in enumerate(self.enc_blocks):
            h = block(h, ctx)
            if self.afno is not None and self.afno_stage is not None and idx == self.afno_stage:
                h = self._apply_afno(h)
            if idx < len(self.downs):
                skips.append(h)
                h = self.downs[idx](h, ctx)
                h = self._apply_theta_mod(h, theta_feats[idx + 1], self.theta_enc_mod[idx + 1])

        h = self.attn(h)
        if self.afno is not None and self.afno_stage is None:
            h = self._apply_afno(h)

        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            h = up(h, skip, ctx)
            h = self._apply_theta_mod(h, theta_dec_feats[idx], self.theta_dec_mod[idx])

        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)


__all__ = ["UVitThermalSurrogate"]
