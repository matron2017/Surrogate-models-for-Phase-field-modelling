"""
Diffusion-ready AFNO U-Net with timestep embeddings and FiLM-conditioned skips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d import UNet2DOutput

from physicsnemo.models.afno.afno import AFNO
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

from .unet_parts import DoubleConv, Down, Up, OutConv
from models.conditioning.skip_condition import ConditionalFiLM, ConditionalScaler


@dataclass
class UAFNODiffusionMetaData(ModelMetaData):
    name: str = "UAFNO_DiffusionResidualPatches"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UAFNO_DiffusionUNet(Module):
    """
    U-Net + AFNO bottleneck used as a diffusion denoiser on residual patches.

    Expects concatenated [noisy_residual, x_t_patch] inputs and predicts noise for the residual.
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 2,
        in_factor: int = 40,
        cond_dim: int = 2,
        afno_inp_shape: Tuple[int, int] = (64, 64),
        afno_depth: int = 12,
        num_blocks: int = 16,
        afno_mlp_ratio: float = 12.0,
        time_embed_dim: int = 256,
        film_hidden: int = 128,
    ) -> None:
        super().__init__(meta=UAFNODiffusionMetaData())
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        C = in_factor

        # encoder
        self.inc = DoubleConv(n_channels, C)
        self.down1 = Down(C, 2 * C)
        self.down2 = Down(2 * C, 4 * C)
        self.down3 = Down(4 * C, 8 * C)
        self.down4 = Down(8 * C, 16 * C)

        self.skip_film = ConditionalFiLM(
            cond_dim=self.cond_dim + self.time_embed_dim,
            widths=[C, 2 * C, 4 * C, 8 * C, 16 * C],
            hidden=film_hidden,
            identity_init=True,
        )

        # bottleneck
        self.bot_pool = nn.MaxPool2d(2)
        self.bot_pre = nn.Sequential(
            nn.Conv2d(16 * C, 16 * C, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        pooled_shape = (afno_inp_shape[0] // 2, afno_inp_shape[1] // 2)
        self.afno_bottleneck = AFNO(
            inp_shape=list(pooled_shape),
            in_channels=16 * C,
            out_channels=16 * C,
            patch_size=(1, 1),
            embed_dim=16 * C,
            depth=afno_depth,
            mlp_ratio=afno_mlp_ratio,
            drop_rate=0.0,
            num_blocks=num_blocks,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        )

        self.bot_scaler = ConditionalScaler(
            cond_dim=self.cond_dim,
            widths=[16 * C],
            hidden=film_hidden,
            identity_init=True,
        )

        self.bot_post = nn.Sequential(
            nn.Conv2d(16 * C, 16 * C, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.bot_up = nn.ConvTranspose2d(16 * C, 16 * C, kernel_size=2, stride=2)

        # decoder
        self.up1 = Up(16 * C, 8 * C, bilinear=False)
        self.up2 = Up(8 * C, 4 * C, bilinear=False)
        self.up3 = Up(4 * C, 2 * C, bilinear=False)
        self.up4 = Up(2 * C, C, bilinear=False)
        self.outc = OutConv(C, n_classes)

        # timestep embeddings
        self.time_proj = Timesteps(
            num_channels=self.time_embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.time_embed_dim,
            time_embed_dim=self.time_embed_dim,
            act_fn="silu",
            out_dim=self.time_embed_dim,
        )
        self.time_to_film = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def timestep_embedding(self, timesteps: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=device)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.to(device=device)
        t_proj = self.time_proj(timesteps)
        return self.time_embedding(t_proj)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond_vec: torch.Tensor,
        return_dict: bool = True,
    ) -> UNet2DOutput | torch.Tensor:
        if cond_vec.dim() != 2 or cond_vec.size(0) != sample.size(0):
            raise ValueError("cond_vec must be (B, cond_dim) aligned with the batch.")

        device = sample.device
        cond_vec = cond_vec.to(device=device, dtype=sample.dtype)

        # timestep embedding
        t_emb = self.timestep_embedding(timestep, device)
        t_emb = self.time_to_film(t_emb)
        film_input = torch.cat([cond_vec, t_emb], dim=-1)

        # encoder
        x1 = self.inc(sample)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        gammas, betas = self.skip_film(film_input)
        x1s, x2s, x3s, x4s, x5s = ConditionalFiLM.apply(
            feats=[x1, x2, x3, x4, x5],
            gammas=gammas,
            betas=betas,
        )

        # bottleneck
        x5s = self.bot_pool(x5s)
        x5s = self.bot_pre(x5s)
        x5s = self.afno_bottleneck(x5s)

        (sb,) = self.bot_scaler(cond_vec)
        x5s = x5s * sb.unsqueeze(-1).unsqueeze(-1)

        x5s = self.bot_post(x5s)
        x5s = self.bot_up(x5s)

        # decoder
        x = self.up1(x5s, x4s)
        x = self.up2(x, x3s)
        x = self.up3(x, x2s)
        x = self.up4(x, x1s)
        x = self.outc(x)

        if return_dict:
            return UNet2DOutput(sample=x)
        return x
