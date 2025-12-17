"""Autoencoder helpers wired to diffusers.AutoencoderKL."""
from __future__ import annotations

from typing import Any, Dict

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


DEFAULT_VAE_CFG: Dict[str, Any] = {
    "in_channels": 2,
    "out_channels": 2,
    "latent_channels": 4,
    "sample_size": 1024,
    "block_out_channels": [256, 512, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
}


def build_phasefield_vae(cfg: Dict[str, Any] | None = None) -> AutoencoderKL:
    params = DEFAULT_VAE_CFG | (cfg or {})
    return AutoencoderKL(**params)
