"""Conditional UNet helpers leveraging diffusers.UNet2DModel."""
from __future__ import annotations

from typing import Any, Dict

from diffusers import UNet2DModel


DEFAULT_UNET_CFG: Dict[str, Any] = {
    "sample_size": 128,
    "in_channels": 10,
    "out_channels": 4,
    "block_out_channels": [320, 640, 960, 960],
    "num_res_blocks": 3,
    "down_block_types": [
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ],
    "attention_head_dim": 8,
}


def build_phasefield_unet(cfg: Dict[str, Any] | None = None) -> UNet2DModel:
    params = DEFAULT_UNET_CFG | (cfg or {})
    return UNet2DModel(**params)
