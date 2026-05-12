from __future__ import annotations

from typing import Any, Dict

import torch

from Phase_field_surrogates.models.unet_film_bottleneck import UNetFiLMAttn
from .config import load_yaml


def build_model_from_config(cfg: Dict[str, Any] | None = None, tiny: bool = False) -> UNetFiLMAttn:
    cfg = cfg or load_yaml()
    params = dict(cfg["model"]["params"])
    if tiny:
        params.update(
            channels=[16, 32, 64, 128],
            num_blocks=[1, 1, 1, 1],
            bottleneck_blocks=[1, 1],
            afno_depth=0,
            afno_inp_shape=None,
            film_dim=64,
            time_emb_dim=32,
            attn_heads=4,
        )
    return UNetFiLMAttn(**params)


def make_model_inputs(sample: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = sample["input"].unsqueeze(0).float()
    y = sample["target"].unsqueeze(0).float()
    source = x[:, :-1]
    thermal = x[:, -1:]
    # UNIDB delta_source_concat contract: 32-channel model state plus 32-channel source context.
    model_x = torch.cat([y, source], dim=1)
    cond_vec = torch.zeros((model_x.shape[0], 1), dtype=model_x.dtype)
    timestep = torch.ones((model_x.shape[0], 1), dtype=model_x.dtype)
    return model_x, cond_vec, timestep, thermal
