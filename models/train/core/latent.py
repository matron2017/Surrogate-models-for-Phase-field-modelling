"""
Helpers to build/apply a latent autoencoder for latent diffusion/flow training.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence

import torch
import torch.nn as nn

from models.latent import LatentAutoencoder2D
from models.train.core.utils import _load_symbol


def build_autoencoder(latent_cfg: Dict[str, Any], state_channels: int, device: torch.device) -> Optional[nn.Module]:
    if not latent_cfg:
        return None
    if bool(latent_cfg.get("use_cached", False)):
        return None
    if not bool(latent_cfg.get("enabled", False)):
        return None
    ae_cfg = latent_cfg.get("autoencoder", {})
    cls_path = ae_cfg.get("file", "models/latent/autoencoder.py")
    cls_name = ae_cfg.get("class", "LatentAutoencoder2D")
    params = dict(ae_cfg.get("params", {}))
    params.setdefault("in_channels", state_channels)
    if params.get("in_channels") != state_channels:
        params["in_channels"] = state_channels
    ModelClass = _load_symbol(cls_path, cls_name)
    ae: nn.Module = ModelClass(**params)
    ckpt = latent_cfg.get("checkpoint")
    if ckpt:
        p = Path(ckpt)
        if p.is_file():
            state = torch.load(p, map_location="cpu")
            # Prefer explicit autoencoder/state_dict entries; fall back to model.* keys if present.
            load_obj = None
            if "autoencoder" in state and state["autoencoder"] is not None:
                load_obj = state["autoencoder"]
            elif "state_dict" in state:
                load_obj = state["state_dict"]
            elif "model" in state:
                model_state = state["model"]
                if any(k.startswith("autoencoder.") for k in model_state):
                    load_obj = {k.replace("autoencoder.", "", 1): v for k, v in model_state.items() if k.startswith("autoencoder.")}
                else:
                    load_obj = model_state
            if load_obj is None:
                load_obj = state
            ae.load_state_dict(load_obj, strict=False)
        else:
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {p}")
    trainable = bool(latent_cfg.get("trainable", False))
    if not trainable:
        for p in ae.parameters():
            p.requires_grad = False
        ae.eval()
    ae._trainable = trainable
    return ae.to(device)


def encode_latent_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    autoencoder: nn.Module,
    use_amp: bool,
    amp_dtype: torch.dtype,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Encode input/target into latents. By default we skip grad (frozen AE); gradients
    # can be enabled when fine-tuning the encoder jointly.
    grad_ctx = nullcontext() if requires_grad else torch.no_grad()
    with grad_ctx:
        with torch.autocast(device_type="cuda", enabled=use_amp and x.is_cuda, dtype=amp_dtype):
            x_lat = autoencoder.encode(x)
            y_lat = autoencoder.encode(y)
    return x_lat, y_lat


def split_latent_fields(z: torch.Tensor, num_fields: int) -> Tuple[torch.Tensor, ...]:
    if z.dim() != 4:
        raise ValueError(f"Expected latent tensor [B,C,H,W], got {tuple(z.shape)}.")
    if num_fields <= 0:
        raise ValueError("num_fields must be positive.")
    if z.size(1) % num_fields != 0:
        raise ValueError(f"Latent channels {z.size(1)} not divisible by num_fields={num_fields}.")
    ch = z.size(1) // num_fields
    return tuple(torch.split(z, ch, dim=1))


def select_latent_fields(fields: Sequence[torch.Tensor], indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        raise ValueError("indices must be non-empty.")
    parts = []
    for idx in indices:
        if idx < 0 or idx >= len(fields):
            raise IndexError(f"Field index {idx} out of range for {len(fields)} fields.")
        parts.append(fields[idx])
    return torch.cat(parts, dim=1)
