"""Bridge model wrapper — wraps UNetFiLMAttn for diffusion bridge use.

Architecture changes vs deterministic surrogate:
  in_channels : 4 (2 source channels + 2 noisy target channels)
  out_channels: 2 (predicted x0, i.e., clean next state)
  use_time    : True  (diffusion timestep t ∈ [0,1] injected via sinusoidal emb)
  Control branch still receives theta (thermal, 1 channel)

Parameterization: x0-prediction
  model(cat(x_noisy, x_source), t_norm) → x0_pred
  loss = MSE(x0_pred, x_target)   (optionally + wavelet terms)

Both UniDB and FracBridge use this same wrapper. The SDE module is passed
in at runtime and is not part of the model itself.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

# The shared UNetFiLMAttn lives in models/ at the workspace root
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # Phase_field_surrogates/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.unet_film_bottleneck import UNetFiLMAttn  # noqa: E402


class BridgePDEModel(nn.Module):
    """UNetFiLMAttn wrapper for diffusion bridge PDE surrogates.

    Parameters
    ----------
    channels : list[int]  — UNet channel widths
    control_channels : list[int]  — ControlNet widths
    afno_depth : int  — AFNO depth at bottleneck
    afno_num_blocks : int  — AFNO blocks (channels[-1] must be divisible by this)
    afno_mlp_ratio : float
    afno_inp_shape : list[int, int]  — spatial size at bottleneck (e.g. [32,32])
    film_dim : int  — FiLM embedding size
    dropout : float
    """

    def __init__(
        self,
        channels: Sequence[int] = (128, 192, 256, 384, 512),
        control_channels: Sequence[int] = (64, 96, 128, 160, 192),
        afno_depth: int = 4,
        afno_num_blocks: int = 16,
        afno_mlp_ratio: float = 4.0,
        afno_inp_shape: Sequence[int] = (32, 32),
        film_dim: int = 512,
        dropout: float = 0.0,
        padding: str = "mixed_y_reflect",
    ) -> None:
        super().__init__()
        self.unet = UNetFiLMAttn(
            in_channels=4,           # cat(x_noisy[2], x_source[2])
            out_channels=2,          # x0_pred
            cond_dim=2,              # (t_diffusion, 0.) — minimal scalar conditioning
            channels=list(channels),
            num_blocks=[2] * len(channels),
            bottleneck_blocks=(2, 2),
            attn_heads=8,
            use_bottleneck_attn=False,
            afno_depth=afno_depth,
            afno_mlp_ratio=afno_mlp_ratio,
            afno_num_blocks=afno_num_blocks,
            afno_inp_shape=list(afno_inp_shape),
            afno_patch_size=[1, 1],
            afno_sparsity_threshold=0.01,
            afno_hard_thresholding_fraction=1.0,
            film_dim=film_dim,
            time_emb_dim=128,
            dropout=dropout,
            padding=padding,
            use_time=True,           # diffusion time via sinusoidal embedding
            skip_film=True,
            film_mode="affine",
            use_control_branch=True,
            hint_channels=1,         # thermal field theta
            control_strength=1.5,
            control_channels=list(control_channels),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,          # (B,2,H,W) noisy target at time t
        x_source: torch.Tensor,         # (B,2,H,W) source / conditioning
        t_diffusion: torch.Tensor,      # (B,) float in [0,1]
        theta: torch.Tensor,            # (B,1,H,W) thermal field
        control_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Predict x0 (clean target) from noisy intermediate.

        Returns x0_pred of shape (B,2,H,W).
        """
        x_in = torch.cat([x_noisy, x_source], dim=1)  # (B,4,H,W)
        # cond_vec: physical scalars (kept minimal; 2 zeros is fine)
        cond = torch.zeros(x_noisy.shape[0], 2, device=x_noisy.device, dtype=x_noisy.dtype)
        # diffusion timestep passed as 3rd positional arg (UNet sinusoidal embedding)
        t_unet = t_diffusion.float().view(-1, 1)
        return self.unet(x_in, cond, t_unet, hint=theta,
                         control_strength=control_strength)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Convenience: build from config dict
# ---------------------------------------------------------------------------

def build_bridge_model(cfg: dict) -> BridgePDEModel:
    mp = cfg.get("model", {}).get("params", {})
    return BridgePDEModel(**mp)
