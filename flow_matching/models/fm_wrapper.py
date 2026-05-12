"""Flow Matching model wrapper — reuses UNetFiLMAttn (identical to BridgePDEModel).

Architecture is EXACTLY the same as BridgePDEModel used for diffusion bridges:
  in_channels : 4  (2 source channels + 2 interpolated-target channels)
  out_channels: 2  (predicted x1 = clean next state, i.e. x0-parameterization)
  use_time    : True  (flow time t ∈ [0,1] injected via sinusoidal embedding)
  Control branch receives theta (thermal, 1 channel)

Parameterization: x1-prediction (equivalent to x0-prediction in bridge)
  At training time t:  x_t = (1-t)*x_source + t*x_target   (no noise added)
  model(cat(x_t, x_source), t) → x1_pred  (predicts clean x_target)
  loss = MSE(x1_pred, x_target)  [+ optional wavelet term]

Inference — Euler ODE from t=0 to t=1:
  x_t = x_source   (start)
  for each step:
      x1_pred = model(x_t, x_source, t, theta)
      v = (x1_pred - x_t) / (1 - t + eps)    # implied velocity
      x_t = x_t + v * dt
  x_pred = x_t  ≈ x_target

Practical difference vs diffusion bridge:
  - Bridge: stochastic OU/fBm path, SDE at inference (can sample multiple times)
  - Flow matching: deterministic ODE path, single forward pass at inference
  - Same backbone, same conditioning, same training budget → honest comparison
"""

from __future__ import annotations
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # Phase_field_surrogates/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.unet_film_bottleneck import UNetFiLMAttn  # noqa: E402


class FMPDEModel(nn.Module):
    """UNetFiLMAttn wrapper for Rectified Flow / Flow Matching PDE surrogates.

    Architecture is identical to BridgePDEModel. Only the training/inference
    objective differs (linear ODE instead of stochastic bridge).

    Parameters mirror BridgePDEModel for easy config sharing.
    """

    def __init__(
        self,
        channels: Sequence[int] = (192, 320, 512, 640, 832),
        control_channels: Sequence[int] = (96, 160, 256, 320, 416),
        afno_depth: int = 12,
        afno_num_blocks: int = 16,
        afno_mlp_ratio: float = 12.0,
        afno_inp_shape: Sequence[int] = (32, 32),
        film_dim: int = 512,
        dropout: float = 0.0,
        padding: str = "mixed_y_reflect",
        hint_channels: int = 1,
        control_strength: float = 1.5,
    ):
        super().__init__()
        self.unet = UNetFiLMAttn(
            in_channels=4,        # cat(x_t, x_source): 2 + 2
            out_channels=2,       # x1_pred: 2 channels (phi, c)
            channels=list(channels),
            afno_depth=afno_depth,
            afno_num_blocks=afno_num_blocks,
            afno_mlp_ratio=afno_mlp_ratio,
            afno_inp_shape=list(afno_inp_shape),
            afno_patch_size=[1, 1],
            afno_sparsity_threshold=0.01,
            afno_hard_thresholding_fraction=1.0,
            film_dim=film_dim,
            time_emb_dim=128,
            dropout=dropout,
            padding=padding,
            use_time=True,
            skip_film=True,
            film_mode="affine",
            use_control_branch=True,
            hint_channels=hint_channels,
            control_strength=control_strength,
            control_channels=list(control_channels),
        )

    def forward(
        self,
        x_interp: torch.Tensor,         # (B,2,H,W) interpolated state at time t
        x_source: torch.Tensor,         # (B,2,H,W) source frame (conditioning)
        t_flow: torch.Tensor,           # (B,) float in [0,1]
        theta: torch.Tensor,            # (B,1,H,W) thermal field
        control_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Predict x1 (clean target) from interpolated intermediate.

        Returns x1_pred of shape (B,2,H,W).
        """
        x_in = torch.cat([x_interp, x_source], dim=1)  # (B,4,H,W)
        cond = torch.zeros(x_interp.shape[0], 2, device=x_interp.device, dtype=x_interp.dtype)
        t_unet = t_flow.float().view(-1, 1)
        return self.unet(x_in, cond, t_unet, hint=theta,
                         control_strength=control_strength)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Euler ODE inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: nn.Module,
    x_source: torch.Tensor,   # (B,2,H,W)
    theta: torch.Tensor,      # (B,1,H,W)
    n_steps: int = 20,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run Euler ODE integration from x_source → x_pred.

    Returns x_pred of shape (B,2,H,W).

    Using x1-parameterization:
        v(x_t, t) = (x1_pred - x_t) / (1 - t + eps)
    which guarantees the ODE trajectory ends at x1_pred at t=1.
    """
    if device is None:
        device = x_source.device
    x_source = x_source.to(device)
    theta = theta.to(device)
    B = x_source.shape[0]

    x_t = x_source.clone()
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t_norm = torch.full((B,), t_val, device=device, dtype=x_t.dtype)
        x1_pred = model(x_t, x_source, t_norm, theta)
        # Euler step using implied velocity
        v = (x1_pred - x_t) / max(1.0 - t_val, 1e-5)
        x_t = x_t + v * dt

    return x_t
