# conditioning/conditional_scaler.py
from __future__ import annotations
from typing import Sequence, List
import torch
import torch.nn as nn

__all__ = ["ConditionalScaler", "ConditionalFiLM"]

class ConditionalScaler(nn.Module):
    """
    Bonneville-style pre-skip multiplicative conditioning.

    Inputs:
        - cond: (B, cond_dim) tensor of conditioning scalars. Example: [Δt, G].

    Outputs:
        - List[Tensor]: five tensors [(B, 32), (B, 64), (B, 128), (B, 256), (B, 512)]
          for channel-wise scaling of encoder features before skip concatenation.

    Design:
        - Trunk MLP: cond_dim -> 128 -> 128 with SiLU activations.
        - Five linear heads map the shared 128-d embedding to per-level scale vectors.
        - Heads are identity-initialised: weights = 0, bias = 1.0.
        - Only multiplicative scaling is applied to features.
    """

    def __init__(
        self,
        cond_dim: int = 2,
        widths: Sequence[int] = (32, 64, 128, 256, 512),
        hidden: int = 128,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.widths = tuple(int(w) for w in widths)
        self.hidden = int(hidden)

        self.trunk = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(self.hidden, c, bias=True) for c in self.widths])

        if identity_init:
            self._init_identity_heads()

        self._reset_trunk()

    def _reset_trunk(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_identity_heads(self) -> None:
        for head in self.heads:
            nn.init.zeros_(head.weight)
            if head.bias is None:
                raise RuntimeError("Expected bias=True for identity initialisation.")
            nn.init.ones_(head.bias)

    @torch.no_grad()
    def to_identity(self) -> None:
        self._init_identity_heads()

    def forward(self, cond: torch.Tensor) -> List[torch.Tensor]:
        if cond.dim() != 2:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}). Got {tuple(cond.shape)}.")
        if cond.size(-1) != self.cond_dim:
            # If the conditioning vector is missing exactly one slot (e.g., time already appended elsewhere),
            # pad a zero column to avoid shape crashes while keeping behaviour deterministic.
            if cond.size(-1) + 1 == self.cond_dim:
                pad = cond.new_zeros(cond.size(0), 1)
                cond = torch.cat([cond, pad], dim=1)
            else:
                raise ValueError(f"cond must have shape (B, {self.cond_dim}). Got {tuple(cond.shape)}.")
        h = self.trunk(cond)
        return [head(h) for head in self.heads]

    @staticmethod
    def apply(scales: Sequence[torch.Tensor], feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(scales) != len(feats):
            raise ValueError(f"Length mismatch: scales={len(scales)} feats={len(feats)}.")
        out = []
        for s, f in zip(scales, feats):
            if s.size(0) != f.size(0) or s.size(1) != f.size(1):
                raise ValueError(f"Batch or channel mismatch: scale {tuple(s.shape)} vs feat {tuple(f.shape)}.")
            out.append(f * s.unsqueeze(-1).unsqueeze(-1))
        return out

    def extra_repr(self) -> str:
        return f"cond_dim={self.cond_dim}, widths={self.widths}, hidden={self.hidden}"


class ConditionalFiLM(nn.Module):
    """
    Produces per-level FiLM (gamma, beta) parameters for U-Net skips.
    """

    def __init__(
        self,
        cond_dim: int,
        widths: Sequence[int],
        hidden: int = 128,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.widths = tuple(int(w) for w in widths)
        self.hidden = int(hidden)
        self.identity_init = bool(identity_init)

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.cond_dim, self.hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(self.hidden, 2 * width, bias=True),
                )
                for width in self.widths
            ]
        )

        if self.identity_init:
            self._init_identity()

    def _init_identity(self) -> None:
        for head, width in zip(self.heads, self.widths, strict=True):
            last = head[-1]
            if not isinstance(last, nn.Linear):
                raise RuntimeError("Expected final layer to be nn.Linear for bias initialisation.")
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias.data[:width] = 1.0  # gamma ≈ 1
            last.bias.data[width:] = 0.0  # beta  ≈ 0

    def forward(self, cond: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if cond.dim() != 2 or cond.size(-1) != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}). Got {tuple(cond.shape)}.")

        gammas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        for head, width in zip(self.heads, self.widths, strict=True):
            h = head(cond)
            gamma, beta = torch.split(h, width, dim=-1)
            gammas.append(gamma)
            betas.append(beta)
        return gammas, betas

    @staticmethod
    def apply(
        feats: Sequence[torch.Tensor],
        gammas: Sequence[torch.Tensor],
        betas: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not (len(feats) == len(gammas) == len(betas)):
            raise ValueError("Feature and FiLM length mismatch.")
        out: list[torch.Tensor] = []
        for f, g, b in zip(feats, gammas, betas, strict=True):
            if f.size(0) != g.size(0) or f.size(0) != b.size(0):
                raise ValueError("Batch size mismatch between features and FiLM params.")
            if f.size(1) != g.size(1) or f.size(1) != b.size(1):
                raise ValueError("Channel mismatch between features and FiLM params.")
            g_view = g.unsqueeze(-1).unsqueeze(-1)
            b_view = b.unsqueeze(-1).unsqueeze(-1)
            out.append(f * g_view + b_view)
        return out

    def extra_repr(self) -> str:
        return f"cond_dim={self.cond_dim}, widths={self.widths}, hidden={self.hidden}"
