# conditioning/conditional_scaler.py
from __future__ import annotations
from typing import Sequence, List
import torch
import torch.nn as nn

__all__ = ["ConditionalScaler"]

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
        if cond.dim() != 2 or cond.size(-1) != self.cond_dim:
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
