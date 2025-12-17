"""Lightweight diffusion operators to wrap backbone models."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from models.diffusion.configs import DiffusionConfig


class DiffusionOperator:
    """Applies a backbone repeatedly according to the diffusion config."""

    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg

    def step_model(self, model, state: torch.Tensor, cond_vec: Optional[torch.Tensor]):
        """Single forward step; override if mixing analytic updates."""
        return model(state, cond_vec)

    def rollout_model(
        self, model, init_state: torch.Tensor, cond_vec: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = init_state
        self.last_trajectory: Sequence[torch.Tensor] | tuple[()] = ()
        traj = [] if self.cfg.record_trajectory else None
        for _ in range(self.cfg.n_steps):
            x = self.step_model(model, x, cond_vec)
            if traj is not None:
                traj.append(x)
        if traj is not None:
            # Expose trajectory for downstream visualisation without changing the return type.
            self.last_trajectory: Sequence[torch.Tensor] = tuple(traj)
        return x
