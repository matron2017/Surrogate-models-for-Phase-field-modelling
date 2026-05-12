"""Fractional diffusion bridge SDE — PyTorch implementation.

Inspired by MAfBM (Markov Approximation of fractional Brownian Motion) from
FDBM_unpaired (Springenberg et al., Fraunhofer HHI), re-implemented in
PyTorch without JAX dependency.

Forward marginal (continuous time t ∈ [0,1]):
  x_t = (1-t)*x_0 + t*x_T + sigma_bridge(t, H)*noise

where:
  sigma_bridge(t, H) = sigma_max * (t*(1-t))^H / peak_normalization
  H = 0.5 → standard Brownian bridge: sqrt(t*(1-t))
  H > 0.5 → smoother paths (persistent increments), variance suppressed near endpoints

Noise calibration for z-scored PDE data (std ≈ 0.54, frame-to-frame Δ ≈ 0.06):
  sigma_max = 0.3  →  peak noise ≈ 0.3  (≈ 5× frame-to-frame dynamics)

Convention (same as UniDB module):
  t = 0   →  x_0  = x_target
  t = T   →  x_T  ≈ x_source (condition, no noise added at endpoint)
  Forward: x_0 → x_T (adding noise toward source)
  Reverse: x_T → x_0 (denoising, 15–30 steps)

Training: x0-parameterization  model(cat(x_t, x_source), t) → x0_pred
"""

from __future__ import annotations
import math
import torch


class FracBridgeSDE:
    """Fractional Brownian bridge SDE.

    Parameters
    ----------
    H : float    — Hurst index (0.5 = standard BB, > 0.5 = smoother)
    sigma_max : float — peak noise std (at t=0.5 for H=0.5)
    T : int      — number of discrete timesteps
    device
    """

    def __init__(
        self,
        H: float = 0.7,
        sigma_max: float = 0.3,
        T: int = 100,
        device: torch.device | str = "cpu",
    ) -> None:
        assert 0.5 <= H <= 1.0, "Hurst index must be in [0.5, 1.0]"
        self.H = H
        self.sigma_max = sigma_max
        self.T = T
        self.device = torch.device(device)
        self._build_schedule()

    def _build_schedule(self) -> None:
        # Continuous time grid, t ∈ (0, 1)
        ts = torch.linspace(0.0, 1.0, self.T + 2)[1:-1]  # (T,), avoids 0 and 1

        # Linear drift coefficients: alpha(t) = t, 1-alpha(t) = 1-t
        alpha = ts                  # weight on x_T (source side at t→1)
        one_minus_alpha = 1.0 - ts  # weight on x_0 (target side at t→0)

        # fBm bridge variance profile: sigma(t) ~ (t*(1-t))^H * sigma_max / peak
        # Normalise so that max sigma = sigma_max
        raw_sigma = (ts * (1.0 - ts)).pow(self.H)
        peak = raw_sigma.max().clamp(min=1e-8)
        sigma_t = raw_sigma / peak * self.sigma_max

        self.ts = ts.to(self.device)
        self.alpha = alpha.to(self.device)
        self.one_minus_alpha = one_minus_alpha.to(self.device)
        self.sigma_t = sigma_t.to(self.device)

    def to(self, device) -> "FracBridgeSDE":
        self.device = torch.device(device)
        self.ts = self.ts.to(device)
        self.alpha = self.alpha.to(device)
        self.one_minus_alpha = self.one_minus_alpha.to(device)
        self.sigma_t = self.sigma_t.to(device)
        return self

    # ------------------------------------------------------------------ #
    # Forward process
    # ------------------------------------------------------------------ #

    def q_sample(
        self,
        x0: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from the forward fractional bridge marginal.

        Args:
            x0   : (B,C,H,W) target (next PDE state, at t=0)
            x_T  : (B,C,H,W) source (current PDE state, at t=T)
            t    : (B,) integer indices in [0, T-1]
            noise: optional pre-sampled Gaussian noise

        Returns:
            x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a  = self.alpha[t][:, None, None, None]
        na = self.one_minus_alpha[t][:, None, None, None]
        s  = self.sigma_t[t][:, None, None, None]
        return a * x_T + na * x0 + s * noise, noise

    # ------------------------------------------------------------------ #
    # Reverse ODE step (x0-parameterization, DDIM-style)
    # ------------------------------------------------------------------ #

    def reverse_ode_step(
        self,
        x0_pred: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic reverse ODE step from t → t-1."""
        t_prev = (t - 1).clamp(min=0)
        a_prev  = self.alpha[t_prev][:, None, None, None]
        na_prev = self.one_minus_alpha[t_prev][:, None, None, None]
        x_prev  = a_prev * x_T + na_prev * x0_pred
        mask = (t == 0)[:, None, None, None].float()
        return x0_pred * mask + x_prev * (1 - mask)

    def reverse_sde_step(
        self,
        x0_pred: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Stochastic reverse step (adds posterior noise)."""
        t_prev = (t - 1).clamp(min=0)
        a_prev  = self.alpha[t_prev][:, None, None, None]
        na_prev = self.one_minus_alpha[t_prev][:, None, None, None]
        # Posterior noise: sigma_{t-1} (smaller near endpoints)
        s_prev = self.sigma_t[t_prev][:, None, None, None]
        x_prev = a_prev * x_T + na_prev * x0_pred + s_prev * torch.randn_like(x0_pred)
        mask = (t == 0)[:, None, None, None].float()
        return x0_pred * mask + x_prev * (1 - mask)

    # ------------------------------------------------------------------ #
    # Inference: run N-step reverse from x_T → x_0
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        x_T: torch.Tensor,
        n_steps: int = 20,
        mode: str = "ode",
        trajectory: list | None = None,
    ) -> torch.Tensor:
        """Reverse bridge: x_T (source ≈ x_source) → x_0 (target).

        Args:
            model_fn   : callable(x_in, t_norm) → x0_pred
                         where x_in = cat(x_t, x_T) (4-ch)
            x_T        : (B,C,H,W) starting point (x_source / current PDE state)
            n_steps    : denoising steps (15–30)
            mode       : 'ode' | 'sde'
            trajectory : optional list; intermediate x_t tensors (detached cpu)
                         are appended after each step if not None
        """
        step_ids = torch.linspace(self.T - 1, 0, n_steps + 1).long().to(self.device)
        B = x_T.shape[0]
        x_t = x_T.clone()

        for i in range(n_steps):
            t_idx = step_ids[i]
            t_batch = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
            t_norm = t_idx.float() / (self.T - 1)

            x_in = torch.cat([x_t, x_T], dim=1)
            x0_pred = model_fn(x_in, t_norm)

            if mode == "ode":
                x_t = self.reverse_ode_step(x0_pred, x_T, t_batch)
            else:
                x_t = self.reverse_sde_step(x0_pred, x_T, t_batch)

            if trajectory is not None:
                trajectory.append(x_t.detach().cpu())

        return x_t
