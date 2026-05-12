"""UniDB diffusion bridge SDE — PyTorch port from UniDB++ (2769433owo/UniDB-plusplus).

Reference: UniDB++ (UniDB/utils/sde_utils.py, class UniDB)
Adapted for: PDE solidification next-state prediction, pixel space.

Convention
----------
  t = 0   →  x_0  = x_target (clean next-state we want to recover)
  t = T   →  x_T  ≈ N(mu, lambda_sq * I)  (noisy, close to x_source)
  mu      =  x_source (current PDE state — OU process attraction point)

Forward marginal (closed form, Eq. 7 UniDB):
  x_t ~ N( m(t)*x_0 + n(t)*mu,  f_sigma(t)^2 * I )

  m(t) = exp(-Θ_t * dt) * (1 + γ * σ_{t:T}²) / (1 + γ * σ_T²)
  n(t) = 1 - m(t)
  f_sigma(t) = sigma_bars(t) * sigma_t_T(t) / sigma_bars(T)    ← bridge std (peaks at midpoint)

Noise calibration for z-scored PDE data (std ≈ 0.54, frame-to-frame Δ ≈ 0.06):
  lambda_sq = 0.1  →  peak f_sigma ≈ 0.32  (≈ 5× typical frame-to-frame dynamics)

Training: x0-parameterization
  model(cat(x_t, mu), t) → x0_pred
  loss = MSE(x0_pred, x_target)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class UniDBSDE:
    """Ornstein-Uhlenbeck diffusion bridge (UniDB).

    Parameters
    ----------
    lambda_sq : float  — noise scale squared (λ²)
    gamma : float      — Bayes-factor regulariser (controls sharpness of bridge)
    T : int            — number of diffusion timesteps (train with all; infer with fewer)
    schedule : str     — 'cosine' | 'linear' | 'constant'
    device : torch.device
    """

    def __init__(
        self,
        lambda_sq: float = 0.1,
        gamma: float = 0.5,
        T: int = 100,
        schedule: str = "cosine",
        eps: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = T
        self.lambda_sq = lambda_sq
        self.gamma = gamma
        self.device = torch.device(device)
        self._build_schedule(lambda_sq, gamma, T, schedule, eps)

    # ------------------------------------------------------------------ #
    # Schedule construction (from UniDB++ verbatim, but in PyTorch)
    # ------------------------------------------------------------------ #

    def _build_schedule(self, lambda_sq: float, gamma: float, T: int, schedule: str, eps: float) -> None:
        def _cosine(steps, s=0.008):
            x = torch.linspace(0, steps, steps + 2)
            ac = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            ac = ac / ac[0]
            return 1 - ac[1:-1]

        def _linear(steps):
            scale = 1000 / steps
            return torch.linspace(scale * 0.0001, scale * 0.02, steps)

        def _constant(steps, v=1.0):
            return torch.full((steps,), v / steps)

        if schedule == "cosine":
            thetas = _cosine(T)
        elif schedule == "linear":
            thetas = _linear(T)
        else:
            thetas = _constant(T)

        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(lambda_sq * (1 - torch.exp(-2 * thetas_cumsum * dt)))
        sigma_t_T = torch.sqrt(
            lambda_sq * (1 - torch.exp(-2 * (thetas_cumsum[-1] - thetas_cumsum) * dt))
        )

        # Bridge conditional std: sigma_bars(t) * sigma_{t:T} / sigma_T
        # gamma enters ONLY through m(t)/n(t), not through f_sigma (follows original UniDB code).
        f_sigmas = sigma_bars * sigma_t_T / sigma_bars[-1].clamp(min=1e-8)

        self.dt = dt
        self.thetas = thetas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)
        self.sigma_t_T = sigma_t_T.to(self.device)
        self.f_sigmas = f_sigmas.to(self.device)

    def to(self, device) -> "UniDBSDE":
        self.device = torch.device(device)
        self.thetas = self.thetas.to(device)
        self.thetas_cumsum = self.thetas_cumsum.to(device)
        self.sigma_bars = self.sigma_bars.to(device)
        self.sigma_t_T = self.sigma_t_T.to(device)
        self.f_sigmas = self.f_sigmas.to(device)
        return self

    # ------------------------------------------------------------------ #
    # Closed-form marginal coefficients  m(t), n(t), sigma(t)
    # ------------------------------------------------------------------ #

    def _m(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient of x_0 (target) in forward marginal."""
        return (torch.exp(-self.thetas_cumsum[t] * self.dt) *
                (1 + self.gamma * self.sigma_t_T[t] ** 2) /
                (1 + self.gamma * self.sigma_bars[-1] ** 2))

    def _n(self, t: torch.Tensor) -> torch.Tensor:
        """Coefficient of mu (source) in forward marginal."""
        return 1.0 - self._m(t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal std at timestep t."""
        return self.f_sigmas[t]

    # ------------------------------------------------------------------ #
    # Forward process
    # ------------------------------------------------------------------ #

    def q_sample(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from the forward bridge marginal.

        Args:
            x0  : (B,C,H,W) target (next PDE state)
            mu  : (B,C,H,W) source (current PDE state)
            t   : (B,) integer timesteps in [0, T-1]
            noise: optional pre-sampled noise

        Returns:
            x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        mt = self._m(t)[:, None, None, None]
        nt = self._n(t)[:, None, None, None]
        st = self._sigma(t)[:, None, None, None]
        return mt * x0 + nt * mu + st * noise, noise

    # ------------------------------------------------------------------ #
    # Reverse ODE step (x0-parameterization)
    # ------------------------------------------------------------------ #

    def reverse_ode_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One step of the deterministic reverse ODE (DDIM-style).

        Given model prediction x0_pred at timestep t, compute x_{t-1}.
        For t=0 returns x0_pred directly.
        """
        # x_{t-1} = m(t-1)*x0_pred + n(t-1)*mu   (mean only, ODE)
        t_prev = (t - 1).clamp(min=0)
        mt_prev = self._m(t_prev)[:, None, None, None]
        nt_prev = self._n(t_prev)[:, None, None, None]
        x_prev = mt_prev * x0_pred + nt_prev * mu
        # at t=0 we're done → return x0_pred
        mask = (t == 0)[:, None, None, None].float()
        return x0_pred * mask + x_prev * (1 - mask)

    def reverse_sde_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One stochastic reverse step (adds posterior noise)."""
        t_prev = (t - 1).clamp(min=0)
        # posterior std: sigma_{t-1|t} = sigma(t-1) * sigma_step / sigma(t)
        sig_t = self._sigma(t)[:, None, None, None]
        sig_prev = self._sigma(t_prev)[:, None, None, None]
        # f_sigma_1 = sqrt(sigma(t)^2 - m(t)^2*sigma(t-1)^2)
        mt = self._m(t)[:, None, None, None]
        f_sig_step = torch.sqrt((sig_t ** 2 - mt ** 2 * sig_prev ** 2).clamp(min=1e-8))
        r_sig = f_sig_step * sig_prev / sig_t

        # reverse mean (DDPM posterior)
        mt_prev = self._m(t_prev)[:, None, None, None]
        nt_prev = self._n(t_prev)[:, None, None, None]
        rev_mean = mt_prev * x0_pred + nt_prev * mu

        noise = torch.randn_like(x_t)
        x_prev = rev_mean + r_sig * noise
        mask = (t == 0)[:, None, None, None].float()
        return x0_pred * mask + x_prev * (1 - mask)

    # ------------------------------------------------------------------ #
    # Inference: run N-step reverse ODE from x_T → x_0
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        mu: torch.Tensor,
        n_steps: int = 20,
        mode: str = "ode",
        trajectory: list | None = None,
    ) -> torch.Tensor:
        """Run the reverse bridge from x_T (≈mu + noise) to x_0_pred.

        Args:
            model_fn   : callable(x_in, t_norm) → x0_pred
                         where x_in = cat(x_t, mu) (4-ch) and t_norm ∈ [0,1]
            mu         : (B,C,H,W) source / conditioning
            n_steps    : number of denoising steps (15–30 recommended)
            mode       : 'ode' (deterministic) | 'sde' (stochastic)
            trajectory : optional list; intermediate x_t tensors (detached cpu)
                         are appended after each step if not None
        """
        step_ids = torch.linspace(self.T - 1, 0, n_steps + 1).long().to(self.device)
        B = mu.shape[0]

        # x_T: start from mu + lambda noise (bridge endpoint near source)
        x_t = mu + math.sqrt(self.lambda_sq) * torch.randn_like(mu)

        for i in range(n_steps):
            t_idx = step_ids[i]
            t_batch = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
            t_norm = t_idx.float() / (self.T - 1)   # scalar in [0,1]

            x_in = torch.cat([x_t, mu], dim=1)
            x0_pred = model_fn(x_in, t_norm)

            if mode == "ode":
                x_t = self.reverse_ode_step(x_t, x0_pred, mu, t_batch)
            else:
                x_t = self.reverse_sde_step(x_t, x0_pred, mu, t_batch)

            if trajectory is not None:
                trajectory.append(x_t.detach().cpu())

        return x_t
