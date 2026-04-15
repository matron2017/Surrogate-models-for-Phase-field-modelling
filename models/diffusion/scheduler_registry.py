"""Noise-schedule registry for diffusion experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch


def _mafbm_gamma_grid(
    k: int,
    gamma_min: float,
    gamma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    gmin = float(gamma_min)
    gmax = float(gamma_max)
    if gmin <= 0 or gmax <= 0:
        raise ValueError("fractional_gamma_min/max must be > 0.")
    if gmax < gmin:
        gmin, gmax = gmax, gmin
    if k == 1:
        return torch.tensor([math.sqrt(gmin * gmax)], device=device, dtype=dtype)
    return torch.exp(
        torch.linspace(
            math.log(gmin),
            math.log(gmax),
            int(k),
            device=device,
            dtype=dtype,
        )
    )


def _mafbm_omega_from_gamma(
    gamma: torch.Tensor,
    hurst: float,
    *,
    horizon_t: float = 1.0,
) -> torch.Tensor:
    """
    Lightweight PyTorch port of MA-fBM quadrature weights used in FDBM code.
    """
    if gamma.numel() == 0:
        return torch.empty_like(gamma)
    h = float(hurst)
    if not (0.0 < h < 1.0):
        raise ValueError(f"fractional_hurst must be in (0,1), got {h}")
    if float(horizon_t) <= 0:
        raise ValueError(f"horizon_t must be > 0, got {horizon_t}")

    g64 = gamma.to(dtype=torch.float64)
    T = float(horizon_t)
    gamma_i, gamma_j = g64[None, :], g64[:, None]
    mat_gamma = gamma_i + gamma_j

    exp_term = torch.exp(-mat_gamma * T)
    A = (T + (exp_term - 1.0) / mat_gamma) / mat_gamma

    low_h = h + 0.5
    high_h = h + 1.5
    low_h_t = torch.full_like(g64, low_h)
    high_h_t = torch.full_like(g64, high_h)
    gammainc_low = torch.special.gammainc(low_h_t, g64 * T)
    gammainc_high = torch.special.gammainc(high_h_t, g64 * T)
    b = (T / torch.pow(g64, low_h)) * gammainc_low - (low_h / torch.pow(g64, high_h)) * gammainc_high

    try:
        omega = torch.linalg.solve(A, b)
    except RuntimeError:
        # Numerical fallback for near-singular systems.
        omega = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
    return omega.to(dtype=gamma.dtype)


def _fractional_time_warp(
    tau: torch.Tensor,
    *,
    hurst: float,
    k: int,
    gamma_min: float,
    gamma_max: float,
    mix: float,
    use_abs_omega: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build a monotone time warp τ -> τ_f using MA-fBM-inspired exponential mixtures.
    """
    if tau.dim() != 1:
        raise ValueError("fractional time warp expects 1-D tau.")
    mix_f = float(mix)
    if k <= 0 or abs(float(hurst) - 0.5) < 1e-6 or mix_f <= 0.0:
        return tau, None, None

    gamma = _mafbm_gamma_grid(
        int(k),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        device=tau.device,
        dtype=tau.dtype,
    )
    omega = _mafbm_omega_from_gamma(gamma, float(hurst), horizon_t=1.0)
    weights = omega.abs() if bool(use_abs_omega) else omega
    if torch.all(weights == 0):
        return tau, omega, gamma

    kernel = 1.0 - torch.exp(-tau[:, None] * gamma[None, :])
    mem = kernel @ weights
    mem = mem - mem[0]
    mem_den = mem[-1].abs().clamp_min(float(eps))
    mem = mem / mem_den
    if not bool(use_abs_omega):
        # Signed omega can cause local non-monotonicity; keep solver-stable monotone warp.
        mem = torch.cummax(mem, dim=0).values
        mem = (mem - mem[0]) / (mem[-1] - mem[0]).clamp_min(float(eps))

    mix_f = max(0.0, min(1.0, mix_f))
    tau_warped = (1.0 - mix_f) * tau + mix_f * mem
    tau_warped = torch.cummax(tau_warped.clamp(0.0, 1.0), dim=0).values
    tau_warped = (tau_warped - tau_warped[0]) / (tau_warped[-1] - tau_warped[0]).clamp_min(float(eps))
    return tau_warped, omega, gamma


@dataclass
class NoiseSchedule:
    betas: torch.Tensor
    kind: str = "vp"

    def __post_init__(self) -> None:
        if self.betas.dim() != 1:
            raise ValueError("betas must be 1-D increasing sequence.")
        if torch.any(self.betas <= 0):
            raise ValueError("betas must be > 0.")
        if torch.any(self.betas >= 1):
            raise ValueError("betas must be < 1.")
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.log_snr = torch.log(self.alpha_bar.clamp_min(1e-12)) - torch.log(
            (1 - self.alpha_bar).clamp_min(1e-12)
        )

    @property
    def timesteps(self) -> int:
        return self.betas.numel()


class VESchedule:
    """Variance-exploding noise schedule σ_t (no betas/alphas)."""

    def __init__(self, sigmas: torch.Tensor):
        if sigmas.dim() != 1:
            raise ValueError("sigmas must be 1-D.")
        if torch.any(sigmas <= 0):
            raise ValueError("sigmas must be > 0 for VE schedule.")
        self.sigmas = sigmas
        self.kind = "ve"
        # For importance sampling compatibility
        self.log_snr = -2.0 * torch.log(self.sigmas)

    @property
    def timesteps(self) -> int:
        return self.sigmas.numel()


@dataclass
class BridgeSchedule:
    """Bridge schedule with coefficients x_t = a_t * xT + b_t * x0 + c_t * eps."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    sigma: float = 1.0
    kind: str = "bridge"

    def __post_init__(self) -> None:
        if self.a.dim() != 1 or self.b.dim() != 1 or self.c.dim() != 1:
            raise ValueError("BridgeSchedule expects 1-D coefficient tensors.")
        if not (self.a.numel() == self.b.numel() == self.c.numel()):
            raise ValueError("BridgeSchedule coefficients must have the same length.")
        if torch.any(self.c < 0):
            raise ValueError("BridgeSchedule noise coefficient c must be >= 0.")
        self.sigma = float(self.sigma)
        self.log_snr = torch.log((self.a**2 + self.b**2).clamp_min(1e-12)) - torch.log(
            (self.c**2).clamp_min(1e-12)
        )

    @property
    def timesteps(self) -> int:
        return self.a.numel()


class UniDBSchedule:
    """
    UniDB/GOUB-style bridge schedule used in DBFM training.

    This keeps the SDE coefficients and helper functions required by the
    reverse-step matching objective:
      - sample_noisy_state(x0, mu, t)
      - get_score_from_noise(noise, t)
      - reverse_sde_step_mean(x_t, score, t, mu)
      - reverse_sde_step(x_t, score, t, mu)
      - reverse_optimum_step(x_t, x0, t, mu)
    """

    def __init__(
        self,
        timesteps: int = 100,
        lambda_square: float = 30.0,
        lambda_rescale_255: bool = True,
        gamma_inv: float = 0.0,
        schedule: str = "cosine",
        eps: float = 0.005,
        input_mode: str = "delta_source_concat",
        residual_mode: str = "none",
        residual_scale: float = 1.0,
        residual_power: float = 1.0,
        residual_normalize: bool = False,
        residual_clip: Optional[float] = None,
        residual_eps: float = 1e-6,
        residual_pi_floor: float = 0.0,
        fractional_hurst: float = 0.5,
        fractional_k: int = 0,
        fractional_gamma_min: float = 0.1,
        fractional_gamma_max: float = 20.0,
        fractional_mix: float = 1.0,
        fractional_use_abs_omega: bool = True,
        fractional_eps: float = 1e-8,
    ):
        self.kind = "unidb"
        self.timesteps = int(timesteps)
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2 for UniDB schedule.")
        self.T = self.timesteps
        self.lambda_rescale_255 = bool(lambda_rescale_255)
        lambda_square_raw = float(lambda_square)
        if self.lambda_rescale_255 and lambda_square_raw >= 1.0:
            self.lambda_square = lambda_square_raw / 255.0
        else:
            self.lambda_square = lambda_square_raw
        self.gamma_inv = float(gamma_inv)
        self.schedule_name = str(schedule).strip().lower()
        self.eps = float(eps)
        self.input_mode = str(input_mode).strip().lower()
        valid_input_modes = {"delta_source_concat", "raw_source_concat"}
        if self.input_mode not in valid_input_modes:
            raise ValueError(
                f"Unknown UniDB input_mode '{self.input_mode}'. Use one of {sorted(valid_input_modes)}."
            )
        self.residual_mode = str(residual_mode).strip().lower()
        self.residual_scale = float(residual_scale)
        self.residual_power = float(residual_power)
        self.residual_normalize = bool(residual_normalize)
        self.residual_clip = None if residual_clip is None else float(residual_clip)
        self.residual_eps = float(residual_eps)
        self.residual_pi_floor = float(residual_pi_floor)
        self.fractional_hurst = float(fractional_hurst)
        self.fractional_k = int(fractional_k)
        self.fractional_gamma_min = float(fractional_gamma_min)
        self.fractional_gamma_max = float(fractional_gamma_max)
        self.fractional_mix = float(fractional_mix)
        self.fractional_use_abs_omega = bool(fractional_use_abs_omega)
        self.fractional_eps = float(fractional_eps)
        self.fractional_enabled = (
            self.fractional_k > 0
            and abs(self.fractional_hurst - 0.5) > 1e-6
            and self.fractional_mix > 0.0
        )
        valid_modes = {"none", "signed", "abs"}
        if self.residual_mode not in valid_modes:
            raise ValueError(f"Unknown residual_mode '{self.residual_mode}'. Use one of {sorted(valid_modes)}.")
        if self.residual_power <= 0:
            raise ValueError("residual_power must be > 0.")
        if self.residual_eps <= 0:
            raise ValueError("residual_eps must be > 0.")
        if self.residual_pi_floor < 0:
            raise ValueError("residual_pi_floor must be >= 0.")
        if not (0.0 < self.fractional_hurst < 1.0):
            raise ValueError(f"fractional_hurst must be in (0,1), got {self.fractional_hurst}.")
        if self.fractional_k < 0:
            raise ValueError(f"fractional_k must be >= 0, got {self.fractional_k}.")
        if self.fractional_mix < 0.0:
            raise ValueError(f"fractional_mix must be >= 0, got {self.fractional_mix}.")
        if self.fractional_eps <= 0:
            raise ValueError(f"fractional_eps must be > 0, got {self.fractional_eps}.")
        self._initialize()

    def _build_thetas(self) -> torch.Tensor:
        T = self.T
        if self.schedule_name == "cosine":
            # Match DBFM's theta construction (length T+1 with indices 0..T).
            s = 0.008
            timesteps = T + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            return 1.0 - alphas_cumprod[1:-1]
        if self.schedule_name == "linear":
            timesteps = T + 1
            scale = 1000.0 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        if self.schedule_name == "constant":
            return torch.ones(T + 1, dtype=torch.float32)
        raise ValueError(
            f"Unknown UniDB schedule '{self.schedule_name}'. Use one of ['cosine','linear','constant']."
        )

    def _initialize(self) -> None:
        thetas_raw = self._build_thetas()
        thetas_cumsum_raw = torch.cumsum(thetas_raw, dim=0) - thetas_raw[0]
        if torch.any(thetas_cumsum_raw < 0):
            raise ValueError("UniDB cumulative schedule must be non-negative.")
        total = thetas_cumsum_raw[-1].clamp_min(self.fractional_eps)
        tau = (thetas_cumsum_raw / total).clamp(0.0, 1.0)
        tau_eff = tau
        fractional_omega = None
        fractional_gamma = None
        if self.fractional_enabled:
            tau_eff, fractional_omega, fractional_gamma = _fractional_time_warp(
                tau,
                hurst=self.fractional_hurst,
                k=self.fractional_k,
                gamma_min=self.fractional_gamma_min,
                gamma_max=self.fractional_gamma_max,
                mix=self.fractional_mix,
                use_abs_omega=self.fractional_use_abs_omega,
                eps=self.fractional_eps,
            )
        thetas_cumsum = tau_eff * total
        thetas = thetas_raw.clone()
        if thetas.numel() > 1:
            thetas[1:] = (thetas_cumsum[1:] - thetas_cumsum[:-1]).clamp_min(self.fractional_eps)

        dt = -1.0 / thetas_cumsum[-1] * math.log(max(self.eps, 1e-8))
        sigmas = torch.sqrt(self.lambda_square**2 * 2.0 * thetas)
        sigma_bars = torch.sqrt(self.lambda_square**2 * (1.0 - torch.exp(-2.0 * thetas_cumsum * dt)))
        sigma_t_T = torch.sqrt(
            self.lambda_square**2
            * (1.0 - torch.exp(-2.0 * (thetas_cumsum[-1] - thetas_cumsum) * dt))
        )
        f_sigmas = sigma_bars * sigma_t_T / sigma_bars[-1].clamp_min(1e-12)

        self.dt = float(dt)
        self.thetas = thetas
        self.sigmas = sigmas
        self.thetas_cumsum = thetas_cumsum
        self.thetas_raw = thetas_raw
        self.sigma_bars = sigma_bars
        self.sigma_t_T = sigma_t_T
        self.f_sigmas = f_sigmas
        self.fractional_omega = fractional_omega
        self.fractional_gamma = fractional_gamma

        # Optional compatibility with existing weighting logic.
        self.log_snr = -2.0 * torch.log(self.f_sigmas.clamp_min(1e-8))

    @staticmethod
    def _ensure_t_index(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.long)
        if t.dim() != 1:
            t = t.view(-1)
        return t.to(dtype=torch.long)

    def _take(self, arr: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        idx = self._ensure_t_index(t).to(device=ref.device)
        vals = arr.to(device=ref.device)[idx]
        view_shape = (idx.shape[0],) + (1,) * (ref.dim() - 1)
        return vals.view(view_shape)

    def _m(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        theta_cumsum_t = self._take(self.thetas_cumsum, t, ref)
        sigma_t_T_t = self._take(self.sigma_t_T, t, ref)
        sigma_bar_T = self.sigma_bars[-1].to(device=ref.device, dtype=ref.dtype)
        num = self.gamma_inv + sigma_t_T_t**2
        den = self.gamma_inv + sigma_bar_T**2
        return torch.exp(-theta_cumsum_t * self.dt) * (num / den.clamp_min(1e-12))

    def _n(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._m(t, ref)

    def f_sigma(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self._take(self.f_sigmas, t, ref)

    def f_mean(self, x0: torch.Tensor, mu: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x0.shape != mu.shape:
            raise ValueError(f"UniDB f_mean requires x0/mu shapes to match (got {x0.shape} vs {mu.shape})")
        return self._m(t, x0) * x0 + self._n(t, x0) * mu

    def compute_residual_modulator(self, x0: torch.Tensor, mu: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute pi-map used for residual-modulated bridge noise.
        - mode=none: returns None (default UniDB behavior).
        - mode=signed: pi = (x0 - mu)
        - mode=abs:    pi = |x0 - mu|
        """
        if self.residual_mode == "none":
            return None
        if x0.shape != mu.shape:
            raise ValueError(f"Residual modulator expects x0/mu shapes to match (got {x0.shape} vs {mu.shape})")
        residual = x0 - mu
        if self.residual_mode == "abs":
            pi = residual.abs()
        else:
            pi = residual

        if self.residual_power != 1.0:
            if self.residual_mode == "abs":
                pi = pi.clamp_min(0.0).pow(self.residual_power)
            else:
                pi = torch.sign(pi) * torch.pow(pi.abs().clamp_min(self.residual_eps), self.residual_power)

        if self.residual_normalize:
            denom = pi.abs().mean(dim=tuple(range(1, pi.dim())), keepdim=True).clamp_min(self.residual_eps)
            pi = pi / denom

        if self.residual_clip is not None:
            c = float(self.residual_clip)
            if self.residual_mode == "abs":
                pi = pi.clamp(min=0.0, max=c)
            else:
                pi = pi.clamp(min=-c, max=c)

        if self.residual_scale != 1.0:
            pi = self.residual_scale * pi

        if self.residual_pi_floor > 0.0:
            floor = float(self.residual_pi_floor)
            if self.residual_mode == "abs":
                pi = pi.clamp_min(floor)
            else:
                pi = torch.sign(pi) * pi.abs().clamp_min(floor)
        return pi

    def sample_noisy_state(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = noise if noise is not None else torch.randn_like(x0)
        mean = self.f_mean(x0, mu, t)
        sigma = self.f_sigma(t, x0)
        pi = self.compute_residual_modulator(x0, mu)
        if pi is None:
            noise_target = eps
        else:
            # RDBM-style target: predict the residual-noise product pi * eps.
            noise_target = pi * eps
        return mean + sigma * noise_target, noise_target

    def get_score_from_noise(
        self,
        noise: torch.Tensor,
        t: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sigma = self.f_sigma(t, noise)
        if pi is None:
            return -noise / sigma.clamp_min(1e-12)
        # If noise is u = pi * eps, then score = -u / (sigma * pi^2).
        den = sigma * pi.pow(2).clamp_min(self.residual_eps)
        return -noise / den.clamp_min(self.residual_eps)

    def _sde_reverse_drift(self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        theta_t = self._take(self.thetas, t, x)
        sigma_t = self._take(self.sigmas, t, x)
        theta_cumsum_t = self._take(self.thetas_cumsum, t, x)
        theta_cumsum_T = self.thetas_cumsum[-1].to(device=x.device, dtype=x.dtype)
        sigma_t_T_t = self._take(self.sigma_t_T, t, x)

        tmp = torch.exp(2.0 * (theta_cumsum_t - theta_cumsum_T) * self.dt)
        drift_h = -(sigma_t**2 * tmp) / (self.gamma_inv + sigma_t_T_t**2).clamp_min(1e-12) * (x - mu)
        t_idx = self._ensure_t_index(t).to(device=x.device)
        mask = (t_idx == self.T).view(-1, *([1] * (x.dim() - 1)))
        drift_h = torch.where(mask, torch.zeros_like(drift_h), drift_h)
        return (theta_t * (mu - x) + drift_h - sigma_t**2 * score) * self.dt

    def reverse_sde_step_mean(self, x_t: torch.Tensor, score: torch.Tensor, t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return x_t - self._sde_reverse_drift(x_t, score, t, mu)

    def reverse_sde_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
        *,
        stochastic: bool = True,
        noise_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        UniDB reverse update:
          x_{t-1} = x_t - drift(x_t,t) - sigma_t * sqrt(dt) * N(0, I)

        With stochastic=False, this is the deterministic mean-ODE step.
        """
        x_next = self.reverse_sde_step_mean(x_t, score, t, mu)
        if not stochastic:
            return x_next
        sigma_t = self._take(self.sigmas, t, x_t)
        eps = noise if noise is not None else torch.randn_like(x_t)
        scale = max(float(noise_scale), 0.0) * math.sqrt(max(self.dt, 0.0))
        return x_next - sigma_t * scale * eps

    def reverse_optimum_step(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        t_idx = self._ensure_t_index(t).to(device=x_t.device)
        if torch.any(t_idx <= 0):
            raise ValueError("UniDB reverse_optimum_step requires timesteps >= 1.")
        t_prev = t_idx - 1

        m_t = self._m(t_idx, x_t)
        m_prev = self._m(t_prev, x_t)
        n_t = self._n(t_idx, x_t)
        n_prev = self._n(t_prev, x_t)
        f_sigma_t = self.f_sigma(t_idx, x_t)
        f_sigma_prev = self.f_sigma(t_prev, x_t)

        f_m = m_t / m_prev.clamp_min(1e-12)
        f_n = n_t - n_prev * m_t / m_prev.clamp_min(1e-12)
        f_sigma_1 = torch.sqrt((f_sigma_t**2 - f_sigma_prev**2 * f_m**2).clamp_min(0.0))
        f_mean_prev = m_prev * x0 + n_prev * mu

        num = f_sigma_prev**2 * f_m * (x_t - f_n * mu) + f_sigma_1**2 * f_mean_prev
        den = (f_sigma_t**2).clamp_min(1e-12)
        return num / den

    def predict_x0_from_noisy(
        self,
        x_t: torch.Tensor,
        noise_target: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover x0 estimate from a noisy UniDB state:
          x_t = m(t) * x0 + n(t) * mu + f_sigma(t) * noise_target
        """
        m_t = self._m(t, x_t)
        n_t = self._n(t, x_t)
        f_sigma_t = self.f_sigma(t, x_t)
        den = m_t.clamp_min(1e-12)
        return (x_t - n_t * mu - f_sigma_t * noise_target) / den


def _linear_betas(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    t = steps / timesteps
    return torch.cos((t + s) / (1 + s) * torch.pi / 2).pow(2).clamp_min(1e-4)


def LinearNoiseSchedule(timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> NoiseSchedule:
    betas = _linear_betas(timesteps, beta_start, beta_end)
    return NoiseSchedule(betas)


def CosineNoiseSchedule(timesteps: int = 1000, s: float = 0.008) -> NoiseSchedule:
    alpha_bar = _cosine_alpha_bar(timesteps, s=s)
    alphas = alpha_bar[1:] / alpha_bar[:-1]
    betas = (1 - alphas).clamp(min=1e-5, max=0.999)
    return NoiseSchedule(betas)


def LogSNRLaplaceSchedule(timesteps: int = 1000, loc: float = 2.0, scale: float = 1.0) -> NoiseSchedule:
    t = torch.linspace(0.0, 1.0, timesteps, dtype=torch.float32)
    centered = t - 0.5
    logsnr = loc - torch.abs(centered) * (2 * scale)
    alpha_bar = torch.sigmoid(logsnr)
    alphas = alpha_bar.clone()
    alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
    betas = (1 - alphas).clamp(min=1e-5, max=0.999)
    return NoiseSchedule(betas)


def LearnedNoiseSchedule(beta_series: Sequence[float]) -> NoiseSchedule:
    betas = torch.as_tensor(beta_series, dtype=torch.float32)
    if betas.dim() != 1:
        raise ValueError("beta_series must be 1-D.")
    return NoiseSchedule(betas)


def ExponentialVESchedule(
    timesteps: int = 1000, sigma_min: float = 0.01, sigma_max: float = 50.0
) -> VESchedule:
    """
    σ_t grows exponentially from sigma_min to sigma_max over timesteps.
    Matches the VE-style exponential schedule used in GenCFD/RecFlow comparisons.
    """
    t = torch.linspace(0.0, 1.0, timesteps, dtype=torch.float32)
    log_sig = torch.log(torch.tensor(sigma_min)) + t * (torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min)))
    sigmas = torch.exp(log_sig)
    return VESchedule(sigmas)


def BrownianBridgeSchedule(timesteps: int = 1000, sigma: float = 1.0) -> BridgeSchedule:
    """Simple Brownian-bridge coefficients with linear interpolation in time."""
    t = torch.linspace(0.0, 1.0, timesteps, dtype=torch.float32)
    a = t
    b = 1.0 - t
    c = torch.sqrt((t * (1.0 - t)).clamp_min(0.0)) * float(sigma)
    return BridgeSchedule(a=a, b=b, c=c, sigma=float(sigma))


def BridgeFractionalSchedule(
    timesteps: int = 1000,
    sigma: float = 1.0,
    fractional_hurst: float = 0.3,
    fractional_k: int = 8,
    fractional_gamma_min: float = 0.1,
    fractional_gamma_max: float = 20.0,
    fractional_mix: float = 1.0,
    fractional_use_abs_omega: bool = True,
    fractional_eps: float = 1e-8,
) -> BridgeSchedule:
    """
    Fractional Brownian bridge schedule in the bridge family (kind='bridge').

    Keeps x_t = a_t * xT + b_t * x0 + c_t * eps, while warping the bridge
    time-grid with MA-fBM-inspired memory terms.
    """
    t = torch.linspace(0.0, 1.0, timesteps, dtype=torch.float32)
    t_eff = t
    if int(fractional_k) > 0 and abs(float(fractional_hurst) - 0.5) > 1e-6 and float(fractional_mix) > 0.0:
        t_eff, _omega, _gamma = _fractional_time_warp(
            t,
            hurst=float(fractional_hurst),
            k=int(fractional_k),
            gamma_min=float(fractional_gamma_min),
            gamma_max=float(fractional_gamma_max),
            mix=float(fractional_mix),
            use_abs_omega=bool(fractional_use_abs_omega),
            eps=float(fractional_eps),
        )
    a = t_eff
    b = 1.0 - t_eff
    c = torch.sqrt((t_eff * (1.0 - t_eff)).clamp_min(0.0)) * float(sigma)
    return BridgeSchedule(a=a, b=b, c=c, sigma=float(sigma))


def UniDBCosineSchedule(
    timesteps: int = 100,
    lambda_square: float = 30.0,
    lambda_rescale_255: bool = True,
    gamma_inv: float = 0.0,
    eps: float = 0.005,
    input_mode: str = "delta_source_concat",
    residual_mode: str = "none",
    residual_scale: float = 1.0,
    residual_power: float = 1.0,
    residual_normalize: bool = False,
    residual_clip: Optional[float] = None,
    residual_eps: float = 1e-6,
    residual_pi_floor: float = 0.0,
    fractional_hurst: float = 0.5,
    fractional_k: int = 0,
    fractional_gamma_min: float = 0.1,
    fractional_gamma_max: float = 20.0,
    fractional_mix: float = 1.0,
    fractional_use_abs_omega: bool = True,
    fractional_eps: float = 1e-8,
) -> UniDBSchedule:
    return UniDBSchedule(
        timesteps=timesteps,
        lambda_square=lambda_square,
        lambda_rescale_255=lambda_rescale_255,
        gamma_inv=gamma_inv,
        schedule="cosine",
        eps=eps,
        input_mode=input_mode,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        residual_power=residual_power,
        residual_normalize=residual_normalize,
        residual_clip=residual_clip,
        residual_eps=residual_eps,
        residual_pi_floor=residual_pi_floor,
        fractional_hurst=fractional_hurst,
        fractional_k=fractional_k,
        fractional_gamma_min=fractional_gamma_min,
        fractional_gamma_max=fractional_gamma_max,
        fractional_mix=fractional_mix,
        fractional_use_abs_omega=fractional_use_abs_omega,
        fractional_eps=fractional_eps,
    )


def UniDBFractionalSchedule(
    timesteps: int = 100,
    lambda_square: float = 30.0,
    lambda_rescale_255: bool = True,
    gamma_inv: float = 0.0,
    eps: float = 0.005,
    input_mode: str = "delta_source_concat",
    residual_mode: str = "none",
    residual_scale: float = 1.0,
    residual_power: float = 1.0,
    residual_normalize: bool = False,
    residual_clip: Optional[float] = None,
    residual_eps: float = 1e-6,
    residual_pi_floor: float = 0.0,
    fractional_hurst: float = 0.3,
    fractional_k: int = 8,
    fractional_gamma_min: float = 0.1,
    fractional_gamma_max: float = 20.0,
    fractional_mix: float = 1.0,
    fractional_use_abs_omega: bool = True,
    fractional_eps: float = 1e-8,
) -> UniDBSchedule:
    return UniDBSchedule(
        timesteps=timesteps,
        lambda_square=lambda_square,
        lambda_rescale_255=lambda_rescale_255,
        gamma_inv=gamma_inv,
        schedule="cosine",
        eps=eps,
        input_mode=input_mode,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        residual_power=residual_power,
        residual_normalize=residual_normalize,
        residual_clip=residual_clip,
        residual_eps=residual_eps,
        residual_pi_floor=residual_pi_floor,
        fractional_hurst=fractional_hurst,
        fractional_k=fractional_k,
        fractional_gamma_min=fractional_gamma_min,
        fractional_gamma_max=fractional_gamma_max,
        fractional_mix=fractional_mix,
        fractional_use_abs_omega=fractional_use_abs_omega,
        fractional_eps=fractional_eps,
    )


_SCHEDULE_REGISTRY: Dict[str, callable] = {
    "linear": LinearNoiseSchedule,
    "cosine": CosineNoiseSchedule,
    "logsnr_laplace": LogSNRLaplaceSchedule,
    "learned": LearnedNoiseSchedule,
    "exponential_ve": ExponentialVESchedule,
    "bridge_linear": BrownianBridgeSchedule,
    "bridge_fractional": BridgeFractionalSchedule,
    "unidb_cosine": UniDBCosineSchedule,
    "unidb_fractional": UniDBFractionalSchedule,
}


def get_noise_schedule(name: str, **kwargs) -> Union[NoiseSchedule, VESchedule, BridgeSchedule, UniDBSchedule]:
    key = str(name).strip().lower()
    if key not in _SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown noise_schedule '{name}'. Registered: {sorted(_SCHEDULE_REGISTRY)}")
    return _SCHEDULE_REGISTRY[key](**kwargs)
