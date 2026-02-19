import torch

from models.diffusion.scheduler_registry import get_noise_schedule
from models.diffusion.timestep_sampler import get_timestep_sampler


def test_noise_schedules_monotone_betas():
    for name in ["linear", "cosine", "logsnr_laplace"]:
        schedule = get_noise_schedule(name, timesteps=100)
        betas = schedule.betas
        assert betas.dim() == 1 and betas.numel() == 100
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
        assert torch.all(schedule.alpha_bar <= 1)
        assert torch.all(schedule.alpha_bar >= 0)


def test_timestep_samplers_basic_range():
    schedule = get_noise_schedule("linear", timesteps=50)
    for name in ["uniform", "logsnr_importance", "adaptive_region"]:
        sampler = get_timestep_sampler(name, schedule=schedule, device=torch.device("cpu"))
        idx = sampler.sample(batch_size=16)
        assert idx.shape == (16,)
        assert idx.dtype == torch.long
        assert torch.all(idx >= 0) and torch.all(idx < schedule.timesteps)


def test_unidb_schedule_shapes_and_finiteness():
    schedule = get_noise_schedule(
        "unidb_cosine",
        timesteps=20,
        lambda_square=30.0,
        gamma_inv=0.0,
        eps=0.005,
    )
    x0 = torch.randn(4, 2, 8, 8)
    mu = torch.randn_like(x0)
    t = torch.randint(1, schedule.timesteps, (x0.shape[0],), dtype=torch.long)

    x_t, eps = schedule.sample_noisy_state(x0=x0, mu=mu, t=t)
    score = schedule.get_score_from_noise(eps, t)
    xt_1_exp = schedule.reverse_sde_step_mean(x_t, score, t, mu)
    xt_1_opt = schedule.reverse_optimum_step(x_t, x0, t, mu)

    assert x_t.shape == x0.shape
    assert eps.shape == x0.shape
    assert xt_1_exp.shape == x0.shape
    assert xt_1_opt.shape == x0.shape
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(xt_1_exp).all()
    assert torch.isfinite(xt_1_opt).all()


def test_unidb_residual_modulation_shapes_and_finiteness():
    schedule = get_noise_schedule(
        "unidb_cosine",
        timesteps=20,
        lambda_square=30.0,
        gamma_inv=0.0,
        eps=0.005,
        residual_mode="abs",
        residual_normalize=True,
    )
    x0 = torch.randn(4, 2, 8, 8)
    mu = torch.randn_like(x0)
    t = torch.randint(1, schedule.timesteps, (x0.shape[0],), dtype=torch.long)

    x_t, u = schedule.sample_noisy_state(x0=x0, mu=mu, t=t)
    pi = schedule.compute_residual_modulator(x0, mu)
    score = schedule.get_score_from_noise(u, t, pi=pi)
    xt_1_exp = schedule.reverse_sde_step_mean(x_t, score, t, mu)
    xt_1_opt = schedule.reverse_optimum_step(x_t, x0, t, mu)

    assert x_t.shape == x0.shape
    assert u.shape == x0.shape
    assert pi is not None and pi.shape == x0.shape
    assert xt_1_exp.shape == x0.shape
    assert xt_1_opt.shape == x0.shape
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(score).all()
    assert torch.isfinite(xt_1_exp).all()
    assert torch.isfinite(xt_1_opt).all()


def test_unidb_fractional_schedule_shapes_and_finiteness():
    schedule = get_noise_schedule(
        "unidb_fractional",
        timesteps=20,
        lambda_square=30.0,
        gamma_inv=0.0,
        eps=0.005,
        fractional_hurst=0.3,
        fractional_k=8,
        fractional_mix=1.0,
        residual_mode="abs",
        residual_normalize=True,
    )
    x0 = torch.randn(4, 2, 8, 8)
    mu = torch.randn_like(x0)
    t = torch.randint(1, schedule.timesteps, (x0.shape[0],), dtype=torch.long)

    x_t, u = schedule.sample_noisy_state(x0=x0, mu=mu, t=t)
    pi = schedule.compute_residual_modulator(x0, mu)
    score = schedule.get_score_from_noise(u, t, pi=pi)
    xt_1_exp = schedule.reverse_sde_step_mean(x_t, score, t, mu)
    xt_1_opt = schedule.reverse_optimum_step(x_t, x0, t, mu)

    assert schedule.fractional_enabled
    assert schedule.fractional_omega is not None
    assert schedule.fractional_gamma is not None
    assert x_t.shape == x0.shape
    assert u.shape == x0.shape
    assert xt_1_exp.shape == x0.shape
    assert xt_1_opt.shape == x0.shape
    assert torch.isfinite(x_t).all()
    assert torch.isfinite(score).all()
    assert torch.isfinite(xt_1_exp).all()
    assert torch.isfinite(xt_1_opt).all()
