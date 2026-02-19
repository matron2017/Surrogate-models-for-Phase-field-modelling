import pytest
import torch

from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.loss_registry import build_diffusion_loss


def test_diffusion_loss_min_snr_weights_high_snr_down():
    cfg = {
        "base": "mse",
        "diffusion_weighting": {
            "kind": "min_snr",
            "min_snr_gamma": 5.0,
        },
    }
    loss_fn = build_diffusion_loss(cfg)

    pred = torch.tensor([[[[2.0]]], [[[2.0]]]], dtype=torch.float32)
    target_eps = torch.zeros_like(pred)
    # snr = [100, 1] -> weights = [0.05, 1.0]
    log_snr = torch.log(torch.tensor([100.0, 1.0], dtype=torch.float32))
    loss = loss_fn(pred, target_eps, log_snr=log_snr)

    expected = ((4.0 * 0.05) + (4.0 * 1.0)) / 2.0
    assert float(loss) == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_diffusion_loss_min_snr_falls_back_without_log_snr():
    cfg = {
        "base": "mse",
        "diffusion_weighting": {"kind": "min_snr", "min_snr_gamma": 5.0},
    }
    loss_fn = build_diffusion_loss(cfg)
    pred = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    target_eps = torch.zeros_like(pred)
    loss = loss_fn(pred, target_eps)
    assert float(loss) == pytest.approx(4.0, rel=1e-6, abs=1e-6)


def test_diffusion_loss_unknown_weighting_raises():
    cfg = {
        "base": "mse",
        "diffusion_weighting": {"kind": "not_a_real_scheme"},
    }
    loss_fn = build_diffusion_loss(cfg)
    pred = torch.zeros(1, 1, 2, 2)
    target_eps = torch.zeros_like(pred)
    with pytest.raises(ValueError, match="Unknown diffusion_weighting.kind"):
        _ = loss_fn(pred, target_eps, log_snr=torch.zeros(1))


def test_diffusion_loss_unidb_reverse_step_smoke():
    cfg = {
        "base": "mse",
        "diffusion_objective": "unidb_reverse_step",
        "diffusion_matching_loss": "l1",
    }
    loss_fn = build_diffusion_loss(cfg)
    schedule = get_noise_schedule(
        "unidb_cosine",
        timesteps=20,
        lambda_square=30.0,
        gamma_inv=0.0,
        eps=0.005,
    )

    x0 = torch.randn(3, 2, 8, 8)
    mu = torch.randn_like(x0)
    t = torch.randint(1, schedule.timesteps, (x0.shape[0],), dtype=torch.long)
    x_t, eps_true = schedule.sample_noisy_state(x0=x0, mu=mu, t=t)
    eps_pred = eps_true + 0.1 * torch.randn_like(eps_true)

    loss = loss_fn(
        eps_pred,
        eps_true,
        target=x0,
        source=mu,
        noisy=x_t,
        t=t,
        schedule=schedule,
    )
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_diffusion_loss_unidb_reverse_step_smoke_residual_mode():
    cfg = {
        "base": "mse",
        "diffusion_objective": "unidb_reverse_step",
        "diffusion_matching_loss": "l1",
    }
    loss_fn = build_diffusion_loss(cfg)
    schedule = get_noise_schedule(
        "unidb_cosine",
        timesteps=20,
        lambda_square=30.0,
        gamma_inv=0.0,
        eps=0.005,
        residual_mode="abs",
        residual_normalize=True,
    )

    x0 = torch.randn(3, 2, 8, 8)
    mu = torch.randn_like(x0)
    t = torch.randint(1, schedule.timesteps, (x0.shape[0],), dtype=torch.long)
    x_t, u_true = schedule.sample_noisy_state(x0=x0, mu=mu, t=t)
    u_pred = u_true + 0.1 * torch.randn_like(u_true)

    loss = loss_fn(
        u_pred,
        u_true,
        target=x0,
        source=mu,
        noisy=x_t,
        t=t,
        schedule=schedule,
    )
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
