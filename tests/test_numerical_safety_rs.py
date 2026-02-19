import pytest
import torch

from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.loss_functions import relative_mass_error
from models.train.core.metric_stats import (
    _compute_vrmse_from_stats,
    _init_channel_stats,
    _update_channel_stats,
)


def test_unidb_rejects_nonpositive_residual_eps():
    with pytest.raises(ValueError):
        get_noise_schedule(
            "unidb_cosine",
            timesteps=20,
            lambda_square=30.0,
            gamma_inv=0.0,
            eps=0.005,
            residual_mode="abs",
            residual_eps=0.0,
        )


def test_vrmse_stats_safe_when_vrmse_eps_is_zero():
    stats = _init_channel_stats(num_channels=2, device=torch.device("cpu"))
    pred = torch.zeros(2, 2, 4, 4)
    target = torch.zeros_like(pred)
    _update_channel_stats(stats, pred, target)
    out = _compute_vrmse_from_stats(stats, vrmse_eps=0.0)

    assert out is not None
    assert torch.isfinite(torch.tensor(out["vmse"])).item()
    assert torch.isfinite(torch.tensor(out["vrmse"])).item()
    assert torch.isfinite(torch.tensor(out["vmse_per_channel"])).all().item()
    assert torch.isfinite(torch.tensor(out["vrmse_per_channel"])).all().item()


def test_relative_mass_error_safe_when_eps_is_zero_and_true_mass_zero():
    u_pred = torch.zeros(3, 8, 8)
    u_true = torch.zeros_like(u_pred)
    rel = relative_mass_error(u_pred, u_true, eps=0.0)
    assert torch.isfinite(rel).all().item()
    assert torch.allclose(rel, torch.zeros_like(rel))
