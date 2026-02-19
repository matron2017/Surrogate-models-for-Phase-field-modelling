import torch

from models.backbones.uvit_film import UVitFiLMVelocity
from models.backbones.uvit_thermal import UVitThermalSurrogate


def _forward_model(model):
    x = torch.randn(2, 2, 64, 64)
    cond = torch.randn(2, 2)
    y = model(x, cond)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 2, 64, 64)
    assert torch.isfinite(y).all()


def test_uvit_forward_cpu():
    model = UVitFiLMVelocity(
        in_channels=2,
        out_channels=2,
        cond_dim=3,
        channels=[16, 32, 32],
        heads=2,
        film_dim=64,
        attn_max_tokens=1024,
    )
    x = torch.randn(1, 2, 128, 128)
    cond = torch.randn(1, 3)
    y = model(x, cond)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 2, 128, 128)
    assert torch.isfinite(y).all()


def test_uvit_thermal_forward_cpu():
    model = UVitThermalSurrogate(
        in_channels=2,
        out_channels=2,
        channels=[16, 32, 32],
        heads=2,
        film_dim=64,
        theta_in_ch=1,
        theta_mode="film",
    )
    x = torch.randn(1, 2, 128, 128)
    theta = torch.randn(1, 1, 1024, 1024)
    y = model(x, theta=theta)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 2, 128, 128)
    assert torch.isfinite(y).all()


def test_uvit_thermal_flow_style_forward_cpu():
    model = UVitThermalSurrogate(
        in_channels=4,
        out_channels=2,
        channels=[16, 32, 32],
        heads=2,
        film_dim=64,
        theta_in_ch=1,
        theta_mode="film",
    )
    x_t = torch.randn(1, 2, 64, 64)
    z_n = torch.randn(1, 2, 64, 64)
    t = torch.rand(1, 1)
    theta = torch.randn(1, 1, 256, 256)
    y = model(x_t, t, z_n, theta)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()


def test_uvit_film_cond_dim0_uses_time_cpu():
    model = UVitFiLMVelocity(
        in_channels=2,
        out_channels=2,
        cond_dim=0,
        channels=[16, 32, 32],
        heads=2,
        film_dim=64,
        attn_max_tokens=1024,
    )
    x = torch.randn(1, 2, 64, 64)
    t1 = torch.tensor([[10.0]])
    t2 = torch.tensor([[900.0]])
    y1 = model(x, t1)
    y2 = model(x, t2)
    assert torch.max(torch.abs(y1 - y2)).item() > 0.0


def test_uvit_film_flow_style_with_theta_cpu():
    model = UVitFiLMVelocity(
        in_channels=4,
        out_channels=2,
        cond_dim=0,
        channels=[16, 32, 32],
        heads=2,
        film_dim=64,
        use_theta=True,
        theta_in_ch=1,
        theta_mode="film",
    )
    x_t = torch.randn(1, 2, 64, 64)
    z_n = torch.randn(1, 2, 64, 64)
    t = torch.rand(1, 1)
    theta = torch.randn(1, 1, 256, 256)
    y = model(x_t, t, z_n, theta)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()
