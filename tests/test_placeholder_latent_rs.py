import torch

from models.train.core.loops import _apply_latent_pipeline


def test_placeholder_latent_shapes_and_theta_resize():
    x = torch.randn(2, 2, 128, 128)
    y = torch.randn(2, 2, 128, 128)
    theta = torch.randn(2, 1, 128, 128)
    latent_cfg = {
        "placeholder": {
            "enabled": True,
            "channels": 32,
            "spatial_size": 64,
            "mode": "bilinear",
        }
    }
    x_lat, y_lat, theta_lat = _apply_latent_pipeline(
        x=x,
        y=y,
        theta=theta,
        autoencoder=None,
        autoencoder_trainable=False,
        latent_amp_enabled=False,
        latent_amp_dtype=torch.float16,
        latent_cfg=latent_cfg,
        model_family="flow_matching",
        use_chlast=False,
    )
    assert x_lat.shape == (2, 32, 64, 64)
    assert y_lat.shape == (2, 32, 64, 64)
    assert theta_lat is not None
    assert theta_lat.shape == (2, 1, 64, 64)
    assert torch.isfinite(x_lat).all()
    assert torch.isfinite(y_lat).all()
    assert torch.isfinite(theta_lat).all()
