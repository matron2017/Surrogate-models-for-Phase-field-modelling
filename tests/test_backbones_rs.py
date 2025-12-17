import torch

from models.backbones.unet_conv_att_cond import UNet_SSA_PreSkip_Full
from models.backbones.uafno_cond import UAFNO_PreSkip_Full
from models.backbones.fno_field import FNO_Field2D
from models.backbones.uvit_film import UVitFiLMVelocity


def _forward_model(model):
    x = torch.randn(2, 2, 64, 64)
    cond = torch.randn(2, 2)
    y = model(x, cond)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 2, 64, 64)
    assert torch.isfinite(y).all()


def test_unet_ssa_forward_cpu():
    model = UNet_SSA_PreSkip_Full(n_channels=2, n_classes=2, cond_dim=2, in_factor=16, ssa_heads=1)
    _forward_model(model)


def test_unet_ssa_forward_with_timestep():
    model = UNet_SSA_PreSkip_Full(n_channels=2, n_classes=2, cond_dim=3, in_factor=16, ssa_heads=1)
    x = torch.randn(1, 2, 64, 64)
    t = torch.tensor([1])
    cond = torch.randn(1, 2)
    y = model(x, t, cond)
    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()


def test_uafno_forward_cpu():
    model = UAFNO_PreSkip_Full(
        n_channels=2,
        n_classes=2,
        cond_dim=2,
        in_factor=8,
        afno_inp_shape=(4, 4),
        afno_depth=2,
        num_blocks=2,
        afno_mlp_ratio=2.0,
    )
    _forward_model(model)


def test_uafno_forward_with_timestep():
    model = UAFNO_PreSkip_Full(
        n_channels=2,
        n_classes=2,
        cond_dim=3,
        in_factor=8,
        afno_inp_shape=(4, 4),
        afno_depth=2,
        num_blocks=2,
        afno_mlp_ratio=2.0,
    )
    x = torch.randn(1, 2, 64, 64)
    t = torch.tensor([2])
    cond = torch.randn(1, 2)
    y = model(x, t, cond)
    assert y.shape == (1, 2, 64, 64)
    assert torch.isfinite(y).all()


def test_fno_forward_cpu():
    model = FNO_Field2D(
        n_channels=2,
        n_classes=2,
        cond_dim=2,
        embed_channels=48,
        fno_inp_shape=(64, 64),
        fno_depth=2,
        fno_num_modes=8,
        fno_decoder_layers=1,
    )
    _forward_model(model)


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
