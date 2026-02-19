import pytest
import torch

from models.train.core.loops import _flow_ensemble_batch_sums, _forward_flow_dbfm, _predict_flow_rollout_single
from models.train.core.setup import _flow_uses_source_concat
from models.train.loss_registry import build_flow_loss


def test_flow_loss_l2_default_and_l1_option():
    pred = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    target = torch.zeros_like(pred)

    l2 = build_flow_loss({})
    l1 = build_flow_loss({"flow_matching_loss": "l1"})

    assert float(l2(pred, target)) == pytest.approx(4.0, rel=1e-6, abs=1e-6)
    assert float(l1(pred, target)) == pytest.approx(2.0, rel=1e-6, abs=1e-6)


def test_flow_uses_source_concat_switches_for_dbfm_objective():
    cfg_concat = {"train": {"objective": "rectified_flow_source_anchored"}}
    cfg_dbfm = {"train": {"objective": "dbfm_source_anchored"}}
    assert _flow_uses_source_concat(cfg_concat) is True
    assert _flow_uses_source_concat(cfg_dbfm) is False


class _TinyModel:
    def __init__(self):
        self.last = None

    def __call__(self, x_t, t_match, theta=None):
        self.last = (x_t, t_match, theta)
        return x_t + t_match.view(-1, 1, 1, 1)


def test_forward_flow_dbfm_passes_xt_t_and_theta():
    model = _TinyModel()
    x_t = torch.randn(2, 3, 4, 4)
    t_match = torch.rand(2, 1)
    theta = torch.randn(2, 1, 4, 4)
    out = _forward_flow_dbfm(model, x_t=x_t, t_match=t_match, theta=theta)

    assert out.shape == x_t.shape
    seen_x, seen_t, seen_theta = model.last
    assert seen_x is x_t
    assert seen_theta is theta
    assert torch.allclose(seen_t, t_match)


def test_flow_ensemble_batch_sums_single_sample():
    samples = torch.tensor([[[[[2.0]]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0]]]], dtype=torch.float32)
    term1_sum, term2_sum, var_sum, elem_count = _flow_ensemble_batch_sums(samples, target)

    assert term1_sum == pytest.approx(2.0, rel=1e-6, abs=1e-6)
    assert term2_sum == pytest.approx(0.0, rel=1e-6, abs=1e-6)
    assert var_sum == pytest.approx(0.0, rel=1e-6, abs=1e-6)
    assert elem_count == 1


def test_flow_ensemble_batch_sums_two_members_matches_closed_form():
    samples = torch.tensor(
        [
            [[[[0.0]]]],
            [[[[2.0]]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    term1_sum, term2_sum, var_sum, elem_count = _flow_ensemble_batch_sums(samples, target)

    assert term1_sum == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert term2_sum == pytest.approx(0.5, rel=1e-6, abs=1e-6)
    assert var_sum == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert elem_count == 1


class _SfmIdentityModel:
    def __call__(self, x_in, cond_t, theta=None):
        c = x_in.shape[1] // 2
        return x_in[:, :c]


def test_sfm_rollout_identity_denoiser_is_stable_at_zero_noise():
    model = _SfmIdentityModel()
    x = torch.randn(2, 3, 4, 4)
    y = _predict_flow_rollout_single(
        model=model,
        x=x,
        cond=None,
        theta=None,
        flow_objective="sfm_latent_source_denoise_concat",
        nfe=8,
        flow_noise_std=0.0,
        flow_noise_mode="scalar",
        flow_noise_perturb_source=True,
    )
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)
