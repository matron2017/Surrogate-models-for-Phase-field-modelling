import torch

from models.train.core.utils import _prepare_batch, _q_sample


def test_prepare_batch_field():
    batch = {
        "input": torch.randn(2, 3, 4, 4),
        "target": torch.randn(2, 3, 4, 4),
        "cond": torch.randn(2, 2),
    }
    cond_cfg = {"enabled": True, "source": "field", "cond_dim": 2}
    x, y, cond = _prepare_batch(batch, torch.device("cpu"), cond_cfg, use_chlast=False)
    assert x.shape == (2, 3, 4, 4)
    assert y.shape == (2, 3, 4, 4)
    assert cond.shape == (2, 2)


def test_q_sample_shapes():
    class DummySchedule:
        def __init__(self):
            self.alpha_bar = torch.linspace(0.1, 0.9, steps=5)

    schedule = DummySchedule()
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.tensor([0, 4], dtype=torch.long)
    x_t, eps = _q_sample(x0, t, schedule)
    assert x_t.shape == x0.shape
    assert eps.shape == x0.shape
