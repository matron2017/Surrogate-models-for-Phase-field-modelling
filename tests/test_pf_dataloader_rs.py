import numpy as np
import torch
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core.pf_dataloader import PFPairDataset


def _make_mock_pf_file(
    path: Path,
    with_weights: bool = False,
    weight_channels: int = 2,
    include_dt_norm: bool = True,
) -> Path:
    T, C, H, W = 3, 4, 6, 5
    pairs_idx = np.array([[0, 1], [1, 2]], dtype=np.int64)
    pairs_dt = np.linspace(0.0, 1.0, len(pairs_idx)).astype(np.float32)
    grad_series = np.linspace(0.0, 1.0, T).astype(np.float32)
    images = np.random.randn(T, C, H, W).astype(np.float32)
    times = np.arange(T, dtype=np.int64)
    time_phys = np.linspace(0.0, 0.5, T).astype(np.float64)
    time_mean = float(time_phys.mean())
    time_std = float(time_phys.std()) or 1.0
    time_norm = ((time_phys - time_mean) / time_std).astype(np.float32)

    with h5py.File(path, "w") as h5:
        g = h5.create_group("sim_0")
        g.create_dataset("pairs_idx", data=pairs_idx)
        g.create_dataset("thermal_gradient_series_norm", data=grad_series)
        g.create_dataset("images", data=images)
        g.attrs["effective_dt"] = 1.0
        g.create_dataset("times", data=times)
        g.create_dataset("time_phys", data=time_phys)
        g.create_dataset("time_phys_norm", data=time_norm)
        g.attrs["time_mean"] = time_mean
        g.attrs["time_std"] = time_std
        g.attrs["zscore_eps_time"] = 1e-12
        if include_dt_norm:
            g.create_dataset("pairs_dt_norm", data=pairs_dt)

    if with_weights:
        weight_path = path.with_name(path.stem + "_weights.h5")
        with h5py.File(weight_path, "w") as hw:
            g = hw.create_group("sim_0")
            hw_data = np.abs(np.random.randn(T, weight_channels, H, W)).astype(np.float32)
            g.create_dataset("wavelet_weights", data=hw_data)
        return weight_path
    return path


def test_pf_pair_dataset_shapes(tmp_path):
    main_path = tmp_path / "pf_mock.h5"
    _make_mock_pf_file(main_path)

    ds = PFPairDataset(
        h5_path=str(main_path),
        input_channels=[0, 1],
        target_channels=[2, 3],
        limit_per_group=1,
    )

    assert len(ds) == 1
    sample = ds[0]
    assert sample["input"].shape == (2, 6, 5)
    assert sample["target"].shape == (2, 6, 5)
    assert sample["cond"].shape == (2,)
    assert torch.allclose(sample["cond"], torch.tensor([0.0, 0.0], dtype=torch.float32), atol=1e-6)


def test_pf_pair_dataset_includes_weights(tmp_path):
    main_path = tmp_path / "pf_mock.h5"
    weight_path = _make_mock_pf_file(main_path, with_weights=True, weight_channels=1)

    ds = PFPairDataset(
        h5_path=str(main_path),
        input_channels=[0],
        target_channels=[1],
        limit_per_group=2,
        weight_h5=str(weight_path),
    )

    sample = ds[0]
    assert "weight" in sample
    assert sample["weight"].shape == (1, 6, 5)
    # pair 0 -> j=1 uses time_phys_norm[1] == 0.0
    assert torch.allclose(sample["cond"], torch.tensor([0.0, 0.0], dtype=torch.float32), atol=1e-6)


def test_pf_pair_dataset_time_fallback(tmp_path):
    main_path = tmp_path / "pf_mock_fallback.h5"
    _make_mock_pf_file(main_path, include_dt_norm=False)

    ds = PFPairDataset(
        h5_path=str(main_path),
        input_channels=[0],
        target_channels=[1],
        limit_per_group=2,
    )

    sample = ds[1]
    assert sample["cond"].shape == (2,)
    # pair 1 -> j=2 uses time_phys_norm[2] = (0.5 - 0.25) / std = 1.2247449...
    assert torch.allclose(sample["cond"][0], torch.tensor(1.2247449, dtype=torch.float32), atol=1e-5)
    assert torch.isfinite(sample["cond"]).all()
