import numpy as np
import torch
import h5py
import pytest
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
    include_thermal_field: bool = False,
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
        g.attrs["dx"] = 1.0
        g.attrs["thermal_gradient_raw"] = 2.0
        g.attrs["pulling_speed"] = 0.0
        g.attrs["x0_dx"] = 0.0
        g.attrs["x_min"] = 0.0
        g.attrs["x_max"] = float(W)
        g.attrs["y_min"] = 0.0
        g.attrs["y_max"] = float(H)
        g.create_dataset("times", data=times)
        g.create_dataset("time_phys", data=time_phys)
        g.create_dataset("time_phys_norm", data=time_norm)
        g.attrs["time_mean"] = time_mean
        g.attrs["time_std"] = time_std
        g.attrs["zscore_eps_time"] = 1e-12
        if include_dt_norm:
            g.create_dataset("pairs_dt_norm", data=pairs_dt)
        if include_thermal_field:
            thermal = np.full((T, 1, H, W), 5.0, dtype=np.float32)
            g.create_dataset("thermal_field", data=thermal)

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
    assert "cond" not in sample


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
    assert "cond" not in sample


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
    assert sample["input"].shape == (1, 6, 5)
    assert sample["target"].shape == (1, 6, 5)
    assert "cond" not in sample


def test_pf_pair_dataset_adds_thermal_channel(tmp_path):
    main_path = tmp_path / "pf_mock_thermal.h5"
    _make_mock_pf_file(main_path)

    ds = PFPairDataset(
        h5_path=str(main_path),
        input_channels=[0, 1],
        target_channels=[2, 3],
        limit_per_group=1,
        add_thermal=True,
        return_cond=False,
        thermal_on_target=False,
    )

    sample = ds[0]
    assert "cond" not in sample
    assert sample["input"].shape == (3, 6, 5)
    assert sample["target"].shape == (2, 6, 5)
    # Thermal map is linear in x with G=2.0 and x centers starting at 0.5.
    assert torch.allclose(sample["input"][-1, 0, 0], torch.tensor(1.0), atol=1e-6)


def test_pf_pair_dataset_uses_precomputed_thermal(tmp_path):
    main_path = tmp_path / "pf_mock_thermal_pre.h5"
    _make_mock_pf_file(main_path, include_thermal_field=True)

    ds = PFPairDataset(
        h5_path=str(main_path),
        input_channels=[0, 1],
        target_channels=[2, 3],
        limit_per_group=1,
        add_thermal=True,
        return_cond=False,
    )

    sample = ds[0]
    assert sample["input"].shape[0] == 3
    assert torch.allclose(sample["input"][-1], torch.full((6, 5), 5.0, dtype=torch.float32))


def test_pf_pair_dataset_rejects_return_cond_true(tmp_path):
    main_path = tmp_path / "pf_mock_no_scalar.h5"
    _make_mock_pf_file(main_path)

    with pytest.raises(ValueError, match="return_cond=True is no longer supported"):
        PFPairDataset(
            h5_path=str(main_path),
            input_channels=[0],
            target_channels=[1],
            return_cond=True,
        )
