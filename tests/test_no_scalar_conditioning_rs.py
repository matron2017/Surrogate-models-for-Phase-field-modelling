from pathlib import Path

import pytest
import torch
import yaml

from models.train.core.config import _load_config
from models.train.core.setup import _build_model_and_task


def test_load_config_rejects_scalar_conditioning(tmp_path):
    cfg = {
        "paths": {"h5": {"train": "/tmp/train.h5"}},
        "dataloader": {"file": "dummy.py", "class": "DummyDataset"},
        "loader": {"batch_size": 1},
        "model": {"backbone": "uvit_thermal", "params": {"in_channels": 32, "out_channels": 32}},
        "trainer": {"epochs": 1},
        "conditioning": {"enabled": True, "cond_dim": 2, "source": "field"},
    }
    cfg_path = Path(tmp_path) / "cfg_scalar.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="Scalar conditioning has been removed"):
        _load_config(str(cfg_path))


def test_build_model_rejects_scalar_conditioning():
    cfg = {
        "train": {"model_family": "surrogate"},
        "task": {"name": "surrogate"},
        "model": {"backbone": "uvit_thermal", "params": {"in_channels": 32, "out_channels": 32}},
        "conditioning": {"enabled": True},
        "trainer": {"channels_last": False},
    }

    with pytest.raises(ValueError, match="Scalar conditioning has been removed"):
        _build_model_and_task(cfg, torch.device("cpu"), model_in_channels=32, cond_dim=0)


def test_scalar_utility_scripts_deleted_from_repository():
    core_root = Path(__file__).resolve().parents[1] / "models" / "train" / "core"
    removed_paths = [
        core_root / "parallel_solid_data.py",
        core_root / "parallel_model_test.py",
        core_root / "number_params_models.py",
        core_root / "smoketest_train.py",
        core_root / "pf_dataloader_old.py",
        core_root / "legacy" / "parallel_solid_data.py",
        core_root / "legacy" / "parallel_model_test.py",
        core_root / "legacy" / "number_params_models.py",
        core_root / "legacy" / "smoketest_train.py",
        core_root / "legacy" / "pf_dataloader_scalar_legacy.py",
        Path(__file__).resolve().parents[1] / "slurm" / "test_parallel.sh",
        Path(__file__).resolve().parents[1] / "slurm" / "test_parallel_single_node.sh",
        Path(__file__).resolve().parents[1] / "slurm" / "test_parallel_multinode.sh",
        Path(__file__).resolve().parents[1] / "slurm" / "params_print.sh",
        Path(__file__).resolve().parents[1] / "models" / "backbones" / "uafno_cond.py",
        Path(__file__).resolve().parents[1] / "models" / "backbones" / "unet_conv_att_cond.py",
        Path(__file__).resolve().parents[1] / "models" / "backbones" / "fno_field.py",
        Path(__file__).resolve().parents[1] / "models" / "backbones" / "uafno_diffusion.py",
        Path(__file__).resolve().parents[1] / "models" / "conditioning" / "skip_condition.py",
        Path(__file__).resolve().parents[1] / "experiments" / "diffusion_prototype" / "src" / "train_ddpm_residual.py",
        Path(__file__).resolve().parents[1] / "experiments" / "diffusion_prototype" / "src" / "infer_ddpm_residual.py",
        Path(__file__).resolve().parents[1] / "slurm" / "train_diffusion.sh",
    ]

    for p in removed_paths:
        assert not p.exists(), f"Scalar-conditioning legacy file should be deleted: {p}"
