import textwrap
from pathlib import Path

import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train import train as train_main


TOY_MODEL = """
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, cond=None):
        return self.net(x)
"""

TOY_DATASET = """
import torch
from torch.utils.data import Dataset

class ToyPairsDataset(Dataset):
    def __init__(self, h5_path=None, length=32):
        gen = torch.Generator().manual_seed(0)
        self.inputs = torch.randn(length, 2, 8, 8, generator=gen)
        self.targets = 2.0 * self.inputs
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx].clone(),
            "target": self.targets[idx].clone(),
            "gid": f"sim_{idx:04d}",
            "pair_index": int(idx),
        }
"""


def _write_file(path: Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_train_one_step_reduces_loss(tmp_path):
    model_py = tmp_path / "toy_model.py"
    data_py = tmp_path / "toy_dataset.py"
    _write_file(model_py, TOY_MODEL)
    _write_file(data_py, TOY_DATASET)

    cfg = {
        "paths": {
            "h5": {
                "train": {"h5_path": str(tmp_path / "dummy_train.h5")},
            },
            "sim_map": None,
        },
        "dataloader": {
            "file": str(data_py),
            "class": "ToyPairsDataset",
            "args": {"length": 32},
        },
        "model": {
            "file": str(model_py),
            "class": "TinyModel",
            "params": {"in_channels": 2, "out_channels": 2},
        },
        "conditioning": {"enabled": False},
        "loader": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        "trainer": {
            "seed": 0,
            "deterministic": True,
            "device": "cpu",
            "epochs": 3,
            "use_val": False,
            "out_dir": str(tmp_path / "runs"),
            "grad_clip": 0.0,
            "resume": False,
            "metrics": {"mae": False, "psnr": False},
            "amp": {"enabled": False},
            "use_wavelet_weights": False,
        },
        "optim": {"name": "adam", "lr": 1e-2, "weight_decay": 0.0},
        "sched": {"name": "none"},
        "mlflow": {"enabled": False},
    }

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    train_main.main(str(cfg_path))

    metrics_path = Path(cfg["trainer"]["out_dir"]) / "TinyModel" / "metrics.csv"
    lines = metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 2  # header + at least one epoch
    measurements = [float(row.split(",")[2]) for row in lines[1:]]
    assert measurements[-1] < measurements[0]
