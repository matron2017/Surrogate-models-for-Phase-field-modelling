from __future__ import annotations

import csv

import torch

from models.train.core.logging import MLFlowContext, RunLogger


def test_metrics_csv_has_objective_and_consistent_row_width(tmp_path):
    cfg = {
        "paths": {"h5": {"train": "dummy_train.h5", "val": "dummy_val.h5"}},
        "trainer": {"epochs": 2, "amp": {}, "early_stop": {}, "metrics": {}},
        "optim": {},
        "sched": {},
        "conditioning": {},
    }
    run = RunLogger(
        cfg=cfg,
        run_dir=tmp_path,
        tag_src=torch.nn.Conv2d(2, 2, kernel_size=3, padding=1),
        device=torch.device("cpu"),
        seed=1,
        deterministic=True,
        x0=torch.zeros(2, 8, 8),
        H=8,
        W=8,
        mlflow_ctx=MLFlowContext(active=False, run_id=None, parent_run_id=None),
        want_mae=True,
        want_psnr=True,
        want_vrmse=True,
        want_spectral=True,
        monitor_split_es="val",
        monitor_mode_es="min",
        resume_path=None,
        start_epoch=1,
    )

    run.log_epoch(
        epoch=1,
        train_metrics={
            "mse": 1.0,
            "rmse": 1.0,
            "objective": 0.9,
            "mae": 0.5,
            "vrmse": 0.8,
            "spectral_rmse": 0.7,
        },
        val_metrics={
            "mse": 1.1,
            "rmse": 1.0488088482,
            "objective": 0.95,
            "mae": 0.55,
            "vrmse": 0.85,
            "spectral_rmse": 0.75,
        },
        gid_stats={"sim_0001": (1.0, 1)},
        val_gid_stats={"sim_1001": (2.0, 2)},
        coverage={"sim_0001": 0.5},
        lr_val=1.0e-4,
        train_task_avg={},
        val_task_avg={},
        dt=0.1,
    )

    with open(tmp_path / "metrics.csv", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0][:6] == ["epoch", "split", "mse", "rmse", "lr", "objective"]
    for row in rows[1:]:
        assert len(row) == len(rows[0])

    train_row = rows[1]
    val_row = rows[2]
    assert train_row[1] == "train"
    assert val_row[1] == "val"
    assert train_row[5] == "0.90000000"
    assert val_row[5] == "0.95000000"
