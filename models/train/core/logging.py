"""
Run-level logging: CSV, run.json, MLflow, checkpoints, and plots.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import platform
import socket
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from models.train.core.utils import _count_params, _fmt_params, _load_json_safe, _now_utc_iso, _rank0, _is_dist, _flatten_dict

try:
    import mlflow
except Exception:
    mlflow = None


@dataclass
class MLFlowContext:
    active: bool
    run_id: Optional[str]
    parent_run_id: Optional[str]


def _start_mlflow(cfg: Dict[str, Any], params: Dict[str, Any]) -> MLFlowContext:
    mlflow_cfg = cfg.setdefault("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    if not (mlflow_enabled and mlflow is not None and _rank0()):
        return MLFlowContext(False, None, None)

    mlflow_run_id = None
    mlflow_parent_run_id = None
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = mlflow_cfg.get("experiment_name", "models")
        mlflow.set_experiment(experiment_name)
        default_run = os.environ.get("SLURM_JOB_ID")
        run_name = mlflow_cfg.get("run_name") or default_run or f"run-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        active = mlflow.start_run(run_name=str(run_name))
        mlflow_run_id = active.info.run_id
        mlflow_parent_run_id = mlflow_run_id
        try:
            mlflow.set_tag("mlflow.parentRunId", mlflow_parent_run_id)
        except Exception:
            pass
        try:
            mlflow.log_params(params)
            cfg_flat: Dict[str, Any] = {}
            _flatten_dict("", cfg, cfg_flat)
            mlflow.log_params(cfg_flat)
        except Exception as e:
            print(f"MLflow param logging failed: {e}", flush=True)
    except Exception as e:
        print(f"MLflow disabled: {e}", flush=True)
        return MLFlowContext(False, None, None)
    return MLFlowContext(True, mlflow_run_id, mlflow_parent_run_id)


class RunLogger:
    """Handles CSV logging, run.json bookkeeping, plotting, MLflow, and checkpoints."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        run_dir: Path,
        tag_src: torch.nn.Module,
        device: torch.device,
        seed: int,
        deterministic: bool,
        x0: torch.Tensor,
        H: int,
        W: int,
        mlflow_ctx: MLFlowContext,
        want_mae: bool,
        want_psnr: bool,
        monitor_split_es: str,
        monitor_mode_es: str,
        resume_path: Optional[str],
        start_epoch: int,
    ):
        self.cfg = cfg
        self.run_dir = run_dir
        self.csv_path = run_dir / "metrics.csv"
        self.plot_path = run_dir / "learning_curve.png"
        self.run_json_path = run_dir / "run.json"
        self.want_mae = want_mae
        self.want_psnr = want_psnr
        self.mlflow_ctx = mlflow_ctx
        self.monitor_split_es = monitor_split_es
        self.monitor_mode_es = monitor_mode_es
        self.resume_path = resume_path
        self.start_epoch = start_epoch

        if _rank0():
            header = ["epoch", "split", "mse", "rmse", "lr"]
            if want_mae:
                header.append("mae")
            if want_psnr:
                header.append("psnr")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

            tot_params, trainable_params = _count_params(tag_src)
            env_blob = {
                "started_utc": _now_utc_iso(),
                "hostname": socket.gethostname(),
                "platform": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                    "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                },
                "distributed": {
                    "enabled": _is_dist(),
                    "backend": ("nccl" if _is_dist() else None),
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")) if _is_dist() else 1,
                    "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                },
                "device": str(device),
                "seed": seed,
                "deterministic": bool(deterministic),
                "model": {
                    "class": tag_src.__class__.__name__,
                    "total_params": int(tot_params),
                    "trainable_params": int(trainable_params),
                },
                "data": {
                    "paths": cfg["paths"],
                    "image_shape": {"C": int(x0.shape[0]), "H": int(H), "W": int(W)},
                },
                "trainer": {
                    "epochs": int(cfg["trainer"]["epochs"]),
                    "amp": cfg["trainer"].get("amp", {}),
                    "channels_last": bool(cfg["trainer"].get("channels_last", False)),
                    "early_stop": cfg["trainer"].get("early_stop", {}),
                    "metrics_cfg": cfg["trainer"].get("metrics", {}),
                },
                "optim": cfg.get("optim", {}),
                "sched": cfg.get("sched", {}),
                "conditioning": cfg.get("conditioning", {}),
                "resume": {
                    "requested": cfg["trainer"].get("resume", None),
                    "loaded_path": (resume_path if resume_path and Path(resume_path).is_file() else None),
                    "start_epoch": int(start_epoch),
                },
                "outputs": {
                    "run_dir": str(run_dir),
                    "csv_path": str(self.csv_path),
                    "plot_path": str(self.plot_path),
                    "checkpoint_last": str(run_dir / "checkpoint.last.pth"),
                    "checkpoint_best": str(run_dir / "checkpoint.best.pth"),
                },
                "mlflow": {
                    "enabled": bool(mlflow_ctx.active),
                    "run_id": mlflow_ctx.run_id,
                    "parent_run_id": mlflow_ctx.parent_run_id,
                },
                "status": {"state": "running"},
            }
            self.run_json_path.write_text(json.dumps(env_blob, indent=2))

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
        gid_stats: Dict[str, Any],
        val_gid_stats: Optional[Dict[str, Any]],
        coverage: Optional[Dict[str, float]],
        lr_val: float,
        train_task_avg: Dict[str, float],
        val_task_avg: Dict[str, float],
        dt: float,
    ):
        if not _rank0():
            return
        row_train = [epoch, "train", f"{train_metrics['mse']:.8f}", f"{train_metrics['rmse']:.8f}", f"{lr_val:.6g}"]
        if self.want_mae:
            row_train.append(f"{train_metrics.get('mae', 0.0):.8f}" if train_metrics.get("mae") is not None else "")
        if self.want_psnr:
            row_train.append("" if train_metrics["mse"] <= 0 else f"{(-10.0 * math.log10(train_metrics['mse'])):.4f}")

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row_train)
            if val_metrics is not None:
                row_val = [
                    epoch,
                    "val",
                    f"{val_metrics['mse']:.8f}",
                    f"{val_metrics['rmse']:.8f}",
                    f"{lr_val:.6g}",
                ]
                if self.want_mae:
                    row_val.append(f"{val_metrics.get('mae', 0.0):.8f}" if val_metrics.get("mae") is not None else "")
                if self.want_psnr:
                    row_val.append("" if val_metrics["mse"] <= 0 else f"{(-10.0 * math.log10(val_metrics['mse'])):.4f}")
                w.writerow(row_val)

            for g, (s, c) in sorted(gid_stats.items()):
                mean_mse_g = s / max(c, 1)
                mean_rmse_g = math.sqrt(max(mean_mse_g, 0.0))
                row = [epoch, f"train:{g}", f"{mean_mse_g:.8f}", f"{mean_rmse_g:.8f}", f"{lr_val:.6g}"]
                if self.want_mae:
                    row.append("")
                if self.want_psnr:
                    row.append("")
                w.writerow(row)

            if val_gid_stats is not None:
                for g, (s, c) in sorted(val_gid_stats.items()):
                    mean_mse_g = s / max(c, 1)
                    mean_rmse_g = math.sqrt(max(mean_mse_g, 0.0))
                    row = [epoch, f"val:{g}", f"{mean_mse_g:.8f}", f"{mean_rmse_g:.8f}", f"{lr_val:.6g}"]
                    if self.want_mae:
                        row.append("")
                    if self.want_psnr:
                        row.append("")
                    w.writerow(row)

            if coverage is not None:
                for g, frac in sorted(coverage.items()):
                    row = [epoch, f"coverage:{g}", f"{frac:.8f}", "", f"{lr_val:.6g}"]
                    if self.want_mae:
                        row.append("")
                    if self.want_psnr:
                        row.append("")
                    w.writerow(row)

        msg = f"epoch={epoch} train_mse={train_metrics['mse']:.6f} train_rmse={train_metrics['rmse']:.6f}"
        if val_metrics is not None:
            msg += f" val_mse={val_metrics['mse']:.6f} val_rmse={val_metrics['rmse']:.6f}"
        if train_task_avg:
            extras = " ".join(f"{k}={v:.6f}" for k, v in sorted(train_task_avg.items()))
            msg += f" task[{extras}]"
        if val_task_avg:
            extras = " ".join(f"{k}={v:.6f}" for k, v in sorted(val_task_avg.items()))
            msg += f" val_task[{extras}]"
        msg += f" dt={dt:.2f}s"
        print(msg, flush=True)

        if self.mlflow_ctx.active:
            metrics = {"train_rmse": train_metrics["rmse"], "train_mse": train_metrics["mse"]}
            if train_metrics.get("mae") is not None:
                metrics["train_mae"] = train_metrics["mae"]
            if val_metrics is not None:
                metrics["val_rmse"] = val_metrics["rmse"]
                metrics["val_mse"] = val_metrics["mse"]
                if val_metrics.get("mae") is not None:
                    metrics["val_mae"] = val_metrics["mae"]
            for k, v in train_task_avg.items():
                metrics[f"train_{k}"] = v
            for k, v in val_task_avg.items():
                metrics[f"val_{k}"] = v
            try:
                mlflow.log_metrics(metrics, step=epoch)
            except Exception as e:
                print(f"MLflow metric logging failed: {e}", flush=True)

    def save_checkpoint(self, state: Dict[str, Any], is_best: bool = False, interrupt: bool = False):
        if not _rank0():
            return
        suffix = "interrupt" if interrupt else ("best" if is_best else "last")
        path = self.run_dir / f"checkpoint.{suffix}.pth"
        torch.save(state, path)
        if not interrupt and not is_best:
            torch.save(state, self.run_dir / "checkpoint.last.pth")

    def finalize(
        self,
        hist_epochs: List[int],
        hist_train: List[float],
        hist_val: List[float],
        best_metric: float,
        status: str,
        monitor_split_es: str,
        monitor_mode_es: str,
        interrupted: bool,
    ):
        if _rank0():
            plt.figure()
            plt.plot(hist_epochs, hist_train, label="train")
            if any(not math.isnan(v) for v in hist_val):
                plt.plot(hist_epochs, hist_val, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=200)
            plt.close()

            try:
                blob = _load_json_safe(self.run_json_path) or {}
                blob["ended_utc"] = _now_utc_iso()
                blob["status"] = {
                    "state": status,
                    "final_epoch": int(hist_epochs[-1] if len(hist_epochs) > 0 else 0),
                    "best_metric": float(best_metric),
                    "monitor": {"split": monitor_split_es, "mode": monitor_mode_es},
                }
                blob["outputs"] = {
                    "run_dir": str(self.run_dir),
                    "csv_path": str(self.csv_path),
                    "plot_path": str(self.plot_path),
                    "checkpoint_last": str(self.run_dir / "checkpoint.last.pth"),
                    "checkpoint_best": str(self.run_dir / "checkpoint.best.pth"),
                    "checkpoint_interrupt": str(self.run_dir / "checkpoint.interrupt.pth") if interrupted else None,
                }
                blob["mlflow"] = {
                    "enabled": bool(self.mlflow_ctx.active),
                    "run_id": self.mlflow_ctx.run_id,
                    "parent_run_id": self.mlflow_ctx.parent_run_id,
                }
                self.run_json_path.write_text(json.dumps(blob, indent=2))
            except Exception as e:
                print(f"run.json finalise failed: {e}", flush=True)

            print(f"CSV: {self.csv_path}")
            print(f"Plot: {self.plot_path}")
            print(f"Checkpoints: {self.run_dir}")
            if self.mlflow_ctx.active:
                try:
                    mlflow.log_artifact(str(self.csv_path))
                    mlflow.log_artifact(str(self.plot_path))
                    mlflow.log_artifact(str(self.run_dir / "config_snapshot.yaml"))
                    mlflow.log_metrics({"best_metric": float(best_metric)})
                except Exception as e:
                    print(f"MLflow artifact logging failed: {e}", flush=True)

        if self.mlflow_ctx.active and _rank0():
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"MLflow end_run failed: {e}", flush=True)
