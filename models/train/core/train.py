#!/usr/bin/env python3
# Orchestrator for the modular trainer.

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.train.core.config import _load_config, _validate_config
from models.train.core.setup import _init_distributed, _build_datasets, _build_model_and_task, _build_loaders
from models.train.core.optim_sched import _make_optimizer, _make_scheduler
from models.train.core.logging import RunLogger, _start_mlflow
from models.train.core.loops import _train_one_epoch, _validate_epoch, _pairs_per_gid_from_dataset
from models.train.core.utils import _barrier_safe, _count_params, _fmt_params, _is_dist, _rank0
from models.train.loss_registry import build_diffusion_loss, build_surrogate_loss
from models.diffusion.scheduler_registry import get_noise_schedule
from models.diffusion.timestep_sampler import get_timestep_sampler
from models.train.adaptive.registry import build_region_selector

try:
    from torchcfm import ConditionalFlowMatcher
except Exception:
    ConditionalFlowMatcher = None


def main(cfg_path: str, resume_arg: Optional[str] = None):
    cfg = _load_config(cfg_path)
    _validate_config(cfg)

    trainer_cfg = cfg["trainer"]
    seed = int(trainer_cfg.get("seed", 17))
    deterministic = bool(trainer_cfg.get("deterministic", False))
    log_interval = int(trainer_cfg.get("log_interval", 0))
    metrics_cfg = cfg["trainer"].get("metrics", {"mae": True, "psnr": True})
    want_mae = bool(metrics_cfg.get("mae", True))
    want_psnr = bool(metrics_cfg.get("psnr", True))

    device, local_rank = _init_distributed(cfg)

    ds_bundle = _build_datasets(cfg, seed)
    train_ds = ds_bundle["train_ds"]
    val_ds = ds_bundle["val_ds"]
    use_val_flag = ds_bundle["use_val_flag"]
    sample = ds_bundle["sample"]
    H, W = ds_bundle["H"], ds_bundle["W"]
    cond_cfg = ds_bundle["cond_cfg"]
    use_weight_loss = ds_bundle["use_weight_loss"]
    model_in_channels = ds_bundle["model_in_channels"]

    model, base_model, backbone, model_family, task_name = _build_model_and_task(cfg, device, model_in_channels, ds_bundle["cond_dim"])

    if _rank0():
        tot, trainable = _count_params(model)
        ws = int(os.environ.get("WORLD_SIZE", "1")) if _is_dist() else 1
        eff_bsz = int(cfg["loader"]["batch_size"]) * ws
        print(f"World size={ws}  batch_per_rank={cfg['loader']['batch_size']}  effective_batch={eff_bsz}")
        print(f"Model parameters: total={_fmt_params(tot)}, trainable={_fmt_params(trainable)}", flush=True)

    use_ddp = _is_dist()
    if use_ddp:
        ddp_kwargs = dict(device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=False)
        try:
            model = DDP(model, static_graph=True, **ddp_kwargs)
        except TypeError:
            model = DDP(model, **ddp_kwargs)

    train_dl, val_dl, train_sampler, val_sampler = _build_loaders(train_ds, val_ds, cfg, use_ddp, seed)

    optim = _make_optimizer(model.parameters(), cfg)
    sched, sched_monitor_cfg = _make_scheduler(optim, cfg)
    grad_clip = float(cfg["trainer"].get("grad_clip", 0.0))
    amp_cfg = cfg["trainer"].get("amp", {"enabled": device.type == "cuda", "dtype": "bf16"})
    amp_enabled = bool(amp_cfg.get("enabled", device.type == "cuda"))
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bf16")).lower() == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    use_chlast = bool(cfg["trainer"].get("channels_last", False))

    loss_fns = {
        "surrogate": build_surrogate_loss(cfg["loss"]),
        "diffusion": build_diffusion_loss(cfg["loss"]),
    }

    diff_cfg = cfg["diffusion"]
    noise_schedule_obj = None
    timestep_sampler_obj = None
    region_selector_obj = None
    if model_family == "diffusion":
        noise_schedule_obj = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
        sampler_kwargs = dict(diff_cfg.get("sampler_kwargs", {}))
        sampler_kwargs.setdefault("device", device)
        timestep_sampler_obj = get_timestep_sampler(diff_cfg["timestep_sampler"], schedule=noise_schedule_obj, **sampler_kwargs)
        region_selector_obj = build_region_selector(cfg["adaptive"].get("region_selector", "none"), **cfg["adaptive"].get("region_kwargs", {}))

    flow_matcher = None
    if model_family == "flow_matching":
        if ConditionalFlowMatcher is None:
            raise ImportError("torchcfm is required for flow_matching but is not installed.")
        flow_matcher = ConditionalFlowMatcher(sigma=float(cfg["flow_matching"].get("sigma", 0.0)))

    out_dir = Path(cfg["trainer"].get("out_dir", "./results"))
    if _rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
    if _is_dist():
        _barrier_safe(local_rank)
    tag_src = model.module if isinstance(model, DDP) else model
    run_dir = out_dir / tag_src.__class__.__name__
    if _rank0():
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_snapshot.yaml", "w") as f:
            import yaml

            yaml.safe_dump(cfg, f)

    resume_cfg = cfg["trainer"].get("resume", None)
    resume_path = None
    if isinstance(resume_cfg, str) and len(resume_cfg) > 0:
        resume_path = resume_cfg
    elif resume_cfg is True:
        resume_path = str(run_dir / "checkpoint.last.pth")
    if resume_arg:
        resume_path = resume_arg

    start_epoch = 1
    best_metric = math.inf
    mlflow_parent_run_id: Optional[str] = None

    default_split = "val" if (use_val_flag and val_dl is not None) else "train"
    default_mode = "min"
    monitor_split_sched = (sched_monitor_cfg or {"split": default_split}).get("split", default_split)
    monitor_mode_sched = (sched_monitor_cfg or {"mode": default_mode}).get("mode", default_mode)
    es_cfg = cfg["trainer"].get("early_stop", {})
    monitor_split_es = es_cfg.get("split", default_split)
    monitor_mode_es = es_cfg.get("mode", default_mode)
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 30))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    best_metric = math.inf if monitor_mode_es == "min" else -math.inf
    es_bad = 0

    mlflow_params = {
        "model_family": model_family,
        "model.backbone": backbone,
        "task.name": task_name,
        "diffusion.noise_schedule": diff_cfg.get("noise_schedule"),
        "diffusion.timestep_sampler": diff_cfg.get("timestep_sampler"),
        "diffusion.dt": float(diff_cfg.get("dt", 1.0)),
        "diffusion.n_steps": int(diff_cfg.get("n_steps", 1)),
        "diffusion.thermal_bc": str(diff_cfg.get("thermal_bc", "dirichlet")),
        "diffusion.record_trajectory": bool(diff_cfg.get("record_trajectory", False)),
        "loss.weight_wavelet_loss": float(cfg["loss"].get("weight_wavelet_loss", 0.0)),
        "adaptive.region_selector": str(cfg["adaptive"].get("region_selector", "none")),
        "adaptive.enable_adaptive_resolution": bool(cfg["adaptive"].get("enable_adaptive_resolution", False)),
        "seed": seed,
    }
    if model_family == "flow_matching":
        mlflow_params["flow_matching.sigma"] = float(cfg["flow_matching"].get("sigma", 0.0))

    mlflow_ctx = _start_mlflow(cfg, mlflow_params)
    mlflow_parent_run_id = mlflow_ctx.parent_run_id

    if resume_path and Path(resume_path).is_file():
        state = torch.load(resume_path, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        if sched and state.get("sched"):
            sched.load_state_dict(state["sched"])
        if state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        best_metric = state.get("best_metric", best_metric)
        start_epoch = int(state.get("epoch", 0)) + 1
        if _rank0():
            print(f"Resumed from {resume_path} at epoch {start_epoch}")
        resume_parent_run_id = state.get("mlflow_parent_run_id")
        if resume_parent_run_id:
            mlflow_parent_run_id = resume_parent_run_id
            if mlflow_ctx.active:
                try:
                    import mlflow

                    if mlflow_ctx.run_id and mlflow_ctx.run_id != resume_parent_run_id:
                        mlflow.set_tags({"mlflow.parentRunId": resume_parent_run_id})
                except Exception as e:
                    print(f"MLflow parent tag failed: {e}", flush=True)

    run_logger = RunLogger(
        cfg=cfg,
        run_dir=run_dir,
        tag_src=tag_src,
        device=device,
        seed=seed,
        deterministic=deterministic,
        x0=sample["input"],
        H=H,
        W=W,
        mlflow_ctx=mlflow_ctx,
        want_mae=want_mae,
        want_psnr=want_psnr,
        monitor_split_es=monitor_split_es,
        monitor_mode_es=monitor_mode_es,
        resume_path=resume_path,
        start_epoch=start_epoch,
    )

    total_pairs_by_gid = _pairs_per_gid_from_dataset(train_ds) if _rank0() else None

    interrupted = {"flag": False}

    def _on_sigint(signum, frame):
        interrupted["flag"] = True

    import signal

    signal.signal(signal.SIGINT, _on_sigint)

    hist_epochs: List[int] = []
    hist_train: List[float] = []
    hist_val: List[float] = []

    epochs = int(cfg["trainer"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        if _is_dist():
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        t0 = time.time()
        train_metrics = _train_one_epoch(
            epoch=epoch,
            model=model,
            train_dl=train_dl,
            device=device,
            cond_cfg=cond_cfg,
            use_chlast=use_chlast,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            optim=optim,
            grad_clip=grad_clip,
            model_family=model_family,
            loss_fns=loss_fns,
            flow_matcher=flow_matcher,
            noise_schedule_obj=noise_schedule_obj,
            timestep_sampler_obj=timestep_sampler_obj,
            region_selector_obj=region_selector_obj,
            use_weight_loss=use_weight_loss,
            log_interval=log_interval,
            want_mae=want_mae,
            total_pairs_by_gid=total_pairs_by_gid,
        )

        val_metrics = None
        if use_val_flag and val_dl is not None:
            val_metrics = _validate_epoch(
                model=model,
                val_dl=val_dl,
                device=device,
                cond_cfg=cond_cfg,
                use_chlast=use_chlast,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                model_family=model_family,
                loss_fns=loss_fns,
                flow_matcher=flow_matcher,
                noise_schedule_obj=noise_schedule_obj,
                timestep_sampler_obj=timestep_sampler_obj,
                region_selector_obj=region_selector_obj,
                want_mae=want_mae,
            )

        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = train_metrics["mse"] if monitor_split_sched == "train" else (
                    val_metrics["mse"] if val_metrics is not None else train_metrics["mse"]
                )
                sched.step(metric)
            else:
                sched.step()

        lr_val = optim.param_groups[0]["lr"]

        dt = time.time() - t0
        run_logger.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            gid_stats=train_metrics["gid_stats"],
            val_gid_stats=val_metrics["gid_stats"] if val_metrics else None,
            coverage=train_metrics["coverage"],
            lr_val=lr_val,
            train_task_avg=train_metrics["task_avg"],
            val_task_avg=val_metrics["task_avg"] if val_metrics else {},
            dt=dt,
        )

        metric_for_es = train_metrics["mse"] if monitor_split_es == "train" else (
            val_metrics["mse"] if val_metrics is not None else train_metrics["mse"]
        )
        improved = (metric_for_es < best_metric - es_min_delta) if monitor_mode_es == "min" else (metric_for_es > best_metric + es_min_delta)
        if improved:
            best_metric = metric_for_es
            es_bad = 0
        else:
            es_bad += 1

        if _rank0():
            state = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optim": optim.state_dict(),
                "sched": (sched.state_dict() if sched is not None else None),
                "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
                "best_metric": best_metric,
                "config": cfg,
                "mlflow_parent_run_id": mlflow_parent_run_id,
            }
            run_logger.save_checkpoint(state, is_best=False, interrupt=False)
            if improved:
                run_logger.save_checkpoint(state, is_best=True, interrupt=False)

        hist_epochs.append(epoch)
        hist_train.append(float(train_metrics["rmse"]))
        hist_val.append(float(val_metrics["rmse"]) if val_metrics is not None else float("nan"))

        if es_enabled and es_bad >= es_patience:
            if _rank0():
                print(f"Early stopping at epoch {epoch} (patience {es_patience})", flush=True)
            break
        if interrupted["flag"]:
            if _rank0():
                state = {
                    "epoch": epoch,
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optim": optim.state_dict(),
                    "sched": (sched.state_dict() if sched is not None else None),
                    "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
                    "best_metric": best_metric,
                    "config": cfg,
                    "mlflow_parent_run_id": mlflow_parent_run_id,
                }
                run_logger.save_checkpoint(state, interrupt=True)
                print(f"Interrupted at epoch {epoch}. Saved checkpoint.interrupt.pth")
            break

    status = "interrupted" if interrupted["flag"] else ("early_stopped" if es_enabled and es_bad >= es_patience else "completed")
    run_logger.finalize(
        hist_epochs=hist_epochs,
        hist_train=hist_train,
        hist_val=hist_val,
        best_metric=best_metric,
        status=status,
        monitor_split_es=monitor_split_es,
        monitor_mode_es=monitor_mode_es,
        interrupted=interrupted["flag"],
    )

    if _is_dist():
        _barrier_safe(local_rank)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import time

    ap = argparse.ArgumentParser(description="Modular DDP trainer")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = ap.parse_args()
    main(args.config, resume_arg=args.resume)
