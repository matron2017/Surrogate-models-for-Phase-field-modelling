"""
Setup helpers: distributed init, datasets, model, dataloaders.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from models.train.core.utils import (
    _collate,
    _infer_backbone_name,
    _load_symbol,
    _prepare_batch,
    _seed_all,
    _seed_worker,
    _print_visible_gpus_once,
    _is_dist,
    _rank0,
)
from models.backbones.registry import build_model as registry_build_model
from models.train.tasks.diffusion_task import DiffusionTask
from models.diffusion.configs import DiffusionConfig


def _init_distributed(cfg: Dict[str, Any]) -> Tuple[torch.device, int]:
    trainer_cfg = cfg["trainer"]
    seed = int(trainer_cfg.get("seed", 17))
    deterministic = bool(trainer_cfg.get("deterministic", False))

    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    _seed_all(seed, deterministic)

    have_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and have_cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(trainer_cfg.get("device", "cuda" if have_cuda else "cpu"))

    if _rank0():
        _print_visible_gpus_once()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not deterministic:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    return device, local_rank


def _build_datasets(cfg: Dict[str, Any], seed: int):
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})

    if isinstance(cfg["paths"]["h5"]["train"], dict):
        train_ds = DSClass(**cfg["paths"]["h5"]["train"], **ds_args)
    else:
        train_ds = DSClass(cfg["paths"]["h5"]["train"], **ds_args)

    use_val_flag = bool(cfg["trainer"].get("use_val", True)) and cfg["paths"]["h5"].get("val")
    val_ds = None
    if use_val_flag:
        if isinstance(cfg["paths"]["h5"]["val"], dict):
            val_ds = DSClass(**cfg["paths"]["h5"]["val"], **ds_args)
        else:
            val_ds = DSClass(cfg["paths"]["h5"]["val"], **ds_args)

    if cfg["loader"].get("overfit_n") is not None:
        n = int(cfg["loader"]["overfit_n"])
        train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))

    sample = train_ds[0]
    x0, y0 = sample["input"], sample["target"]
    assert x0.dim() == 3 and y0.dim() == 3
    H, W = int(x0.shape[-2]), int(x0.shape[-1])

    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = cond_cfg.get("enabled", True)
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    cond_source = str(cond_cfg.get("source", "field")).lower()
    if cond_enabled:
        if cond_source == "channels":
            assert x0.size(0) >= cond_dim
        else:
            assert "cond" in sample and sample["cond"].dim() == 1 and sample["cond"].numel() == cond_dim

    has_weight = "weight" in sample
    use_weight_loss = bool(cfg["trainer"].get("use_wavelet_weights", False))
    if use_weight_loss and not has_weight:
        raise RuntimeError("trainer.use_wavelet_weights=True but dataset samples do not contain 'weight'.")

    model_family = str(cfg["train"].get("model_family", "surrogate")).lower()
    state_channels = x0.shape[0] - (cond_dim if cond_enabled and cond_source == "channels" else 0)
    if state_channels <= 0:
        raise ValueError(f"Non-positive state_channels inferred from input shape {tuple(x0.shape)} and cond_dim={cond_dim}")
    model_in_channels = state_channels * (2 if model_family in {"diffusion", "flow_matching"} else 1)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "use_val_flag": use_val_flag,
        "sample": sample,
        "H": H,
        "W": W,
        "cond_cfg": cond_cfg,
        "cond_dim": cond_dim,
        "cond_source": cond_source,
        "use_weight_loss": use_weight_loss,
        "state_channels": state_channels,
        "model_in_channels": model_in_channels,
    }


def _build_model_and_task(cfg: Dict[str, Any], device, model_in_channels: int, cond_dim: int):
    model_family = str(cfg["train"].get("model_family", "surrogate")).lower()
    task_name = str(cfg.get("task", {}).get("name", "surrogate")).lower()
    model_cfg = cfg["model"]
    backbone = str(model_cfg.get("backbone") or _infer_backbone_name(model_cfg))
    use_legacy_model = "file" in model_cfg and "class" in model_cfg and model_cfg.get("file") and model_cfg.get("class")

    if use_legacy_model:
        ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
        base_model = ModelClass(**model_cfg.get("params", {}))
    else:
        model_cfg_local = dict(model_cfg)
        backbone_name_local = str(model_cfg_local.get("backbone", backbone)).lower()
        if model_family == "flow_matching":
            params = dict(model_cfg_local.get("params", model_cfg_local))
            if "cond_dim" in params:
                if backbone_name_local in {"unet_film_attn", "unet_bottleneck_attn"}:
                    params["cond_dim"] = int(params["cond_dim"])
                else:
                    params["cond_dim"] = int(params["cond_dim"]) + 1
            model_cfg_local["params"] = params
        params = dict(model_cfg_local.get("params", model_cfg_local))
        for key in ("in_channels", "n_channels"):
            if key in params and model_family in {"diffusion", "flow_matching"}:
                params[key] = model_in_channels
        model_cfg_local["params"] = params
        base_model = registry_build_model(model_family, backbone, model_cfg_local)

    base_model = base_model.to(device)
    model = base_model
    if task_name == "diffusion":
        diff_cfg = cfg["diffusion"]
        model = DiffusionTask(
            base_model,
            DiffusionConfig(
                dt=float(diff_cfg.get("dt", 1.0)),
                n_steps=int(diff_cfg.get("n_steps", 1)),
                thermal_bc=str(diff_cfg.get("thermal_bc", "dirichlet")),
                record_trajectory=bool(diff_cfg.get("record_trajectory", False)),
            ),
        ).to(device)
    if bool(cfg["trainer"].get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, base_model, backbone, model_family, task_name


def _build_loaders(train_ds, val_ds, cfg: Dict[str, Any], use_ddp: bool, seed: int):
    num_workers = int(cfg["loader"].get("num_workers", 4))
    batch_size = int(cfg["loader"]["batch_size"])
    pin_mem = bool(cfg["loader"].get("pin_memory", True))
    gen = torch.Generator().manual_seed(seed)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True, seed=seed) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False, seed=seed) if (use_ddp and val_ds is not None) else None

    def _dl_kwargs(nw: int):
        kw = dict(
            num_workers=nw,
            pin_memory=pin_mem,
            persistent_workers=(nw > 0),
            worker_init_fn=_seed_worker,
            generator=gen,
            collate_fn=_collate,
        )
        if nw > 0:
            kw["prefetch_factor"] = 4
        return kw

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not use_ddp), sampler=train_sampler, drop_last=True, **_dl_kwargs(num_workers)
    )
    val_dl = None
    use_val_flag = bool(cfg["trainer"].get("use_val", True)) and cfg["paths"]["h5"].get("val")
    if use_val_flag and val_ds is not None:
        val_dl = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, drop_last=False, **_dl_kwargs(num_workers)
        )
    return train_dl, val_dl, train_sampler, val_sampler
