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


_FLOW_NO_SOURCE_CONCAT_OBJECTIVES = {
    "dbfm_source_anchored",
    "dbfm_rectified_flow",
    "dbfm_flow",
    "dbfm",
}


def _flow_uses_source_concat(cfg: Dict[str, Any]) -> bool:
    objective = str(cfg.get("train", {}).get("objective", "default")).lower()
    return objective not in _FLOW_NO_SOURCE_CONCAT_OBJECTIVES


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
    train_args = cfg["dataloader"].get("train_args", {})
    val_args = cfg["dataloader"].get("val_args", {})
    ds_args_train = {**ds_args, **train_args}
    ds_args_val = {**ds_args, **val_args}

    if isinstance(cfg["paths"]["h5"]["train"], dict):
        train_ds = DSClass(**cfg["paths"]["h5"]["train"], **ds_args_train)
    else:
        train_ds = DSClass(cfg["paths"]["h5"]["train"], **ds_args_train)

    use_val_flag = bool(cfg["trainer"].get("use_val", True)) and cfg["paths"]["h5"].get("val")
    val_ds = None
    if use_val_flag:
        if isinstance(cfg["paths"]["h5"]["val"], dict):
            val_ds = DSClass(**cfg["paths"]["h5"]["val"], **ds_args_val)
        else:
            val_ds = DSClass(cfg["paths"]["h5"]["val"], **ds_args_val)

    overfit_indices = cfg["loader"].get("overfit_indices")
    overfit_n = cfg["loader"].get("overfit_n")
    if overfit_indices is not None:
        if isinstance(overfit_indices, str):
            idxs = [int(tok.strip()) for tok in overfit_indices.split(",") if tok.strip()]
        else:
            idxs = [int(v) for v in overfit_indices]
        if not idxs:
            raise ValueError("loader.overfit_indices was provided but empty.")
        if overfit_n is not None and _rank0():
            print(
                "[loader] both overfit_indices and overfit_n were set; using overfit_indices.",
                flush=True,
            )

        def _subset_exact(ds, name: str):
            bad = [i for i in idxs if i < 0 or i >= len(ds)]
            if bad:
                raise ValueError(
                    f"loader.overfit_indices contains out-of-range values for {name} dataset "
                    f"(len={len(ds)}): {bad[:8]}"
                )
            return Subset(ds, idxs)

        train_ds = _subset_exact(train_ds, "train")
        if val_ds is not None:
            val_ds = _subset_exact(val_ds, "val")
    elif overfit_n is not None:
        n = int(overfit_n)
        train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))

    sample = train_ds[0]
    x0, y0 = sample["input"], sample["target"]
    assert x0.dim() == 3 and y0.dim() == 3
    H, W = int(x0.shape[-2]), int(x0.shape[-1])

    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", False))
    if cond_enabled:
        raise ValueError(
            "Scalar conditioning has been removed from the active training path. "
            "Set conditioning.enabled=false and use conditioning.use_theta with add_thermal."
        )
    cond_dim = 0
    cond_source = "none"
    cond_cfg["cond_dim"] = 0
    cond_cfg["source"] = "none"
    cond_cfg.pop("cond_indices", None)

    has_weight = "weight" in sample
    use_weight_loss = bool(cfg["trainer"].get("use_wavelet_weights", False))
    if use_weight_loss and not has_weight:
        raise RuntimeError("trainer.use_wavelet_weights=True but dataset samples do not contain 'weight'.")

    model_family = str(cfg["train"].get("model_family", "surrogate")).lower()
    flow_uses_source_concat = _flow_uses_source_concat(cfg)
    state_channels = x0.shape[0]
    if cond_cfg.get("use_theta", False):
        if not bool(ds_args.get("add_thermal", False)):
            raise ValueError(
                "conditioning.use_theta=true requires dataloader.args.add_thermal=true "
                "so the thermal field is present in input channels."
            )
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if theta_channels <= 0:
            raise ValueError("conditioning.theta_channels must be positive when use_theta is enabled.")
        theta_norm = str(cond_cfg.get("theta_normalization", "none")).strip().lower()
        if _rank0() and theta_norm in {"none", ""}:
            print(
                "[warning] conditioning.use_theta=true with theta_normalization='none'. "
                "Raw thermal magnitudes may destabilize training; consider affine normalization.",
                flush=True,
            )
        state_channels -= theta_channels
    if state_channels <= 0:
        raise ValueError(f"Non-positive state_channels inferred from input shape {tuple(x0.shape)} and cond_dim={cond_dim}")
    if model_family == "diffusion":
        model_multiplier = 2
    elif model_family == "flow_matching":
        model_multiplier = 2 if flow_uses_source_concat else 1
    else:
        model_multiplier = 1
    model_in_channels = state_channels * model_multiplier

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

    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    cond_enabled = bool(cond_cfg.get("enabled", False))
    if cond_enabled:
        raise ValueError(
            "Scalar conditioning has been removed from the active training path. "
            "Set conditioning.enabled=false and use conditioning.use_theta with add_thermal."
        )

    if use_legacy_model:
        ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
        base_model = ModelClass(**model_cfg.get("params", {}))
    else:
        model_cfg_local = dict(model_cfg)
        backbone_name_local = str(model_cfg_local.get("backbone", backbone)).lower()
        params = dict(model_cfg_local.get("params", model_cfg_local))
        if "cond_dim" in params:
            # Keep FiLM-UNet scalar channel enabled for the t-input path used in
            # diffusion/flow calls (model(x, t, ...)). For other backbones, keep
            # legacy behavior and disable scalar conditioning.
            if backbone_name_local in {"unet_film_attn", "unet_bottleneck_attn"}:
                params["cond_dim"] = max(1, int(params.get("cond_dim", 1)))
            else:
                params["cond_dim"] = 0
        model_cfg_local["params"] = params
        params = dict(model_cfg_local.get("params", model_cfg_local))
        theta_extra = 0
        use_control_branch = bool(params.get("use_control_branch", False))
        if (
            model_family in {"diffusion", "flow_matching"}
            and bool(cfg.get("conditioning", {}).get("use_theta", False))
            and backbone_name_local in {"unet_film_attn", "unet_bottleneck_attn"}
            and not use_control_branch
        ):
            theta_extra = int(cfg.get("conditioning", {}).get("theta_channels", 1))
        for key in ("in_channels", "n_channels"):
            if key in params and model_family in {"diffusion", "flow_matching"}:
                params[key] = model_in_channels + theta_extra
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
