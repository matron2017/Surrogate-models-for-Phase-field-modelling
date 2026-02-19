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
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from models.train.core.config import _load_config, _validate_config
from models.train.core.setup import (
    _init_distributed,
    _build_datasets,
    _build_model_and_task,
    _build_loaders,
    _flow_uses_source_concat,
)
from models.train.core.optim_sched import _make_optimizer, _make_scheduler
from models.train.core.logging import RunLogger, _start_mlflow
from models.train.core.loops import _train_one_epoch, _validate_epoch, _pairs_per_gid_from_dataset
from models.train.core.latent import build_autoencoder
from models.train.core.utils import _barrier_safe, _count_params, _fmt_params, _is_dist, _rank0
from models.train.loss_registry import build_diffusion_loss, build_surrogate_loss, build_flow_loss
from models.diffusion.scheduler_registry import get_noise_schedule
from models.diffusion.timestep_sampler import get_timestep_sampler
from models.train.adaptive.registry import build_region_selector

try:
    from torchcfm import ConditionalFlowMatcher
except Exception:
    ConditionalFlowMatcher = None


def _check_step_sync(train_metrics: Dict[str, Any], device: torch.device, local_rank: int) -> None:
    if not _is_dist():
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    step_count = int(train_metrics.get("step_count", 0))
    opt_step_count = int(train_metrics.get("opt_step_count", 0))
    t = torch.tensor([step_count, opt_step_count], device=device, dtype=torch.int64)
    t_min = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    if _rank0() and (t_min != t_max).any():
        raise RuntimeError(
            f"DDP step desync detected: local steps={step_count}, opt_steps={opt_step_count}, "
            f"min={t_min.tolist()}, max={t_max.tolist()}"
        )
    _barrier_safe(local_rank)


def _collect_control_branch_params(model: torch.nn.Module, include_time_embed: bool = False) -> List[torch.nn.Parameter]:
    """
    Collects ControlNet-style branch parameters from the active backbone.
    Mirrors the module grouping used by ControlNet-XS optimizer setup.
    """
    base = model.module if isinstance(model, DDP) else model
    if hasattr(base, "backbone"):
        base = base.backbone
    module_names = (
        "control_model",
        "enc_zero_convs_out",
        "enc_zero_convs_in",
        "middle_block_out",
        "middle_block_in",
        "dec_zero_convs_out",
        "dec_zero_convs_in",
        "input_hint_block",
    )
    if include_time_embed:
        module_names = module_names + ("time_mlp",)

    params: List[torch.nn.Parameter] = []
    seen = set()
    for name in module_names:
        mod = getattr(base, name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
    return params


def main(cfg_path: str, resume_arg: Optional[str] = None):
    cfg = _load_config(cfg_path)
    _validate_config(cfg)

    trainer_cfg = cfg["trainer"]
    seed = int(trainer_cfg.get("seed", 17))
    deterministic = bool(trainer_cfg.get("deterministic", False))
    log_interval = int(trainer_cfg.get("log_interval", 0))
    nan_debug = bool(trainer_cfg.get("nan_debug", False))
    nan_debug_steps = int(trainer_cfg.get("nan_debug_steps", 2))
    nan_debug_param = trainer_cfg.get("nan_debug_param", None)
    nan_debug_input_stats = bool(trainer_cfg.get("nan_debug_input_stats", False))
    nan_tolerate_steps = int(trainer_cfg.get("nan_tolerate_steps", 0))
    detect_anomaly = bool(trainer_cfg.get("detect_anomaly", False))
    detect_anomaly_steps = int(trainer_cfg.get("detect_anomaly_steps", nan_debug_steps))
    debug_barrier = bool(trainer_cfg.get("debug_barrier", False))
    debug_phase_markers = bool(trainer_cfg.get("debug_phase_markers", debug_barrier))
    use_ddp_join = bool(trainer_cfg.get("ddp_join", False))
    lr_warmup_steps = int(trainer_cfg.get("lr_warmup_steps", 0))
    lr_warmup_start_lr = trainer_cfg.get("lr_warmup_start_lr", None)
    lr_warmup_start_lr = None if lr_warmup_start_lr is None else float(lr_warmup_start_lr)
    lr_warmup_phases = trainer_cfg.get("lr_warmup_phases", None)
    lr_warmup_epoch_phases_cfg = trainer_cfg.get("lr_warmup_epoch_phases", None)
    lr_warmup_epoch_phases = []
    lr_warmup_epoch_total = 0
    if lr_warmup_epoch_phases_cfg:
        for phase in lr_warmup_epoch_phases_cfg:
            try:
                epochs = int(phase.get("epochs", 0))
                factor = float(phase.get("factor", 1.0))
            except AttributeError:
                continue
            if epochs <= 0:
                continue
            lr_warmup_epoch_total += epochs
            lr_warmup_epoch_phases.append((lr_warmup_epoch_total, factor))
    accumulation_steps = int(trainer_cfg.get("accumulation_steps", 1))
    steps_per_epoch = trainer_cfg.get("steps_per_epoch", None)
    if steps_per_epoch is not None:
        steps_per_epoch = int(steps_per_epoch)
        if steps_per_epoch <= 0:
            steps_per_epoch = None
    metrics_cfg = cfg["trainer"].get("metrics", {"mae": True, "psnr": True})
    want_mae = bool(metrics_cfg.get("mae", True))
    want_psnr = bool(metrics_cfg.get("psnr", True))
    want_vrmse = bool(metrics_cfg.get("vrmse", False))
    vrmse_eps = float(metrics_cfg.get("vrmse_eps", 1e-6))
    want_spectral = bool(metrics_cfg.get("spectral_rmse", False))
    spectral_train = bool(metrics_cfg.get("spectral_train", False))
    want_endpoint_rmse = bool(metrics_cfg.get("endpoint_rmse", False))
    spectral_cfg = {
        "bands": int(metrics_cfg.get("spectral_bands", 3)),
        "eps": float(metrics_cfg.get("spectral_eps", 1e-6)),
    }

    if _rank0():
        train_h5 = cfg.get("paths", {}).get("h5", {}).get("train", None)
        val_h5 = cfg.get("paths", {}).get("h5", {}).get("val", None)
        print(
            f"[config] cfg_path={cfg_path} train_h5={train_h5} val_h5={val_h5}",
            flush=True,
        )

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
    model_family = str(cfg["train"].get("model_family", "surrogate")).lower()
    flow_uses_source_concat = _flow_uses_source_concat(cfg)
    latent_cfg = cfg.get("latent", {})
    autoencoder = build_autoencoder(latent_cfg, state_channels=ds_bundle["state_channels"], device=device)
    autoencoder_trainable = autoencoder is not None and any(p.requires_grad for p in autoencoder.parameters())
    metric_channels = int(sample["target"].shape[0])
    if autoencoder is not None:
        latent_spec = dict(latent_cfg) if isinstance(latent_cfg, dict) else {}
        split_fields = bool(latent_spec.get("split_fields", False))
        drop_thermal = bool(latent_spec.get("drop_thermal_target", False))
        if model_family in {"diffusion", "flow_matching"} and split_fields and drop_thermal:
            num_fields = int(latent_spec.get("num_fields", 3))
            total_latent = int(autoencoder.latent_channels)
            if num_fields <= 0:
                raise ValueError("latent.num_fields must be positive when split_fields is enabled.")
            if total_latent % num_fields != 0:
                raise ValueError(f"latent_channels={total_latent} not divisible by num_fields={num_fields}.")
            field_ch = total_latent // num_fields
            cond_fields = [int(i) for i in latent_spec.get("cond_fields", list(range(num_fields)))]
            target_fields = [int(i) for i in latent_spec.get("target_fields", [0, 1])]
            target_ch = field_ch * len(target_fields)
            cond_ch = field_ch * len(cond_fields)
            metric_channels = target_ch
            if model_family == "diffusion":
                model_in_channels = target_ch + cond_ch
            elif model_family == "flow_matching":
                model_in_channels = (target_ch + cond_ch) if flow_uses_source_concat else target_ch
            else:
                model_in_channels = target_ch
            model_cfg = cfg.get("model", {})
            params = model_cfg.get("params", model_cfg) if isinstance(model_cfg, dict) else {}
            for key in ("out_channels", "n_classes"):
                if key in params:
                    params[key] = target_ch
            if isinstance(model_cfg, dict):
                model_cfg["params"] = params
                cfg["model"] = model_cfg
        else:
            if model_family == "diffusion":
                model_multiplier = 2
            elif model_family == "flow_matching":
                model_multiplier = 2 if flow_uses_source_concat else 1
            else:
                model_multiplier = 1
            model_in_channels = autoencoder.latent_channels * model_multiplier
            metric_channels = int(autoencoder.latent_channels)
    else:
        latent_spec = dict(latent_cfg) if isinstance(latent_cfg, dict) else {}
        placeholder_cfg = latent_spec.get("placeholder", {}) if isinstance(latent_spec, dict) else {}
        if (
            isinstance(placeholder_cfg, dict)
            and bool(placeholder_cfg.get("enabled", False))
            and model_family in {"diffusion", "flow_matching"}
        ):
            placeholder_ch = int(placeholder_cfg.get("channels", metric_channels))
            if placeholder_ch <= 0:
                raise ValueError(f"latent.placeholder.channels must be > 0 (got {placeholder_ch}).")
            if model_family == "diffusion":
                model_multiplier = 2
            elif model_family == "flow_matching":
                model_multiplier = 2 if flow_uses_source_concat else 1
            else:
                model_multiplier = 1
            model_in_channels = placeholder_ch * model_multiplier
            metric_channels = placeholder_ch

    model, base_model, backbone, model_family, task_name = _build_model_and_task(cfg, device, model_in_channels, ds_bundle["cond_dim"])

    if _rank0():
        tot, trainable = _count_params(model)
        ws = int(os.environ.get("WORLD_SIZE", "1")) if _is_dist() else 1
        batch_per_rank = int(cfg["loader"]["batch_size"])
        global_micro_batch = batch_per_rank * ws
        eff_bsz = global_micro_batch * max(accumulation_steps, 1)
        train_len = len(train_ds)
        val_len = len(val_ds) if val_ds is not None else 0
        model_cfg = cfg.get("model", {})
        model_params = model_cfg.get("params", model_cfg) if isinstance(model_cfg, dict) else {}
        patch_size = model_params.get("patch_size", None)
        wavelet_cfg = cfg.get("loss", {}).get("ae_wavelet", {})
        wavelet_enabled = bool(wavelet_cfg.get("enabled", False))
        wavelet_weight = float(cfg.get("loss", {}).get("weight_wavelet_loss", 0.0))
        use_wavelet_weights = bool(cfg.get("trainer", {}).get("use_wavelet_weights", False))
        print(
            "World size="
            f"{ws}  batch_per_rank={batch_per_rank}  global_micro_batch={global_micro_batch} "
            f"accumulation_steps={max(accumulation_steps, 1)}  effective_batch={eff_bsz}"
        )
        print(f"Model parameters: total={_fmt_params(tot)}, trainable={_fmt_params(trainable)}")
        print(f"Dataset sizes: train={train_len}  val={val_len}")
        print(f"Input shape: C={sample['input'].shape[0]} H={H} W={W}  patch_size={patch_size}")
        print(
            "Wavelet: ae_wavelet_enabled="
            f"{wavelet_enabled}  weight_wavelet_loss={wavelet_weight:.3g}  use_wavelet_weights={use_wavelet_weights}",
            flush=True,
        )

    use_ddp = _is_dist()
    if use_ddp and autoencoder_trainable:
        raise ValueError("latent.trainable=True is only supported for single-process runs; freeze the AE or run single-GPU.")
    if use_ddp:
        ddp_kwargs = dict(device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=False)
        try:
            model = DDP(model, static_graph=True, **ddp_kwargs)
        except TypeError:
            model = DDP(model, **ddp_kwargs)

    train_dl, val_dl, train_sampler, val_sampler = _build_loaders(train_ds, val_ds, cfg, use_ddp, seed)
    if debug_phase_markers and _rank0():
        loader_cfg = cfg.get("loader", {})
        drop_last = bool(loader_cfg.get("drop_last", False))
        sampler_name = train_sampler.__class__.__name__ if train_sampler is not None else "None"
        val_sampler_name = val_sampler.__class__.__name__ if val_sampler is not None else "None"
        print(
            f"[phase] loader drop_last={drop_last} train_sampler={sampler_name} val_sampler={val_sampler_name}",
            flush=True,
        )

    control_train_only = bool(cfg["trainer"].get("control_train_only", False))
    control_train_time_embed = bool(cfg["trainer"].get("control_train_time_embed", False))
    if control_train_only:
        optim_params = _collect_control_branch_params(model, include_time_embed=control_train_time_embed)
        if len(optim_params) == 0:
            raise ValueError(
                "trainer.control_train_only=true but no ControlNet-style branch params were found. "
                "Enable model.params.use_control_branch and verify control modules are present."
            )
        if _rank0():
            control_trainable = sum(p.numel() for p in optim_params)
            print(
                f"[control] training control branch only: params={_fmt_params(control_trainable)} "
                f"(include_time_embed={control_train_time_embed})",
                flush=True,
            )
    else:
        optim_params = list(model.parameters())
    if autoencoder is not None:
        optim_params.extend([p for p in autoencoder.parameters() if p.requires_grad])
    optim = _make_optimizer(optim_params, cfg)
    sched, sched_monitor_cfg = _make_scheduler(optim, cfg)
    warmup_cfg = cfg.get("sched", {})
    warmup_epochs = int(warmup_cfg.get("warmup_epochs", 0))
    warmup_start_lr = float(warmup_cfg.get("warmup_start_lr", 0.0))
    base_lrs = [pg.get("initial_lr", pg.get("lr", 0.0)) for pg in optim.param_groups]
    grad_clip = float(cfg["trainer"].get("grad_clip", 0.0))
    amp_cfg = cfg["trainer"].get("amp", {"enabled": device.type == "cuda", "dtype": "bf16"})
    amp_enabled = bool(amp_cfg.get("enabled", device.type == "cuda"))
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bf16")).lower() == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    use_chlast = bool(cfg["trainer"].get("channels_last", False))

    loss_fns = {
        "surrogate": build_surrogate_loss(cfg["loss"]),
        "diffusion": build_diffusion_loss(cfg["loss"]),
        "flow": build_flow_loss(cfg["loss"]),
    }
    physics_aux_cfg = dict(cfg.get("loss", {}).get("physics_aux", {}))
    physics_aux_enabled = bool(physics_aux_cfg.get("enabled", False))
    physics_aux_pixel_w = float(physics_aux_cfg.get("pixel_weight", 0.0))
    physics_aux_cmean_w = float(
        physics_aux_cfg.get(
            "concentration_mean_weight",
            physics_aux_cfg.get("cmean_weight", 0.0),
        )
    )
    physics_aux_mass_w = float(physics_aux_cfg.get("mass_rel_weight", 0.0))

    diff_cfg = cfg["diffusion"]
    flow_objective = str(cfg["train"].get("objective", "default")).lower()
    diffusion_objective = str(cfg["loss"].get("diffusion_objective", "epsilon_mse")).lower()
    flow_stochastic_cfg = cfg.get("flow_matching", {})
    if not isinstance(flow_stochastic_cfg, dict):
        flow_stochastic_cfg = {}
    flow_stochastic_std = float(flow_stochastic_cfg.get("noise_stochastic_std", 0.0))
    flow_stochastic_mode = str(flow_stochastic_cfg.get("noise_stochastic_mode", "scalar")).strip().lower()
    flow_stochastic_perturb_source = bool(flow_stochastic_cfg.get("noise_stochastic_perturb_source", True))
    flow_val_nfe = max(1, int(flow_stochastic_cfg.get("val_rollout_nfe", 20)))
    flow_val_num_samples = max(1, int(flow_stochastic_cfg.get("val_num_samples", 1)))
    flow_val_deterministic = bool(flow_stochastic_cfg.get("val_deterministic", False))
    flow_val_prob_metrics = bool(flow_stochastic_cfg.get("val_probabilistic_metrics", True))
    flow_val_monitor_metric_stochastic = str(
        flow_stochastic_cfg.get("val_monitor_metric_stochastic", "endpoint_crps")
    ).strip().lower()
    sfm_sigma_z = float(flow_stochastic_cfg.get("sfm_sigma_z", max(flow_stochastic_std, 0.05)))
    sfm_sigma_min = float(flow_stochastic_cfg.get("sfm_sigma_min", 1e-3))
    sfm_sigma_max = float(flow_stochastic_cfg.get("sfm_sigma_max", 0.25))
    sfm_adaptive_sigma = bool(flow_stochastic_cfg.get("sfm_adaptive_sigma", True))
    sfm_sigma_ema_beta = float(flow_stochastic_cfg.get("sfm_sigma_ema_beta", 0.02))
    sfm_encoder_reg_lambda = float(flow_stochastic_cfg.get("sfm_encoder_reg_lambda", 0.0))
    sfm_objectives = {"sfm_latent_source_denoise_concat", "sfm_latent_source_concat"}
    if model_family == "flow_matching":
        flow_backbone = base_model.backbone if hasattr(base_model, "backbone") else base_model
        flow_backbone_cond_dim = getattr(flow_backbone, "cond_dim", None)
        flow_requires_stochastic_scalar = bool(flow_objective in sfm_objectives) or (flow_stochastic_std > 0.0)
        if flow_requires_stochastic_scalar:
            if flow_backbone_cond_dim is None:
                raise ValueError(
                    "Flow stochastic configuration requires a backbone exposing cond_dim "
                    "(expected cond_dim>=2 for [t, noise])."
                )
            flow_backbone_cond_dim = int(flow_backbone_cond_dim)
            if flow_backbone_cond_dim < 2:
                raise ValueError(
                    "Flow stochastic conditioning is enabled, but model.params.cond_dim is too small.\n"
                    f"Got cond_dim={flow_backbone_cond_dim}, expected >=2 for [t, noise] scalars.\n"
                    "Set model.params.cond_dim=2 in the flow config to avoid silently ignoring stochastic noise."
                )
    if _rank0() and model_family == "flow_matching":
        if flow_stochastic_std > 0.0 and (not flow_val_deterministic) and flow_val_num_samples <= 1:
            print(
                "[flow.val] stochastic rollout configured with single-sample validation. "
                "For stable endpoint metrics, set flow_matching.val_num_samples>1 or "
                "flow_matching.val_deterministic=true.",
                flush=True,
            )
        print(
            "[flow.val] "
            f"rollout_nfe={flow_val_nfe} "
            f"num_samples={flow_val_num_samples} "
            f"deterministic={flow_val_deterministic} "
            f"prob_metrics={flow_val_prob_metrics} "
            f"stochastic_monitor_metric={flow_val_monitor_metric_stochastic}",
            flush=True,
        )
        if flow_objective in sfm_objectives:
            print(
                "[flow.sfm] "
                f"sigma_z={sfm_sigma_z} sigma_min={sfm_sigma_min} sigma_max={sfm_sigma_max} "
                f"adaptive_sigma={sfm_adaptive_sigma} ema_beta={sfm_sigma_ema_beta} "
                f"encoder_reg_lambda={sfm_encoder_reg_lambda}",
                flush=True,
            )
    noise_schedule_obj = None
    timestep_sampler_obj = None
    region_selector_obj = None
    if model_family == "diffusion":
        noise_schedule_obj = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
        unidb_objectives = {"unidb_reverse_step", "unidb", "reverse_step_matching"}
        unidb_predict_objectives = {"unidb_predict_next", "predict_next", "predict_x0", "x0_mse", "next_field_mse"}
        unidb_allowed_objectives = unidb_objectives | unidb_predict_objectives
        sched_kind = getattr(noise_schedule_obj, "kind", "")
        if sched_kind == "unidb" and diffusion_objective not in unidb_allowed_objectives:
            raise ValueError(
                "UniDB noise_schedule requires diffusion_objective in "
                f"{sorted(unidb_allowed_objectives)} (got '{diffusion_objective}')."
            )
        if sched_kind != "unidb" and diffusion_objective in unidb_allowed_objectives:
            raise ValueError(
                "UniDB-specific diffusion_objective requires noise_schedule='unidb_*' "
                f"(got diffusion_objective='{diffusion_objective}', schedule kind '{sched_kind}')."
            )
        sampler_kwargs = dict(diff_cfg.get("sampler_kwargs", {}))
        if _rank0() and sched_kind == "unidb":
            input_mode = str(getattr(noise_schedule_obj, "input_mode", "delta_source_concat")).lower()
            t_min = sampler_kwargs.get("t_min", None)
            t_max = sampler_kwargs.get("t_max", None)
            residual_mode = str(getattr(noise_schedule_obj, "residual_mode", "none")).lower()
            residual_norm = bool(getattr(noise_schedule_obj, "residual_normalize", False))
            residual_power = float(getattr(noise_schedule_obj, "residual_power", 1.0))
            residual_clip = getattr(noise_schedule_obj, "residual_clip", None)
            residual_pi_floor = float(getattr(noise_schedule_obj, "residual_pi_floor", 0.0))
            print(
                "[unidb] "
                f"input_mode={input_mode} timesteps={getattr(noise_schedule_obj, 'timesteps', 'na')} "
                f"lambda_square={getattr(noise_schedule_obj, 'lambda_square', 'na')} "
                f"gamma_inv={getattr(noise_schedule_obj, 'gamma_inv', 'na')} "
                f"eps={getattr(noise_schedule_obj, 'eps', 'na')} "
                f"residual_mode={residual_mode} residual_normalize={residual_norm} "
                f"residual_power={residual_power} residual_clip={residual_clip} residual_pi_floor={residual_pi_floor} "
                f"sampler_t_min={t_min} sampler_t_max={t_max}",
                flush=True,
            )
        sampler_kwargs.setdefault("device", device)
        timestep_sampler_obj = get_timestep_sampler(diff_cfg["timestep_sampler"], schedule=noise_schedule_obj, **sampler_kwargs)
        if _rank0() and sched_kind == "unidb":
            sample_lo = int(getattr(timestep_sampler_obj, "t_min", -1))
            sample_hi_excl = int(getattr(timestep_sampler_obj, "_rand_high", -1))
            include_terminal = bool(getattr(timestep_sampler_obj, "include_terminal_unidb", False))
            sample_hi = sample_hi_excl - 1
            print(
                "[unidb.sampler] "
                f"sampled_t_range=[{sample_lo},{sample_hi}] high_exclusive={sample_hi_excl} "
                f"include_terminal={include_terminal} "
                "(policy: set t_min=1 and include_terminal_unidb=false for UniDB t=1..T-1 training)",
                flush=True,
            )
        region_selector_obj = build_region_selector(cfg["adaptive"].get("region_selector", "none"), **cfg["adaptive"].get("region_kwargs", {}))

    flow_matcher = None
    no_cfm_objectives = {
        "rectified_flow_constant_displacement",
        "rectified_flow_constant_displacement_concat",
        "rectified_flow_source_anchored",
        "rectified_flow_source_anchored_concat",
        "rectified_flow_noise_source_concat",
        "rectified_flow_noise_cond_concat",
        "sfm_latent_source_denoise_concat",
        "sfm_latent_source_concat",
        "dbfm_source_anchored",
        "dbfm_rectified_flow",
        "dbfm_flow",
        "dbfm",
    }
    if model_family == "flow_matching" and flow_objective not in no_cfm_objectives:
        if ConditionalFlowMatcher is None:
            raise ImportError("torchcfm is required for flow_matching but is not installed.")
        flow_matcher = ConditionalFlowMatcher(sigma=float(cfg["flow_matching"].get("sigma", 0.0)))
    if _rank0() and (physics_aux_enabled or physics_aux_pixel_w > 0.0 or physics_aux_cmean_w > 0.0 or physics_aux_mass_w > 0.0):
        print(
            "[physics_aux] "
            f"enabled={physics_aux_enabled} "
            f"pixel_weight={physics_aux_pixel_w} "
            f"concentration_mean_weight={physics_aux_cmean_w} "
            f"mass_rel_weight={physics_aux_mass_w} "
            f"concentration_channels={physics_aux_cfg.get('concentration_channels', [1])} "
            f"mass_channels={physics_aux_cfg.get('mass_channels', [])} "
            f"sub_batch_size={physics_aux_cfg.get('sub_batch_size', 0)}",
            flush=True,
        )

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
    default_metric = (
        "endpoint_rmse"
        if (default_split == "val" and bool(metrics_cfg.get("endpoint_rmse", False)))
        else "mse"
    )
    monitor_split_sched = (sched_monitor_cfg or {"split": default_split}).get("split", default_split)
    monitor_mode_sched = (sched_monitor_cfg or {"mode": default_mode}).get("mode", default_mode)
    monitor_metric_sched = (sched_monitor_cfg or {"metric": default_metric}).get("metric", default_metric)
    es_cfg = cfg["trainer"].get("early_stop", {})
    monitor_split_es = es_cfg.get("split", default_split)
    monitor_mode_es = es_cfg.get("mode", default_mode)
    monitor_metric_es = es_cfg.get("metric", monitor_metric_sched)
    flow_monitor_force_endpoint = bool(cfg["trainer"].get("flow_monitor_force_endpoint", True))
    if (
        model_family == "flow_matching"
        and flow_monitor_force_endpoint
        and bool(use_val_flag and val_dl is not None)
        and want_endpoint_rmse
    ):
        objective_like_metrics = {
            "",
            "mse",
            "rmse",
            "mae",
            "objective",
            "spectral_rmse",
            "vrmse",
        }
        endpoint_like_metrics = {
            "endpoint_mse",
            "endpoint_rmse",
            "endpoint_spectral_rmse",
        }
        prob_override_enabled = bool(
            flow_val_prob_metrics and (not flow_val_deterministic) and flow_val_num_samples > 1
        )
        preferred_flow_metric = (
            flow_val_monitor_metric_stochastic if prob_override_enabled else "endpoint_rmse"
        )
        sched_split = str(monitor_split_sched).strip().lower()
        sched_metric = str(monitor_metric_sched or "").strip().lower()
        es_split = str(monitor_split_es).strip().lower()
        es_metric = str(monitor_metric_es or "").strip().lower()
        if sched_split == "val" and sched_metric in (objective_like_metrics | endpoint_like_metrics):
            if _rank0():
                print(
                    f"[monitor] overriding scheduler metric to '{preferred_flow_metric}' for flow rollout consistency "
                    f"(was '{monitor_metric_sched}')",
                    flush=True,
                )
            monitor_metric_sched = preferred_flow_metric
        if es_split == "val" and es_metric in (objective_like_metrics | endpoint_like_metrics):
            if _rank0():
                print(
                    f"[monitor] overriding checkpoint metric to '{preferred_flow_metric}' for flow rollout consistency "
                    f"(was '{monitor_metric_es}')",
                    flush=True,
                )
            monitor_metric_es = preferred_flow_metric
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 30))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    best_metric = math.inf if monitor_mode_es == "min" else -math.inf
    es_bad = 0

    def _resolve_monitor_metric(
        *,
        split: str,
        metric_name: str,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
    ) -> float:
        src = train_metrics if str(split).lower() == "train" else (val_metrics if val_metrics is not None else train_metrics)
        requested = str(metric_name or "").strip()
        if requested and (requested in src) and (src.get(requested) is not None):
            return float(src[requested])

        endpoint_keys = (
            "endpoint_crps",
            "endpoint_rmse",
            "endpoint_mse",
            "endpoint_spectral_rmse",
            "endpoint_ssr_distance",
            "endpoint_ssr",
            "endpoint_spread",
        )
        generic_keys = ("rmse", "mse", "spectral_rmse", "objective")
        fallback_order = endpoint_keys + generic_keys if requested.startswith("endpoint_") else generic_keys + endpoint_keys
        for key in fallback_order:
            if key in src and src.get(key) is not None:
                if _rank0() and requested and requested != key:
                    print(
                        f"[monitor] requested metric '{requested}' unavailable on split='{split}', using '{key}'",
                        flush=True,
                    )
                return float(src[key])
        raise KeyError(f"No monitorable metric found for split='{split}'.")

    if _rank0():
        print(
            "[monitor] "
            f"scheduler(split={monitor_split_sched}, mode={monitor_mode_sched}, metric={monitor_metric_sched}) "
            f"checkpoint/earlystop(split={monitor_split_es}, mode={monitor_mode_es}, metric={monitor_metric_es})",
            flush=True,
        )
        print(
            "[metrics] "
            f"val_objective_logged={bool(use_val_flag and val_dl is not None)} "
            f"val_endpoint_rmse_logged={bool(want_endpoint_rmse and use_val_flag and val_dl is not None)} "
            f"val_endpoint_spectral_logged={bool(want_endpoint_rmse and want_spectral and use_val_flag and val_dl is not None)}",
            flush=True,
        )

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
        mlflow_params["train.objective"] = flow_objective
        mlflow_params["flow_matching.noise_stochastic_std"] = flow_stochastic_std
        mlflow_params["flow_matching.noise_stochastic_mode"] = flow_stochastic_mode
        mlflow_params["flow_matching.noise_stochastic_perturb_source"] = flow_stochastic_perturb_source
        mlflow_params["flow_matching.val_rollout_nfe"] = int(flow_val_nfe)
        mlflow_params["flow_matching.val_num_samples"] = int(flow_val_num_samples)
        mlflow_params["flow_matching.val_deterministic"] = bool(flow_val_deterministic)
        if flow_objective in sfm_objectives:
            mlflow_params["flow_matching.sfm_sigma_z"] = float(sfm_sigma_z)
            mlflow_params["flow_matching.sfm_sigma_min"] = float(sfm_sigma_min)
            mlflow_params["flow_matching.sfm_sigma_max"] = float(sfm_sigma_max)
            mlflow_params["flow_matching.sfm_adaptive_sigma"] = bool(sfm_adaptive_sigma)
            mlflow_params["flow_matching.sfm_sigma_ema_beta"] = float(sfm_sigma_ema_beta)
            mlflow_params["flow_matching.sfm_encoder_reg_lambda"] = float(sfm_encoder_reg_lambda)
    if model_family == "diffusion":
        mlflow_params["loss.diffusion_objective"] = diffusion_objective

    mlflow_ctx = _start_mlflow(cfg, mlflow_params)
    mlflow_parent_run_id = mlflow_ctx.parent_run_id

    if resume_path and Path(resume_path).is_file():
        state = torch.load(resume_path, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        if autoencoder is not None and state.get("autoencoder"):
            autoencoder.load_state_dict(state["autoencoder"])
        optim.load_state_dict(state["optim"])
        if sched and state.get("sched"):
            try:
                sched.load_state_dict(state["sched"])
            except Exception as e:
                if _rank0():
                    print(f"Scheduler resume skipped (state mismatch): {e}")
        if sched is not None:
            # Ensure resumed runs continue the scheduler epoch count even if state restore is skipped.
            try:
                target_epoch = max(0, start_epoch - 1)
                sched.last_epoch = target_epoch
                if hasattr(sched, "_step_count"):
                    sched._step_count = target_epoch + 1
            except Exception:
                pass
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
        want_vrmse=want_vrmse,
        want_spectral=want_spectral,
        want_endpoint_rmse=want_endpoint_rmse,
        want_endpoint_prob_metrics=flow_val_prob_metrics,
        monitor_split_es=monitor_split_es,
        monitor_mode_es=monitor_mode_es,
        resume_path=resume_path,
        start_epoch=start_epoch,
    )

    total_pairs_by_gid = _pairs_per_gid_from_dataset(train_ds) if _rank0() else None
    sfm_state: Dict[str, float] = {"sigma_z_ema": float(sfm_sigma_z)}

    interrupted = {"flag": False}

    def _mark_interrupted(signum, frame):
        _ = frame
        interrupted["flag"] = True
        if _rank0():
            print(f"[signal] received {signum}; will save interrupt checkpoint at epoch boundary", flush=True)

    import signal

    signal.signal(signal.SIGINT, _mark_interrupted)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _mark_interrupted)

    hist_epochs: List[int] = []
    hist_train: List[float] = []
    hist_val: List[float] = []

    epochs = int(cfg["trainer"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        if debug_phase_markers:
            rank_env = int(os.environ.get("RANK", "-1"))
            print(f"[phase] rank={rank_env} epoch={epoch} start", flush=True)
        if debug_barrier and _is_dist():
            use_monitored = hasattr(dist, "monitored_barrier") and dist.get_backend() == "gloo"
            if use_monitored:
                dist.monitored_barrier()
            else:
                _barrier_safe(local_rank)
        if _is_dist():
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        epoch_warmup_factor = None
        if lr_warmup_epoch_phases:
            for limit, factor in lr_warmup_epoch_phases:
                if epoch <= limit:
                    epoch_warmup_factor = factor
                    break
        do_warmup = False
        if epoch_warmup_factor is not None:
            do_warmup = True
            for pg, base_lr in zip(optim.param_groups, base_lrs):
                pg["lr"] = base_lr * epoch_warmup_factor
        else:
            do_warmup = warmup_epochs > 0 and epoch <= warmup_epochs
            if do_warmup:
                frac = float(epoch) / float(max(warmup_epochs, 1))
                for pg, base_lr in zip(optim.param_groups, base_lrs):
                    pg["lr"] = warmup_start_lr + (base_lr - warmup_start_lr) * frac

        # Optional epoch-based PSGD preconditioner warmup/ramp schedule.
        if cfg["optim"]["name"].lower() == "psgd" and hasattr(optim, "precond_schedule"):
            opt_cfg = cfg["optim"]
            warm_epochs = int(opt_cfg.get("preconditioner_warmup_epochs", 0))
            ramp_epochs = int(opt_cfg.get("preconditioner_ramp_epochs", 0))
            if warm_epochs > 0 or ramp_epochs > 0:
                ramp_mults = opt_cfg.get(
                    "preconditioner_ramp_multipliers", [0.01, 0.05, 0.1, 0.5, 1.0]
                )
                warm_freq_mult = float(opt_cfg.get("preconditioner_warmup_frequency_mult", 1.0))
                ramp_freq_mult = float(opt_cfg.get("preconditioner_ramp_frequency_mult", 1.0))
                p_target = opt_cfg.get("preconditioner_update_probability", None)
                if p_target is None:
                    p_target = opt_cfg.get("precondition_update_probability", None)
                if p_target is None:
                    p_target = 1.0 / float(opt_cfg.get("precondition_frequency", 256))
                p_target = float(p_target)
                if warm_epochs > 0 and epoch <= warm_epochs:
                    p_eff = 0.0
                    freq_mult = warm_freq_mult
                elif ramp_epochs > 0 and epoch <= warm_epochs + ramp_epochs:
                    idx = max(epoch - warm_epochs - 1, 0)
                    mult = float(ramp_mults[min(idx, len(ramp_mults) - 1)])
                    freq_mult = ramp_freq_mult
                    p_eff = p_target * mult / max(freq_mult, 1.0)
                else:
                    p_eff = p_target
                    freq_mult = 1.0
                try:
                    optim.precond_schedule = p_eff
                except Exception:
                    pass
                if debug_phase_markers and _rank0():
                    print(
                        f"[phase] epoch={epoch} psgd_precond_schedule prob={p_eff:.6g} "
                        f"freq_mult={freq_mult:g}",
                        flush=True,
                    )

        t0 = time.time()
        if use_ddp_join and use_ddp and isinstance(model, DDP):
            if debug_phase_markers and _rank0():
                print("[phase] DDP join enabled for uneven inputs", flush=True)
            with model.join():
                train_metrics = _train_one_epoch(
                    epoch=epoch,
                    model=model,
                    train_dl=train_dl,
                    device=device,
                    cond_cfg=cond_cfg,
                    autoencoder=autoencoder,
                    autoencoder_trainable=autoencoder_trainable,
                    latent_amp_enabled=amp_enabled,
                    latent_amp_dtype=amp_dtype,
                    latent_cfg=latent_cfg,
                    use_chlast=use_chlast,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                    optim=optim,
                    grad_clip=grad_clip,
                    model_family=model_family,
                    loss_fns=loss_fns,
                    flow_matcher=flow_matcher,
                    flow_objective=flow_objective,
                    diffusion_objective=diffusion_objective,
                    noise_schedule_obj=noise_schedule_obj,
                    timestep_sampler_obj=timestep_sampler_obj,
                    region_selector_obj=region_selector_obj,
                    use_weight_loss=use_weight_loss,
                    log_interval=log_interval,
                    want_mae=want_mae,
                    want_vrmse=want_vrmse,
                    vrmse_eps=vrmse_eps,
                    metric_channels=metric_channels,
                    want_spectral=want_spectral,
                    spectral_cfg=spectral_cfg,
                    spectral_train=spectral_train,
                    steps_per_epoch=steps_per_epoch,
                    total_pairs_by_gid=total_pairs_by_gid,
                    nan_debug=nan_debug,
                    nan_debug_steps=nan_debug_steps,
                    nan_debug_param=nan_debug_param,
                    nan_debug_input_stats=nan_debug_input_stats,
                    nan_tolerate_steps=nan_tolerate_steps,
                    detect_anomaly=detect_anomaly,
                    detect_anomaly_steps=detect_anomaly_steps,
                    lr_warmup_steps=lr_warmup_steps,
                    lr_warmup_start_lr=lr_warmup_start_lr,
                    lr_warmup_phases=lr_warmup_phases,
                    accumulation_steps=accumulation_steps,
                    flow_stochastic_std=flow_stochastic_std,
                    flow_stochastic_mode=flow_stochastic_mode,
                    flow_stochastic_perturb_source=flow_stochastic_perturb_source,
                    sfm_sigma_z=sfm_sigma_z,
                    sfm_sigma_min=sfm_sigma_min,
                    sfm_sigma_max=sfm_sigma_max,
                    sfm_adaptive_sigma=sfm_adaptive_sigma,
                    sfm_sigma_ema_beta=sfm_sigma_ema_beta,
                    sfm_encoder_reg_lambda=sfm_encoder_reg_lambda,
                    sfm_state=sfm_state,
                )
        else:
                train_metrics = _train_one_epoch(
                    epoch=epoch,
                    model=model,
                    train_dl=train_dl,
                    device=device,
                cond_cfg=cond_cfg,
                autoencoder=autoencoder,
                autoencoder_trainable=autoencoder_trainable,
                latent_amp_enabled=amp_enabled,
                latent_amp_dtype=amp_dtype,
                latent_cfg=latent_cfg,
                use_chlast=use_chlast,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
                optim=optim,
                grad_clip=grad_clip,
                model_family=model_family,
                loss_fns=loss_fns,
                flow_matcher=flow_matcher,
                flow_objective=flow_objective,
                diffusion_objective=diffusion_objective,
                noise_schedule_obj=noise_schedule_obj,
                timestep_sampler_obj=timestep_sampler_obj,
                region_selector_obj=region_selector_obj,
                use_weight_loss=use_weight_loss,
                log_interval=log_interval,
                want_mae=want_mae,
                want_vrmse=want_vrmse,
                vrmse_eps=vrmse_eps,
                metric_channels=metric_channels,
                want_spectral=want_spectral,
                spectral_cfg=spectral_cfg,
                spectral_train=spectral_train,
                steps_per_epoch=steps_per_epoch,
                total_pairs_by_gid=total_pairs_by_gid,
                nan_debug=nan_debug,
                nan_debug_steps=nan_debug_steps,
                nan_debug_param=nan_debug_param,
                nan_debug_input_stats=nan_debug_input_stats,
                nan_tolerate_steps=nan_tolerate_steps,
                detect_anomaly=detect_anomaly,
                detect_anomaly_steps=detect_anomaly_steps,
                lr_warmup_steps=lr_warmup_steps,
                lr_warmup_start_lr=lr_warmup_start_lr,
                lr_warmup_phases=lr_warmup_phases,
                    accumulation_steps=accumulation_steps,
                    flow_stochastic_std=flow_stochastic_std,
                    flow_stochastic_mode=flow_stochastic_mode,
                    flow_stochastic_perturb_source=flow_stochastic_perturb_source,
                    sfm_sigma_z=sfm_sigma_z,
                    sfm_sigma_min=sfm_sigma_min,
                    sfm_sigma_max=sfm_sigma_max,
                    sfm_adaptive_sigma=sfm_adaptive_sigma,
                    sfm_sigma_ema_beta=sfm_sigma_ema_beta,
                    sfm_encoder_reg_lambda=sfm_encoder_reg_lambda,
                    sfm_state=sfm_state,
                )
        _check_step_sync(train_metrics, device, local_rank)

        val_metrics = None
        if use_val_flag and val_dl is not None:
            if debug_phase_markers:
                rank_env = int(os.environ.get("RANK", "-1"))
                print(f"[phase] rank={rank_env} epoch={epoch} validation_start", flush=True)
            if debug_barrier and _is_dist():
                use_monitored = hasattr(dist, "monitored_barrier") and dist.get_backend() == "gloo"
                if use_monitored:
                    dist.monitored_barrier()
                else:
                    _barrier_safe(local_rank)
            vrmse_ref_var = train_metrics.get("var_per_channel") if want_vrmse else None
            val_metrics = _validate_epoch(
                model=model,
                val_dl=val_dl,
                device=device,
                cond_cfg=cond_cfg,
                autoencoder=autoencoder,
                latent_amp_enabled=amp_enabled,
                latent_amp_dtype=amp_dtype,
                latent_cfg=latent_cfg,
                use_chlast=use_chlast,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                model_family=model_family,
                loss_fns=loss_fns,
                flow_matcher=flow_matcher,
                flow_objective=flow_objective,
                diffusion_objective=diffusion_objective,
                noise_schedule_obj=noise_schedule_obj,
                timestep_sampler_obj=timestep_sampler_obj,
                region_selector_obj=region_selector_obj,
                want_mae=want_mae,
                want_vrmse=want_vrmse,
                vrmse_eps=vrmse_eps,
                metric_channels=metric_channels,
                vrmse_ref_var=vrmse_ref_var,
                want_spectral=want_spectral,
                spectral_cfg=spectral_cfg if want_spectral else None,
                want_endpoint_rmse=want_endpoint_rmse,
                flow_stochastic_std=flow_stochastic_std,
                flow_stochastic_mode=flow_stochastic_mode,
                flow_stochastic_perturb_source=flow_stochastic_perturb_source,
                flow_val_nfe=flow_val_nfe,
                flow_val_num_samples=flow_val_num_samples,
                flow_val_deterministic=flow_val_deterministic,
                flow_val_prob_metrics=flow_val_prob_metrics,
                sfm_sigma_z=sfm_sigma_z,
                sfm_sigma_min=sfm_sigma_min,
                sfm_sigma_max=sfm_sigma_max,
                sfm_adaptive_sigma=sfm_adaptive_sigma,
                sfm_sigma_ema_beta=sfm_sigma_ema_beta,
                sfm_encoder_reg_lambda=sfm_encoder_reg_lambda,
                sfm_state=sfm_state,
            )
            if debug_phase_markers:
                rank_env = int(os.environ.get("RANK", "-1"))
                print(f"[phase] rank={rank_env} epoch={epoch} validation_end", flush=True)
        if _is_dist():
            use_monitored = debug_barrier and hasattr(dist, "monitored_barrier") and dist.get_backend() == "gloo"
            if use_monitored:
                dist.monitored_barrier()
            else:
                _barrier_safe(local_rank)

        # Scheduler updates should follow actual optimizer updates; skip epochs with 0 optimizer steps.
        # This matches PyTorch guidance ("scheduler.step() after optimizer.step()").
        did_optimizer_step = int(train_metrics.get("opt_step_count", 0)) > 0
        if sched is not None and not do_warmup:
            if did_optimizer_step:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = _resolve_monitor_metric(
                        split=monitor_split_sched,
                        metric_name=monitor_metric_sched,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                    sched.step(metric)
                else:
                    sched.step()
            elif _rank0():
                print(
                    f"[lr] epoch={epoch} skip_scheduler_step reason=no_optimizer_step",
                    flush=True,
                )

        lr_val = optim.param_groups[0]["lr"]

        dt = time.time() - t0
        if _rank0():
            sched_name = str(cfg.get("sched", {}).get("name", "none"))
            if epoch_warmup_factor is not None:
                print(
                    f"[lr] epoch={epoch} lr={lr_val:.6g} warmup_factor={epoch_warmup_factor:g} sched={sched_name}",
                    flush=True,
                )
            else:
                print(f"[lr] epoch={epoch} lr={lr_val:.6g} sched={sched_name}", flush=True)
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

        metric_for_es = _resolve_monitor_metric(
            split=monitor_split_es,
            metric_name=monitor_metric_es,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        improved = (metric_for_es < best_metric - es_min_delta) if monitor_mode_es == "min" else (metric_for_es > best_metric + es_min_delta)
        if improved:
            best_metric = metric_for_es
            es_bad = 0
        else:
            es_bad += 1

        if debug_phase_markers:
            rank_env = int(os.environ.get("RANK", "-1"))
            print(f"[phase] rank={rank_env} epoch={epoch} checkpoint_start", flush=True)
        if _is_dist():
            use_monitored = debug_barrier and hasattr(dist, "monitored_barrier") and dist.get_backend() == "gloo"
            if use_monitored:
                dist.monitored_barrier()
            else:
                _barrier_safe(local_rank)
        if _rank0():
            state = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "autoencoder": (autoencoder.state_dict() if autoencoder is not None else None),
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
        if debug_phase_markers:
            rank_env = int(os.environ.get("RANK", "-1"))
            print(f"[phase] rank={rank_env} epoch={epoch} checkpoint_end", flush=True)
        if _is_dist():
            use_monitored = debug_barrier and hasattr(dist, "monitored_barrier") and dist.get_backend() == "gloo"
            if use_monitored:
                dist.monitored_barrier()
            else:
                _barrier_safe(local_rank)

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
                    "autoencoder": (autoencoder.state_dict() if autoencoder is not None else None),
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
