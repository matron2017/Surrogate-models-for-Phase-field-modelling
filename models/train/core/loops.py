"""
Train/validation loop helpers and per-epoch strategies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from models.train.core.utils import (
    _prepare_batch,
    _q_sample,
    _allreduce_sum_count,
    _all_gather_object,
    _rank0,
)


def _pairs_per_gid_from_dataset(ds):
    if hasattr(ds, "items"):
        return Counter(g for g, _ in ds.items)
    from torch.utils.data import Subset

    if isinstance(ds, Subset) and hasattr(ds.dataset, "items"):
        base_items = ds.dataset.items
        idxs = ds.indices
        return Counter(base_items[i][0] for i in idxs)
    return None


def _train_one_epoch(
    epoch: int,
    model,
    train_dl,
    device,
    cond_cfg,
    use_chlast: bool,
    amp_enabled: bool,
    amp_dtype,
    scaler: GradScaler,
    optim,
    grad_clip: float,
    model_family: str,
    loss_fns: Dict[str, Any],
    flow_matcher,
    noise_schedule_obj,
    timestep_sampler_obj,
    region_selector_obj,
    use_weight_loss: bool,
    log_interval: int,
    want_mae: bool,
    total_pairs_by_gid: Optional[Counter],
):
    model.train()
    se_sum, elem_count = 0.0, 0
    mae_sum = 0.0 if want_mae else None
    log_se_sum, log_elem_count = 0.0, 0
    log_mae_sum = 0.0 if want_mae else None
    num_steps = len(train_dl) if hasattr(train_dl, "__len__") else None

    gid_stats_local = defaultdict(lambda: [0.0, 0])
    seen_pairs_local = set()
    train_task_metric_sum: Dict[str, float] = {}
    train_task_metric_count: Dict[str, int] = {}

    for step_idx, batch in enumerate(train_dl, start=1):
        x, y, cond = _prepare_batch(batch, device, cond_cfg, use_chlast)
        weight = batch.get("weight", None)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)
            if use_chlast and weight.dim() == 4:
                weight = weight.contiguous(memory_format=torch.channels_last)

        optim.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            if model_family == "diffusion":
                t = timestep_sampler_obj.sample(batch_size=y.shape[0])
                t = torch.tensor(t, device=device, dtype=torch.long) if not torch.is_tensor(t) else t.to(device=device, dtype=torch.long)
                x_noisy, eps = _q_sample(y, t, noise_schedule_obj)
                if use_chlast:
                    x_noisy = x_noisy.contiguous(memory_format=torch.channels_last)
                x_in = torch.cat([x_noisy, x], dim=1)
                region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                pred = model(x_in, t, cond, region_info=region_info) if cond is not None else model(x_in, t, region_info=region_info)
                loss = loss_fns["diffusion"](pred, eps, target=y, region_info=region_info)
                metric_target = eps
            elif model_family == "flow_matching":
                x0_noise = torch.randn_like(y)
                t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x0_noise, x1=y)
                t_match = t_match.to(device).view(-1, 1)
                x_t = x_t.to(device)
                u_t = u_t.to(device)
                if use_chlast:
                    x_t = x_t.contiguous(memory_format=torch.channels_last)
                cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                x_in = torch.cat([x_t, x], dim=1)
                pred = model(x_in, cond_t)
                loss = F.mse_loss(pred, u_t, reduction="mean")
                metric_target = u_t
            else:
                pred = model(x, cond) if cond is not None else model(x)
                dataset_weight = weight if (use_weight_loss and weight is not None) else None
                loss = loss_fns["surrogate"](pred, y, weight=dataset_weight)
                metric_target = y
                metric_owner = model.module if isinstance(model, DDP) else model
                if hasattr(metric_owner, "compute_metrics"):
                    try:
                        task_metrics = metric_owner.compute_metrics(pred.detach(), metric_target.detach())  # type: ignore[arg-type]
                    except Exception:
                        task_metrics = {}
                    for name, val in task_metrics.items():
                        train_task_metric_sum[name] = train_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                        train_task_metric_count[name] = train_task_metric_count.get(name, 0) + 1

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optim.step()

        with torch.no_grad():
            mse_batch = F.mse_loss(pred, metric_target, reduction="mean")
            if want_mae:
                mae_batch = F.l1_loss(pred, metric_target, reduction="mean")

            elems = metric_target.numel()
            se_sum += float(mse_batch.detach().cpu()) * elems
            elem_count += elems
            if want_mae:
                mae_sum += float(mae_batch.detach().cpu()) * elems
            log_se_sum += float(mse_batch.detach().cpu()) * elems
            log_elem_count += elems
            if want_mae and log_mae_sum is not None:
                log_mae_sum += float(mae_batch.detach().cpu()) * elems

            ps_mse = (pred.detach() - metric_target.detach())
            ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()

        for g, k, m in zip(batch["gid"], batch["pair_index"], ps_mse):
            gid_stats_local[g][0] += float(m)
            gid_stats_local[g][1] += 1
            seen_pairs_local.add((g, int(k)))

        if log_interval > 0 and _rank0() and (step_idx % log_interval == 0):
            int_mse = log_se_sum / max(log_elem_count, 1)
            int_rmse = math.sqrt(max(int_mse, 0.0))
            msg = f"epoch={epoch} step={step_idx}"
            if num_steps:
                msg += f"/{num_steps}"
            msg += f" train_rmse={int_rmse:.6f}"
            if want_mae and log_mae_sum is not None:
                int_mae = log_mae_sum / max(log_elem_count, 1)
                msg += f" train_mae={int_mae:.6f}"
            msg += f" lr={optim.param_groups[0]['lr']:.6g}"
            print(msg, flush=True)
            log_se_sum, log_elem_count = 0.0, 0
            if want_mae:
                log_mae_sum = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    se_sum_g, elem_count_g = _allreduce_sum_count(device, se_sum, elem_count)
    train_mse = se_sum_g / max(elem_count_g, 1)
    train_rmse = math.sqrt(max(train_mse, 0.0))
    train_mae = None
    if want_mae:
        mae_sum_g, _ = _allreduce_sum_count(device, mae_sum, elem_count)
        train_mae = mae_sum_g / max(elem_count_g, 1)

    gid_stats_all = _all_gather_object(dict(gid_stats_local))
    seen_sets_all = _all_gather_object(seen_pairs_local)
    gid_stats_merged = defaultdict(lambda: [0.0, 0])
    if _rank0():
        for d in gid_stats_all:
            for g, (s, c) in d.items():
                gid_stats_merged[g][0] += float(s)
                gid_stats_merged[g][1] += int(c)
        seen_all = set().union(*seen_sets_all)
    else:
        seen_all = set()

    coverage = None
    if _rank0() and total_pairs_by_gid is not None:
        coverage = {}
        cov_counts = Counter(g for g, _ in seen_all)
        for g, tot in total_pairs_by_gid.items():
            frac = cov_counts.get(g, 0) / max(tot, 1)
            coverage[g] = frac

    train_task_avg = {k: v / max(train_task_metric_count.get(k, 1), 1) for k, v in train_task_metric_sum.items()}
    return {
        "mse": train_mse,
        "rmse": train_rmse,
        "mae": train_mae,
        "gid_stats": gid_stats_merged,
        "coverage": coverage,
        "task_avg": train_task_avg,
    }


def _validate_epoch(
    model,
    val_dl,
    device,
    cond_cfg,
    use_chlast: bool,
    amp_enabled: bool,
    amp_dtype,
    model_family: str,
    loss_fns: Dict[str, Any],
    flow_matcher,
    noise_schedule_obj,
    timestep_sampler_obj,
    region_selector_obj,
    want_mae: bool,
):
    if val_dl is None:
        return None
    model.eval()
    v_se_sum, v_elem_count = 0.0, 0
    v_mae_sum = 0.0 if want_mae else None
    v_gid_stats_local = defaultdict(lambda: [0.0, 0]) if model_family != "diffusion" else None
    val_task_metric_sum: Dict[str, float] = {}
    val_task_metric_count: Dict[str, int] = {}

    with torch.inference_mode():
        for batch in val_dl:
            x, y, cond = _prepare_batch(batch, device, cond_cfg, use_chlast)
            if model_family == "diffusion":
                t = timestep_sampler_obj.sample(batch_size=y.shape[0])
                t = torch.tensor(t, device=device, dtype=torch.long) if not torch.is_tensor(t) else t.to(device=device, dtype=torch.long)
                x_noisy, eps = _q_sample(y, t, noise_schedule_obj)
                if use_chlast:
                    x_noisy = x_noisy.contiguous(memory_format=torch.channels_last)
                x_in = torch.cat([x_noisy, x], dim=1)
                region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    pred = model(x_in, t, cond, region_info=region_info) if cond is not None else model(
                        x_in, t, region_info=region_info
                    )
                    vloss = F.mse_loss(pred, eps)
                    if want_mae:
                        vmae = F.l1_loss(pred, eps)
                elems = eps.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
            elif model_family == "flow_matching":
                x0_noise = torch.randn_like(y)
                t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x0_noise, x1=y)
                t_match = t_match.to(device).view(-1, 1)
                x_t = x_t.to(device)
                u_t = u_t.to(device)
                if use_chlast:
                    x_t = x_t.contiguous(memory_format=torch.channels_last)
                cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    x_in = torch.cat([x_t, x], dim=1)
                    pred = model(x_in, cond_t)
                    vloss = F.mse_loss(pred, u_t)
                    if want_mae:
                        vmae = F.l1_loss(pred, u_t)
                elems = u_t.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
                ps_mse = (pred.detach() - u_t.detach())
                ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                for g, m in zip(batch["gid"], ps_mse):
                    v_gid_stats_local[g][0] += float(m)
                    v_gid_stats_local[g][1] += 1
            else:
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    pred = model(x, cond) if cond is not None else model(x)
                    vloss = F.mse_loss(pred, y)
                    if want_mae:
                        vmae = F.l1_loss(pred, y)
                    metric_owner = model.module if isinstance(model, DDP) else model
                    if hasattr(metric_owner, "compute_metrics"):
                        try:
                            task_metrics = metric_owner.compute_metrics(pred.detach(), y.detach())  # type: ignore[arg-type]
                        except Exception:
                            task_metrics = {}
                        for name, val in task_metrics.items():
                            val_task_metric_sum[name] = val_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                            val_task_metric_count[name] = val_task_metric_count.get(name, 0) + 1
                elems = y.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
                ps_mse = (pred.detach() - y.detach())
                ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                for g, m in zip(batch["gid"], ps_mse):
                    v_gid_stats_local[g][0] += float(m)
                    v_gid_stats_local[g][1] += 1

    v_se_sum_g, v_elem_count_g = _allreduce_sum_count(device, v_se_sum, v_elem_count)
    val_mse = v_se_sum_g / max(v_elem_count_g, 1)
    val_rmse = math.sqrt(max(val_mse, 0.0))
    val_mae = None
    if want_mae and v_mae_sum is not None:
        v_mae_sum_g, _ = _allreduce_sum_count(device, v_mae_sum, v_elem_count)
        val_mae = v_mae_sum_g / max(v_elem_count_g, 1)

    val_gid_stats_merged = None
    if v_gid_stats_local is not None:
        v_gid_stats_all = _all_gather_object(dict(v_gid_stats_local))
        if _rank0():
            val_gid_stats_merged = defaultdict(lambda: [0.0, 0])
            for d in v_gid_stats_all:
                for g, (s, c) in d.items():
                    val_gid_stats_merged[g][0] += float(s)
                    val_gid_stats_merged[g][1] += int(c)

    val_task_avg = (
        {k: v / max(val_task_metric_count.get(k, 1), 1) for k, v in val_task_metric_sum.items()}
        if val_task_metric_sum
        else {}
    )
    return {
        "mse": val_mse,
        "rmse": val_rmse,
        "mae": val_mae,
        "gid_stats": val_gid_stats_merged,
        "task_avg": val_task_avg,
    }
