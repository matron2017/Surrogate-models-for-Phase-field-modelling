"""
Optimizer and scheduler builders.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import os
from functools import partial
from bisect import bisect_right

import torch

try:
    import heavyball
    from heavyball import (
        ForeachCachedDelayedPSGDKron,
        ForeachCachedPSGDKron,
        ForeachDelayedPSGD,
        ForeachPSGDKron,
    )
except Exception:
    heavyball = None
    ForeachCachedDelayedPSGDKron = None
    ForeachCachedPSGDKron = None
    ForeachDelayedPSGD = None
    ForeachPSGDKron = None


def _precond_prob_schedule(n, max_prob=1.0, min_prob=0.01, decay=0.999, flat_start=0, warmup=0):
    if warmup and n < warmup:
        return 0.0
    return max(min_prob, max_prob * decay ** max(n - warmup - flat_start, 0))


def _make_optimizer(params, cfg: Dict[str, Any]):
    name = cfg["optim"]["name"].lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"].get("weight_decay", 0.0))
    betas = tuple(cfg["optim"].get("betas", [0.9, 0.999]))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "psgd":
        if ForeachCachedDelayedPSGDKron is None or heavyball is None:
            raise ImportError("Optimizer 'psgd' requires the heavyball package.")
        hb_compile = os.getenv("HEAVYBALL_COMPILE_MODE", "none").strip().lower()
        if hb_compile in ("", "none", "off", "0", "false"):
            heavyball.utils.compile_mode = None
        else:
            heavyball.utils.compile_mode = hb_compile
        precond_freq = int(cfg["optim"].get("precondition_frequency", 16))
        precond_decay = float(cfg["optim"].get("precondition_frequency_decay", 0.999))
        precond_size = int(cfg["optim"].get("precondition_size", 4096))
        precond_warmup = int(cfg["optim"].get("precondition_warmup", 0))
        precond_override = cfg["optim"].get("preconditioner_update_probability", None)
        if precond_override is None:
            precond_override = cfg["optim"].get("precondition_update_probability", None)
        merge_dims = bool(cfg["optim"].get("merge_dims", False))
        cached = bool(cfg["optim"].get("psgd_cached", True))
        delayed = bool(cfg["optim"].get("psgd_delayed", True))
        if precond_freq <= 0:
            raise ValueError("optim.precondition_frequency must be > 0 for PSGD.")
        if precond_override is not None:
            precond_override = float(precond_override)
            if precond_override < 0.0 or precond_override > 1.0:
                raise ValueError("optim.preconditioner_update_probability must be within [0, 1].")
            precond_prob = precond_override
        else:
            precond_prob = partial(
                _precond_prob_schedule,
                min_prob=1 / precond_freq,
                decay=precond_decay,
                warmup=precond_warmup,
            )
        if cached and delayed:
            opt_cls = ForeachCachedDelayedPSGDKron
        elif cached and not delayed:
            opt_cls = ForeachCachedPSGDKron
        elif not cached and delayed:
            opt_cls = ForeachDelayedPSGD
        else:
            opt_cls = ForeachPSGDKron
        return opt_cls(
            params,
            lr=lr,
            beta=betas[0],
            weight_decay=wd,
            preconditioner_update_probability=precond_prob,
            max_size_triangular=precond_size,
            merge_dims=merge_dims,
        )
    raise ValueError(f"Unknown optimizer {name}")


def _make_scheduler(optim, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    s = cfg["sched"]["name"].lower()
    if s == "none":
        return None, None
    if s == "cosine":
        T_max = int(cfg["sched"].get("T_max", 100))
        speedup = float(cfg["sched"].get("cosine_speedup", 1.0))
        eta_min = float(cfg["sched"].get("eta_min", 0.0))
        if speedup <= 1.0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=eta_min), None
        base_lrs = [pg["lr"] for pg in optim.param_groups]

        def _make_lambda(base_lr: float):
            if base_lr <= 0:
                return lambda t: 0.0
            eta_ratio = eta_min / base_lr

            def _f(t: int):
                t_eff = min(float(t) * speedup, float(T_max))
                return eta_ratio + (1.0 - eta_ratio) * 0.5 * (1.0 + math.cos(math.pi * t_eff / T_max))

            return _f

        lambdas = [_make_lambda(lr) for lr in base_lrs]
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambdas), None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=int(cfg["sched"].get("step_size", 30)), gamma=float(cfg["sched"].get("gamma", 0.1))
        ), None
    if s == "multistep":
        milestones_cfg = cfg["sched"].get("milestones", None)
        if milestones_cfg is None:
            raise ValueError("sched.milestones must be provided for multistep schedule.")
        if isinstance(milestones_cfg, str):
            parts = [p for p in milestones_cfg.replace(",", " ").split() if p]
            milestones = [int(p) for p in parts]
        else:
            milestones = [int(m) for m in milestones_cfg]
        milestones = sorted(set(milestones))
        gamma_cfg = cfg["sched"].get("gamma", 0.1)
        if isinstance(gamma_cfg, (list, tuple)):
            gamma_seq = [float(g) for g in gamma_cfg]
            if len(gamma_seq) == 1:
                gamma_seq = gamma_seq * len(milestones)
            if len(gamma_seq) != len(milestones):
                raise ValueError(
                    f"len(sched.gamma)={len(gamma_seq)} must be 1 or equal to len(sched.milestones)={len(milestones)}."
                )
            base_lrs = [pg["lr"] for pg in optim.param_groups]

            def _make_lambda(base_lr: float, gamma_sequence):
                del base_lr  # keep pyright happy for unused captures in older torch versions

                def _f(t: int):
                    n = bisect_right(milestones, t)
                    lr_mult = 1.0
                    for i in range(n):
                        lr_mult *= gamma_sequence[i]
                    return lr_mult

                return _f

            lambdas = [_make_lambda(lr, gamma_seq) for lr in base_lrs]
            return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambdas), None

        return torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=milestones, gamma=float(gamma_cfg)
        ), None
    if s == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optim, gamma=float(cfg["sched"].get("gamma", 0.97))), None
    if s == "plateau":
        monitor = cfg["sched"].get("monitor", {"split": "train", "mode": "min"})
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode=monitor.get("mode", "min"),
                factor=float(cfg["sched"].get("factor", 0.5)),
                patience=int(cfg["sched"].get("patience", 10)),
                min_lr=float(cfg["sched"].get("min_lr", 1e-7)),
                threshold=float(cfg["sched"].get("threshold", 1e-4)),
            ),
            monitor,
        )
    raise ValueError(f"Unknown scheduler {s}")
