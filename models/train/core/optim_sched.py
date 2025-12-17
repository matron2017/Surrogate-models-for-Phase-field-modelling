"""
Optimizer and scheduler builders.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _make_optimizer(params, cfg: Dict[str, Any]):
    name = cfg["optim"]["name"].lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"].get("weight_decay", 0.0))
    betas = tuple(cfg["optim"].get("betas", [0.9, 0.999]))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")


def _make_scheduler(optim, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    s = cfg["sched"]["name"].lower()
    if s == "none":
        return None, None
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg["sched"].get("T_max", 100))), None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=int(cfg["sched"].get("step_size", 30)), gamma=float(cfg["sched"].get("gamma", 0.1))
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
