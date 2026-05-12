#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/scratch/project_2008261/pf_surrogate_modelling")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_field_surrogates.models.unet_film_bottleneck import UNetFiLMAttn
from Phase_field_surrogates.utils.pf_dataloader import PFPairDataset
from Phase_field_surrogates.utils.wavelet_weight import wavelet_multiband_importance_per_channel


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str, max_items: int | None = None):
    ds_args = dict(cfg["dataloader"].get("args", {}))
    split_args = dict(cfg["dataloader"].get(f"{split}_args", {}))
    ds_args.update(split_args)
    if max_items is not None:
        ds_args["max_items"] = max_items
    return PFPairDataset(cfg["paths"]["h5"][split], **ds_args)


def weighted_wavelet_mse(pred: torch.Tensor, target: torch.Tensor, wcfg: dict):
    wmap, _ = wavelet_multiband_importance_per_channel(
        target,
        J=int(wcfg.get("J", 3)),
        wave=str(wcfg.get("wave", "haar")),
        mode=str(wcfg.get("mode", "zero")),
        level_weights=list(wcfg.get("level_weights", [2.0, 1.4, 0.7])),
        lowpass_weight=float(wcfg.get("lowpass_weight", 1.2)),
        beta_w=float(wcfg.get("beta_w", 120.0)),
        power=float(wcfg.get("power", 1.8)),
        norm_quantile=float(wcfg.get("norm_quantile", 0.95)),
        normalize_mean=bool(wcfg.get("normalize", True)),
        rescale_max=bool(wcfg.get("rescale_max", False)),
        clip_max=float(wcfg.get("clip_max", 16.0)),
        combine_norm=bool(wcfg.get("combine_norm", True)),
    )
    return torch.mean(wmap * (pred - target) ** 2), wmap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-items", type=int, default=8)
    ap.add_argument("--max-val-items", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = int(cfg.get("trainer", {}).get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"

    train_ds = build_dataset(cfg, "train", max_items=args.max_train_items)
    val_ds = build_dataset(cfg, "val", max_items=args.max_val_items)
    dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = UNetFiLMAttn(**dict(cfg["model"]["params"]))
    model = model.to(device).train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"].get("lr", 1.0e-4)),
        betas=tuple(cfg["optim"].get("betas", [0.9, 0.999])),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.01)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    wavelet_mix = float(cfg["loss"].get("weight_wavelet_loss", 0.0))
    wavelet_cfg = dict(cfg["loss"].get("wavelet_weight", {}))

    rows = []
    last_wavelet_shape = None
    for step, batch in enumerate(dl, start=1):
        if step > int(args.steps):
            break
        x_full = batch["input"].float().to(device)
        y = batch["target"].float().to(device)
        x = x_full[:, :2]
        theta = x_full[:, 2:3]
        cond_vec = torch.zeros((x.shape[0], int(cfg["model"]["params"].get("cond_dim", 2))), dtype=x.dtype, device=device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(x, cond_vec, theta=theta)
            base_mse = torch.mean((pred - y) ** 2)
            if wavelet_mix > 0.0:
                wloss, wmap = weighted_wavelet_mse(pred, y, wavelet_cfg)
                loss = (1.0 - wavelet_mix) * base_mse + wavelet_mix * wloss
                last_wavelet_shape = list(wmap.shape)
            else:
                wloss = torch.tensor(0.0, device=device)
                loss = base_mse
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_norm_(model.parameters(), max_norm=float(cfg["trainer"].get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()

        rows.append({
            "step": step,
            "loss": float(loss.item()),
            "base_mse": float(base_mse.item()),
            "wavelet_mse": float(wloss.item()),
        })

    model.eval()
    with torch.inference_mode():
        v = val_ds[0]
        vx_full = v["input"].unsqueeze(0).float().to(device)
        vy = v["target"].unsqueeze(0).float().to(device)
        vx = vx_full[:, :2]
        vtheta = vx_full[:, 2:3]
        vcond = torch.zeros((1, int(cfg["model"]["params"].get("cond_dim", 2))), dtype=vx.dtype, device=device)
        vpred = model(vx, vcond, theta=vtheta)
        val_mse = float(torch.mean((vpred - vy) ** 2).item())

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "base_mse", "wavelet_mse"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "train_items": len(train_ds),
        "val_items": len(val_ds),
        "steps_ran": len(rows),
        "wavelet_enabled": wavelet_mix > 0.0,
        "wavelet_mix": wavelet_mix,
        "wavelet_shape": last_wavelet_shape,
        "first_loss": rows[0]["loss"] if rows else None,
        "last_loss": rows[-1]["loss"] if rows else None,
        "val_mse": val_mse,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
