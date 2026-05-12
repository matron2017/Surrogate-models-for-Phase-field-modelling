#!/usr/bin/env python3
"""Train the custom PDEConvAE on 3-channel 512×512 solidification data.

Handles single-GPU and torchrun multi-GPU / multi-node DDP.
Logs CSV metrics, saves checkpoints, cosine LR schedule with warmup.

Usage (single GPU):
  python scripts/train_pde_conv_ae.py --config configs_current/autoencoder/train_pde_conv_ae_f16c32_512.yaml

Usage (multi-GPU torchrun):
  torchrun --nproc_per_node=4 scripts/train_pde_conv_ae.py --config ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT",
    Path(__file__).resolve().parent.parent)).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.pde_conv_ae import PDEConvAE  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _log(msg: str) -> None:
    if _is_main():
        print(msg, flush=True)


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PDEFieldDataset(Dataset):
    """Reads (phi, c, theta) frames from HDF5, normalises to [-1, 1] per channel."""

    def __init__(self, h5_path: str, t_start: int = 0, t_step: int = 1,
                 max_frames: Optional[int] = None, augment: bool = False,
                 norm_stats: Optional[dict] = None) -> None:
        self.h5_path = h5_path
        self.augment = augment
        self.index: list[tuple[str, int]] = []
        with h5py.File(h5_path, "r") as f:
            for sim in sorted(f.keys()):
                n_t = f[sim]["images"].shape[0]
                for t in range(t_start, n_t, t_step):
                    self.index.append((sim, t))
                    if max_frames and len(self.index) >= max_frames:
                        break
                if max_frames and len(self.index) >= max_frames:
                    break
        if norm_stats is not None:
            self.norm_min = np.array(norm_stats["min"], dtype=np.float32)
            self.norm_scale = np.array(norm_stats["scale"], dtype=np.float32)
        else:
            self._compute_stats()

    def _compute_stats(self) -> None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(self.index), size=min(200, len(self.index)), replace=False)
        mins = np.full(3, np.inf, dtype=np.float32)
        maxs = np.full(3, -np.inf, dtype=np.float32)
        with h5py.File(self.h5_path, "r") as f:
            for i in idx:
                sim, t = self.index[i]
                x = self._read_frame(f, sim, t)
                for c in range(3):
                    mins[c] = min(mins[c], float(x[c].min()))
                    maxs[c] = max(maxs[c], float(x[c].max()))
        scale = maxs - mins
        scale[scale == 0] = 1.0
        self.norm_min = mins
        self.norm_scale = scale

    @property
    def norm_stats(self) -> dict:
        return {"min": self.norm_min.tolist(), "scale": self.norm_scale.tolist()}

    @staticmethod
    def _read_frame(f: h5py.File, sim: str, t: int) -> np.ndarray:
        g = f[sim]
        x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
        th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
        return np.concatenate([x2, th], axis=0)  # (3, H, W)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sim, t = self.index[idx]
        with h5py.File(self.h5_path, "r") as f:
            x = self._read_frame(f, sim, t)
        x = (x - self.norm_min[:, None, None]) / self.norm_scale[:, None, None]
        x = x * 2.0 - 1.0
        x = torch.from_numpy(x)
        if self.augment:
            k = int(torch.randint(4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
            if torch.rand(1).item() > 0.5:
                x = x.flip(dims=[2])
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                      dtype=x.dtype, device=x.device).unsqueeze(0)
    ky = kx.transpose(-2, -1)
    B, C, H, W = x.shape
    xf = x.reshape(B * C, 1, H, W)
    xf = F.pad(xf, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xf, kx)
    gy = F.conv2d(xf, ky)
    # eps avoids grad(sqrt(0))=inf → GradScaler skip-all-steps
    return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt().reshape(B, C, H, W)


def pde_ae_loss(pred: torch.Tensor, target: torch.Tensor,
                lambda_l1: float = 1.0, lambda_grad: float = 0.5,
                lambda_spec: float = 0.0) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(pred, target)
    w = torch.ones(pred.shape[1], device=pred.device, dtype=pred.dtype)
    w[0] = 2.0  # extra weight on phi (sharp interface)
    grad_l = (F.l1_loss(_sobel_edges(pred), _sobel_edges(target),
                        reduction="none") * w[None, :, None, None]).mean()
    total = lambda_l1 * l1 + lambda_grad * grad_l
    return {"total": total, "l1": l1, "grad": grad_l}


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule with optional warmup
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(epoch: int, total_epochs: int, lr_base: float, lr_min: float,
              warmup_epochs: int = 0, warmup_lr: float = 0.0) -> float:
    if epoch < warmup_epochs:
        return warmup_lr + (lr_base - warmup_lr) * epoch / max(warmup_epochs, 1)
    t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * t))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = _expand_env(yaml.safe_load(f))

    # DDP init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    world_size = dist.get_world_size() if is_ddp else 1
    rank = dist.get_rank() if is_ddp else 0

    # Paths
    data_root = Path(os.environ.get("DATA_ROOT",
        str(Path(args.config).parent.parent.parent / "data")))
    runs_root = Path(os.environ.get("RUNS_ROOT",
        str(Path(args.config).parent.parent.parent / "runs")))
    out_dir = Path(str(cfg["trainer"]["out_dir"]).replace("${RUNS_ROOT}", str(runs_root)))
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    train_h5 = str(Path(str(cfg["data"]["train_h5"]).replace("${DATA_ROOT}", str(data_root))))
    val_h5   = str(Path(str(cfg["data"]["val_h5"]).replace("${DATA_ROOT}", str(data_root))))

    # Dataset (share norm_stats from train to val so scales match)
    train_ds = PDEFieldDataset(train_h5,
        t_start=cfg["data"].get("t_start", 0),
        t_step=cfg["data"].get("t_step", 1),
        max_frames=cfg["data"].get("max_frames_train"),
        augment=cfg["data"].get("augment", False))
    val_ds = PDEFieldDataset(val_h5,
        t_start=cfg["data"].get("t_start", 0),
        t_step=cfg["data"].get("t_step", 1),
        max_frames=cfg["data"].get("max_frames_val"),
        augment=False,
        norm_stats=train_ds.norm_stats)

    bs = cfg["trainer"]["batch_size"]
    nw = cfg["trainer"].get("num_workers", 4)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_ddp else None
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=nw,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, sampler=val_sampler,
                              shuffle=False, num_workers=nw, pin_memory=True)

    # Model
    model_params = cfg["model"].get("params", {})
    model = PDEConvAE(**model_params).to(device)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    n_params = sum(p.numel() for p in model.parameters())
    raw = model.module if is_ddp else model
    _log(f"PDEConvAE  params={n_params/1e6:.1f}M  compression={raw.compression_info}")
    _log(f"Dataset:   train={len(train_ds)}  val={len(val_ds)}")
    _log(f"Out:       {out_dir}")

    # Optimiser
    oc = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(oc["lr"]),
                            weight_decay=float(oc.get("weight_decay", 0.0)),
                            betas=tuple(oc.get("betas", [0.9, 0.999])))
    scaler = GradScaler(enabled=cfg["trainer"]["amp"]["enabled"])

    lc = cfg.get("loss", {})
    lambda_l1   = float(lc.get("lambda_l1",   1.0))
    lambda_grad = float(lc.get("lambda_grad",  0.5))
    lambda_spec = float(lc.get("lambda_spec",  0.0))

    sc = cfg.get("sched", {})
    lr_base   = float(oc["lr"])
    lr_min    = float(sc.get("eta_min", 1e-6))
    warmup_ep = int(sc.get("warmup_epochs", 0))
    total_ep  = int(cfg["trainer"]["epochs"])
    grad_clip = float(cfg["trainer"].get("grad_clip", 1.0))
    ckpt_interval = int(cfg["trainer"].get("checkpoint_interval_epochs", 10))

    # Metrics CSV
    csv_path = out_dir / "metrics.csv"
    if _is_main() and not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("epoch,step,lr,train_total,train_l1,train_grad,val_l1,dt_s\n")

    start_epoch = 0
    global_step = 0

    # Optionally resume
    last_ckpt = out_dir / "checkpoint_last.pt"
    if cfg["trainer"].get("resume", False) and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu")
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        _log(f"[resume] epoch={start_epoch}  step={global_step}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_ep):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # LR
        lr = cosine_lr(epoch, total_ep, lr_base, lr_min, warmup_ep)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        run_total = run_l1 = run_grad = 0.0
        n_steps = 0
        t0 = time.time()

        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["trainer"]["amp"]["enabled"]):
                xr = model(x)
                losses = pde_ae_loss(xr, x, lambda_l1, lambda_grad, lambda_spec)
            scaler.scale(losses["total"]).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            run_total += losses["total"].item()
            run_l1    += losses["l1"].item()
            run_grad  += losses["grad"].item()
            n_steps += 1
            global_step += 1

        # Average across DDP ranks
        def _avg(v):
            t = torch.tensor(v / n_steps, device=device)
            if is_ddp:
                dist.all_reduce(t); t /= world_size
            return t.item()

        train_total = _avg(run_total)
        train_l1    = _avg(run_l1)
        train_grad  = _avg(run_grad)

        # Validation
        model.eval()
        val_l1_acc = 0.0; val_n = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with autocast(enabled=cfg["trainer"]["amp"]["enabled"]):
                    xr = model(x)
                val_l1_acc += F.l1_loss(xr, x).item() * x.shape[0]
                val_n += x.shape[0]
        val_l1_t = torch.tensor(val_l1_acc / max(val_n, 1), device=device)
        if is_ddp:
            dist.all_reduce(val_l1_t); val_l1_t /= world_size
        val_l1 = val_l1_t.item()
        dt = time.time() - t0

        _log(f"epoch={epoch+1:3d}  lr={lr:.2e}  "
             f"train_total={train_total:.5f}  train_l1={train_l1:.5f}  "
             f"train_grad={train_grad:.5f}  val_l1={val_l1:.5f}  dt={dt:.1f}s")

        if _is_main():
            with open(csv_path, "a") as f:
                f.write(f"{epoch+1},{global_step},{lr:.6e},"
                        f"{train_total:.8f},{train_l1:.8f},{train_grad:.8f},"
                        f"{val_l1:.8f},{dt:.3f}\n")

            # Checkpoint
            raw_model = model.module if is_ddp else model
            state = {"model": raw_model.state_dict(), "optim": opt.state_dict(),
                     "epoch": epoch, "step": global_step,
                     "norm_stats": train_ds.norm_stats}
            torch.save(state, out_dir / "checkpoint_last.pt")
            if (epoch + 1) % ckpt_interval == 0:
                torch.save(state, out_dir / f"checkpoint_ep{epoch+1:04d}.pt")

    _log("Training complete.")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
