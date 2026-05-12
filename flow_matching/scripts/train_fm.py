#!/usr/bin/env python3
"""DDP training script for Rectified Flow / Flow Matching PDE surrogates.

Baseline comparison for diffusion bridge models. Uses the SAME backbone
(UNetFiLMAttn with ControlNet-XS branch) and data — only the interpolation
scheme and inference procedure differ:

  Bridge   : stochastic OU/fBm path, q(x_t|x0,x1) SDE, reverse SDE at inference
  Flow Matching: deterministic linear path, x_t = (1-t)*x_src + t*x_tgt,
                 Euler ODE at inference

Training signal (x1-parameterization, same loss as bridge x0-param):
  1. Sample t ~ U(0, 1)  (continuous, no discrete T needed)
  2. x_t = (1-t)*x_source + t*x_target          [linear interpolation, no noise]
  3. model(cat(x_t, x_source), t, theta) → x1_pred
  4. loss = MSE(x1_pred, x_target) + λ_wav * WaveletMSE(x1_pred, x_target)

Input layout (from PFPairDataset with add_thermal=True):
    batch["input"]  — (B, 3, H, W): channels [phi, c, theta_normalized]
    batch["target"] — (B, 2, H, W): channels [phi, c]

Launch (torchrun):
    torchrun --nproc_per_node NUM_GPUS \\
        scripts/train_fm.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pf_dataloader import PFPairDataset                              # noqa: E402
from flow_matching.models.fm_wrapper import FMPDEModel                    # noqa: E402

try:
    from utils.wavelet_weight import wavelet_importance_weighted_mse       # noqa: E402
    _WAVELET_AVAILABLE = True
except Exception:
    _WAVELET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> FMPDEModel:
    mc = cfg.get("model", {})
    return FMPDEModel(
        channels=list(mc.get("channels", [192, 320, 512, 640, 832])),
        control_channels=list(mc.get("control_channels", [96, 160, 256, 320, 416])),
        afno_depth=int(mc.get("afno_depth", 12)),
        afno_num_blocks=int(mc.get("afno_num_blocks", 16)),
        afno_mlp_ratio=float(mc.get("afno_mlp_ratio", 12.0)),
        afno_inp_shape=list(mc.get("afno_inp_shape", [32, 32])),
        film_dim=int(mc.get("film_dim", 512)),
        dropout=float(mc.get("dropout", 0.0)),
    )


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

_DATASET_KWARGS = dict(
    input_channels=[0, 1],
    target_channels=[0, 1],
    add_thermal=True,
    thermal_axis="x",
    thermal_use_x0=True,
    thermal_require_precomputed=True,
    normalize_images=True,
    normalize_source="file",
    normalize_thermal=True,
    thermal_norm_source="file",
    use_pairs_idx=True,
)


def build_loaders(
    cfg: dict, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    dc = cfg.get("data", {})
    tc = cfg.get("training", {})
    bs = int(tc.get("batch_size_per_gpu", 1))
    nw = int(dc.get("num_workers", 8))

    train_ds = PFPairDataset(dc["h5_path"],     **_DATASET_KWARGS)
    val_ds   = PFPairDataset(dc["val_h5_path"], **_DATASET_KWARGS)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, sampler=val_sampler,
        num_workers=nw, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    val_mse: float,
) -> None:
    torch.save(
        {
            "model": model_ddp.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_mse": val_mse,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)
    model_ddp.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]), int(ckpt["global_step"]), float(ckpt.get("val_mse", float("inf")))


# ---------------------------------------------------------------------------
# Validation — random t, MSE (same metric as bridge for comparability)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model_ddp: DDP,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean MSE across all validation batches (all-reduced across ranks).

    Uses random t ~ U(0,1) per sample, same as training — measures how well
    the model predicts x1 from x_t. This is directly comparable to bridge validation.
    """
    model_ddp.eval()

    total_sq_err = 0.0
    total_n = 0

    for batch in val_loader:
        x_in     = batch["input"].to(device, non_blocking=True)      # (B, 3, H, W)
        x_target = batch["target"].to(device, non_blocking=True)     # (B, 2, H, W)
        x_source = x_in[:, :2]
        theta    = x_in[:, 2:]

        B = x_target.shape[0]
        t_norm = torch.rand(B, device=device)
        t_w    = t_norm.view(B, 1, 1, 1)
        x_t    = (1.0 - t_w) * x_source + t_w * x_target

        x1_pred = model_ddp(x_t, x_source, t_norm, theta)
        mse = F.mse_loss(x1_pred, x_target, reduction="mean").item()
        total_sq_err += mse * B
        total_n += B

    sq_t = torch.tensor(total_sq_err, device=device)
    n_t  = torch.tensor(float(total_n), device=device)
    dist.all_reduce(sq_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_t,  op=dist.ReduceOp.SUM)

    model_ddp.train()
    return (sq_t / n_t.clamp(min=1.0)).item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, rank: int, local_rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    run_dir = Path(cfg["run_dir"])
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tc = cfg.get("training", {})
    epochs         = int(tc.get("epochs",          300))
    grad_accum     = int(tc.get("grad_accum",       1))
    lr             = float(tc.get("lr",            1e-4))
    weight_decay   = float(tc.get("weight_decay",  1e-5))
    warmup_frac    = float(tc.get("warmup_frac",   0.05))
    lambda_wavelet = float(tc.get("lambda_wavelet", 0.1))
    val_every      = int(tc.get("val_every",        5))
    ckpt_every     = int(tc.get("ckpt_every",      10))
    use_fp16       = str(tc.get("dtype", "fp16")).lower() == "fp16"

    if lambda_wavelet > 0 and not _WAVELET_AVAILABLE:
        if rank == 0:
            print("[warn] pytorch_wavelets not available — wavelet loss disabled")
        lambda_wavelet = 0.0

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)

    # ── LR schedule params ──────────────────────────────────────────────────
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps     = max(1, steps_per_epoch * epochs)
    warmup_steps    = max(1, int(total_steps * warmup_frac))

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[rank 0] Parameters: {n_params / 1e6:.2f}M  |  "
            f"steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}"
        )

    # ── Optimizer + scheduler + scaler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = GradScaler("cuda", enabled=use_fp16)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, total_steps, warmup_steps),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch   = 0
    global_step   = 0
    best_val_mse  = float("inf")
    ckpt_last     = run_dir / "checkpoint.last.pth"

    if ckpt_last.exists():
        last_epoch, global_step, best_val_mse = load_checkpoint(
            ckpt_last, model, optimizer, scaler, device
        )
        start_epoch = last_epoch + 1
        if rank == 0:
            print(
                f"[resume] Loaded {ckpt_last}  "
                f"epoch={last_epoch}  step={global_step}  best_val_mse={best_val_mse:.6f}"
            )
        for _ in range(global_step):
            scheduler.step()

    model.train()

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)

        epoch_mse_sum = 0.0
        epoch_n       = 0
        t_start       = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            x_in     = batch["input"].to(device, non_blocking=True)       # (B, 3, H, W)
            x_target = batch["target"].to(device, non_blocking=True)       # (B, 2, H, W)
            x_source = x_in[:, :2]
            theta    = x_in[:, 2:]

            B = x_target.shape[0]

            # Flow matching: linear interpolation between source and target
            t_norm = torch.rand(B, device=device)                          # U(0,1) per sample
            t_w    = t_norm.view(B, 1, 1, 1)
            x_t    = (1.0 - t_w) * x_source + t_w * x_target             # (B,2,H,W)

            is_update_step = (
                (batch_idx + 1) % grad_accum == 0
                or batch_idx == len(train_loader) - 1
            )

            with autocast("cuda", enabled=use_fp16):
                x1_pred  = model(x_t, x_source, t_norm, theta)
                loss_mse = F.mse_loss(x1_pred, x_target)

                if lambda_wavelet > 0:
                    loss_wav = wavelet_importance_weighted_mse(
                        x1_pred.float(), x_target.float()
                    )
                    loss = loss_mse + lambda_wavelet * loss_wav
                else:
                    loss = loss_mse

                loss = loss / grad_accum

            scaler.scale(loss).backward()

            epoch_mse_sum += loss_mse.detach().item() * B
            epoch_n       += B

            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        # ── Aggregate train MSE across ranks ────────────────────────────────
        mse_t = torch.tensor(epoch_mse_sum, device=device)
        n_t   = torch.tensor(float(epoch_n), device=device)
        dist.all_reduce(mse_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_t,   op=dist.ReduceOp.SUM)
        avg_train_mse = (mse_t / n_t.clamp(min=1.0)).item()

        dt         = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Validation ──────────────────────────────────────────────────────
        val_mse = float("nan")
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            val_mse = validate(model, val_loader, device)
            if rank == 0 and val_mse < best_val_mse:
                best_val_mse = val_mse
                save_checkpoint(
                    run_dir / "checkpoint.best.pth",
                    model, optimizer, scaler, epoch, global_step, best_val_mse,
                )

        # ── Logging ─────────────────────────────────────────────────────────
        if rank == 0:
            val_str = f"{val_mse:.6f}" if not math.isnan(val_mse) else "        ----"
            print(
                f"[epoch {epoch + 1:4d}/{epochs}]  "
                f"train_mse={avg_train_mse:.6f}  "
                f"val_mse={val_str}  "
                f"lr={current_lr:.3e}  "
                f"dt={dt:.1f}s"
            )

        # ── Checkpoint (last) ────────────────────────────────────────────────
        if rank == 0 and ((epoch + 1) % ckpt_every == 0 or epoch == epochs - 1):
            save_checkpoint(
                ckpt_last,
                model, optimizer, scaler, epoch, global_step, val_mse,
            )

    dist.barrier()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDP flow matching training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))

    # device_id required on Puhti to avoid NCCL ALLREDUCE hangs across nodes
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    try:
        train(cfg, rank, local_rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
