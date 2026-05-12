#!/usr/bin/env python3
"""DDP training script for diffusion bridge PDE surrogates.

Supports UniDB (Ornstein-Uhlenbeck) and FracBridge (fractional Brownian) bridges.

Input layout (from PFPairDataset with add_thermal=True):
    batch["input"]  — (B, 3, H, W): channels [phi, c, theta_normalized]
    batch["target"] — (B, 2, H, W): channels [phi, c]

The thermal channel is split out of "input" before the model call:
    x_source = batch["input"][:, :2]   # (B, 2, H, W)
    theta    = batch["input"][:, 2:]   # (B, 1, H, W)

Launch (torchrun):
    torchrun --nproc_per_node NUM_GPUS \\
        scripts/train_bridge.py --config path/to/config.yaml
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

# Phase_field_surrogates/ must be on sys.path (set by SLURM launcher via PYTHONPATH,
# or added here as a fallback when running interactively).
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pf_dataloader import PFPairDataset                          # noqa: E402
from diffusion_bridge.sde.unidb_sde import UniDBSDE                    # noqa: E402
from diffusion_bridge.sde.frac_bridge_sde import FracBridgeSDE         # noqa: E402
from diffusion_bridge.models.bridge_wrapper import BridgePDEModel       # noqa: E402

try:
    from utils.wavelet_weight import wavelet_importance_weighted_mse    # noqa: E402
    _WAVELET_AVAILABLE = True
except Exception:
    _WAVELET_AVAILABLE = False


# ---------------------------------------------------------------------------
# SDE / model builders
# ---------------------------------------------------------------------------

def build_sde(cfg: dict, device: torch.device):
    sc = cfg.get("sde", {})
    T = int(sc.get("T", 100))
    bridge_type = str(cfg.get("bridge_type", "unidb")).lower()
    if bridge_type == "unidb":
        return UniDBSDE(
            lambda_sq=float(sc.get("lambda_sq", 0.1)),
            gamma=float(sc.get("gamma", 0.5)),
            T=T,
            schedule=str(sc.get("schedule", "cosine")),
            device=device,
        )
    if bridge_type == "fractal":
        return FracBridgeSDE(
            H=float(sc.get("H", 0.7)),
            sigma_max=float(sc.get("sigma_max", 0.3)),
            T=T,
            device=device,
        )
    raise ValueError(f"Unknown bridge_type: {bridge_type!r}. Expected 'unidb' or 'fractal'.")


def build_model(cfg: dict) -> BridgePDEModel:
    mc = cfg.get("model", {})
    return BridgePDEModel(
        channels=list(mc.get("channels", [192, 320, 512, 640, 832])),
        control_channels=list(mc.get("control_channels", [96, 160, 256, 320, 416])),
        afno_depth=int(mc.get("afno_depth", 12)),
        afno_num_blocks=int(mc.get("afno_num_blocks", 16)),
        afno_mlp_ratio=float(mc.get("afno_mlp_ratio", 12.0)),
        afno_inp_shape=list(mc.get("afno_inp_shape", [32, 32])),
        film_dim=int(mc.get("film_dim", 512)),
        dropout=float(mc.get("dropout", 0.1)),
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
    bs = int(tc.get("batch_size_per_gpu", 6))
    nw = int(dc.get("num_workers", 4))

    train_ds = PFPairDataset(
        h5_path=dc["h5_path"],
        augment=True,
        augment_flip=True,
        augment_roll=False,
        **_DATASET_KWARGS,
    )
    val_ds = PFPairDataset(
        h5_path=dc["val_h5_path"],
        augment=False,
        **_DATASET_KWARGS,
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(nw > 0),
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    """Returns LR multiplier ∈ (0, 1] for a given optimizer step index."""
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
# Validation (no AMP, no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model_ddp: DDP,
    val_loader: DataLoader,
    sde,
    device: torch.device,
) -> float:
    """Compute mean MSE across all validation batches (all-reduced across ranks)."""
    model_ddp.eval()

    total_sq_err = 0.0
    total_n = 0

    for batch in val_loader:
        x_in = batch["input"].to(device, non_blocking=True)      # (B, 3, H, W)
        x_target = batch["target"].to(device, non_blocking=True)  # (B, 2, H, W)
        x_source = x_in[:, :2]   # (B, 2, H, W)
        theta    = x_in[:, 2:]   # (B, 1, H, W)

        B = x_target.shape[0]
        T = sde.T
        t_rand = torch.randint(0, T, (B,), device=device)
        x_t, _ = sde.q_sample(x_target, x_source, t_rand)
        t_norm = t_rand.float() / float(T - 1)

        x0_pred = model_ddp(x_t, x_source, t_norm, theta)
        # Accumulate mean-per-sample MSE (mean over C*H*W, sum over batch)
        mse = F.mse_loss(x0_pred, x_target, reduction="mean").item()
        total_sq_err += mse * B
        total_n += B

    # Reduce across ranks
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
    dist.barrier()  # all ranks wait until run_dir exists

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

    # ── SDE ─────────────────────────────────────────────────────────────────
    sde = build_sde(cfg, device)

    # ── Optimizer + scheduler + scaler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = GradScaler("cuda", enabled=use_fp16)

    # Scheduler constructed before checkpoint load so base_lrs = [lr] (original lr).
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
        # Fast-forward scheduler to match the number of completed optimizer steps.
        # Each call uses self.base_lrs (original lr), not the restored optimizer lr.
        for _ in range(global_step):
            scheduler.step()

    model.train()

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)

        epoch_mse_sum = 0.0
        epoch_n       = 0
        t_start       = time.time()
        T             = sde.T

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            x_in     = batch["input"].to(device, non_blocking=True)      # (B, 3, H, W)
            x_target = batch["target"].to(device, non_blocking=True)      # (B, 2, H, W)
            x_source = x_in[:, :2]   # (B, 2, H, W)
            theta    = x_in[:, 2:]   # (B, 1, H, W)

            B = x_target.shape[0]

            # Forward bridge: corrupt x_target toward x_source
            t_rand = torch.randint(0, T, (B,), device=device)
            x_t, _ = sde.q_sample(x_target, x_source, t_rand)
            t_norm  = t_rand.float() / float(T - 1)   # ∈ [0, 1]

            # Decide whether to perform an optimizer update this iteration
            is_update_step = (
                (batch_idx + 1) % grad_accum == 0
                or batch_idx == len(train_loader) - 1
            )

            with autocast("cuda", enabled=use_fp16):
                x0_pred  = model(x_t, x_source, t_norm, theta)
                loss_mse = F.mse_loss(x0_pred, x_target)

                if lambda_wavelet > 0:
                    # Wavelet loss requires float32; weights are computed without grad
                    loss_wav = wavelet_importance_weighted_mse(
                        x0_pred.float(), x_target.float()
                    )
                    loss = loss_mse + lambda_wavelet * loss_wav
                else:
                    loss = loss_mse

                loss = loss / grad_accum  # scale for accumulation

            scaler.scale(loss).backward()

            # Track plain MSE for logging (no wavelet weighting)
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
            val_mse = validate(model, val_loader, sde, device)
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
    parser = argparse.ArgumentParser(description="DDP diffusion bridge training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE",  1))

    dist.init_process_group("nccl")

    try:
        train(cfg, rank, local_rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
