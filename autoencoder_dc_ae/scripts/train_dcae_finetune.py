#!/usr/bin/env python3
"""Fine-tune DC-AE (dc-ae-f32c32-in-1.0) on 3-channel PDE solidification data.

Input channels:
  0: phi          (phase field)
  1: c            (concentration)
  2: theta        (thermal field)

Losses:
  L1              – pixel-level fidelity
  Gradient (Sobel)– penalises interface blurring for phi channel
  Spectral        – L1 on FFT magnitude spectrum (preserves high-frequency structure)

Usage (single node, single GPU):
  python scripts/train_dcae_finetune.py --config configs/autoencoder/finetune/dc_ae_f32c32_pde_512.yaml

Usage (torchrun, N GPUs):
  torchrun --nproc_per_node=N scripts/train_dcae_finetune.py --config ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

# ── DC-Gen repo path ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
REPO_ROOT = Path(os.environ.get("DC_GEN_REPO_ROOT", PROJECT_ROOT / "external_refs" / "DC-Gen")).expanduser().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _expand_env_placeholders(value):
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PDEFieldDataset(Dataset):
    """Reads (phi, c, theta) frames from a HDF5 file.

    Each call returns a float32 tensor (3, 512, 512) normalised to [-1, 1]
    per channel using the *dataset-wide* min/max computed at init time.
    """

    def __init__(
        self,
        h5_path: str,
        t_start: int = 0,
        t_step: int = 1,
        max_frames: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.h5_path = h5_path
        self.augment = augment

        # Collect all (sim_key, t_index) pairs
        self.index: list[tuple[str, int]] = []
        with h5py.File(h5_path, "r") as f:
            for sim in sorted(f.keys()):
                g = f[sim]
                n_t = g["images"].shape[0]
                for t in range(t_start, n_t, t_step):
                    self.index.append((sim, t))
                    if max_frames is not None and len(self.index) >= max_frames:
                        break
                if max_frames is not None and len(self.index) >= max_frames:
                    break

        # Compute per-channel statistics for normalisation (on a small subset)
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Estimate per-channel min/max over a random subset of ≤200 frames."""
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(self.index), size=min(200, len(self.index)), replace=False)
        mins = np.full(3, np.inf, dtype=np.float32)
        maxs = np.full(3, -np.inf, dtype=np.float32)
        with h5py.File(self.h5_path, "r") as f:
            for i in subset_idx:
                sim, t = self.index[i]
                x3 = self._read_frame(f, sim, t)
                for c in range(3):
                    mins[c] = min(mins[c], float(x3[c].min()))
                    maxs[c] = max(maxs[c], float(x3[c].max()))
        scale = maxs - mins
        scale[scale == 0] = 1.0
        self.norm_min = mins
        self.norm_scale = scale

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
            x3 = self._read_frame(f, sim, t)

        # Normalise to [-1, 1]
        x3 = (x3 - self.norm_min[:, None, None]) / self.norm_scale[:, None, None]
        x3 = x3 * 2.0 - 1.0

        x = torch.from_numpy(x3)  # (3, H, W)

        if self.augment:
            # Random 90° rotations + horizontal flip (physics-appropriate)
            k = int(torch.randint(4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
            if torch.rand(1).item() > 0.5:
                x = x.flip(dims=[2])

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge map for a (B, C, H, W) tensor.  Returns same shape."""
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                      dtype=x.dtype, device=x.device).unsqueeze(0)  # (1,1,3,3)
    ky = kx.transpose(-2, -1)
    B, C, H, W = x.shape
    xf = x.reshape(B * C, 1, H, W)
    xf = F.pad(xf, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xf, kx)
    gy = F.conv2d(xf, ky)
    mag = (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()  # eps avoids grad(sqrt(0))=inf
    return mag.reshape(B, C, H, W)


def _fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """2-D FFT magnitude spectrum (log-scaled), shape (B, C, H, W//2+1)."""
    # Use float32 for FFT to avoid unstable ComplexHalf kernels under AMP.
    x32 = x.float()
    f = torch.fft.rfft2(x32, norm="ortho")
    return (f.abs() + 1e-6).log()


def pde_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_grad: float = 0.5,
    lambda_spec: float = 0.1,
    phi_ch: int = 0,
) -> dict[str, torch.Tensor]:
    """Combined PDE reconstruction loss.

    Args:
        pred:        (B, 3, H, W) — model output in [-1, 1]
        target:      (B, 3, H, W) — ground truth in [-1, 1]
        lambda_grad: weight for Sobel gradient loss
        lambda_spec: weight for spectral (FFT) loss
        phi_ch:      channel index for the phase field (extra gradient weight)

    Returns dict with keys: total, l1, grad, spec
    """
    l1 = F.l1_loss(pred, target)

    # Gradient loss — upweight the phi channel (sharp interface)
    g_pred = _sobel_edges(pred)
    g_tgt = _sobel_edges(target)
    # Extra weight on phi channel to preserve dendritic interface sharpness
    w = torch.ones(pred.shape[1], device=pred.device, dtype=pred.dtype)
    w[phi_ch] = 2.0
    grad_loss = (F.l1_loss(g_pred, g_tgt, reduction="none") * w[None, :, None, None]).mean()

    # Spectral loss (FFT)
    s_pred = _fft_magnitude(pred)
    s_tgt = _fft_magnitude(target)
    spec_loss = F.l1_loss(s_pred, s_tgt)

    total = l1 + lambda_grad * grad_loss + lambda_spec * spec_loss
    return {"total": total, "l1": l1, "grad": grad_loss, "spec": spec_loss}


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _log(msg: str) -> None:
    if _is_main():
        print(msg, flush=True)


def _save_checkpoint(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict,
    out_dir: Path,
    tag: str = "last",
) -> None:
    if not _is_main():
        return
    state = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optim": optim.state_dict(),
    }
    p = out_dir / f"checkpoint.{tag}.pth"
    torch.save(state, p)
    _log(f"  [ckpt] saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    # ── DDP init ─────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    _log(f"[train] rank={local_rank}/{world_size}  device={device}")

    # ── Config ───────────────────────────────────────────────────────────────
    dc = cfg.get("data", {})
    mc = cfg.get("model", {})
    lc = cfg.get("loss", {})
    oc = cfg.get("optim", {})
    sc = cfg.get("sched", {})
    tc = cfg.get("trainer", {})

    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_interval = int(tc.get("checkpoint_interval_epochs", 5))
    epochs = int(tc.get("epochs", 50))
    steps_per_epoch = tc.get("steps_per_epoch", None)
    amp_enabled = tc.get("amp", {}).get("enabled", True)
    grad_clip = float(tc.get("grad_clip", 1.0))
    resume = bool(tc.get("resume", False))

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = PDEFieldDataset(
        dc["train_h5"],
        t_start=dc.get("t_start", 0),
        t_step=dc.get("t_step", 1),
        max_frames=dc.get("max_frames_train", None),
        augment=dc.get("augment", True),
    )
    val_ds = PDEFieldDataset(
        dc["val_h5"],
        t_start=dc.get("t_start", 0),
        t_step=dc.get("t_step", 2),
        max_frames=dc.get("max_frames_val", 200),
        augment=False,
    )
    # Share normalisation stats from train to val
    val_ds.norm_min = train_ds.norm_min
    val_ds.norm_scale = train_ds.norm_scale

    bs = int(tc.get("batch_size", 4))
    nw = int(tc.get("num_workers", 4))
    grad_accum = int(tc.get("grad_accum_steps", 1))  # gradient accumulation

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    total_train = len(train_ds)
    eff_batch = bs * (torch.distributed.get_world_size() if use_ddp else 1) * grad_accum
    _log(f"[data] train={total_train} val={len(val_ds)} frames  batch={bs}  grad_accum={grad_accum}  eff_global_batch={eff_batch}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from dc_gen.ae_model_zoo import DCAE_HF

    model_key = mc.get("pretrained_key", "dc-ae-f32c32-in-1.0")
    model_source = mc.get("pretrained_source") or os.environ.get("MODEL_SOURCE") or f"mit-han-lab/{model_key}"
    _log(f"[model] Loading DCAE_HF from '{model_source}' …")
    model = DCAE_HF.from_pretrained(model_source).to(device)
    _log(f"[model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Optionally freeze encoder (fine-tune decoder only)
    if mc.get("freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        _log("[model] Encoder frozen — fine-tuning decoder only")

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr = float(oc.get("lr", 1e-5))
    wd = float(oc.get("weight_decay", 1e-4))
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd, betas=(0.9, 0.999),
    )

    # ── AMP ───────────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_l1 = float("inf")
    reset_lr_on_resume = bool(tc.get("reset_lr_on_resume", False))
    if resume:
        ckpt_path = out_dir / "checkpoint.last.pth"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            raw = model.module if hasattr(model, "module") else model
            raw.load_state_dict(state["model"])
            optim.load_state_dict(state["optim"])
            start_epoch = state["epoch"] + 1
            global_step = state["step"]
            best_val_l1 = state["metrics"].get("val_l1", float("inf"))
            _log(f"[resume] Resumed from epoch {start_epoch}, step {global_step}")
            if reset_lr_on_resume:
                # Reset optimizer LR to the configured value so the cosine
                # schedule restarts from base_lr rather than the checkpoint's
                # near-zero decayed value.
                for pg in optim.param_groups:
                    pg["lr"] = lr
                _log(f"[sched] LR reset to {lr:.2e} (reset_lr_on_resume=true)")

    # ── Scheduler (cosine, created AFTER resume so base_lrs reflects reset LR) ─
    # T_max: use remaining epochs when reset_lr_on_resume so the schedule
    # reaches eta_min exactly at the end of training.
    remaining = epochs - start_epoch
    if reset_lr_on_resume and remaining > 0:
        t_max = remaining
        _log(f"[sched] CosineAnnealingLR  T_max={t_max}  (remaining epochs, restarts from lr={lr:.2e})")
    else:
        t_max = int(sc.get("T_max", epochs))
        _log(f"[sched] CosineAnnealingLR  T_max={t_max}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr * 0.01)

    # ── Loss weights ─────────────────────────────────────────────────────────
    lambda_grad = float(lc.get("lambda_grad", 0.5))
    lambda_spec = float(lc.get("lambda_spec", 0.1))

    # ── Training ─────────────────────────────────────────────────────────────
    metrics_history: list[dict] = []

    for epoch in range(start_epoch, epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        accum = {"total": 0.0, "l1": 0.0, "grad": 0.0, "spec": 0.0}
        n_steps = 0

        for batch_idx, x in enumerate(train_loader):
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)

            # gradient accumulation: accumulate over grad_accum micro-steps
            accum_step = batch_idx % grad_accum
            is_sync_step = (accum_step == grad_accum - 1) or (batch_idx == len(train_loader) - 1)

            if accum_step == 0:
                optim.zero_grad(set_to_none=True)

            ctx = model.no_sync() if (use_ddp and not is_sync_step) else contextlib.nullcontext()
            with ctx:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    z = model.encode(x) if not hasattr(model, "module") else model.module.encode(x)
                    y = model.decode(z) if not hasattr(model, "module") else model.module.decode(z)
                    losses = pde_reconstruction_loss(y, x, lambda_grad=lambda_grad, lambda_spec=lambda_spec)
                    loss_scaled = losses["total"] / grad_accum

                scaler.scale(loss_scaled).backward()

            if is_sync_step:
                if grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optim)
                scaler.update()

            for k in accum:
                accum[k] += losses[k].item()
            n_steps += 1
            global_step += 1

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        val_l1 = 0.0
        val_n = 0
        if _is_main():
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.eval()
            with torch.no_grad():
                for x_val in val_loader:
                    x_val = x_val.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        z_val = raw_model.encode(x_val)
                        y_val = raw_model.decode(z_val)
                    val_l1 += F.l1_loss(y_val, x_val).item() * x_val.shape[0]
                    val_n += x_val.shape[0]
            val_l1 /= max(val_n, 1)

        # ── Logging ──────────────────────────────────────────────────────────
        dt = time.time() - t0
        m = {
            "epoch": epoch,
            "step": global_step,
            "lr": scheduler.get_last_lr()[0],
            "train_total": accum["total"] / max(n_steps, 1),
            "train_l1": accum["l1"] / max(n_steps, 1),
            "train_grad": accum["grad"] / max(n_steps, 1),
            "train_spec": accum["spec"] / max(n_steps, 1),
            "val_l1": val_l1,
            "dt_s": dt,
        }
        _log(
            f"[epoch {epoch:04d}] "
            f"loss={m['train_total']:.4f}  l1={m['train_l1']:.4f}  "
            f"grad={m['train_grad']:.4f}  spec={m['train_spec']:.4f}  "
            f"val_l1={val_l1:.4f}  lr={m['lr']:.2e}  t={dt:.0f}s"
        )
        metrics_history.append(m)

        if _is_main():
            # Write metrics CSV
            metrics_path = out_dir / "metrics.csv"
            write_header = not metrics_path.exists()
            with metrics_path.open("a") as fcsv:
                if write_header:
                    fcsv.write(",".join(m.keys()) + "\n")
                fcsv.write(",".join(str(v) for v in m.values()) + "\n")

            # Save last checkpoint every epoch
            raw_model = model.module if hasattr(model, "module") else model
            _save_checkpoint(raw_model, optim, epoch, global_step, m, out_dir, "last")

            # Save best checkpoint
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                _save_checkpoint(raw_model, optim, epoch, global_step, m, out_dir, "best")
                _log(f"  [best] val_l1={val_l1:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % ckpt_interval == 0:
                _save_checkpoint(raw_model, optim, epoch, global_step, m, out_dir, f"epoch{epoch:04d}")

    _log("[train] DONE")
    if _is_main():
        (out_dir / "train_metrics.json").write_text(json.dumps(metrics_history, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune DC-AE on PDE solidification data.")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("DC_GEN_REPO_ROOT", str(REPO_ROOT))
    os.environ.setdefault("DATA_ROOT", str(PROJECT_ROOT / "data"))
    os.environ.setdefault("TMP_ROOT", str(PROJECT_ROOT / "tmp"))
    os.environ.setdefault("RUNS_ROOT", str(PROJECT_ROOT / "runs"))

    with open(args.config) as f:
        cfg = _expand_env_placeholders(yaml.safe_load(f))

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
