#!/usr/bin/env python3
"""Flow Matching smoke test + visualization for PDE solidification surrogates.

Tests the FM model (UNetFiLMAttn backbone, same as bridges) end-to-end:
  1. Data load + normalization check
  2. One forward pass at several interpolation fractions t
  3. Euler ODE inference (x_source → x_pred, n_steps=20)
  4. 5-panel plot: [x_source | x_interp(t=0.5) | x_pred | x_target | |error|]

Usage (single GPU, no DDP):
  python scripts/smoke_fm.py

GPU smoke (via SLURM):
  sbatch slurm/fm_smoke.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent  # Phase_field_surrogates/
sys.path.insert(0, str(ROOT))

from flow_matching.models.fm_wrapper import FMPDEModel, euler_sample   # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loader (same logic as bridge smoke test)
# ---------------------------------------------------------------------------

def _load_batch(h5_path: str, batch_size: int = 2, device: torch.device = torch.device("cpu")):
    with h5py.File(h5_path, "r") as f:
        sim = list(f.keys())[0]
        g = f[sim]
        imgs  = np.asarray(g["images"][:batch_size + 1, :2], dtype=np.float32)
        therm = np.asarray(g["thermal_field"][:batch_size + 1, :1], dtype=np.float32)

        th_mean = float(np.asarray(f.attrs.get("thermal_field_channel_mean", [0.0]))[0])
        th_std  = float(np.asarray(f.attrs.get("thermal_field_channel_std",  [1.0]))[0])
        th_eps  = float(f.attrs.get("zscore_eps_thermal_field", 1e-6))
        th_already  = str(f.attrs.get("thermal_field_norm", "")).lower() == "zscore"
        img_already = str(f.attrs.get("normalization_schema", "")).lower() == "zscore"

    if not img_already:
        for c in range(2):
            imgs[:, c] = (imgs[:, c] - imgs[:, c].mean()) / (imgs[:, c].std() + 1e-6)
        print("[warn] images not marked as z-scored in H5, applied local z-score")

    if not th_already:
        therm = (therm - th_mean) / (th_std + th_eps)

    x_src = torch.from_numpy(imgs[:batch_size]).to(device)
    x_tgt = torch.from_numpy(imgs[1:batch_size + 1]).to(device)
    theta  = torch.from_numpy(therm[:batch_size]).to(device)

    print(f"  [data] x_src  mean={x_src.mean():.4f}  std={x_src.std():.4f}  "
          f"min={x_src.min():.4f}  max={x_src.max():.4f}")
    print(f"  [data] theta  mean={theta.mean():.4f}  std={theta.std():.4f}  "
          f"min={theta.min():.4f}  max={theta.max():.4f}")
    print(f"  [data] frame-to-frame MAE (src→tgt) = {(x_tgt - x_src).abs().mean():.4f}")
    return x_src, x_tgt, theta


# ---------------------------------------------------------------------------
# Interpolation profile
# ---------------------------------------------------------------------------

def _interp_profile(x_src: torch.Tensor, x_tgt: torch.Tensor):
    """Print stats of linear interpolation at several t values."""
    print("\n[FM] Interpolation profile (linear, no noise):")
    for t_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        x_t = (1.0 - t_val) * x_src + t_val * x_tgt
        dist_from_src = (x_t - x_src).abs().mean().item()
        dist_from_tgt = (x_t - x_tgt).abs().mean().item()
        print(f"  t={t_val:.2f}  xt_std={x_t.std():.4f}  "
              f"d(src)={dist_from_src:.4f}  d(tgt)={dist_from_tgt:.4f}")


# ---------------------------------------------------------------------------
# 5-panel plot
# ---------------------------------------------------------------------------

def _plot_panel(x_src, x_interp, x_pred, x_tgt, out_path: Path, title: str):
    if not HAS_MPL:
        print("[plot] matplotlib not available, skipping")
        return

    C = x_src.shape[1]
    ch_names = ["phi", "c"]

    def _t(x, c):
        return x[0, c].cpu().numpy()

    fig, axes = plt.subplots(C, 5, figsize=(18, 4 * C))
    if C == 1:
        axes = axes[None, :]

    col_titles = ["x_source", "x_interp (t=0.5)", "x_pred (Euler ODE)", "x_target", "|error|"]
    panels = [x_src, x_interp, x_pred, x_tgt, (x_pred - x_tgt).abs()]

    for row in range(C):
        for col, (panel, ctitle) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            img = _t(panel, row)
            if col == 4:
                im = ax.imshow(img, cmap="hot", vmin=0, vmax=max(img.max(), 0.1))
            else:
                lim = max(abs(img.min()), abs(img.max()), 0.1)
                im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(ctitle, fontsize=9)
            ax.set_ylabel(ch_names[row], fontsize=9)
            ax.axis("off")

    mae  = (x_pred - x_tgt).abs().mean().item()
    smae = (x_src  - x_tgt).abs().mean().item()
    fig.suptitle(f"{title}  |  pred MAE={mae:.4f}  src_baseline={smae:.4f}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ---------------------------------------------------------------------------
# Single forward pass check at several t values
# ---------------------------------------------------------------------------

def _forward_profile(model: FMPDEModel, x_src, x_tgt, theta, device):
    """Check model output shape, finite values, and loss at a few t values."""
    print("\n[FM] Forward pass profile (no gradient):")
    model.eval()
    with torch.no_grad():
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            B = x_src.shape[0]
            t_w = torch.full((B, 1, 1, 1), t_val, device=device)
            x_t = (1.0 - t_w) * x_src + t_w * x_tgt
            t_norm = torch.full((B,), t_val, device=device)
            x1_pred = model(x_t, x_src, t_norm, theta)
            mse = F.mse_loss(x1_pred, x_tgt).item()
            finite = torch.isfinite(x1_pred).all().item()
            print(f"  t={t_val:.2f}  out_shape={tuple(x1_pred.shape)}  "
                  f"MSE={mse:.4f}  finite={finite}")
    model.train()


# ---------------------------------------------------------------------------
# Main smoke run
# ---------------------------------------------------------------------------

def run_smoke(
    n_steps: int,
    h5_path: str,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 2,
):
    print(f"\n{'='*60}")
    print(f" Flow Matching smoke  n_steps={n_steps}  device={device}")
    print(f"{'='*60}")

    x_src, x_tgt, theta = _load_batch(h5_path, batch_size=batch_size, device=device)
    print(f"Data: x_src={tuple(x_src.shape)}  x_tgt={tuple(x_tgt.shape)}  theta={tuple(theta.shape)}")

    # ── Build model (same 300M+ backbone as bridges) ─────────────────────
    model = FMPDEModel(
        channels=[192, 320, 512, 640, 832],
        control_channels=[96, 160, 256, 320, 416],
        afno_depth=12,
        afno_num_blocks=16,
        afno_mlp_ratio=12.0,
        afno_inp_shape=[32, 32],
        film_dim=512,
        dropout=0.0,
    ).to(device)
    print(f"Model: {model.n_params/1e6:.1f}M params")

    # ── Interpolation profile ─────────────────────────────────────────────
    _interp_profile(x_src, x_tgt)

    # ── Forward pass at several t values ─────────────────────────────────
    _forward_profile(model, x_src, x_tgt, theta, device)

    # ── Build x_interp at t=0.5 for visualization ─────────────────────────
    with torch.no_grad():
        x_interp = 0.5 * x_src + 0.5 * x_tgt

    # ── Euler ODE inference ───────────────────────────────────────────────
    print(f"\n[FM] Running {n_steps}-step Euler ODE (random weights — checks shape/memory) ...")
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        x_pred = euler_sample(model, x_src, theta, n_steps=n_steps, device=device)
    dt = time.time() - t0

    mae  = (x_pred - x_tgt).abs().mean().item()
    smae = (x_src  - x_tgt).abs().mean().item()
    print(f"  Done in {dt:.2f}s  |  pred MAE={mae:.4f}  src_baseline={smae:.4f}")
    print(f"  (random weights → MAE meaningless; checks for shape/OOM/finite outputs)")

    # ── Memory ────────────────────────────────────────────────────────────
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU memory: {mem:.2f} GB")

    # ── Single training step (forward + backward) ─────────────────────────
    print("\n[FM] Training step (forward + backward, batch_size=2) ...")
    model.train()
    B = x_src.shape[0]
    t_norm = torch.rand(B, device=device)
    t_w    = t_norm.view(B, 1, 1, 1)
    x_t    = (1.0 - t_w) * x_src + t_w * x_tgt
    x1_pred = model(x_t, x_src, t_norm, theta)
    loss = F.mse_loss(x1_pred, x_tgt)
    loss.backward()
    print(f"  Training loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}")
    if device.type == "cuda":
        mem_bwd = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU memory (after backward): {mem_bwd:.2f} GB")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_path = out_dir / f"smoke_fm_n{n_steps}.png"
    _plot_panel(x_src, x_interp, x_pred, x_tgt, plot_path,
                title=f"Flow Matching (random weights, {n_steps} Euler steps)")

    print(f"\n[FM] SMOKE PASS ✓")
    return {"mae": mae, "src_baseline": smae, "dt_s": dt}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps",    type=int, default=20)
    parser.add_argument("--h5",         type=str,
                        default=str(ROOT / "autoencoder_dc_ae/data/train.h5"))
    parser.add_argument("--out_dir",    type=str,
                        default=str(ROOT / "flow_matching/runs/smoke"))
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_smoke(args.n_steps, args.h5, out_dir, device, args.batch_size)

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"  pred MAE={res['mae']:.4f}  src_baseline={res['src_baseline']:.4f}  "
          f"t={res['dt_s']:.1f}s")
    print("\nOutput plots:", out_dir)


if __name__ == "__main__":
    main()
