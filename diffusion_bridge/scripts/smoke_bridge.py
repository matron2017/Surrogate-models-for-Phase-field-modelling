#!/usr/bin/env python3
"""Diffusion bridge smoke test + visualization for PDE solidification surrogates.

Tests both UniDB and FracBridge in a single script:
  1. Forward bridge: corrupt x_target at several timesteps → show noise profile
  2. 20-step reverse ODE: x_source+noise → x_pred
  3. 5-panel plot per channel: [x_source | x_noisy(t=T/2) | x_pred | x_target | |error|]

Usage (single GPU, no DDP):
  python scripts/smoke_bridge.py --bridge unidb    --n_steps 20
  python scripts/smoke_bridge.py --bridge fractal  --n_steps 20

GPU smoke (via slurm, see slurm/bridge_smoke.sh):
  sbatch slurm/bridge_smoke.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent  # Phase_field_surrogates/
sys.path.insert(0, str(ROOT))

from diffusion_bridge.sde.unidb_sde import UniDBSDE
from diffusion_bridge.sde.frac_bridge_sde import FracBridgeSDE
from diffusion_bridge.models.bridge_wrapper import BridgePDEModel

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Tiny H5 loader (no DataLoader — smoke only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_batch(h5_path: str, batch_size: int = 2, device: torch.device = torch.device("cpu")):
    """Load a small batch using same normalization as det pixel training.

    Images: already z-scored at file creation (normalization_schema=zscore) → use as-is.
    Thermal: raw Kelvin → apply z-score using stored thermal_field_channel_mean/std.
    """
    with h5py.File(h5_path, "r") as f:
        sim = list(f.keys())[0]
        g = f[sim]
        imgs  = np.asarray(g["images"][:batch_size + 1, :2], dtype=np.float32)
        therm = np.asarray(g["thermal_field"][:batch_size + 1, :1], dtype=np.float32)

        # Read thermal normalization stats from file attrs
        th_mean = float(np.asarray(f.attrs.get("thermal_field_channel_mean", [0.0]))[0])
        th_std  = float(np.asarray(f.attrs.get("thermal_field_channel_std",  [1.0]))[0])
        th_eps  = float(f.attrs.get("zscore_eps_thermal_field", 1e-6))
        th_already = str(f.attrs.get("thermal_field_norm", "")).lower() == "zscore"
        img_already = str(f.attrs.get("normalization_schema", "")).lower() == "zscore"

    # Images: already z-scored at file creation → use as-is (same as det pixel with normalize_source=file)
    if not img_already:
        # Fallback: manual channel z-score if schema not set
        for c in range(2):
            imgs[:, c] = (imgs[:, c] - imgs[:, c].mean()) / (imgs[:, c].std() + 1e-6)
        print("[warn] images not marked as z-scored in H5, applied local z-score")

    # Thermal: apply z-score using stored stats (thermal_field_norm='none' in H5 → not yet normalized)
    if not th_already:
        therm = (therm - th_mean) / (th_std + th_eps)

    x_src = torch.from_numpy(imgs[:batch_size]).to(device)
    x_tgt = torch.from_numpy(imgs[1:batch_size + 1]).to(device)
    theta  = torch.from_numpy(therm[:batch_size]).to(device)

    print(f"  [data] imgs  mean={x_src.mean():.4f} std={x_src.std():.4f} "
          f"min={x_src.min():.4f} max={x_src.max():.4f}")
    print(f"  [data] theta mean={theta.mean():.4f} std={theta.std():.4f} "
          f"min={theta.min():.4f} max={theta.max():.4f}")
    print(f"  [data] frame-to-frame MAE (src→tgt) = {(x_tgt - x_src).abs().mean():.4f}")
    return x_src, x_tgt, theta


# ─────────────────────────────────────────────────────────────────────────────
# Noise profile test
# ─────────────────────────────────────────────────────────────────────────────

def _noise_profile(sde, x0, mu, label: str):
    """Print sigma at a few timestep fractions and verify bridge endpoints."""
    T = sde.T
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"\n[{label}] Noise profile (data_std={x0.std():.4f}, delta={( x0-mu).abs().mean():.4f}):")
    for fv in fracs:
        t_idx = int(fv * (T - 1))
        t_batch = torch.tensor([t_idx], device=mu.device)
        xt, _  = sde.q_sample(x0[:1], mu[:1], t_batch)
        noise_contribution = (xt - (1e-8 + xt)).std()  # placeholder
        std = (xt - (x0[:1] * (1 - fv) + mu[:1] * fv)).std().item()   # residual noise
        print(f"  t={fv:.2f} (idx={t_idx:3d}): xt_std={xt.std():.4f}  noise_std≈{std:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _plot_panel(x_src, x_noisy, x_pred, x_tgt, out_path: Path, title: str):
    """5-panel plot: source | noisy | pred | target | |error|  (per channel row)."""
    if not HAS_MPL:
        print("[plot] matplotlib not available, skipping plot")
        return

    C = x_src.shape[1]
    ch_names = ["phi", "c"] if C >= 2 else [f"ch{i}" for i in range(C)]

    def _t(x, c):
        return x[0, c].cpu().numpy()

    fig, axes = plt.subplots(C, 5, figsize=(18, 4 * C))
    if C == 1:
        axes = axes[None, :]

    col_titles = ["x_source", "x_noisy (t=T/2)", "x_pred (20-step)", "x_target", "|error|"]
    panels = [x_src, x_noisy, x_pred, x_tgt,
              (x_pred - x_tgt).abs()]

    for row, ch in enumerate(range(C)):
        for col, (panel, ctitle) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            img = _t(panel, ch)
            vmin, vmax = img.min(), img.max()
            if col == 4:
                im = ax.imshow(img, cmap="hot", vmin=0, vmax=max(img.max(), 0.1))
            else:
                lim = max(abs(vmin), abs(vmax), 0.1)
                im = ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(ctitle, fontsize=9)
            ax.set_ylabel(ch_names[ch], fontsize=9)
            ax.axis("off")

    mae = (x_pred - x_tgt).abs().mean().item()
    src_mae = (x_src - x_tgt).abs().mean().item()
    fig.suptitle(f"{title}  |  pred MAE={mae:.4f}  src_baseline={src_mae:.4f}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke run
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(
    bridge_type: str,
    n_steps: int,
    h5_path: str,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 2,
):
    print(f"\n{'='*60}")
    print(f" Bridge smoke: {bridge_type.upper()}  n_steps={n_steps}  device={device}")
    print(f"{'='*60}")

    x_src, x_tgt, theta = _load_batch(h5_path, batch_size=batch_size, device=device)
    print(f"Data: x_src={tuple(x_src.shape)}  x_tgt={tuple(x_tgt.shape)}  theta={tuple(theta.shape)}")

    # ── Build SDE ────────────────────────────────────────────────────────
    T = 100
    if bridge_type == "unidb":
        sde = UniDBSDE(lambda_sq=0.1, gamma=0.5, T=T, schedule="cosine", device=device)
        sde_label = "UniDB (OU bridge, λ²=0.1, γ=0.5, cosine)"
    else:
        sde = FracBridgeSDE(H=0.7, sigma_max=0.3, T=T, device=device)
        sde_label = "FracBridge (H=0.7, σ_max=0.3)"

    # ── Build model (300M+ backbone — same as det pixel training) ────────
    model = BridgePDEModel(
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

    model.eval()

    # ── Forward noise profile ─────────────────────────────────────────────
    with torch.no_grad():
        if bridge_type == "unidb":
            _noise_profile(sde, x_tgt, x_src, sde_label)
        else:
            _noise_profile(sde, x_tgt, x_src, sde_label)

    # ── Sample x_noisy at t = T/2 for visualization ───────────────────────
    with torch.no_grad():
        t_mid = torch.full((batch_size,), T // 2, device=device, dtype=torch.long)
        if bridge_type == "unidb":
            x_noisy, _ = sde.q_sample(x_tgt, x_src, t_mid)
        else:
            x_noisy, _ = sde.q_sample(x_tgt, x_src, t_mid)

    # ── Reverse (inference) ───────────────────────────────────────────────
    print(f"\n[{bridge_type}] Running {n_steps}-step reverse ODE...")
    t0 = time.time()

    with torch.no_grad():
        def model_fn(x_in, t_norm):
            t_diff = torch.full((x_in.shape[0],), t_norm, device=device)
            x_n, x_s = x_in[:, :2], x_in[:, 2:]
            return model(x_n, x_s, t_diff, theta)

        if bridge_type == "unidb":
            x_pred = sde.sample(model_fn, mu=x_src, n_steps=n_steps, mode="ode")
        else:
            x_pred = sde.sample(model_fn, x_T=x_src, n_steps=n_steps, mode="ode")

    dt = time.time() - t0
    mae  = (x_pred - x_tgt).abs().mean().item()
    smae = (x_src  - x_tgt).abs().mean().item()
    print(f"  Done in {dt:.2f}s  |  pred MAE={mae:.4f}  src_baseline={smae:.4f}")

    # ── Memory ────────────────────────────────────────────────────────────
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU memory: {mem:.2f} GB")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_path = out_dir / f"smoke_{bridge_type}_n{n_steps}.png"
    _plot_panel(x_src, x_noisy, x_pred, x_tgt, plot_path,
                title=f"{sde_label}  ({n_steps} steps)")

    print(f"\n[{bridge_type}] SMOKE PASS ✓")
    return {"bridge": bridge_type, "mae": mae, "src_baseline": smae, "dt_s": dt}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", choices=["unidb", "fractal", "both"],
                        default="both", help="Which bridge to test")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Number of reverse denoising steps")
    parser.add_argument("--h5", type=str,
                        default=str(ROOT / "autoencoder_dc_ae/data/train.h5"))
    parser.add_argument("--out_dir", type=str,
                        default=str(ROOT / "diffusion_bridge/runs/smoke"))
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bridges = ["unidb", "fractal"] if args.bridge == "both" else [args.bridge]
    results = []
    for b in bridges:
        res = run_smoke(b, args.n_steps, args.h5, out_dir, device, args.batch_size)
        results.append(res)

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['bridge']:8s}  MAE={r['mae']:.4f}  baseline={r['src_baseline']:.4f}  t={r['dt_s']:.1f}s")
    print("\nOutput plots:", out_dir)


if __name__ == "__main__":
    main()
