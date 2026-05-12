#!/usr/bin/env python3
"""
DC-AE comparison: pretrained (no fine-tune) vs fine-tuned (best checkpoint).
Shows 2 data points — first and last frame of test set.
Each row: [GT φ | GT c×3 | GT θ] | [Pretrained φ | c×3 | θ] | [Fine-tuned φ | c×3 | θ]

Physical unit display:
  φ  → clipped to [-1, 1]          (RdBu_r: blue=solid +1, red=liquid -1)
  c  → clipped to [0, max] × 3     (dark-blue/teal/green/violet sequential)
  θ  → raw Kelvin from h5          (plasma)
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DCAE_ROOT  = ROOT / "autoencoder_dc_ae"
DC_GEN     = DCAE_ROOT / "external_refs/DC-Gen"
TEST_H5    = DCAE_ROOT / "data/test.h5"
CKPT_BEST  = DCAE_ROOT / "runs/autoencoder/finetune/dc_ae_f32c32_lr2e5/checkpoint.best.pth"
OUT_PATH   = ROOT / "eval/plots/dcae_pretrained_vs_finetuned.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if str(DC_GEN) not in sys.path:
    sys.path.insert(0, str(DC_GEN))

# ── Colormaps ─────────────────────────────────────────────────────────────────
# Phase field: RdBu_r — matches eval_row_det_pixel.png exactly (blue=solid, red=liquid)
CMAP_PHI   = "RdBu_r"
# Concentration ×3: dark blue → teal → green → violet (sequential, values ≥ 0)
CMAP_CONC  = mcolors.LinearSegmentedColormap.from_list(
    "conc_bgtv",
    ["#0D3B8E", "#1565C0", "#00838F", "#2E7D32", "#558B2F", "#6A1B9A", "#4A148C"])
# Thermal: plasma (works well for narrow Kelvin ranges)
CMAP_THERM = "plasma"


def _read_frame(h5_path: Path, sim: str, t: int) -> np.ndarray:
    """Return (3, H, W) float32 in physical units: [phi, c, theta_K]."""
    with h5py.File(h5_path, "r") as f:
        g = f[sim]
        x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
        th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
    return np.concatenate([x2, th], axis=0)


def _compute_norm_stats(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (min, scale) from a random subset of all frames in h5."""
    index = []
    with h5py.File(h5_path, "r") as f:
        for sim in sorted(f.keys()):
            n_t = f[sim]["images"].shape[0]
            for t in range(n_t):
                index.append((sim, t))
    rng = np.random.default_rng(42)
    subset = rng.choice(len(index), size=min(400, len(index)), replace=False)
    mins = np.full(3, np.inf, dtype=np.float32)
    maxs = np.full(3, -np.inf, dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        for i in subset:
            sim, t_idx = index[i]
            x3 = _read_frame(h5_path, sim, t_idx)
            for c in range(3):
                mins[c] = min(mins[c], x3[c].min())
                maxs[c] = max(maxs[c], x3[c].max())
    scale = maxs - mins
    scale[scale == 0] = 1.0
    return mins, scale


def to_norm(x_phys: np.ndarray, norm_min, norm_scale) -> torch.Tensor:
    """(3,H,W) physical → (1,3,H,W) normalised [-1,1] tensor."""
    x = (x_phys - norm_min[:, None, None]) / norm_scale[:, None, None]
    x = x * 2.0 - 1.0
    return torch.from_numpy(x[None]).float()


def from_norm(t: torch.Tensor, norm_min, norm_scale) -> np.ndarray:
    """(1,3,H,W) normalised [-1,1] → (3,H,W) raw h5 values (phi, c_raw, theta_K)."""
    x = t.squeeze(0).cpu().float().numpy()
    return (x * 0.5 + 0.5) * norm_scale[:, None, None] + norm_min[:, None, None]


def to_display(raw3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (3,H,W) raw h5 values to display-ready physical units.

    φ  (ch 0): raw range ~[-1, 2.7] → clip to [-1, 1]
    c  (ch 1): raw range ~[-1.6, 5.9] → clip to [0, +inf] then ×3
    θ  (ch 2): raw Kelvin, pass through unchanged
    """
    phi  = np.clip(raw3[0], -1.0, 1.0)
    c    = np.clip(raw3[1],  0.0, None) * 3.0
    th   = raw3[2].copy()
    return phi, c, th


@torch.no_grad()
def run_model(model, x_norm_t: torch.Tensor, device) -> torch.Tensor:
    x = x_norm_t.to(device)
    z = model.encode(x)
    y = model.decode(z)
    return y.cpu()


def load_pretrained(device) -> torch.nn.Module:
    from dc_gen.ae_model_zoo import DCAE_HF
    model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f32c32-in-1.0").to(device)
    model.eval()
    print("[pretrained] Loaded original mit-han-lab/dc-ae-f32c32-in-1.0 (no PDE fine-tune)")
    return model


def load_finetuned(device) -> torch.nn.Module:
    from dc_gen.ae_model_zoo import DCAE_HF
    model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f32c32-in-1.0").to(device)
    ck = torch.load(CKPT_BEST, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {re.sub(r'\bcontext_main\b', 'context_module.main',
                 re.sub(r'\blocal_main\b', 'local_module.main', k)): v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    ep = ck.get("epoch", "?")
    metrics = ck.get("metrics", {})
    val_l1 = metrics.get("val_l1", "?") if isinstance(metrics, dict) else "?"
    print(f"[finetuned]  Loaded lr2e5 best checkpoint  epoch={ep}  val_l1={val_l1}")
    return model


def _stats_str(arr: np.ndarray) -> str:
    return f"μ={arr.mean():.3g}  [{arr.min():.3g}, {arr.max():.3g}]"


def _panel(ax, img: np.ndarray, title: str, cmap, vmin, vmax,
           cbar_label: str = "", stats: bool = True):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=8.5, pad=3, fontweight="bold")
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="vertical")
    cb.ax.tick_params(labelsize=6.5)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6.5)
    if stats:
        s = _stats_str(img)
        ax.set_xlabel(s, fontsize=6, labelpad=2)
        ax.xaxis.set_visible(True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ax.set_xticks([])


def make_comparison_plot(samples: list[dict], norm_min, norm_scale,
                         model_pretrained, model_finetuned, device, out_path: Path):
    """
    Layout: 2 rows × 9 columns
      cols 0-2: GT  (φ [-1,1] | c×3 [≥0] | θ [K])
      cols 3-5: Pretrained recon
      cols 6-8: Fine-tuned recon
    All channels in physical/display units:
      φ   clipped to [-1, 1],  RdBu_r,  fixed vmin=-1 vmax=+1
      c×3 clipped to [0, ...], CMAP_CONC, vmin=0, vmax=shared across models per row
      θ   raw Kelvin,           plasma,    vmin/vmax=shared 2–98 percentile per row
    """
    n_rows = len(samples)
    n_cols = 9
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.2 * n_cols, 4.8 * n_rows),
                              gridspec_kw={"wspace": 0.38, "hspace": 0.60})
    if n_rows == 1:
        axes = axes[None, :]

    ch_labels = ["φ (phase field)",  "c×3 (concentration)", "θ (thermal, K)"]
    ch_units  = ["[-1, 1]",          "[×3 units]",          "K"]
    ch_cmaps  = [CMAP_PHI, CMAP_CONC, CMAP_THERM]
    group_titles = ["Ground Truth",
                    "Pretrained  (no fine-tune)",
                    "Fine-tuned  (lr=2e-5, best ckpt)"]
    group_colors = ["#2c3e50", "#922b21", "#154360"]

    for row_idx, sample in enumerate(samples):
        raw_gt  = sample["phys"]   # (3, H, W) — raw h5 values
        x_norm  = to_norm(raw_gt, norm_min, norm_scale).to(device)

        with torch.no_grad():
            y_pre = model_pretrained.decode(model_pretrained.encode(x_norm)).cpu()
            y_ft  = model_finetuned.decode( model_finetuned.encode( x_norm)).cpu()

        raw_pre = from_norm(y_pre, norm_min, norm_scale)
        raw_ft  = from_norm(y_ft,  norm_min, norm_scale)

        # Convert each to display-ready physical units
        disp = []
        for raw in [raw_gt, raw_pre, raw_ft]:
            phi, c3, th = to_display(raw)
            disp.append((phi, c3, th))

        # Shared colour limits per channel (across GT + both models)
        # φ: always fixed [-1, 1]
        phi_vmin, phi_vmax = -1.0, 1.0
        # c×3: [0, max across all three]
        c_vmax = max(float(d[1].max()) for d in disp)
        c_vmin = 0.0
        # θ: 2–98th percentile across all three
        all_th = np.concatenate([d[2].ravel() for d in disp])
        th_vmin = float(np.percentile(all_th, 2))
        th_vmax = float(np.percentile(all_th, 98))

        vmins = [phi_vmin, c_vmin, th_vmin]
        vmaxs = [phi_vmax, c_vmax, th_vmax]

        for gi, ((phi, c3, th), gcol, gtitle) in enumerate(zip(disp, group_colors, group_titles)):
            imgs = [phi, c3, th]
            for ch in range(3):
                col = gi * 3 + ch
                ax  = axes[row_idx, col]
                _panel(ax, imgs[ch],
                       f"{gtitle}\n{ch_labels[ch]}",
                       ch_cmaps[ch], vmins[ch], vmaxs[ch],
                       cbar_label=ch_units[ch], stats=True)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(gcol)
                    spine.set_linewidth(2.2)

        # Row annotation
        axes[row_idx, 0].set_ylabel(sample["label"], fontsize=9,
                                    fontweight="bold", labelpad=6)
        axes[row_idx, 0].yaxis.set_visible(True)

    fig.suptitle(
        "DC-AE Reconstruction — Pretrained vs Fine-tuned  |  Test set: first & last frame",
        fontsize=13, fontweight="bold", y=1.02,
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group_colors[0], label="Ground Truth"),
        Patch(facecolor=group_colors[1],
              label="Pretrained  (mit-han-lab/dc-ae-f32c32-in-1.0, ImageNet only)"),
        Patch(facecolor=group_colors[2],
              label="Fine-tuned  (lr=2e-5 cosine 1000 ep, best val_l1 checkpoint)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, 1.005))

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[done] Saved → {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Norm stats from test h5 ───────────────────────────────────────────────
    print("[norm] Computing stats from test.h5 …")
    norm_min, norm_scale = _compute_norm_stats(TEST_H5)
    print(f"  phi:   min={norm_min[0]:.3f}  scale={norm_scale[0]:.3f}")
    print(f"  c:     min={norm_min[1]:.3f}  scale={norm_scale[1]:.3f}")
    print(f"  theta: min={norm_min[2]:.1f}  scale={norm_scale[2]:.1f}")

    # ── Select frames: first and last of test set ─────────────────────────────
    with h5py.File(TEST_H5, "r") as f:
        sims = sorted(f.keys())
        first_sim, first_t = sims[0],  0
        last_sim,  last_t  = sims[-1], f[sims[-1]]["images"].shape[0] - 1

    phys_first = _read_frame(TEST_H5, first_sim, first_t)
    phys_last  = _read_frame(TEST_H5, last_sim,  last_t)

    samples = [
        {"label": f"Start of test\n({first_sim}, t=0)", "phys": phys_first},
        {"label": f"End of test\n({last_sim}, t={last_t})", "phys": phys_last},
    ]
    print(f"[samples] {first_sim}/t={first_t}  and  {last_sim}/t={last_t}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[models] Loading pretrained …")
    model_pre = load_pretrained(device)

    print("[models] Loading fine-tuned …")
    model_ft  = load_finetuned(device)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[plot] Generating comparison …")
    make_comparison_plot(samples, norm_min, norm_scale,
                         model_pre, model_ft, device, OUT_PATH)


if __name__ == "__main__":
    main()
