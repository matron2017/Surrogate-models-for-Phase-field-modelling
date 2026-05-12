#!/usr/bin/env python3
"""
DC-AE comparison: pretrained (no fine-tune) vs fine-tuned (best checkpoint).
Shows 2 data points — first and last frame of test set.
Each row: [GT φ | GT c | GT θ] | [Pretrained φ | c | θ] | [Fine-tuned φ | c | θ]
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
from matplotlib.colorbar import Colorbar

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
CMAP_PHI   = "RdBu_r"
CMAP_CONC  = mcolors.LinearSegmentedColormap.from_list(
    "conc_bgc", ["#1565C0", "#29B6F6", "#E3F2FD", "#C8E6C9", "#1B5E20"])
CMAP_THERM = "inferno"
CMAP_ERR   = "bwr"


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
    """(1,3,H,W) normalised → (3,H,W) physical numpy."""
    x = t.squeeze(0).cpu().float().numpy()
    return (x * 0.5 + 0.5) * norm_scale[:, None, None] + norm_min[:, None, None]


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
    samples: list of dicts with keys 'label', 'phys' (3,H,W)
    Layout: 2 rows × 9 columns
      cols 0-2: GT (φ, c, θ)
      cols 3-5: Pretrained recon (φ, c, θ)
      cols 6-8: Fine-tuned recon (φ, c, θ)
    """
    n_rows = len(samples)
    n_cols = 9
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.2 * n_cols, 4.8 * n_rows),
                              gridspec_kw={"wspace": 0.35, "hspace": 0.55})
    if n_rows == 1:
        axes = axes[None, :]

    ch_labels  = ["φ (phase field)", "c (concentration)", "θ (thermal, K)"]
    ch_units   = ["—", "mol fraction", "K"]
    ch_cmaps   = [CMAP_PHI, CMAP_CONC, CMAP_THERM]

    # column group headers
    group_titles = ["Ground Truth", "Pretrained (no fine-tune)", "Fine-tuned (lr=2e-5, best ckpt)"]
    group_colors = ["#2c3e50",      "#c0392b",                    "#1a5276"]

    # Add group column headers as figure text
    for gi, (gtitle, gcol) in enumerate(zip(group_titles, group_colors)):
        col_center = (gi * 3 + 1) / (n_cols)  # normalised x center of group
        # Will add per-row via ax titles instead
        pass

    for row_idx, sample in enumerate(samples):
        phys_gt = sample["phys"]          # (3, H, W)
        x_norm  = to_norm(phys_gt, norm_min, norm_scale).to(device)

        with torch.no_grad():
            z_pre  = model_pretrained.encode(x_norm)
            y_pre  = model_pretrained.decode(z_pre).cpu()
            z_ft   = model_finetuned.encode(x_norm)
            y_ft   = model_finetuned.decode(z_ft).cpu()

        phys_pre = from_norm(y_pre, norm_min, norm_scale)
        phys_ft  = from_norm(y_ft,  norm_min, norm_scale)

        groups = [
            ("GT",           phys_gt,  group_colors[0]),
            ("Pretrained",   phys_pre, group_colors[1]),
            ("Fine-tuned",   phys_ft,  group_colors[2]),
        ]

        # Determine shared vmin/vmax per channel across all three (GT + both models)
        vmins, vmaxs = [], []
        for ch in range(3):
            all_ch = np.stack([phys_gt[ch], phys_pre[ch], phys_ft[ch]])
            vmins.append(float(all_ch.min()))
            vmaxs.append(float(all_ch.max()))

        for gi, (glabel, phys, gcol) in enumerate(groups):
            for ch in range(3):
                col = gi * 3 + ch
                ax  = axes[row_idx, col]
                img = phys[ch]
                cmap = ch_cmaps[ch]
                title = f"{glabel}\n{ch_labels[ch]}"
                _panel(ax, img, title, cmap, vmins[ch], vmaxs[ch],
                       cbar_label=ch_units[ch], stats=True)

                # Highlight group with colored spine
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(gcol)
                    spine.set_linewidth(2.0)

        # Row label
        fig.text(0.005, 1.0 - (row_idx + 0.5) / n_rows,
                 sample["label"], va="center", ha="left",
                 fontsize=10, fontweight="bold", rotation=90,
                 color="#333333")

    # Overall title + group legend
    fig.suptitle(
        "DC-AE Reconstruction: Pretrained vs Fine-tuned  |  Test set (first & last frame)",
        fontsize=13, fontweight="bold", y=1.01
    )
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group_colors[0], label="Ground Truth"),
        Patch(facecolor=group_colors[1], label="Pretrained  (mit-han-lab/dc-ae-f32c32-in-1.0, no PDE adaptation)"),
        Patch(facecolor=group_colors[2], label="Fine-tuned  (lr=2e-5 cosine, best val checkpoint)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 1.0))

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
