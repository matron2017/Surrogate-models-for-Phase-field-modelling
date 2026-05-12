#!/usr/bin/env python3
"""Evaluation & visualisation script for all trained PDE surrogate models.

Produces one PNG per model showing:
  - Columns: x_source (input frame) | x_target (GT next frame) | x_pred | |error|
  - Rows: phi channel, c channel  (repeating for N_SAMPLES)
  - Bridge models: one extra column for noisy intermediate x_t at t=T/2
  - Bridge trajectory figure: denoising steps from t=T → t=0
  - DC-AE: input | reconstruction | |error| for (phi, c, thermal)

Usage (single GPU, see slurm launcher):
  python eval/scripts/eval_all_models.py --out_dir eval/plots --n_samples 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent  # Phase_field_surrogates/
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── colour helpers ──────────────────────────────────────────────────────────
CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot"


# ─── Normalization stats ──────────────────────────────────────────────────────

class NormStats:
    """Z-score stats loaded from H5 for back-converting to physical units."""
    def __init__(self, phi_mean, phi_std, c_mean, c_std, th_mean, th_std):
        self.phi_mean = phi_mean; self.phi_std = phi_std
        self.c_mean   = c_mean;   self.c_std   = c_std
        self.th_mean  = th_mean;  self.th_std  = th_std

    @classmethod
    def from_h5(cls, h5_path: str) -> "NormStats":
        with h5py.File(h5_path, "r") as f:
            ch_mean = np.asarray(f.attrs["channel_mean"], dtype=np.float32)
            ch_std  = np.asarray(f.attrs["channel_std"],  dtype=np.float32)
            th_mean = float(np.asarray(f.attrs["thermal_field_channel_mean"])[0])
            th_std  = float(np.asarray(f.attrs["thermal_field_channel_std"])[0])
        return cls(float(ch_mean[0]), float(ch_std[0]),
                   float(ch_mean[1]), float(ch_std[1]),
                   th_mean, th_std)

    def phi_to_phys(self, z: np.ndarray) -> np.ndarray:
        """Z-scored phi → physical [-1,1] space (may slightly overshoot)."""
        return z * self.phi_std + self.phi_mean

    def c_to_phys(self, z: np.ndarray) -> np.ndarray:
        """Z-scored c → physical × 3 display units."""
        return (z * self.c_std + self.c_mean) * 3.0

    def th_to_kelvin(self, z: np.ndarray) -> np.ndarray:
        """Z-scored thermal → Kelvin."""
        return z * self.th_std + self.th_mean


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_surrogate_batch(h5_path: str, n: int, device: torch.device,
                         last_n_frames: int = 30):
    """Load n (source, target, theta) triples from the last `last_n_frames`
    of val.h5 simulations — same z-score norm as PFPairDataset.
    Returns tensors in z-scored space (as used by the models) + NormStats.
    """
    norm = NormStats.from_h5(h5_path)
    th_mean = norm.th_mean; th_std = norm.th_std
    th_eps  = 1e-6

    sources, targets, thetas = [], [], []
    with h5py.File(h5_path, "r") as f:
        for sim in sorted(f.keys()):
            if len(sources) >= n:
                break
            g = f[sim]
            imgs  = np.asarray(g["images"][:, :2], dtype=np.float32)
            therm = np.asarray(g["thermal_field"][:, :1], dtype=np.float32)
            T_len = imgs.shape[0]
            # Sample from last `last_n_frames` of this simulation
            t_start = max(T_len - last_n_frames, 0)
            # Pick one representative frame near the end of this simulation
            t = min(t_start + last_n_frames // 2, T_len - 2)
            sources.append(imgs[t])
            targets.append(imgs[t + 1])
            th = therm[t]
            th = (th - th_mean) / (th_std + th_eps)
            thetas.append(th)

    x_src = torch.from_numpy(np.stack(sources[:n])).to(device)
    x_tgt = torch.from_numpy(np.stack(targets[:n])).to(device)
    theta  = torch.from_numpy(np.stack(thetas[:n])).to(device)
    return x_src, x_tgt, theta, norm


def load_dcae_batch(h5_path: str, n: int, device: torch.device):
    """Load n frames for DC-AE evaluation — normalise to [-1,1] per dataset stats.
    Also returns NormStats for physical-unit display.
    """
    frames, norm_min, norm_scale = _dcae_collect_frames(h5_path, n)
    x = torch.from_numpy(frames).to(device)
    norm = NormStats.from_h5(h5_path)
    return x, norm_min, norm_scale, norm


def _dcae_collect_frames(h5_path: str, n: int, last_n_frames: int = 30):
    """Mirror PDEFieldDataset normalisation: per-channel min/max → [-1,1].
    Samples from the last `last_n_frames` of each simulation.
    """
    index = []
    with h5py.File(h5_path, "r") as f:
        for sim in sorted(f.keys()):
            n_t = f[sim]["images"].shape[0]
            t_start = max(n_t - last_n_frames, 0)
            for t in range(t_start, n_t):
                index.append((sim, t))
            if len(index) >= max(n, 200):
                break

    # Compute stats on ≤200 frames (deterministic subset)
    rng = np.random.default_rng(42)
    subset = rng.choice(len(index), size=min(200, len(index)), replace=False)
    mins  = np.full(3, np.inf,  dtype=np.float32)
    maxs  = np.full(3, -np.inf, dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        for i in subset:
            sim, t = index[i]
            x3 = _read_frame3(f, sim, t)
            for c in range(3):
                mins[c] = min(mins[c], float(x3[c].min()))
                maxs[c] = max(maxs[c], float(x3[c].max()))
    scale = maxs - mins
    scale[scale == 0] = 1.0

    frames = []
    with h5py.File(h5_path, "r") as f:
        for sim, t in index[:n]:
            x3 = _read_frame3(f, sim, t)
            x3 = (x3 - mins[:, None, None]) / scale[:, None, None]
            x3 = x3 * 2.0 - 1.0
            frames.append(x3)
    return np.stack(frames), mins, scale


def _read_frame3(f, sim, t):
    g = f[sim]
    x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
    th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
    return np.concatenate([x2, th], axis=0)  # (3, H, W)


# ─── Model builders ───────────────────────────────────────────────────────────

def build_det_pixel(ckpt_path: str, device: torch.device):
    from models.unet_film_bottleneck import UNetFiLMAttn
    model = UNetFiLMAttn(
        in_channels=2, out_channels=2, cond_dim=2,
        channels=[192, 320, 512, 640, 832],
        num_blocks=[2, 2, 2, 2, 2],
        bottleneck_blocks=(2, 2),
        attn_heads=8,
        use_bottleneck_attn=False,
        afno_depth=12, afno_mlp_ratio=12.0, afno_num_blocks=16,
        afno_inp_shape=[32, 32], afno_patch_size=[1, 1],
        afno_sparsity_threshold=0.01, afno_hard_thresholding_fraction=1.0,
        film_dim=512, time_emb_dim=128,
        use_time=False,
        skip_film=True, film_mode="affine",
        use_control_branch=True, hint_channels=1,
        control_strength=1.5,
        control_channels=[96, 160, 256, 320, 416],
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("model", ck)
    # Strip DDP prefix if present
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[det_pixel] Loaded {n/1e6:.1f}M params from epoch {ck.get('epoch', '?')}")
    return model


def build_bridge(ckpt_path: str, device: torch.device):
    from diffusion_bridge.models.bridge_wrapper import BridgePDEModel
    model = BridgePDEModel(
        channels=[192, 320, 512, 640, 832],
        control_channels=[96, 160, 256, 320, 416],
        afno_depth=12, afno_num_blocks=16, afno_mlp_ratio=12.0,
        afno_inp_shape=[32, 32], film_dim=512, dropout=0.0,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("model", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[bridge] Loaded {n/1e6:.1f}M params from epoch {ck.get('epoch', '?')} val_mse={ck.get('val_mse','?')}")
    return model


def _remap_dcae_keys(sd: dict) -> dict:
    """Remap old DC-Gen checkpoint key names to current DCAE_HF attribute paths.

    Old: *.context_main.*  / *.local_main.*
    New: *.context_module.main.*  / *.local_module.main.*
    """
    import re
    out = {}
    for k, v in sd.items():
        k2 = re.sub(r'\bcontext_main\b', 'context_module.main', k)
        k2 = re.sub(r'\blocal_main\b',   'local_module.main',   k2)
        out[k2] = v
    return out


def build_dcae(ckpt_path: str, device: torch.device):
    # DC-Gen repo is not installed as a package — add it to sys.path
    dc_gen_repo = ROOT / "autoencoder_dc_ae/external_refs/DC-Gen"
    if str(dc_gen_repo) not in sys.path:
        sys.path.insert(0, str(dc_gen_repo))
    from dc_gen.ae_model_zoo import DCAE_HF
    model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f32c32-in-1.0").to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("model", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = _remap_dcae_keys(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[dcae] Loaded {n/1e6:.1f}M params from epoch {ck.get('epoch', '?')}")
    return model


# ─── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def run_det_pixel(model, x_src, theta):
    """Deterministic UNet-AFNO prediction. Returns (N,2,H,W) in float32."""
    cond = torch.zeros(x_src.shape[0], 2, device=x_src.device, dtype=x_src.dtype)
    with torch.amp.autocast("cuda", enabled=True):
        pred = model(x_src, cond, hint=theta)
    return pred.float()


@torch.no_grad()
def run_bridge(model, sde, x_src, x_tgt, theta, n_steps: int = 20, return_trajectory: bool = False):
    """Run bridge reverse ODE. Returns x_pred (N,2,H,W) and noisy x_t at t=T/2."""
    from diffusion_bridge.sde.unidb_sde import UniDBSDE
    from diffusion_bridge.sde.frac_bridge_sde import FracBridgeSDE

    N = x_src.shape[0]
    T = sde.T
    theta_N = theta[:N]

    # ── Noisy intermediate for visualisation (forward bridge at t=T/2) ──────
    t_half = torch.full((N,), T // 2, device=x_src.device, dtype=torch.long)
    if isinstance(sde, UniDBSDE):
        x_t_half, _ = sde.q_sample(x_tgt[:N], x_src[:N], t_half)
    else:
        x_t_half, _ = sde.q_sample(x_tgt[:N], x_src[:N], t_half)

    # ── model_fn wrapper: SDE calls model_fn(x_in_4ch, t_norm_scalar) ───────
    # Split 4-ch x_in back into (x_noisy, x_source), broadcast t_norm to batch
    def model_fn(x_in_4ch, t_norm):
        x_noisy  = x_in_4ch[:, :2]
        x_source = x_in_4ch[:, 2:]
        B = x_noisy.shape[0]
        # t_norm may be 0-dim scalar tensor → reshape to (B,)
        t_b = t_norm.reshape(1).expand(B).to(x_noisy.device)
        with torch.amp.autocast("cuda", enabled=True):
            return model(x_noisy, x_source, t_b, theta_N[:B])

    trajectory = [] if return_trajectory else None

    if isinstance(sde, UniDBSDE):
        x_pred = sde.sample(model_fn, x_src[:N], n_steps=n_steps, trajectory=trajectory)
    else:
        x_pred = sde.sample(model_fn, x_src[:N], n_steps=n_steps, trajectory=trajectory)

    return x_pred.float(), x_t_half.float(), trajectory or []


@torch.no_grad()
def run_dcae(model, x_frames):
    """Encode + decode. Returns reconstruction (N,3,H,W) in float32."""
    with torch.amp.autocast("cuda", enabled=True):
        latent = model.encode(x_frames)
        if hasattr(latent, "latent_dist"):
            latent = latent.latent_dist.sample()
        elif hasattr(latent, "latent"):
            latent = latent.latent
        recon = model.decode(latent)
        if hasattr(recon, "sample"):
            recon = recon.sample
    return recon.float().clamp(-1, 1)


# ─── Plotting helpers ──────────────────────────────────────────────────────────

def _np(t):
    return t.squeeze().cpu().numpy()


def _sym_lim(arr):
    """Symmetric colour limits centred at 0, 2nd–98th percentile robust."""
    v = np.abs(arr).ravel()
    vmax = float(np.percentile(v, 98))
    return -vmax, vmax


def _err_lim(arr):
    return 0.0, float(np.percentile(np.abs(arr).ravel(), 98))


def _add_colorbar(ax, im, label=""):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    if label:
        cb.set_label(label, fontsize=7)


# ─── Per-model plot functions ──────────────────────────────────────────────────

def plot_surrogate(x_src, x_tgt, x_pred, theta, title: str, out_path: Path,
                   norm: "NormStats | None" = None,
                   x_noisy_half=None, n_steps_label=""):
    """5 or 6 column comparison figure for surrogate/bridge models.

    Rows: phi, c  (repeated per sample)
    Cols: [source | GT target | prediction | |error| | thermal] (+ noisy for bridge)
    Physical units applied when norm is provided:
      - phi  → physical [-1,1] scale, cmap limits clamped to [-1,1]
      - c    → (z * c_std + c_mean) × 3  display units
      - theta → Kelvin
    """
    N = x_src.shape[0]

    # ── Physical conversion helpers ───────────────────────────────────────────
    def _phys(arr_np, ch):
        if norm is None:
            return arr_np
        if ch == 0:
            return norm.phi_to_phys(arr_np)
        else:
            return norm.c_to_phys(arr_np)

    def _phys_th(arr_np):
        return norm.th_to_kelvin(arr_np) if norm is not None else arr_np

    # ── Channel display settings ──────────────────────────────────────────────
    if norm is not None:
        ch_names   = ["φ  (phase field)", "c × 3  (composition)"]
        ch_units   = ["[-1, 1]",          "[× 3 units]"]
        th_unit    = "K"
    else:
        ch_names   = ["φ  (phase)", "c  (composition)"]
        ch_units   = ["", ""]
        th_unit    = "z-score"

    n_cols = 6 if x_noisy_half is not None else 5
    col_labels_base = ["Source (input)", "Target (GT)", f"Pred ({n_steps_label})", "|Error|", f"Thermal [{th_unit}]"]
    if x_noisy_half is not None:
        col_labels_base = ["Source (input)", "Noisy x_t (t=T/2)", "Target (GT)", f"Pred ({n_steps_label})", "|Error|", f"Thermal [{th_unit}]"]

    n_rows = N * 2
    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.002)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.08)

    for s in range(N):
        for ch in range(2):
            row = s * 2 + ch
            src_ch  = _phys(_np(x_src[s, ch]),  ch)
            tgt_ch  = _phys(_np(x_tgt[s, ch]),  ch)
            pred_ch = _phys(_np(x_pred[s, ch]), ch)
            err_ch  = pred_ch - tgt_ch
            th_ch   = _phys_th(_np(theta[s, 0]))

            # Colour limits for field channels
            if ch == 0 and norm is not None:
                vlo, vhi = -1.0, 1.0          # phi: fixed physical range
            else:
                vlo, vhi = _sym_lim(np.stack([src_ch, tgt_ch, pred_ch]))
            elo, ehi = _err_lim(err_ch)

            col = 0
            # Source
            ax = fig.add_subplot(gs[row, col]); col += 1
            im = ax.imshow(src_ch, cmap=CMAP_FIELD, vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title(col_labels_base[0], fontsize=9, fontweight="bold")
            ax.set_ylabel(f"S{s+1} {ch_names[ch]}\n{ch_units[ch]}", fontsize=7)
            ax.axis("off"); _add_colorbar(ax, im)

            # Noisy (bridge only)
            if x_noisy_half is not None:
                noisy_ch = _phys(_np(x_noisy_half[s, ch]), ch)
                ax = fig.add_subplot(gs[row, col]); col += 1
                im = ax.imshow(noisy_ch, cmap=CMAP_FIELD, vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
                if row == 0: ax.set_title(col_labels_base[1], fontsize=9, fontweight="bold")
                ax.axis("off"); _add_colorbar(ax, im)

            # GT target
            ax = fig.add_subplot(gs[row, col]); col += 1
            im = ax.imshow(tgt_ch, cmap=CMAP_FIELD, vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title("Target (GT)", fontsize=9, fontweight="bold")
            ax.axis("off"); _add_colorbar(ax, im)

            # Prediction
            ax = fig.add_subplot(gs[row, col]); col += 1
            im = ax.imshow(pred_ch, cmap=CMAP_FIELD, vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title([l for l in col_labels_base if "Pred" in l][0], fontsize=9, fontweight="bold")
            ax.axis("off"); _add_colorbar(ax, im)

            # |Error|
            ax = fig.add_subplot(gs[row, col]); col += 1
            im = ax.imshow(np.abs(err_ch), cmap=CMAP_ERR, vmin=elo, vmax=ehi, origin="lower", aspect="equal")
            mae = np.abs(err_ch).mean()
            if row == 0: ax.set_title("|Error|", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"MAE={mae:.4f}", fontsize=8)
            ax.axis("off"); _add_colorbar(ax, im)

            # Thermal
            ax = fig.add_subplot(gs[row, col]); col += 1
            th_vlo = float(np.percentile(th_ch, 2))
            th_vhi = float(np.percentile(th_ch, 98))
            im = ax.imshow(th_ch, cmap="plasma", vmin=th_vlo, vmax=th_vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title(f"Thermal [{th_unit}]", fontsize=9, fontweight="bold")
            ax.axis("off"); _add_colorbar(ax, im, label=th_unit)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_bridge_trajectory(trajectory, x_src, x_tgt, title: str, out_path: Path,
                           norm: "NormStats | None" = None,
                           n_samples: int = 3):
    """Show denoising trajectory for n_samples examples side-by-side.

    Layout: 8 columns = [Source | step1 | ... | step6 | GT target]
    Rows: one per sample × 2 channels (phi, c)
    All values shown in physical units when norm is provided.
    """
    if not trajectory:
        print("  [trajectory] no trajectory to plot")
        return

    def _phys(arr_np, ch):
        if norm is None:
            return arr_np
        return norm.phi_to_phys(arr_np) if ch == 0 else norm.c_to_phys(arr_np)

    ch_names = ["φ  [-1,1]", "c × 3"] if norm else ["φ", "c"]
    n_s = min(n_samples, x_src.shape[0])

    # Sub-sample trajectory to 6 evenly spaced steps
    keep = min(6, len(trajectory))
    step_indices = np.linspace(0, len(trajectory) - 1, keep, dtype=int)
    frames = [trajectory[i] for i in step_indices]
    n_cols = len(frames) + 2   # source + steps + GT

    n_rows = n_s * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    fig.suptitle(title + f" — Denoising trajectory ({n_s} samples)", fontsize=12, fontweight="bold")

    for s in range(n_s):
        for ch in range(2):
            row = s * 2 + ch
            src_arr  = _phys(_np(x_src[s, ch]), ch)
            tgt_arr  = _phys(_np(x_tgt[s, ch]), ch)
            step_arrs = [_phys(_np(f[s, ch]), ch) for f in frames]

            all_vals = np.concatenate([src_arr.ravel(), tgt_arr.ravel()] +
                                       [a.ravel() for a in step_arrs])
            if ch == 0 and norm is not None:
                vmin, vmax = -1.0, 1.0
            else:
                vabs = float(np.percentile(np.abs(all_vals), 98))
                vmin, vmax = -vabs, vabs

            # Source
            ax = axes[row, 0]
            ax.imshow(src_arr, cmap=CMAP_FIELD, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
            if row == 0: ax.set_title("Source", fontsize=9, fontweight="bold")
            ax.set_ylabel(f"S{s+1} {ch_names[ch]}", fontsize=7)
            ax.axis("off")

            # Trajectory steps
            for j, (arr, si) in enumerate(zip(step_arrs, step_indices)):
                ax = axes[row, j + 1]
                ax.imshow(arr, cmap=CMAP_FIELD, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
                if row == 0: ax.set_title(f"step {si+1}/{len(trajectory)}", fontsize=7)
                ax.axis("off")

            # GT
            ax = axes[row, n_cols - 1]
            im = ax.imshow(tgt_arr, cmap=CMAP_FIELD, vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
            if row == 0: ax.set_title("GT target", fontsize=9, fontweight="bold")
            ax.axis("off")

    unit_lbl = "phys" if norm else "z-score"
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, label=unit_lbl)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_dcae(x_in, x_recon, norm_min, norm_scale, title: str, out_path: Path,
              norm: "NormStats | None" = None):
    """3-col (input | recon | |error|) × N×3 rows (per sample: phi, c, thermal).
    Physical units applied via norm when provided.
    """
    N = x_in.shape[0]

    # De-normalise from DC-AE [-1,1] space back to raw, then to physical
    def _to_phys_dcae(arr_norm, ch):
        """arr_norm ∈ [-1,1] (DC-AE space) → physical display value."""
        raw = (arr_norm * 0.5 + 0.5) * norm_scale[ch] + norm_min[ch]
        if norm is None:
            return raw
        if ch == 0:
            # raw is z-scored phi → physical
            return norm.phi_to_phys(raw)
        elif ch == 1:
            # raw is z-scored c → physical × 3
            return norm.c_to_phys(raw)
        else:
            # ch==2: raw is Kelvin (DC-AE trained on raw thermal)
            return raw  # already Kelvin

    ch_names = (["φ  (phase, [-1,1])", "c × 3  (composition)", "θ  (Kelvin)"]
                if norm else ["φ", "c", "θ"])
    unit_strs = (["[-1,1]", "[×3 units]", "[K]"] if norm else ["", "", ""])

    n_rows = N * 3
    fig = plt.figure(figsize=(3.2 * 3, 3.0 * n_rows))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.35, wspace=0.08)

    for s in range(N):
        for ch in range(3):
            row = s * 3 + ch
            inp_ch  = _to_phys_dcae(_np(x_in[s, ch]),    ch)
            rec_ch  = _to_phys_dcae(_np(x_recon[s, ch]), ch)
            err_ch  = rec_ch - inp_ch

            if ch == 0 and norm is not None:
                vlo, vhi = -1.0, 1.0
            elif ch == 2 and norm is not None:
                # Kelvin: use actual range
                vlo = float(np.percentile(inp_ch, 2))
                vhi = float(np.percentile(inp_ch, 98))
            else:
                vlo, vhi = _sym_lim(np.stack([inp_ch, rec_ch]))
            elo, ehi = _err_lim(err_ch)

            # Input
            ax = fig.add_subplot(gs[row, 0])
            im = ax.imshow(inp_ch, cmap=CMAP_FIELD if ch < 2 else "plasma",
                           vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title("Input", fontsize=9, fontweight="bold")
            ax.set_ylabel(f"S{s+1} {ch_names[ch]}\n{unit_strs[ch]}", fontsize=7)
            ax.axis("off"); _add_colorbar(ax, im, unit_strs[ch])

            # Reconstruction
            ax = fig.add_subplot(gs[row, 1])
            im = ax.imshow(rec_ch, cmap=CMAP_FIELD if ch < 2 else "plasma",
                           vmin=vlo, vmax=vhi, origin="lower", aspect="equal")
            if row == 0: ax.set_title("Reconstruction", fontsize=9, fontweight="bold")
            ax.axis("off"); _add_colorbar(ax, im, unit_strs[ch])

            # |Error|
            ax = fig.add_subplot(gs[row, 2])
            mae = np.abs(err_ch).mean()
            im = ax.imshow(np.abs(err_ch), cmap=CMAP_ERR, vmin=elo, vmax=ehi, origin="lower", aspect="equal")
            if row == 0: ax.set_title("|Error|", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"MAE={mae:.4f}", fontsize=8)
            ax.axis("off"); _add_colorbar(ax, im)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",   default="eval/plots")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--n_steps",   type=int, default=20, help="Bridge denoising steps")
    ap.add_argument("--skip_dcae", action="store_true")
    ap.add_argument("--skip_det",  action="store_true")
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--h5", default=None,
                    help="Path to HDF5 data file (default: autoencoder_dc_ae/data/val.h5)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  n_samples={args.n_samples}  n_steps={args.n_steps}")

    val_h5 = Path(args.h5) if args.h5 else ROOT / "autoencoder_dc_ae/data/val.h5"
    print(f"[eval] data={val_h5}")

    # ── Checkpoint paths ───────────────────────────────────────────────────────
    DET_CKPT = ROOT / "deterministic_pixel/runs" \
        / "big_det_unet_afno_controlxs_wavelet_512_20260511T113243Z_n3_ws12_bpg1_acc6_34391106" \
        / "UNetFiLMAttn/checkpoint.last.pth"
    UNIDB_CKPT = ROOT / "diffusion_bridge/runs/unidb_big_20260511T131120Z/checkpoint.last.pth"
    FRAC_CKPT  = ROOT / "diffusion_bridge/runs/frac_big_20260511T131120Z/checkpoint.last.pth"
    DCAE_CKPT  = ROOT / "autoencoder_dc_ae/runs/autoencoder/finetune/dc_ae_f32c32_lr2e5/checkpoint.best.pth"

    # ── Load surrogate batch (last 30 frames of each simulation) ──────────────
    print("[eval] Loading surrogate batch from last 30 frames …")
    x_src, x_tgt, theta, norm = load_surrogate_batch(
        str(val_h5), args.n_samples, device, last_n_frames=30
    )
    print(f"  x_src={tuple(x_src.shape)}  x_tgt={tuple(x_tgt.shape)}  theta={tuple(theta.shape)}")

    # ── Det pixel ─────────────────────────────────────────────────────────────
    if not args.skip_det and DET_CKPT.exists():
        print("\n[det_pixel] Running inference …")
        model = build_det_pixel(str(DET_CKPT), device)
        x_pred = run_det_pixel(model, x_src, theta)
        del model; torch.cuda.empty_cache()
        plot_surrogate(
            x_src, x_tgt, x_pred, theta,
            title=f"Deterministic UNet-AFNO  (392M params)  —  {args.n_samples} late-traj samples",
            out_path=out_dir / "eval_det_pixel.png",
            norm=norm, n_steps_label="det",
        )
    else:
        print(f"\n[det_pixel] Skipped (skip_det={args.skip_det}, ckpt_exists={DET_CKPT.exists()})")

    # ── UniDB bridge ──────────────────────────────────────────────────────────
    if not args.skip_bridge and UNIDB_CKPT.exists():
        print("\n[unidb] Running inference …")
        from diffusion_bridge.sde.unidb_sde import UniDBSDE
        model = build_bridge(str(UNIDB_CKPT), device)
        sde   = UniDBSDE(T=1000, device=device)
        x_pred, x_noisy_half, traj = run_bridge(
            model, sde, x_src, x_tgt, theta,
            n_steps=args.n_steps, return_trajectory=True,
        )
        del model; torch.cuda.empty_cache()
        plot_surrogate(
            x_src, x_tgt, x_pred, theta,
            title=f"UniDB Bridge  (421M params, {args.n_steps}-step ODE)  —  {args.n_samples} late-traj samples",
            out_path=out_dir / "eval_bridge_unidb.png",
            norm=norm, x_noisy_half=x_noisy_half,
            n_steps_label=f"{args.n_steps}-step",
        )
        plot_bridge_trajectory(
            traj, x_src, x_tgt,
            title=f"UniDB Bridge denoising trajectory ({args.n_steps} steps)",
            out_path=out_dir / "eval_bridge_unidb_trajectory.png",
            norm=norm, n_samples=args.n_samples,
        )
    else:
        print(f"\n[unidb] Skipped (ckpt_exists={UNIDB_CKPT.exists()})")

    # ── FracBridge ────────────────────────────────────────────────────────────
    if not args.skip_bridge and FRAC_CKPT.exists():
        print("\n[fractal] Running inference …")
        from diffusion_bridge.sde.frac_bridge_sde import FracBridgeSDE
        model = build_bridge(str(FRAC_CKPT), device)
        sde   = FracBridgeSDE(T=1000, device=device)
        x_pred, x_noisy_half, traj = run_bridge(
            model, sde, x_src, x_tgt, theta,
            n_steps=args.n_steps, return_trajectory=True,
        )
        del model; torch.cuda.empty_cache()
        plot_surrogate(
            x_src, x_tgt, x_pred, theta,
            title=f"Fractional Bridge  (421M params, {args.n_steps}-step ODE)  —  {args.n_samples} late-traj samples",
            out_path=out_dir / "eval_bridge_frac.png",
            norm=norm, x_noisy_half=x_noisy_half,
            n_steps_label=f"{args.n_steps}-step",
        )
        plot_bridge_trajectory(
            traj, x_src, x_tgt,
            title=f"Fractional Bridge denoising trajectory ({args.n_steps} steps)",
            out_path=out_dir / "eval_bridge_frac_trajectory.png",
            norm=norm, n_samples=args.n_samples,
        )
    else:
        print(f"\n[fractal] Skipped (ckpt_exists={FRAC_CKPT.exists()})")

    # ── DC-AE ─────────────────────────────────────────────────────────────────
    if not args.skip_dcae and DCAE_CKPT.exists():
        print("\n[dcae] Running inference …")
        x_frames, norm_min, norm_scale, dcae_norm = load_dcae_batch(
            str(val_h5), args.n_samples, device
        )
        model  = build_dcae(str(DCAE_CKPT), device)
        x_recon = run_dcae(model, x_frames)
        del model; torch.cuda.empty_cache()
        plot_dcae(
            x_frames.cpu(), x_recon.cpu(), norm_min, norm_scale,
            title=f"DC-AE f32c32  (fine-tuned, lr2e5)  —  {args.n_samples} late-traj samples",
            out_path=out_dir / "eval_dcae.png",
            norm=dcae_norm,
        )
    else:
        print(f"\n[dcae] Skipped (ckpt_exists={DCAE_CKPT.exists()})")

    print(f"\n[eval] All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
