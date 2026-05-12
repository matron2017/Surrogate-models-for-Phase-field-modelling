#!/usr/bin/env python3
"""Single-row high-quality summary figures for each PDE surrogate model.

One figure per model, one row of panels (one sample, all channels laid out
side-by-side).  Every panel is annotated with its min / mean / max value in
physical units.  Concentration (c) uses a cyan/green/blue diverging colormap.
Surrogate models include two GT-delta panels (GT out − GT in) so the expected
temporal change is immediately visible.

Layout — surrogate models (11 panels):
  φ src | c src | φ GT | c GT | GT Δφ | GT Δc | φ pred | c pred | |err φ| | |err c| | θ K

Layout — DC-AE (9 panels):
  φ in | c in | θ in | φ rec | c rec | θ rec | |err φ| | |err c| | |err θ|

Default data: test.h5

Models produced:
  eval_row_det_pixel.png
  eval_row_bridge_unidb.png
  eval_row_bridge_frac.png
  eval_row_dcae_lr2e5.png   (best-checkpoint, lr=2e-5)
  eval_row_dcae_lr5e6.png   (best-checkpoint, lr=5e-6)

Usage (SLURM launcher also provided):
  python eval/scripts/eval_summary_rows.py --out_dir eval/plots/test --n_steps 20
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot"
CMAP_THERM = "plasma"
# Blue → sky-blue → near-white → mint-green → dark green (for concentration field)
CMAP_CONC = LinearSegmentedColormap.from_list(
    "conc_bgc",
    ["#1565C0", "#29B6F6", "#E3F2FD", "#C8E6C9", "#1B5E20"],
    N=256,
)


# ─── Normalization stats ──────────────────────────────────────────────────────

class NormStats:
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

    def phi_to_phys(self, z): return z * self.phi_std + self.phi_mean
    def c_to_phys(self, z):   return (z * self.c_std + self.c_mean) * 3.0
    def th_to_kelvin(self, z): return z * self.th_std + self.th_mean


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _np(t): return t.squeeze().cpu().numpy()


def load_surrogate_sample(h5_path: str, device: torch.device, last_n_frames: int = 30):
    """Load one (source, target, theta) triple from val.h5."""
    norm = NormStats.from_h5(h5_path)
    th_mean, th_std = norm.th_mean, norm.th_std
    with h5py.File(h5_path, "r") as f:
        sim = sorted(f.keys())[0]
        g = f[sim]
        imgs  = np.asarray(g["images"][:, :2], dtype=np.float32)
        therm = np.asarray(g["thermal_field"][:, :1], dtype=np.float32)
        T_len = imgs.shape[0]
        t_start = max(T_len - last_n_frames, 0)
        t = min(t_start + last_n_frames // 2, T_len - 2)
        src = imgs[t]
        tgt = imgs[t + 1]
        th  = therm[t]
        th  = (th - th_mean) / (th_std + 1e-6)
    x_src   = torch.from_numpy(src[None]).to(device)   # (1,2,H,W)
    x_tgt   = torch.from_numpy(tgt[None]).to(device)
    theta   = torch.from_numpy(th[None]).to(device)    # (1,1,H,W)
    return x_src, x_tgt, theta, norm


def load_dcae_sample(h5_path: str, device: torch.device, last_n_frames: int = 30):
    """Load one frame for DC-AE evaluation, normalised to [-1,1]."""
    norm = NormStats.from_h5(h5_path)
    # Collect frames to compute stats
    index = []
    with h5py.File(h5_path, "r") as f:
        for sim in sorted(f.keys()):
            n_t = f[sim]["images"].shape[0]
            t_start = max(n_t - last_n_frames, 0)
            for t in range(t_start, n_t):
                index.append((sim, t))
            if len(index) >= 200:
                break
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
    sim, t = index[0]
    with h5py.File(h5_path, "r") as f:
        x3 = _read_frame3(f, sim, t)
    x3 = (x3 - mins[:, None, None]) / scale[:, None, None]
    x3 = x3 * 2.0 - 1.0
    x = torch.from_numpy(x3[None]).to(device)   # (1,3,H,W)
    return x, mins, scale, norm


def _read_frame3(f, sim, t):
    g = f[sim]
    x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
    th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
    return np.concatenate([x2, th], axis=0)


# ─── Model builders (reuse same logic as eval_all_models) ────────────────────

def build_det_pixel(ckpt_path: str, device: torch.device):
    from models.unet_film_bottleneck import UNetFiLMAttn
    model = UNetFiLMAttn(
        in_channels=2, out_channels=2, cond_dim=2,
        channels=[192, 320, 512, 640, 832],
        num_blocks=[2, 2, 2, 2, 2],
        bottleneck_blocks=(2, 2),
        attn_heads=8, use_bottleneck_attn=False,
        afno_depth=12, afno_mlp_ratio=12.0, afno_num_blocks=16,
        afno_inp_shape=[32, 32], afno_patch_size=[1, 1],
        afno_sparsity_threshold=0.01, afno_hard_thresholding_fraction=1.0,
        film_dim=512, time_emb_dim=128, use_time=False,
        skip_film=True, film_mode="affine",
        use_control_branch=True, hint_channels=1, control_strength=1.5,
        control_channels=[96, 160, 256, 320, 416],
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = {k.replace("module.", "", 1): v for k, v in ck.get("model", ck).items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[det_pixel] epoch={ck.get('epoch','?')}")
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
    sd = {k.replace("module.", "", 1): v for k, v in ck.get("model", ck).items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[bridge] epoch={ck.get('epoch','?')}  val_mse={ck.get('val_mse','?')}")
    return model


def build_dcae(ckpt_path: str, device: torch.device):
    dc_gen_repo = ROOT / "autoencoder_dc_ae/external_refs/DC-Gen"
    if str(dc_gen_repo) not in sys.path:
        sys.path.insert(0, str(dc_gen_repo))
    from dc_gen.ae_model_zoo import DCAE_HF
    import re
    model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f32c32-in-1.0").to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("model", ck)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {re.sub(r'\bcontext_main\b', 'context_module.main',
                 re.sub(r'\blocal_main\b', 'local_module.main', k)): v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[dcae] epoch={ck.get('epoch','?')}")
    return model


# ─── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def run_det_pixel(model, x_src, theta):
    cond = torch.zeros(x_src.shape[0], 2, device=x_src.device, dtype=x_src.dtype)
    with torch.amp.autocast("cuda", enabled=True):
        pred = model(x_src, cond, hint=theta)
    return pred.float()


@torch.no_grad()
def run_bridge(model, sde, x_src, x_tgt, theta, n_steps: int = 20):
    from diffusion_bridge.sde.unidb_sde import UniDBSDE
    N = x_src.shape[0]
    theta_N = theta[:N]

    def model_fn(x_in_4ch, t_norm):
        x_noisy  = x_in_4ch[:, :2]
        x_source = x_in_4ch[:, 2:]
        B = x_noisy.shape[0]
        t_b = t_norm.reshape(1).expand(B).to(x_noisy.device)
        with torch.amp.autocast("cuda", enabled=True):
            return model(x_noisy, x_source, t_b, theta_N[:B])

    x_pred = sde.sample(model_fn, x_src[:N], n_steps=n_steps)
    return x_pred.float()


@torch.no_grad()
def run_dcae(model, x_frames):
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


# ─── Annotation helper ────────────────────────────────────────────────────────

def _annotate_stats(ax, arr: np.ndarray, fontsize: int = 7):
    """Overlay min / mean / max as a small text box in lower-left corner."""
    mn  = float(arr.min())
    mu  = float(arr.mean())
    mx  = float(arr.max())
    txt = f"min {mn:.3g}\nmean {mu:.3g}\nmax {mx:.3g}"
    ax.text(
        0.02, 0.02, txt,
        transform=ax.transAxes,
        fontsize=fontsize, color="white",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
    )


def _panel(ax, img: np.ndarray, title: str, cmap: str,
           vmin=None, vmax=None, cbar_label: str = "",
           fontsize_title: int = 9, annotate: bool = True):
    """Draw one image panel with title, colorbar, and stats overlay."""
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold", pad=4)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
    cb.ax.tick_params(labelsize=6)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6)
    if annotate:
        _annotate_stats(ax, img)
    return im


def _sym_lim(arrays):
    v = np.abs(np.concatenate([a.ravel() for a in arrays]))
    return -float(np.percentile(v, 98)), float(np.percentile(v, 98))


def _err_lim(arr):
    return 0.0, float(np.percentile(np.abs(arr).ravel(), 98))


# ─── Per-model single-row figure ─────────────────────────────────────────────

def save_row_surrogate(x_src, x_tgt, x_pred, theta, norm: NormStats,
                       title: str, out_path: Path, tag: str = "det"):
    """Single-row figure:
    [φ src | c src | φ GT | c GT | GT Δφ | GT Δc | φ pred | c pred | φ err | c err | thermal K]

    GT Δφ and GT Δc show the signed change (target − source) = what the model must predict.
    Each panel annotated with min / mean / max in physical units.
    Concentration channel (c) uses cyan/green/blue colormap.
    """
    # Convert to physical
    phi_src  = norm.phi_to_phys(_np(x_src[0, 0]))
    c_src    = norm.c_to_phys(_np(x_src[0, 1]))
    phi_gt   = norm.phi_to_phys(_np(x_tgt[0, 0]))
    c_gt     = norm.c_to_phys(_np(x_tgt[0, 1]))
    phi_pred = norm.phi_to_phys(_np(x_pred[0, 0]))
    c_pred   = norm.c_to_phys(_np(x_pred[0, 1]))
    phi_err  = phi_pred - phi_gt
    c_err    = c_pred   - c_gt
    phi_delta = phi_gt  - phi_src   # GT temporal change
    c_delta   = c_gt    - c_src
    therm    = norm.th_to_kelvin(_np(theta[0, 0]))

    # Colour limits
    phi_vlo, phi_vhi = -1.0, 1.0
    c_vlo,   c_vhi   = _sym_lim([c_src, c_gt, c_pred])
    delta_phi_lo, delta_phi_hi = _sym_lim([phi_delta])
    delta_c_lo,   delta_c_hi   = _sym_lim([c_delta])
    phi_elo, phi_ehi = _err_lim(phi_err)
    c_elo,   c_ehi   = _err_lim(c_err)
    th_vlo  = float(np.percentile(therm, 2))
    th_vhi  = float(np.percentile(therm, 98))

    panels = [
        (phi_src,        "Source  φ",          CMAP_FIELD, phi_vlo,       phi_vhi,       "[-1,1]"),
        (c_src,          "Source  c×3",         CMAP_CONC,  c_vlo,         c_vhi,         "[×3 u]"),
        (phi_gt,         "GT  φ  (t+1)",        CMAP_FIELD, phi_vlo,       phi_vhi,       "[-1,1]"),
        (c_gt,           "GT  c×3  (t+1)",      CMAP_CONC,  c_vlo,         c_vhi,         "[×3 u]"),
        (phi_delta,      "GT Δφ  (out−in)",     CMAP_FIELD, delta_phi_lo,  delta_phi_hi,  "[-1,1]"),
        (c_delta,        "GT Δc×3  (out−in)",   CMAP_CONC,  delta_c_lo,    delta_c_hi,    "[×3 u]"),
        (phi_pred,       f"Pred  φ  ({tag})",   CMAP_FIELD, phi_vlo,       phi_vhi,       "[-1,1]"),
        (c_pred,         f"Pred  c×3  ({tag})", CMAP_CONC,  c_vlo,         c_vhi,         "[×3 u]"),
        (np.abs(phi_err),"|Err|  φ",            CMAP_ERR,   phi_elo,       phi_ehi,       ""),
        (np.abs(c_err),  "|Err|  c",            CMAP_ERR,   c_elo,         c_ehi,         ""),
        (therm,          "Thermal  θ",          CMAP_THERM, th_vlo,        th_vhi,        "K"),
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5),
                             gridspec_kw={"wspace": 0.22})
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.03)

    for ax, (img, ttl, cmap, vlo, vhi, lbl) in zip(axes, panels):
        _panel(ax, img, ttl, cmap, vlo, vhi, cbar_label=lbl)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_row_bridge(x_src, x_tgt, x_pred, theta, norm: NormStats,
                    title: str, out_path: Path, tag: str = "bridge"):
    """Same as surrogate but tagged with bridge model name."""
    save_row_surrogate(x_src, x_tgt, x_pred, theta, norm, title, out_path, tag=tag)


def save_row_dcae(x_in, x_recon, norm_min, norm_scale, norm: NormStats,
                  title: str, out_path: Path, tag: str = "dcae"):
    """Single-row for DC-AE: [φ in | c in | θ in | φ rec | c rec | θ rec | φ err | c err | θ err]
    Concentration channel uses cyan/green/blue colormap.
    """

    def _to_phys(arr_norm, ch):
        """DC-AE [-1,1] → physical display units."""
        raw = (arr_norm * 0.5 + 0.5) * norm_scale[ch] + norm_min[ch]
        if ch == 0:
            return norm.phi_to_phys(raw)
        elif ch == 1:
            return norm.c_to_phys(raw)
        else:
            return raw  # already Kelvin after de-normalisation

    phi_in  = _to_phys(_np(x_in[0, 0]),    0)
    c_in    = _to_phys(_np(x_in[0, 1]),    1)
    th_in   = _to_phys(_np(x_in[0, 2]),    2)
    phi_rec = _to_phys(_np(x_recon[0, 0]), 0)
    c_rec   = _to_phys(_np(x_recon[0, 1]), 1)
    th_rec  = _to_phys(_np(x_recon[0, 2]), 2)

    phi_err = phi_rec - phi_in
    c_err   = c_rec   - c_in
    th_err  = th_rec  - th_in

    phi_vlo, phi_vhi = -1.0, 1.0
    c_vlo,   c_vhi   = _sym_lim([c_in, c_rec])
    th_vlo  = float(np.percentile(th_in, 2))
    th_vhi  = float(np.percentile(th_in, 98))
    phi_elo, phi_ehi = _err_lim(phi_err)
    c_elo,   c_ehi   = _err_lim(c_err)
    th_elo,  th_ehi  = _err_lim(th_err)

    panels = [
        (phi_in,           "Input  φ",          CMAP_FIELD, phi_vlo, phi_vhi, "[-1,1]"),
        (c_in,             "Input  c×3",         CMAP_CONC,  c_vlo,   c_vhi,   "[×3 u]"),
        (th_in,            "Input  θ",           CMAP_THERM, th_vlo,  th_vhi,  "K"),
        (phi_rec,          f"Recon  φ  ({tag})", CMAP_FIELD, phi_vlo, phi_vhi, "[-1,1]"),
        (c_rec,            f"Recon  c×3  ({tag})", CMAP_CONC, c_vlo,  c_vhi,  "[×3 u]"),
        (th_rec,           f"Recon  θ  ({tag})", CMAP_THERM, th_vlo, th_vhi,  "K"),
        (np.abs(phi_err),  "|Err|  φ",           CMAP_ERR,   phi_elo, phi_ehi, ""),
        (np.abs(c_err),    "|Err|  c",           CMAP_ERR,   c_elo,   c_ehi,  ""),
        (np.abs(th_err),   "|Err|  θ",           CMAP_ERR,   th_elo,  th_ehi, "K"),
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5),
                             gridspec_kw={"wspace": 0.22})
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.03)

    for ax, (img, ttl, cmap, vlo, vhi, lbl) in zip(axes, panels):
        _panel(ax, img, ttl, cmap, vlo, vhi, cbar_label=lbl)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",   default="eval/plots")
    ap.add_argument("--n_steps",   type=int, default=20)
    ap.add_argument("--skip_det",     action="store_true")
    ap.add_argument("--skip_bridge",  action="store_true")
    ap.add_argument("--skip_dcae",    action="store_true")
    ap.add_argument("--h5", default=None,
                    help="Path to HDF5 data file (default: autoencoder_dc_ae/data/val.h5)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_rows] device={device}  n_steps={args.n_steps}")

    val_h5 = Path(args.h5) if args.h5 else ROOT / "autoencoder_dc_ae/data/test.h5"
    print(f"[eval_rows] data={val_h5}")

    DET_CKPT = (ROOT / "deterministic_pixel/runs"
                / "big_det_unet_afno_controlxs_wavelet_512_20260511T113243Z_n3_ws12_bpg1_acc6_34391106"
                / "UNetFiLMAttn/checkpoint.last.pth")
    UNIDB_CKPT = ROOT / "diffusion_bridge/runs/unidb_big_20260511T131120Z/checkpoint.last.pth"
    FRAC_CKPT  = ROOT / "diffusion_bridge/runs/frac_big_20260511T131120Z/checkpoint.last.pth"
    DCAE_LR2E5 = ROOT / "autoencoder_dc_ae/runs/autoencoder/finetune/dc_ae_f32c32_lr2e5/checkpoint.best.pth"
    DCAE_LR5E6 = ROOT / "autoencoder_dc_ae/runs/autoencoder/finetune/dc_ae_f32c32_lr5e6/checkpoint.best.pth"

    # ── Load one surrogate sample ──────────────────────────────────────────────
    print("[eval_rows] Loading surrogate sample …")
    x_src, x_tgt, theta, norm = load_surrogate_sample(str(val_h5), device)
    print(f"  x_src={tuple(x_src.shape)}  x_tgt={tuple(x_tgt.shape)}")

    # ── Deterministic pixel ───────────────────────────────────────────────────
    if not args.skip_det and DET_CKPT.exists():
        print("\n[det_pixel] …")
        model = build_det_pixel(str(DET_CKPT), device)
        x_pred = run_det_pixel(model, x_src, theta)
        del model; torch.cuda.empty_cache()
        save_row_surrogate(
            x_src, x_tgt, x_pred, theta, norm,
            title="Deterministic UNet-AFNO  (392M)  — val sample, physical units",
            out_path=out_dir / "eval_row_det_pixel.png",
            tag="det",
        )
    else:
        print(f"[skip] det_pixel  (ckpt={'missing' if not DET_CKPT.exists() else 'skipped'})")

    # ── Bridge models ─────────────────────────────────────────────────────────
    if not args.skip_bridge:
        for name, ckpt, cfg_yaml in [
            ("unidb", UNIDB_CKPT, ROOT / "diffusion_bridge/configs/unidb_bridge_pde_512_big.yaml"),
            ("frac",  FRAC_CKPT,  ROOT / "diffusion_bridge/configs/frac_bridge_pde_512_big.yaml"),
        ]:
            if not ckpt.exists():
                print(f"[skip] bridge_{name}  (checkpoint missing)")
                continue
            print(f"\n[bridge_{name}] …")
            import yaml
            cfg = yaml.safe_load(cfg_yaml.read_text())
            sde_cfg = cfg["sde"]
            bridge_type = cfg["bridge_type"]

            if bridge_type == "unidb":
                from diffusion_bridge.sde.unidb_sde import UniDBSDE
                sde = UniDBSDE(
                    lambda_sq=sde_cfg["lambda_sq"],
                    gamma=sde_cfg["gamma"],
                    T=sde_cfg["T"],
                    schedule=sde_cfg.get("schedule", "cosine"),
                    eps=sde_cfg.get("eps", 0.01),
                )
            else:
                from diffusion_bridge.sde.frac_bridge_sde import FracBridgeSDE
                sde = FracBridgeSDE(
                    H=sde_cfg["H"],
                    sigma_max=sde_cfg["sigma_max"],
                    T=sde_cfg["T"],
                )

            model = build_bridge(str(ckpt), device)
            x_pred = run_bridge(model, sde, x_src, x_tgt, theta, n_steps=args.n_steps)
            del model; torch.cuda.empty_cache()

            epoch_tag = torch.load(ckpt, map_location="cpu").get("epoch", "?")
            label = ("UniDB" if name == "unidb" else "FracBridge") + f"  (ep {epoch_tag}, {args.n_steps} steps)"
            save_row_bridge(
                x_src, x_tgt, x_pred, theta, norm,
                title=f"Diffusion Bridge — {label}  — val sample, physical units",
                out_path=out_dir / f"eval_row_bridge_{name}.png",
                tag=name,
            )
    else:
        print("[skip] bridge models")

    # ── DC-AE ─────────────────────────────────────────────────────────────────
    if not args.skip_dcae:
        print("\n[dcae] Loading DC-AE sample …")
        x_in, norm_min, norm_scale, norm_dcae = load_dcae_sample(str(val_h5), device)

        for label, ckpt in [("lr2e5", DCAE_LR2E5), ("lr5e6", DCAE_LR5E6)]:
            if not ckpt.exists():
                print(f"[skip] dcae_{label}  (checkpoint missing)")
                continue
            print(f"\n[dcae_{label}] …")
            model = build_dcae(str(ckpt), device)
            x_recon = run_dcae(model, x_in)
            del model; torch.cuda.empty_cache()
            save_row_dcae(
                x_in, x_recon, norm_min, norm_scale, norm_dcae,
                title=f"DC-AE f32c32  ({label})  — val sample, physical units",
                out_path=out_dir / f"eval_row_dcae_{label}.png",
                tag=label,
            )
    else:
        print("[skip] dcae")

    print(f"\n[eval_rows] Done.  Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
