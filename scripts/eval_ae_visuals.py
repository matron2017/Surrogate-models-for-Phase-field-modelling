#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core.utils import _load_symbol, _collate
from models.train.core.pf_dataloader import PFPairDataset


def _resolve_h5_path(cfg: Dict[str, Any], split: str) -> str:
    entry = cfg.get("paths", {}).get("h5", {}).get(split)
    if entry is None:
        raise KeyError(f"Missing paths.h5.{split} in config.")
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if "h5_path" in entry:
            return entry["h5_path"]
        if len(entry) == 1:
            k, v = next(iter(entry.items()))
            if isinstance(v, str) and v:
                return v
            return str(k)
    raise ValueError(f"Unsupported paths.h5.{split} entry: {entry!r}")


def _filter_pf_args(args: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "h5_path",
        "input_channels",
        "target_channels",
        "limit_per_group",
        "max_items",
        "weight_h5",
        "weight_key",
        "identity_pairs",
        "use_pairs_idx",
        "data_key",
        "return_cond",
        "add_thermal",
        "thermal_axis",
        "thermal_use_x0",
        "thermal_T0",
        "thermal_on_target",
        "thermal_debug",
        "thermal_debug_prob",
        "augment",
        "augment_flip",
        "augment_flip_prob",
        "augment_roll",
        "augment_roll_prob",
        "augment_roll_max",
        "augment_swap",
        "augment_swap_prob",
        "augment_rotate",
        "augment_rotate_prob",
        "normalize_images",
        "normalize_force",
        "normalize_source",
    }
    return {k: v for k, v in args.items() if k in allowed}


def _build_dataset(cfg: Dict[str, Any], split: str, h5_override: str | None = None) -> PFPairDataset:
    dl_cfg = cfg.get("dataloader", {}) or {}
    args = dict(dl_cfg.get("args", {}) or {})
    split_args = dict(dl_cfg.get(f"{split}_args", {}) or {})
    args.update(split_args)
    args["h5_path"] = h5_override or _resolve_h5_path(cfg, split)
    return PFPairDataset(**_filter_pf_args(args))


def _load_model(ckpt_path: Path, cfg_fallback: Dict[str, Any]) -> torch.nn.Module:
    try:
        from collections import defaultdict
        def trust_region_clip_(*args, **kwargs):
            return None
        trust_region_clip_.__module__ = "heavyball.utils"
        class ForeachPSGDKron:  # pragma: no cover - stub for checkpoint loading
            pass
        class ForeachCachedPSGDKron:  # pragma: no cover
            pass
        class ForeachDelayedPSGD:  # pragma: no cover
            pass
        class ForeachCachedDelayedPSGDKron:  # pragma: no cover
            pass
        ForeachPSGDKron.__module__ = "heavyball"
        ForeachCachedPSGDKron.__module__ = "heavyball"
        ForeachDelayedPSGD.__module__ = "heavyball"
        ForeachCachedDelayedPSGDKron.__module__ = "heavyball"
        safe_list = [
            defaultdict,
            dict,
            trust_region_clip_,
            ForeachPSGDKron,
            ForeachCachedPSGDKron,
            ForeachDelayedPSGD,
            ForeachCachedDelayedPSGDKron,
        ]
        with torch.serialization.safe_globals(safe_list):
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                # Fallback for older checkpoints requiring full unpickling.
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    cfg = cfg_fallback
    model_cfg = cfg.get("model") or cfg_fallback.get("model")
    if not model_cfg:
        raise KeyError("Model config missing from checkpoint and config file.")
    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    model = ModelClass(**(model_cfg.get("params", {}) or {}))
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _select_indices(n_items: int, count: int) -> List[int]:
    count = max(1, min(count, n_items))
    if count == 1:
        return [0]
    idx = np.linspace(0, n_items - 1, count, dtype=int)
    return [int(i) for i in idx.tolist()]


def _safe_vminmax(*arrays: np.ndarray) -> Tuple[float | None, float | None]:
    vals = np.concatenate([a.ravel() for a in arrays if a.size > 0])
    if vals.size == 0:
        return None, None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None, None
    return vmin, vmax


def _load_h5_norm_stats(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if "channel_mean" not in f.attrs or "channel_std" not in f.attrs:
            raise RuntimeError("H5 root missing channel_mean/channel_std attrs.")
        mean = np.array(f.attrs["channel_mean"], dtype=np.float32)
        std = np.array(f.attrs["channel_std"], dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise RuntimeError("channel_mean/std must be 1D arrays with same shape.")
    return mean, std


def _stats_text(arr: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> str:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return "min: nan\nmax: nan\nmean: nan"
    vmin_val = float(np.min(a)) if vmin is None else float(vmin)
    vmax_val = float(np.max(a)) if vmax is None else float(vmax)
    mean = float(np.mean(a))
    return f"min: {vmin_val:.4g}\nmax: {vmax_val:.4g}\nmean: {mean:.4g}"


def _annotate_stats(
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    fontsize: int = 9,
) -> None:
    ax.text(
        0.98,
        0.98,
        _stats_text(arr, vmin=vmin, vmax=vmax),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6),
    )


def _autocorr2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.mean(x)
    f = np.fft.fft2(x)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    norm = ac.max()
    if np.isfinite(norm) and norm > 0:
        ac = ac / norm
    return ac


def _power_spectrum(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    f = np.fft.fft2(x)
    p = np.abs(f) ** 2
    p = np.fft.fftshift(p)
    return np.log1p(p)


def _plot_pair_map(
    out_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    dpi: int,
    figscale: float,
    cmap: str = "viridis",
) -> Dict[str, Any]:
    diff = np.abs(pred - gt)
    vmin, vmax = _safe_vminmax(gt, pred)
    dmin, dmax = _safe_vminmax(diff)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3.6 * 3 * figscale, 3.2 * figscale), squeeze=False)
    for ax, data, label, v0, v1 in [
        (axes[0, 0], gt, "gt", vmin, vmax),
        (axes[0, 1], pred, "pred", vmin, vmax),
        (axes[0, 2], diff, "abs diff", dmin, dmax),
    ]:
        im = ax.imshow(data, origin="lower", vmin=v0, vmax=v1, cmap=cmap)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return {"abs_diff": diff}


def _plot_single_map(
    out_path: Path,
    data: np.ndarray,
    title: str,
    dpi: int,
    figscale: float,
    cmap: str = "viridis",
) -> None:
    vmin, vmax = _safe_vminmax(data)
    fig, ax = plt.subplots(figsize=(3.6 * figscale, 3.2 * figscale))
    im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_sample(
    out_path: Path,
    x_np: np.ndarray,
    y_np: np.ndarray,
    yhat_np: np.ndarray,
    eps: float,
    dpi: int,
    title: str,
    mode: str,
    figscale: float,
    panel_stats: bool,
    *,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    stats_fontsize: int = 9,
) -> Dict[str, Any]:
    # Phase field (channel 0) rescaled to [-1, 1] for plotting only.
    phase_vmin = None
    phase_vmax = None
    phase_scaled = None
    if y_np.shape[0] > 0:
        phase_pool = np.concatenate(
            [x_np[0].ravel(), y_np[0].ravel(), yhat_np[0].ravel()]
        )
        if phase_pool.size:
            lo = float(np.nanpercentile(phase_pool, 2))
            hi = float(np.nanpercentile(phase_pool, 98))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                phase_vmin, phase_vmax = lo, hi

        def _scale_phase(arr: np.ndarray) -> np.ndarray:
            if phase_vmin is None or phase_vmax is None:
                return arr
            norm = (arr - phase_vmin) / (phase_vmax - phase_vmin)
            return np.clip(norm, 0.0, 1.0) * 2.0 - 1.0

        phase_scaled = (
            _scale_phase(x_np[0]),
            _scale_phase(y_np[0]),
            _scale_phase(yhat_np[0]),
        )

    resid = yhat_np - y_np
    err = np.abs(resid)
    pct = err / (np.abs(y_np) + eps) * 100.0

    C = y_np.shape[0]
    mode = str(mode).lower()
    if mode == "input_pred":
        cols = ["input", "pred"]
    elif mode == "input_pred_residual":
        cols = ["input", "pred", "residual"]
    elif mode == "input_pred_err":
        cols = ["input", "pred", "abs err", "% err"]
    else:
        cols = ["input", "gt", "pred", "abs err", "% err"]
    fig_w = 3.4 * len(cols) * figscale
    fig_h = 3.0 * C * figscale
    fig, axes = plt.subplots(nrows=C, ncols=len(cols), figsize=(fig_w, fig_h), squeeze=False)

    for c in range(C):
        if c == 0 and phase_scaled is not None:
            x_plot, y_plot, yhat_plot = phase_scaled
            resid_plot = yhat_plot - y_plot
            err_plot = np.abs(resid_plot)
            vmin, vmax = -1.0, 1.0
            err_vmin, err_vmax = _safe_vminmax(err_plot)
        else:
            x_plot, y_plot, yhat_plot = x_np[c], y_np[c], yhat_np[c]
            resid_plot = resid[c]
            err_plot = err[c]
            vmin, vmax = _safe_vminmax(x_plot, y_plot, yhat_plot)
            err_vmin, err_vmax = _safe_vminmax(err_plot)
        pct_vmin, pct_vmax = _safe_vminmax(pct[c])
        if pct_vmax is not None:
            pct_vmax = float(np.nanpercentile(pct[c], 99)) if np.isfinite(pct_vmax) else pct_vmax
        for j, label in enumerate(cols):
            ax = axes[c, j]
            if label == "input":
                img = x_plot; v0, v1 = vmin, vmax; cmap = "viridis"
            elif label == "gt":
                img = y_plot; v0, v1 = vmin, vmax; cmap = "viridis"
            elif label == "pred":
                img = yhat_plot; v0, v1 = vmin, vmax; cmap = "viridis"
            elif label == "residual":
                img = resid_plot
                abs_pct = np.nanpercentile(np.abs(img), 99) if img.size else 0.0
                v0, v1 = (-abs_pct, abs_pct) if np.isfinite(abs_pct) and abs_pct > 0 else (None, None)
                cmap = "seismic"
            elif label == "abs err":
                img = err_plot; v0, v1 = err_vmin, err_vmax; cmap = "magma"
            else:
                img = pct[c]; v0, v1 = pct_vmin, pct_vmax; cmap = "magma"
            im = ax.imshow(img, origin="lower", vmin=v0, vmax=v1, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(label, fontsize=label_fontsize, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=max(8, label_fontsize - 2))
            if panel_stats:
                _annotate_stats(ax, img, vmin=v0, vmax=v1, fontsize=stats_fontsize)

    fig.suptitle(title, fontsize=title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

    return {"abs_err": err, "pct_err": pct, "residual": resid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pct-eps", type=float, default=1e-3)
    ap.add_argument("--indices", type=str, default="")
    ap.add_argument("--h5-path", type=str, default="")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--plot-mode", type=str, default="input_pred", choices=["input_pred", "input_pred_err", "input_pred_residual", "full"])
    ap.add_argument("--figscale", type=float, default=2.0)
    ap.add_argument("--gid", type=str, default="")
    ap.add_argument("--group-index", type=int, default=-1)
    ap.add_argument("--skip-first", action="store_true")
    ap.add_argument("--sample-strategy", type=str, default="linspace", choices=["linspace", "random"])
    ap.add_argument("--phys-scale", dest="phys_scale", action="store_true", default=True)
    ap.add_argument("--no-phys-scale", dest="phys_scale", action="store_false")
    ap.add_argument("--phi-rescale", dest="phi_rescale", action="store_true", default=True)
    ap.add_argument("--no-phi-rescale", dest="phi_rescale", action="store_false")
    ap.add_argument("--panel-stats", dest="panel_stats", action="store_true", default=True)
    ap.add_argument("--no-panel-stats", dest="panel_stats", action="store_false")
    ap.add_argument("--residual-autocorr", dest="residual_autocorr", action="store_true", default=True)
    ap.add_argument("--no-residual-autocorr", dest="residual_autocorr", action="store_false")
    ap.add_argument("--residual-spectrum", dest="residual_spectrum", action="store_true", default=True)
    ap.add_argument("--no-residual-spectrum", dest="residual_spectrum", action="store_false")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ckpt_path.parent / "eval_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(ckpt_path, cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    h5_override = args.h5_path.strip() or None
    ds = _build_dataset(cfg, args.split, h5_override=h5_override)
    h5_path = h5_override or _resolve_h5_path(cfg, args.split)
    ch_mean = None
    ch_std = None
    if args.phys_scale:
        ch_mean, ch_std = _load_h5_norm_stats(h5_path)

    gid = args.gid.strip()
    if not gid and args.group_index >= 0:
        gids = sorted(ds.shapes.keys())
        if args.group_index >= len(gids):
            raise ValueError(f"group-index {args.group_index} out of range (num groups={len(gids)})")
        gid = gids[args.group_index]
    if gid:
        ds.items = [item for item in ds.items if item[0] == gid]
        if not ds.items:
            raise RuntimeError(f"No items found for gid {gid}")
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    if args.indices:
        idx = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        start = 1 if args.skip_first else 0
        if len(ds) <= start:
            raise RuntimeError("Dataset too small after skip_first.")
        if args.sample_strategy == "random":
            rng = np.random.default_rng(0)
            idx = rng.choice(np.arange(start, len(ds)), size=min(args.num_samples, len(ds) - start), replace=False).tolist()
        else:
            idx = _select_indices(len(ds) - start, args.num_samples)
            idx = [i + start for i in idx]

    h5_handle = h5py.File(h5_path, "r")
    results = []
    for k, i in enumerate(idx, start=1):
        sample = ds[i]
        x = sample["input"].unsqueeze(0).to(device)
        y = sample["target"].unsqueeze(0).to(device)
        cond = sample.get("cond")
        cond = cond.unsqueeze(0).to(device) if cond is not None else None
        with torch.no_grad():
            yhat = model(x, cond) if cond is not None else model(x)

        x_np = x[0].detach().cpu().numpy()
        y_np = y[0].detach().cpu().numpy()
        yhat_np = yhat[0].detach().cpu().numpy()
        if args.phys_scale:
            x_np = x_np * ch_std[:, None, None] + ch_mean[:, None, None]
            y_np = y_np * ch_std[:, None, None] + ch_mean[:, None, None]
            yhat_np = yhat_np * ch_std[:, None, None] + ch_mean[:, None, None]

        gid = sample.get("gid")
        pair_index = int(sample.get("pair_index", -1))
        stride = None
        G_raw = float("nan")
        if gid in h5_handle:
            grp = h5_handle[gid]
            if "pairs_dt_euler" in grp:
                stride = int(grp["pairs_dt_euler"][pair_index])
            elif "pairs_stride" in grp:
                stride = int(grp["pairs_stride"][pair_index])
            elif "pairs_idx" in grp:
                i0, i1 = grp["pairs_idx"][pair_index]
                stride = int(i1 - i0)
            G_raw = float(grp.attrs.get("thermal_gradient_raw", grp.attrs.get("thermal_gradient", float("nan"))))

        phi_rscld_mean_gt = None
        phi_rscld_mean_pred = None
        conc_mean_gt = None
        conc_mean_pred = None
        if y_np.shape[0] > 0:
            phi_rscld_mean_gt = float(np.mean(0.5 * (y_np[0] + 1.0)))
            phi_rscld_mean_pred = float(np.mean(0.5 * (yhat_np[0] + 1.0)))
        if y_np.shape[0] > 1:
            conc_mean_gt = float(np.mean(y_np[1]))
            conc_mean_pred = float(np.mean(yhat_np[1]))

        title_lines = [
            f"Thermal Grad={G_raw:.3e}  Euler step={stride if stride is not None else 'NA'}  pair={pair_index}  idx={i:05d}"
        ]
        if phi_rscld_mean_gt is not None and phi_rscld_mean_pred is not None:
            title_lines.append(
                f"phi_rscld mean GT={phi_rscld_mean_gt:.4f} | Pred={phi_rscld_mean_pred:.4f} (phase plots mapped to [-1,1])"
            )
        if conc_mean_gt is not None and conc_mean_pred is not None:
            title_lines.append(
                f"conc mean GT={conc_mean_gt:.4g} | Pred={conc_mean_pred:.4g}"
            )
        title = "\n".join(title_lines)
        out_path = out_dir / f"sample_{k:02d}_idx_{i:05d}.png"
        stats = _plot_sample(
            out_path,
            x_np,
            y_np,
            yhat_np,
            eps=args.pct_eps,
            dpi=args.dpi,
            title=title,
            mode=args.plot_mode,
            figscale=args.figscale,
            panel_stats=args.panel_stats,
            title_fontsize=14,
            label_fontsize=12,
            stats_fontsize=9,
        )

        err = stats["abs_err"]
        pct = stats["pct_err"]
        resid = stats["residual"]
        rmse = np.sqrt(np.mean((yhat_np - y_np) ** 2, axis=(1, 2))).tolist()
        mae = np.mean(np.abs(yhat_np - y_np), axis=(1, 2)).tolist()
        mape = (np.mean(pct, axis=(1, 2))).tolist()
        gt_min = np.min(y_np, axis=(1, 2)).tolist()
        gt_max = np.max(y_np, axis=(1, 2)).tolist()
        gt_mean = np.mean(y_np, axis=(1, 2)).tolist()
        pred_min = np.min(yhat_np, axis=(1, 2)).tolist()
        pred_max = np.max(yhat_np, axis=(1, 2)).tolist()
        pred_mean = np.mean(yhat_np, axis=(1, 2)).tolist()
        res_min = np.min(resid, axis=(1, 2)).tolist()
        res_max = np.max(resid, axis=(1, 2)).tolist()
        res_mean = np.mean(resid, axis=(1, 2)).tolist()

        if args.phi_rescale and y_np.shape[0] > 0:
            phi_rscld_mean_gt = float(np.mean(0.5 * (y_np[0] + 1.0)))
            phi_rscld_mean_pred = float(np.mean(0.5 * (yhat_np[0] + 1.0)))

        autocorr_mse = []
        spectrum_mse = []
        autocorr_imgs = []
        spectrum_imgs = []
        autocorr_residual_imgs = []
        spectrum_residual_imgs = []
        for ch in range(y_np.shape[0]):
            ac_gt = _autocorr2d(y_np[ch])
            ac_pred = _autocorr2d(yhat_np[ch])
            sp_gt = _power_spectrum(y_np[ch])
            sp_pred = _power_spectrum(yhat_np[ch])
            autocorr_mse.append(float(np.mean((ac_pred - ac_gt) ** 2)))
            spectrum_mse.append(float(np.mean((sp_pred - sp_gt) ** 2)))
            ac_path = out_dir / f"sample_{k:02d}_idx_{i:05d}_ch{ch}_autocorr.png"
            sp_path = out_dir / f"sample_{k:02d}_idx_{i:05d}_ch{ch}_spectrum.png"
            _plot_pair_map(
                ac_path,
                ac_gt,
                ac_pred,
                title=f"autocorr ch={ch} idx={i}",
                dpi=args.dpi,
                figscale=args.figscale,
                cmap="viridis",
            )
            _plot_pair_map(
                sp_path,
                sp_gt,
                sp_pred,
                title=f"spectrum ch={ch} idx={i}",
                dpi=args.dpi,
                figscale=args.figscale,
                cmap="magma",
            )
            autocorr_imgs.append(str(ac_path.name))
            spectrum_imgs.append(str(sp_path.name))
            if args.residual_autocorr:
                ac_res = _autocorr2d(resid[ch])
                ac_res_path = out_dir / f"sample_{k:02d}_idx_{i:05d}_ch{ch}_autocorr_residual.png"
                _plot_single_map(
                    ac_res_path,
                    ac_res,
                    title=f"autocorr residual ch={ch} idx={i}",
                    dpi=args.dpi,
                    figscale=args.figscale,
                    cmap="viridis",
                )
                autocorr_residual_imgs.append(str(ac_res_path.name))
            if args.residual_spectrum:
                sp_res = _power_spectrum(resid[ch])
                sp_res_path = out_dir / f"sample_{k:02d}_idx_{i:05d}_ch{ch}_spectrum_residual.png"
                _plot_single_map(
                    sp_res_path,
                    sp_res,
                    title=f"spectrum residual ch={ch} idx={i}",
                    dpi=args.dpi,
                    figscale=args.figscale,
                    cmap="magma",
                )
                spectrum_residual_imgs.append(str(sp_res_path.name))
        results.append({
            "index": int(i),
            "gid": sample.get("gid"),
            "pair_index": int(sample.get("pair_index", -1)),
            "rmse": rmse,
            "mae": mae,
            "mape_percent": mape,
            "gt_min": gt_min,
            "gt_max": gt_max,
            "gt_mean": gt_mean,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "pred_mean": pred_mean,
            "residual_min": res_min,
            "residual_max": res_max,
            "residual_mean": res_mean,
            "phi_rscld_mean_gt": phi_rscld_mean_gt,
            "phi_rscld_mean_pred": phi_rscld_mean_pred,
            "autocorr_mse": autocorr_mse,
            "spectrum_mse": spectrum_mse,
            "autocorr_images": autocorr_imgs,
            "spectrum_images": spectrum_imgs,
            "autocorr_residual_images": autocorr_residual_imgs,
            "spectrum_residual_images": spectrum_residual_imgs,
            "image": out_path.name,
        })

    h5_handle.close()
    summary = {
        "checkpoint": str(ckpt_path),
        "config": str(args.config),
        "split": args.split,
        "num_samples": len(results),
        "phys_scale": bool(args.phys_scale),
        "phi_rescale": bool(args.phi_rescale),
        "panel_stats": bool(args.panel_stats),
        "residual_autocorr": bool(args.residual_autocorr),
        "residual_spectrum": bool(args.residual_spectrum),
        "results": results,
    }
    (out_dir / "metrics_samples.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved {len(results)} visuals to {out_dir}")
    for r in results:
        print(f"idx={r['index']} rmse={r['rmse']} mape%={r['mape_percent']} file={r['image']}")


if __name__ == "__main__":
    main()
