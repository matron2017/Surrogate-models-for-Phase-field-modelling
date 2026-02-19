#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core.utils import _load_symbol
from models.train.core import wavelet_weight as ww


def _parse_frames(spec: str) -> List[Tuple[str, int]]:
    """
    Parse frames string like: "sim_0001:0,1,396;sim_0031:236"
    """
    frames: List[Tuple[str, int]] = []
    for block in (s.strip() for s in spec.split(";") if s.strip()):
        gid, idxs = block.split(":")
        for idx in idxs.split(","):
            frames.append((gid.strip(), int(idx.strip())))
    return frames


def _parse_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _get_norm_stats(h5f: h5py.File, normalize_images: bool, normalize_source: str) -> Tuple[np.ndarray, np.ndarray, float, bool] | None:
    if not normalize_images:
        return None
    mean = std = None
    eps = 1e-6
    already = False
    if normalize_source == "file":
        if "channel_mean" in h5f.attrs and "channel_std" in h5f.attrs:
            mean = np.array(h5f.attrs["channel_mean"], dtype=np.float32)
            std = np.array(h5f.attrs["channel_std"], dtype=np.float32)
            eps = float(h5f.attrs.get("zscore_eps_images", 1e-6))
            already = str(h5f.attrs.get("normalization_schema", "")).lower() == "zscore"
    if mean is None or std is None:
        return None
    return mean, std, eps, already


def _apply_norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float, channels: Sequence[int] | None) -> np.ndarray:
    if channels is None:
        m = mean
        s = std
    else:
        m = mean[list(channels)]
        s = std[list(channels)]
    s = np.where(s > 0, s, eps)
    return (arr - m[:, None, None]) / s[:, None, None]


def _apply_denorm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, channels: Sequence[int] | None) -> np.ndarray:
    if channels is None:
        m = mean
        s = std
    else:
        m = mean[list(channels)]
        s = std[list(channels)]
    return arr * s[:, None, None] + m[:, None, None]


def _plot_map(arr: np.ndarray, title: str, out_path: Path, cmap: str = "viridis", vmin=None, vmax=None, dpi: int = 250) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _field_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
    }


def _residual_limits(arr: np.ndarray, mode: str) -> Tuple[float | None, float | None]:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).any():
        return None, None
    m = str(mode).strip().lower()
    if m == "p99":
        abs_pct = float(np.nanpercentile(np.abs(a), 99))
        if np.isfinite(abs_pct) and abs_pct > 0:
            return -abs_pct, abs_pct
        return None, None
    if m == "symmetric":
        amin = float(np.nanmin(a))
        amax = float(np.nanmax(a))
        abs_max = max(abs(amin), abs(amax))
        if np.isfinite(abs_max) and abs_max > 0:
            return -abs_max, abs_max
        return None, None
    if m == "full":
        amin = float(np.nanmin(a))
        amax = float(np.nanmax(a))
        if not np.isfinite(amin) or not np.isfinite(amax):
            return None, None
        if amax <= amin:
            amax = amin + 1e-12
        return amin, amax
    raise ValueError(f"Unsupported residual-vlim mode '{mode}'. Use one of: full, symmetric, p99.")


def _plot_wavelet_band_maps(
    y_raw: np.ndarray,
    *,
    gid: str,
    frame_idx: int,
    channels: Sequence[int],
    out_dir: Path,
    J: int,
    wave: str,
    mode: str,
    dpi: int,
) -> Dict[str, Dict[int, List[str]]]:
    """Plot per-level wavelet energy maps + lowpass energy in a separate folder."""
    if ww.DWTForward is None:
        raise ImportError("pytorch_wavelets is required for band maps.")

    out_dir.mkdir(parents=True, exist_ok=True)
    y_t = torch.from_numpy(y_raw).unsqueeze(0).to(dtype=torch.float32)
    dwt = ww._get_dwt_forward(J, wave, mode, y_t.device, y_t.dtype)
    Yl, Yh = dwt(y_t)  # Yh: list of [B,C,3,Hj,Wj]
    outputs: Dict[str, Dict[int, List[str]]] = {"details": {}, "lowpass": {}}

    # Detail levels
    for lvl, h in enumerate(Yh, start=1):
        # energy per level
        F_j = (h * h).sum(dim=2)  # [B,C,Hj,Wj]
        F_j_up = torch.nn.functional.interpolate(F_j, size=y_t.shape[-2:], mode="bilinear", align_corners=False)
        F_np = F_j_up[0].cpu().numpy()
        for c in channels:
            outputs["details"].setdefault(c, [])
            out_path = out_dir / f"band_detail_L{lvl}_sim_{gid}_frame{frame_idx:04d}_ch{c}.png"
            _plot_map(
                F_np[c],
                f"Detail L{lvl} sim={gid} frame={frame_idx} ch{c}",
                out_path,
                cmap="magma",
                dpi=dpi,
            )
            outputs["details"][c].append(str(out_path))

    # Lowpass energy
    low = (Yl * Yl)
    low_up = torch.nn.functional.interpolate(low, size=y_t.shape[-2:], mode="bilinear", align_corners=False)
    low_np = low_up[0].cpu().numpy()
    for c in channels:
        outputs["lowpass"].setdefault(c, [])
        out_path = out_dir / f"band_lowpass_sim_{gid}_frame{frame_idx:04d}_ch{c}.png"
        _plot_map(
            low_np[c],
            f"Lowpass sim={gid} frame={frame_idx} ch{c}",
            out_path,
            cmap="magma",
            dpi=dpi,
        )
        outputs["lowpass"][c].append(str(out_path))

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--train-h5", required=True, type=Path)
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--frames", required=True, type=str, help="Format: sim_0001:0,1,396;sim_0031:236")
    ap.add_argument("--channels", default="0", type=str, help="Comma-separated channel indices to plot")
    ap.add_argument("--thetas", default="0.92,0.85", type=str, help="Comma-separated theta values for weights")
    ap.add_argument("--beta-w", default=6.0, type=float)
    ap.add_argument("--bp-beta-w", default=None, type=float, help="Bandpass beta_w override")
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--J", default=1, type=int)
    ap.add_argument("--wave", default="haar", type=str)
    ap.add_argument("--mode", default="zero", type=str)
    ap.add_argument("--method", default="quantile", choices=("quantile", "bandpass", "multiband"))
    ap.add_argument("--weight-methods", default=None, type=str,
                    help="Comma-separated list: quantile,bandpass (overrides --method)")
    ap.add_argument("--only-weights", action="store_true", help="Skip GT/residual plots")
    ap.add_argument("--skip-pct-residual", action="store_true", help="Do not output percent residual maps")
    ap.add_argument(
        "--residual-vlim",
        type=str,
        default="full",
        choices=("full", "symmetric", "p99"),
        help="Residual colorbar limits: full=min/max, symmetric=+/-max(abs), p99=+/-p99(abs).",
    )
    ap.add_argument("--phys-scale", action="store_true", help="Convert normalized channels back to physical scale.")
    ap.add_argument("--phase-rescale", action="store_true", help="Map phase channel to [-1,1] for visualization.")
    ap.add_argument("--phase-channel", type=int, default=0, help="Phase channel index for --phase-rescale.")
    ap.add_argument("--phase-pmin", type=float, default=2.0, help="Lower percentile for phase scaling.")
    ap.add_argument("--phase-pmax", type=float, default=98.0, help="Upper percentile for phase scaling.")
    ap.add_argument("--bp-sigma-low", type=float, default=1.5)
    ap.add_argument("--bp-sigma-high", type=float, default=10.0)
    ap.add_argument("--bp-band-sigma", type=float, default=16.0)
    ap.add_argument("--bp-mask-quantile", type=float, default=0.95)
    ap.add_argument("--bp-power", type=float, default=1.5)
    ap.add_argument("--bp-norm-quantile", type=float, default=0.99)
    ap.add_argument("--bp-normalize-mean", action="store_true")
    ap.add_argument("--bp-clip-min", type=float, default=None)
    ap.add_argument("--bp-clip-max", type=float, default=500.0)
    ap.add_argument("--mb-level-weights", type=str, default="1.0,0.7,0.4")
    ap.add_argument("--mb-lowpass-weight", type=float, default=0.2)
    ap.add_argument("--mb-norm-quantile", type=float, default=0.99)
    ap.add_argument("--mb-power", type=float, default=1.5)
    ap.add_argument("--mb-normalize-mean", action="store_true")
    ap.add_argument("--mb-rescale-max", action="store_true", help="Rescale weights so max == beta_w")
    ap.add_argument("--mb-clip-min", type=float, default=None)
    ap.add_argument("--mb-clip-max", type=float, default=500.0)
    ap.add_argument("--mb-combine-norm", action="store_true")
    ap.add_argument("--mb-tag", type=str, default="multiband", help="Suffix tag for multiband outputs")
    ap.add_argument(
        "--bp-variants",
        type=str,
        default=None,
        help=(
            "Optional list of bandpass variants. Format: "
            "'name:sigma_low,sigma_high,band_sigma;name2:...'. "
            "Overrides --bp-sigma-* when provided."
        ),
    )
    ap.add_argument("--dpi", default=250, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--band-maps", action="store_true", help="Export per-level wavelet energy maps")
    ap.add_argument("--combo-weights", action="store_true", help="Combine multiband + bandpass weights")
    ap.add_argument("--combo-rescale-max", action="store_true", help="Rescale combined weights so max == beta_w")
    ap.add_argument(
        "--combo-soft-lowfreq-weight",
        type=float,
        default=0.25,
        help="Low-frequency weight in the soft combo (default: 0.25)",
    )
    ap.add_argument(
        "--combo-soft-highfreq-weight",
        type=float,
        default=0.75,
        help="High-frequency weight in the soft combo (default: 0.75)",
    )
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = cfg["model"]
    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = None
    ckpt_epoch = -1
    best_metric = None
    if not args.only_weights:
        model = ModelClass(**(model_cfg.get("params", {}) or {}))
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        ckpt_epoch = int(state.get("epoch", -1)) if isinstance(state, dict) else -1
        best_metric = state.get("best_metric", None) if isinstance(state, dict) else None
        model.to(device)

    dl_args = cfg.get("dataloader", {}).get("args", {}) or {}
    input_channels = dl_args.get("input_channels", None)
    target_channels = dl_args.get("target_channels", None)
    normalize_images = bool(dl_args.get("normalize_images", False))
    normalize_force = bool(dl_args.get("normalize_force", False))
    normalize_source = str(dl_args.get("normalize_source", "file"))

    if input_channels is None or input_channels == "all":
        input_channels = None
    if target_channels is None or target_channels == "all":
        target_channels = None

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    thetas = _parse_floats(args.thetas)
    if args.weight_methods:
        weight_methods = [m.strip() for m in args.weight_methods.split(",") if m.strip()]
    else:
        weight_methods = [args.method]

    bandpass_variants = None
    if args.bp_variants:
        variants = []
        for item in args.bp_variants.split(";"):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid --bp-variants entry: {item}")
            name, vals = item.split(":", 1)
            parts = [p.strip() for p in vals.split(",") if p.strip()]
            if len(parts) != 3:
                raise ValueError(f"Invalid --bp-variants entry: {item}")
            sigma_low, sigma_high, band_sigma = map(float, parts)
            variants.append(
                {
                    "name": name.strip(),
                    "sigma_low": sigma_low,
                    "sigma_high": sigma_high,
                    "band_sigma": band_sigma,
                }
            )
        if variants:
            bandpass_variants = variants
    frames = _parse_frames(args.frames)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Dict[str, object]] = []
    for gid, frame_idx in frames:
        h5_path = args.train_h5 if gid.startswith("sim_0001") else args.test_h5
        stats = None
        norm_applied = False
        with h5py.File(h5_path, "r") as hf:
            img = hf[gid]["images"][frame_idx]
            x_raw = img if input_channels is None else img[input_channels]
            y_raw = img if target_channels is None else img[target_channels]
            stats = _get_norm_stats(hf, normalize_images, normalize_source)
            if stats is not None:
                mean, std, eps, already = stats
                if normalize_force or not already:
                    x_norm = _apply_norm(x_raw, mean, std, eps, input_channels)
                    y_norm = _apply_norm(y_raw, mean, std, eps, target_channels)
                else:
                    x_norm = x_raw
                    y_norm = y_raw
                norm_applied = True
            else:
                x_norm = x_raw
                y_norm = y_raw

        x_np = np.array(x_norm, copy=True)
        y_np = np.array(y_norm, copy=True)
        residual = None
        field_stats: Dict[str, Dict[str, Dict[str, float]]] = {str(c): {} for c in channels}
        if not args.only_weights:
            x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                yhat = model(x_t)
            yhat_np = np.array(yhat[0].detach().cpu().numpy(), copy=True)

            # Optional conversion to physical units.
            if bool(args.phys_scale) and stats is not None and norm_applied:
                mean, std, _eps, _already = stats
                x_np = _apply_denorm(x_np, mean, std, input_channels)
                y_np = _apply_denorm(y_np, mean, std, target_channels)
                yhat_np = _apply_denorm(yhat_np, mean, std, target_channels)

            # Optional phase remap to [-1, 1] for visual intuition.
            if bool(args.phase_rescale):
                pc = int(args.phase_channel)
                if 0 <= pc < int(y_np.shape[0]) and pc < int(x_np.shape[0]) and pc < int(yhat_np.shape[0]):
                    pool = np.concatenate([x_np[pc].ravel(), y_np[pc].ravel(), yhat_np[pc].ravel()])
                    lo = float(np.nanpercentile(pool, float(args.phase_pmin)))
                    hi = float(np.nanpercentile(pool, float(args.phase_pmax)))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        def _s(a: np.ndarray) -> np.ndarray:
                            n = (a - lo) / (hi - lo)
                            return np.clip(n, 0.0, 1.0) * 2.0 - 1.0
                        x_np[pc] = _s(x_np[pc])
                        y_np[pc] = _s(y_np[pc])
                        yhat_np[pc] = _s(yhat_np[pc])

            residual = yhat_np - y_np

            for c in channels:
                vmin = float(np.nanmin(y_np[c]))
                vmax = float(np.nanmax(y_np[c]))
                mean_gt = float(np.nanmean(y_np[c]))
                mean_pred = float(np.nanmean(yhat_np[c]))
                denom = max(abs(mean_gt), 1e-8)
                mean_pct = 100.0 * (mean_pred - mean_gt) / denom
                mean_line = f"mean GT={mean_gt:.4g} | Pred={mean_pred:.4g} | Δ%={mean_pct:+.2f}%"
                if c < x_np.shape[0]:
                    field_stats[str(c)]["input"] = _field_stats(x_np[c])
                    input_path = args.out_dir / f"input_{gid}_frame{frame_idx:04d}_ch{c}.png"
                    _plot_map(
                        x_np[c],
                        f"{gid} frame {frame_idx} ch{c} input\n{mean_line}",
                        input_path,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        dpi=args.dpi,
                    )
                gt_path = args.out_dir / f"gt_{gid}_frame{frame_idx:04d}_ch{c}.png"
                field_stats[str(c)]["gt"] = _field_stats(y_np[c])
                _plot_map(
                    y_np[c],
                    f"{gid} frame {frame_idx} ch{c} GT\n{mean_line}",
                    gt_path,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    dpi=args.dpi,
                )
                pred_path = args.out_dir / f"pred_{gid}_frame{frame_idx:04d}_ch{c}.png"
                field_stats[str(c)]["pred"] = _field_stats(yhat_np[c])
                _plot_map(
                    yhat_np[c],
                    f"{gid} frame {frame_idx} ch{c} pred\n{mean_line}",
                    pred_path,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    dpi=args.dpi,
                )
                v0, v1 = _residual_limits(residual[c], args.residual_vlim)
                residual_path = args.out_dir / f"residual_{gid}_frame{frame_idx:04d}_ch{c}.png"
                field_stats[str(c)]["residual"] = _field_stats(residual[c])
                _plot_map(
                    residual[c],
                    f"{gid} frame {frame_idx} ch{c} residual\n{mean_line}",
                    residual_path,
                    cmap="seismic",
                    vmin=v0,
                    vmax=v1,
                    dpi=args.dpi,
                )
                if not args.skip_pct_residual:
                    denom = np.abs(y_np[c]) + 1e-6
                    pct = 100.0 * np.abs(residual[c]) / denom
                    field_stats[str(c)]["pct_residual"] = _field_stats(pct)
                    pct_vmax = float(np.nanpercentile(pct, 99)) if np.isfinite(pct).any() else None
                    pct_path = args.out_dir / f"pct_residual_{gid}_frame{frame_idx:04d}_ch{c}.png"
                    _plot_map(
                        pct,
                        f"{gid} frame {frame_idx} ch{c} % residual\n{mean_line}",
                        pct_path,
                        cmap="magma",
                        vmin=0.0,
                        vmax=pct_vmax,
                        dpi=args.dpi,
                    )

        # Wavelet weights from raw target frame (no normalization)
        x_wave = torch.from_numpy(y_raw).unsqueeze(0).to(dtype=torch.float32)
        weight_paths: Dict[str, Dict[int, str]] = {}
        weight_arrays: Dict[str, np.ndarray] = {}
        band_paths: Dict[str, Dict[int, List[str]]] | None = None
        if "bandpass" in weight_methods:
            try:
                x_wave_dev = x_wave.to(device=device)
            except Exception:
                x_wave_dev = x_wave
            variants = bandpass_variants or [
                {
                    "name": "bandpass",
                    "sigma_low": float(args.bp_sigma_low),
                    "sigma_high": float(args.bp_sigma_high),
                    "band_sigma": float(args.bp_band_sigma),
                }
            ]
            for v in variants:
                with torch.no_grad():
                    w_map, _ = ww.wavelet_bandpass_importance_per_channel(
                        x_wave_dev,
                        J=args.J,
                        wave=args.wave,
                        mode=args.mode,
                        beta_w=float(args.bp_beta_w if args.bp_beta_w is not None else args.beta_w),
                        power=float(args.bp_power),
                        sigma_low=float(v["sigma_low"]),
                        sigma_high=float(v["sigma_high"]),
                        band_sigma=float(v["band_sigma"]),
                        mask_quantile=float(args.bp_mask_quantile),
                        norm_quantile=float(args.bp_norm_quantile),
                        normalize_mean=bool(args.bp_normalize_mean),
                        clip_min=args.bp_clip_min,
                        clip_max=args.bp_clip_max,
                    )
                w_np = w_map[0].detach().cpu().numpy()
                tag = f'bandpass_{v["name"]}'
                weight_paths[tag] = {}
                weight_arrays[tag] = w_np
                for c in channels:
                    weight_path = args.out_dir / f"weight_{gid}_frame{frame_idx:04d}_ch{c}_{tag}.png"
                    _plot_map(
                        w_np[c],
                        f"{gid} frame {frame_idx} ch{c} {tag}",
                        weight_path,
                        cmap="magma",
                        dpi=args.dpi,
                    )
                    weight_paths[tag][c] = str(weight_path)
        if "quantile" in weight_methods:
            for theta in thetas:
                try:
                    x_wave_dev = x_wave.to(device=device)
                except Exception:
                    x_wave_dev = x_wave
                with torch.no_grad():
                    w_map, _ = ww.wavelet_importance_per_channel(
                        x_wave_dev,
                        J=args.J,
                        wave=args.wave,
                        mode=args.mode,
                        theta=float(theta),
                        alpha=float(args.alpha),
                        beta_w=float(args.beta_w),
                    )
                w_np = w_map[0].detach().cpu().numpy()
                tag = f"theta{theta:.2f}".replace(".", "")
                weight_paths[tag] = {}
                weight_arrays[tag] = w_np
                for c in channels:
                    weight_path = args.out_dir / f"weight_{gid}_frame{frame_idx:04d}_ch{c}_{tag}.png"
                    _plot_map(
                        w_np[c],
                        f"{gid} frame {frame_idx} ch{c} weight theta={theta:.2f}",
                        weight_path,
                        cmap="magma",
                        dpi=args.dpi,
                    )
                    weight_paths[tag][c] = str(weight_path)

        if "multiband" in weight_methods:
            try:
                x_wave_dev = x_wave.to(device=device)
            except Exception:
                x_wave_dev = x_wave
            level_weights = [float(x) for x in args.mb_level_weights.split(",") if x.strip()]
            with torch.no_grad():
                w_map, _ = ww.wavelet_multiband_importance_per_channel(
                    x_wave_dev,
                    J=args.J,
                    wave=args.wave,
                    mode=args.mode,
                    level_weights=level_weights,
                    lowpass_weight=float(args.mb_lowpass_weight),
                    beta_w=float(args.bp_beta_w if args.bp_beta_w is not None else args.beta_w),
                    power=float(args.mb_power),
                    norm_quantile=float(args.mb_norm_quantile),
                    normalize_mean=bool(args.mb_normalize_mean),
                    rescale_max=bool(args.mb_rescale_max),
                    clip_min=args.mb_clip_min,
                    clip_max=args.mb_clip_max,
                    combine_norm=bool(args.mb_combine_norm),
                )
            w_np = w_map[0].detach().cpu().numpy()
            tag = str(args.mb_tag)
            weight_paths[tag] = {}
            weight_arrays[tag] = w_np
            for c in channels:
                weight_path = args.out_dir / f"weight_{gid}_frame{frame_idx:04d}_ch{c}_{tag}.png"
                _plot_map(
                    w_np[c],
                    f"{gid} frame {frame_idx} ch{c} {tag}",
                    weight_path,
                    cmap="magma",
                    dpi=args.dpi,
                )
                weight_paths[tag][c] = str(weight_path)

        if args.combo_weights and weight_arrays:
            # Combine multiband (low/mid) with a bandpass (high freq) variant if available.
            mb_tag = str(args.mb_tag)
            bp_tag = None
            for tag in weight_arrays.keys():
                if tag.startswith("bandpass_highfreq"):
                    bp_tag = tag
                    break
            if bp_tag is None:
                for tag in weight_arrays.keys():
                    if tag.startswith("bandpass_"):
                        bp_tag = tag
                        break
            if bp_tag is not None and mb_tag in weight_arrays:
                w_mb = weight_arrays[mb_tag]
                w_bp = weight_arrays[bp_tag]
                w_combo = 1.0 + 0.5 * ((w_mb - 1.0) + (w_bp - 1.0))
                if args.combo_rescale_max:
                    vmax = float(np.nanmax(w_combo))
                    if np.isfinite(vmax) and vmax > 0:
                        w_combo = w_combo / vmax * float(args.beta_w)
                w_combo = np.clip(w_combo, 1.0, float(args.beta_w))
                combo_tag = f"combo_{bp_tag}_{mb_tag}"
                weight_paths[combo_tag] = {}
                weight_arrays[combo_tag] = w_combo
                for c in channels:
                    weight_path = args.out_dir / f"weight_{gid}_frame{frame_idx:04d}_ch{c}_{combo_tag}.png"
                    _plot_map(
                        w_combo[c],
                        f"{gid} frame {frame_idx} ch{c} {combo_tag}",
                        weight_path,
                        cmap="magma",
                        dpi=args.dpi,
                    )
                    weight_paths[combo_tag][c] = str(weight_path)
                # Optional: softer low-frequency blend if a lowfreq multiband exists.
                low_mb_tag = "multiband_lowfreq"
                if low_mb_tag in weight_arrays:
                    w_mb_low = weight_arrays[low_mb_tag]
                    low_w = float(args.combo_soft_lowfreq_weight)
                    high_w = float(args.combo_soft_highfreq_weight)
                    w_soft = 1.0 + low_w * (w_mb_low - 1.0) + high_w * (w_bp - 1.0)
                    if args.combo_rescale_max:
                        vmax = float(np.nanmax(w_soft))
                        if np.isfinite(vmax) and vmax > 0:
                            w_soft = w_soft / vmax * float(args.beta_w)
                    w_soft = np.clip(w_soft, 1.0, float(args.beta_w))
                    soft_tag = f"combo_{bp_tag}_{low_mb_tag}_soft"
                    weight_paths[soft_tag] = {}
                    weight_arrays[soft_tag] = w_soft
                    for c in channels:
                        weight_path = args.out_dir / f"weight_{gid}_frame{frame_idx:04d}_ch{c}_{soft_tag}.png"
                        _plot_map(
                            w_soft[c],
                            f"{gid} frame {frame_idx} ch{c} {soft_tag}",
                            weight_path,
                            cmap="magma",
                            dpi=args.dpi,
                        )
                        weight_paths[soft_tag][c] = str(weight_path)

        if args.band_maps:
            band_dir = args.out_dir / "band_maps"
            band_paths = _plot_wavelet_band_maps(
                y_raw=y_raw,
                gid=gid,
                frame_idx=frame_idx,
                channels=channels,
                out_dir=band_dir,
                J=int(args.J),
                wave=str(args.wave),
                mode=str(args.mode),
                dpi=int(args.dpi),
            )

        outputs.append(
            {
                "gid": gid,
                "frame": frame_idx,
                "channels": channels,
                "input": None if args.only_weights else {c: str(args.out_dir / f"input_{gid}_frame{frame_idx:04d}_ch{c}.png") for c in channels},
                "gt": None if args.only_weights else {c: str(args.out_dir / f"gt_{gid}_frame{frame_idx:04d}_ch{c}.png") for c in channels},
                "pred": None if args.only_weights else {c: str(args.out_dir / f"pred_{gid}_frame{frame_idx:04d}_ch{c}.png") for c in channels},
                "residual": None if args.only_weights else {c: str(args.out_dir / f"residual_{gid}_frame{frame_idx:04d}_ch{c}.png") for c in channels},
                "pct_residual": None
                if (args.only_weights or args.skip_pct_residual)
                else {c: str(args.out_dir / f"pct_residual_{gid}_frame{frame_idx:04d}_ch{c}.png") for c in channels},
                "field_stats": field_stats,
                "weights": weight_paths,
                "band_maps": band_paths,
            }
        )

    metadata = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_metric": best_metric,
        "frames": args.frames,
        "channels": channels,
        "thetas": thetas,
        "beta_w": float(args.beta_w),
        "alpha": float(args.alpha),
        "J": int(args.J),
        "wave": str(args.wave),
        "mode": str(args.mode),
        "method": str(args.method),
        "weight_methods": weight_methods,
        "only_weights": bool(args.only_weights),
        "skip_pct_residual": bool(args.skip_pct_residual),
        "phys_scale": bool(args.phys_scale),
        "phase_rescale": bool(args.phase_rescale),
        "phase_channel": int(args.phase_channel),
        "phase_pmin": float(args.phase_pmin),
        "phase_pmax": float(args.phase_pmax),
        "bandpass": {
            "beta_w": float(args.bp_beta_w) if args.bp_beta_w is not None else float(args.beta_w),
            "sigma_low": float(args.bp_sigma_low),
            "sigma_high": float(args.bp_sigma_high),
            "band_sigma": float(args.bp_band_sigma),
            "mask_quantile": float(args.bp_mask_quantile),
            "power": float(args.bp_power),
            "norm_quantile": float(args.bp_norm_quantile),
            "normalize_mean": bool(args.bp_normalize_mean),
            "clip_min": args.bp_clip_min,
            "clip_max": args.bp_clip_max,
        },
        "multiband": {
            "level_weights": [float(x) for x in args.mb_level_weights.split(",") if x.strip()],
            "lowpass_weight": float(args.mb_lowpass_weight),
            "norm_quantile": float(args.mb_norm_quantile),
            "power": float(args.mb_power),
            "normalize_mean": bool(args.mb_normalize_mean),
            "rescale_max": bool(args.mb_rescale_max),
            "clip_min": args.mb_clip_min,
            "clip_max": args.mb_clip_max,
            "combine_norm": bool(args.mb_combine_norm),
        },
        "bandpass_variants": bandpass_variants,
        "outputs": outputs,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
