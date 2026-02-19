#!/usr/bin/env python3
"""
Visualize AE wavelet-weight strategies on specific frames/channels.

Creates one composite PNG per (frame, channel) on the same samples used in AE comparisons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core import wavelet_weight as ww  # noqa: E402


@dataclass
class Strategy:
    name: str
    method: str
    params: Dict[str, object]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _stats(arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float64)
    return f"min={np.min(a):.4g}\nmax={np.max(a):.4g}\nmean={np.mean(a):.4g}"


def _default_strategies(J: int, beta_w: float) -> List[Strategy]:
    return [
        Strategy(
            name="multiband_lowfreq_strong",
            method="multiband",
            params={
                "J": J,
                "beta_w": beta_w,
                "level_weights": [0.8, 0.5, 0.35],
                "lowpass_weight": 2.8,
                "power": 1.15,
                "norm_quantile": 0.99,
                "normalize_mean": True,
                "rescale_max": False,
                "clip_min": None,
                "clip_max": 250.0,
                "combine_norm": True,
            },
        ),
        Strategy(
            name="multiband_lowfreq_mild",
            method="multiband",
            params={
                "J": J,
                "beta_w": beta_w,
                "level_weights": [0.9, 0.7, 0.5],
                "lowpass_weight": 1.8,
                "power": 1.2,
                "norm_quantile": 0.99,
                "normalize_mean": True,
                "rescale_max": False,
                "clip_min": None,
                "clip_max": 250.0,
                "combine_norm": True,
            },
        ),
        Strategy(
            name="multiband_balanced_allfreq",
            method="multiband",
            params={
                "J": J,
                "beta_w": beta_w,
                "level_weights": [1.0, 1.0, 1.0],
                "lowpass_weight": 1.0,
                "power": 1.3,
                "norm_quantile": 0.99,
                "normalize_mean": True,
                "rescale_max": False,
                "clip_min": None,
                "clip_max": 250.0,
                "combine_norm": True,
            },
        ),
        Strategy(
            name="quantile_baseline",
            method="quantile",
            params={
                "J": J,
                "theta": 0.85,
                "alpha": 1.1,
                "beta_w": beta_w,
            },
        ),
        Strategy(
            name="bandpass_highfreq_baseline",
            method="bandpass",
            params={
                "J": J,
                "beta_w": beta_w,
                "power": 1.5,
                "sigma_low": 1.5,
                "sigma_high": 10.0,
                "band_sigma": 16.0,
                "mask_quantile": 0.95,
                "norm_quantile": 0.99,
                "normalize_mean": True,
                "clip_min": None,
                "clip_max": 250.0,
            },
        ),
    ]


def _compute_weights(y: torch.Tensor, s: Strategy, wave: str, mode: str) -> np.ndarray:
    p = s.params
    if s.method == "multiband":
        w, _ = ww.wavelet_multiband_importance_per_channel(
            y,
            J=int(p.get("J", 3)),
            wave=wave,
            mode=mode,
            level_weights=[float(x) for x in p.get("level_weights", [1.0, 1.0, 1.0])],
            lowpass_weight=float(p.get("lowpass_weight", 1.0)),
            beta_w=float(p.get("beta_w", 40.0)),
            power=float(p.get("power", 1.2)),
            norm_quantile=float(p.get("norm_quantile", 0.99)),
            normalize_mean=bool(p.get("normalize_mean", True)),
            rescale_max=bool(p.get("rescale_max", False)),
            clip_min=p.get("clip_min", None),
            clip_max=p.get("clip_max", 250.0),
            combine_norm=bool(p.get("combine_norm", True)),
        )
    elif s.method == "quantile":
        w, _ = ww.wavelet_importance_per_channel(
            y,
            J=int(p.get("J", 3)),
            wave=wave,
            mode=mode,
            theta=float(p.get("theta", 0.85)),
            alpha=float(p.get("alpha", 1.1)),
            beta_w=float(p.get("beta_w", 40.0)),
        )
    elif s.method == "bandpass":
        w, _ = ww.wavelet_bandpass_importance_per_channel(
            y,
            J=int(p.get("J", 3)),
            wave=wave,
            mode=mode,
            beta_w=float(p.get("beta_w", 40.0)),
            power=float(p.get("power", 1.5)),
            sigma_low=float(p.get("sigma_low", 1.5)),
            sigma_high=float(p.get("sigma_high", 10.0)),
            band_sigma=float(p.get("band_sigma", 16.0)),
            mask_quantile=float(p.get("mask_quantile", 0.95)),
            norm_quantile=float(p.get("norm_quantile", 0.99)),
            normalize_mean=bool(p.get("normalize_mean", True)),
            clip_min=p.get("clip_min", None),
            clip_max=p.get("clip_max", 250.0),
        )
    else:
        raise ValueError(f"Unsupported strategy method '{s.method}'")
    return w[0, 0].detach().cpu().numpy()


def _render_one(
    *,
    sim_id: str,
    frame: int,
    channel: int,
    field: np.ndarray,
    maps: Dict[str, np.ndarray],
    out_path: Path,
    dpi: int,
) -> None:
    names = list(maps.keys())
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.8 * nrows), constrained_layout=True)
    ax_list = axes.flatten()

    field_vmin = float(np.min(field))
    field_vmax = float(np.max(field))
    if abs(field_vmax - field_vmin) < 1e-12:
        field_vmax = field_vmin + 1e-12

    all_w = [maps[k] for k in names]
    w_vmin = float(min(np.min(a) for a in all_w))
    w_vmax = float(max(np.max(a) for a in all_w))
    if abs(w_vmax - w_vmin) < 1e-12:
        w_vmax = w_vmin + 1e-12

    # Panel 0: target field
    im0 = ax_list[0].imshow(field, origin="lower", cmap="viridis", vmin=field_vmin, vmax=field_vmax, interpolation="nearest")
    ax_list[0].set_title("Target Field (same as AE input/GT)", fontsize=11, fontweight="bold")
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    ax_list[0].text(
        0.02,
        0.02,
        _stats(field),
        transform=ax_list[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.65),
    )
    fig.colorbar(im0, ax=ax_list[0], fraction=0.046, pad=0.03)

    # Strategy panels
    for i, name in enumerate(names, start=1):
        arr = maps[name]
        ax = ax_list[i]
        im = ax.imshow(arr, origin="lower", cmap="magma", vmin=w_vmin, vmax=w_vmax, interpolation="nearest")
        ax.set_title(name.replace("_", " "), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.02,
            _stats(arr),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.65),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Hide unused panel if any.
    for j in range(1 + len(names), len(ax_list)):
        ax_list[j].axis("off")

    fig.suptitle(
        f"AE Wavelet Strategy Comparison | {sim_id} frame={frame} ch={channel}",
        fontsize=13,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize AE wavelet strategy maps on selected frames/channels.")
    ap.add_argument(
        "--h5-path",
        type=Path,
        default=Path("/scratch/project_2008261/pf_surrogate_modelling/data/stochastic/rightclean/simulation_val_rightclean_fixed34_gradshared.h5"),
    )
    ap.add_argument("--sim-id", type=str, default="sim_0041")
    ap.add_argument("--frames", type=str, default="0,50,289,313")
    ap.add_argument("--channels", type=str, default="0,1")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/scratch/project_2008261/pf_surrogate_modelling/results/eval_best_adamw_vs_psgd_2026-02-11/"
            "psgd_best_e279_frames_0_50_289_313/wavelet_strategy_visuals"
        ),
    )
    ap.add_argument("--J", type=int, default=3)
    ap.add_argument("--beta-w", type=float, default=40.0)
    ap.add_argument("--wave", type=str, default="haar")
    ap.add_argument("--mode", type=str, default="zero")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    frames = _parse_int_list(args.frames)
    channels = _parse_int_list(args.channels)
    if not frames or not channels:
        raise ValueError("Need at least one frame and one channel.")

    strategies = _default_strategies(J=int(args.J), beta_w=float(args.beta_w))
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = {
        "h5_path": str(args.h5_path.resolve()),
        "sim_id": str(args.sim_id),
        "frames": frames,
        "channels": channels,
        "device": str(dev),
        "wavelet": {"J": int(args.J), "wave": str(args.wave), "mode": str(args.mode), "beta_w": float(args.beta_w)},
        "strategies": [{"name": s.name, "method": s.method, "params": s.params} for s in strategies],
        "outputs": [],
    }

    with h5py.File(args.h5_path, "r") as hf:
        if args.sim_id not in hf:
            raise KeyError(f"Simulation '{args.sim_id}' not found in {args.h5_path}")
        ds = hf[args.sim_id]["images"]
        tmax = int(ds.shape[0])
        cmax = int(ds.shape[1])
        for frame in frames:
            if frame < 0 or frame >= tmax:
                continue
            for ch in channels:
                if ch < 0 or ch >= cmax:
                    continue
                field = ds[frame, ch].astype(np.float32)
                y = torch.from_numpy(field).view(1, 1, field.shape[0], field.shape[1]).to(device=dev, dtype=torch.float32)
                with torch.no_grad():
                    maps = {s.name: _compute_weights(y, s, wave=str(args.wave), mode=str(args.mode)) for s in strategies}
                out_path = args.out_dir / f"wavelet_strategies_{args.sim_id}_frame{frame:04d}_ch{ch}.png"
                _render_one(
                    sim_id=str(args.sim_id),
                    frame=int(frame),
                    channel=int(ch),
                    field=field,
                    maps=maps,
                    out_path=out_path,
                    dpi=int(args.dpi),
                )
                manifest["outputs"].append(str(out_path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "wavelet_strategy_visuals_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(args.out_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()

