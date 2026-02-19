#!/usr/bin/env python3
"""
Lightweight wavelet-weight strategy comparison for AE datasets.

Goal:
- Compare multiple weighting strategies before full precompute/training.
- Keep outputs compact (JSON/CSV/command txt only by default).
- Explicitly rank strategies by low-frequency emphasis while penalizing bulk.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch


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


def _parse_int_list(text: str | None) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str | None) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _select_frames(num_frames: int, frames_per_sim: int, explicit: List[int]) -> List[int]:
    if explicit:
        return [i for i in explicit if 0 <= i < num_frames]
    if frames_per_sim <= 0:
        return []
    if frames_per_sim == 1:
        return [0]
    idx = np.linspace(0, num_frames - 1, frames_per_sim).round().astype(int)
    return sorted(set(idx.tolist()))


def _downsample_block_mean(x: np.ndarray, target_size: int) -> np.ndarray:
    h, w = x.shape
    if h == target_size and w == target_size:
        return x
    if h % target_size == 0 and w % target_size == 0:
        sh = h // target_size
        sw = w // target_size
        return x.reshape(target_size, sh, target_size, sw).mean(axis=(1, 3))
    # Fallback: nearest-ish resize via torch interpolate (keeps script dependency-light)
    xt = torch.from_numpy(x).view(1, 1, h, w).float()
    yt = torch.nn.functional.interpolate(xt, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return yt[0, 0].cpu().numpy()


def _radius_map(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    m = rr.max()
    if m <= 0:
        return rr
    return rr / m


def _freq_fractions(x: np.ndarray, low_cut: float, mid_cut: float) -> Dict[str, float]:
    f = np.fft.fftshift(np.fft.fft2(x))
    p = (np.abs(f) ** 2).astype(np.float64)
    rr = _radius_map(*x.shape)
    low = p[rr <= low_cut].sum()
    mid = p[(rr > low_cut) & (rr <= mid_cut)].sum()
    high = p[rr > mid_cut].sum()
    tot = low + mid + high
    if tot <= 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    return {"low": float(low / tot), "mid": float(mid / tot), "high": float(high / tot)}


def _default_strategies(J: int, beta_w: float) -> List[Strategy]:
    # Includes all-frequency multiband options and two baselines.
    return [
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


def _compute_weights(y: torch.Tensor, s: Strategy, wave: str, mode: str) -> torch.Tensor:
    p = s.params
    if s.method == "quantile":
        w, _ = ww.wavelet_importance_per_channel(
            y,
            J=int(p.get("J", 1)),
            wave=wave,
            mode=mode,
            theta=float(p.get("theta", 0.85)),
            alpha=float(p.get("alpha", 1.1)),
            beta_w=float(p.get("beta_w", 6.0)),
        )
        return w
    if s.method == "bandpass":
        w, _ = ww.wavelet_bandpass_importance_per_channel(
            y,
            J=int(p.get("J", 1)),
            wave=wave,
            mode=mode,
            beta_w=float(p.get("beta_w", 50.0)),
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
        return w
    if s.method == "multiband":
        w, _ = ww.wavelet_multiband_importance_per_channel(
            y,
            J=int(p.get("J", 3)),
            wave=wave,
            mode=mode,
            level_weights=[float(x) for x in p.get("level_weights", [1.0, 1.0, 1.0])],
            lowpass_weight=float(p.get("lowpass_weight", 1.0)),
            beta_w=float(p.get("beta_w", 50.0)),
            power=float(p.get("power", 1.3)),
            norm_quantile=float(p.get("norm_quantile", 0.99)),
            normalize_mean=bool(p.get("normalize_mean", True)),
            rescale_max=bool(p.get("rescale_max", False)),
            clip_min=p.get("clip_min", None),
            clip_max=p.get("clip_max", 250.0),
            combine_norm=bool(p.get("combine_norm", True)),
        )
        return w
    raise ValueError(f"Unsupported strategy method '{s.method}'")


def _empty_acc(channels: Iterable[int]) -> Dict[str, object]:
    per_ch = {}
    for c in channels:
        per_ch[str(c)] = {
            "count_maps": 0,
            "sum_mean": 0.0,
            "sum_std": 0.0,
            "sum_p95": 0.0,
            "sum_p99": 0.0,
            "global_min": float("inf"),
            "global_max": float("-inf"),
            "sum_bulk_1_2": 0.0,
            "sum_bulk_1_5": 0.0,
            "sum_bulk_2_0": 0.0,
            "sum_low_frac": 0.0,
            "sum_mid_frac": 0.0,
            "sum_high_frac": 0.0,
        }
    return {
        "count_maps": 0,
        "sum_mean": 0.0,
        "sum_std": 0.0,
        "sum_p95": 0.0,
        "sum_p99": 0.0,
        "global_min": float("inf"),
        "global_max": float("-inf"),
        "sum_bulk_1_2": 0.0,
        "sum_bulk_1_5": 0.0,
        "sum_bulk_2_0": 0.0,
        "sum_low_frac": 0.0,
        "sum_mid_frac": 0.0,
        "sum_high_frac": 0.0,
        "by_channel": per_ch,
    }


def _update_acc(acc: Dict[str, object], w_map: np.ndarray, ch_idx: int, low_cut: float, mid_cut: float, freq_size: int) -> None:
    # w_map: [H, W]
    a = np.asarray(w_map, dtype=np.float64)
    flat = a.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    p95 = float(np.percentile(flat, 95.0))
    p99 = float(np.percentile(flat, 99.0))
    amin = float(np.min(flat))
    amax = float(np.max(flat))
    bulk_1_2 = float(np.mean(flat > 1.2))
    bulk_1_5 = float(np.mean(flat > 1.5))
    bulk_2_0 = float(np.mean(flat > 2.0))

    ff = a
    if freq_size > 0 and (a.shape[0] != freq_size or a.shape[1] != freq_size):
        ff = _downsample_block_mean(a, target_size=freq_size)
    freq = _freq_fractions(ff, low_cut=low_cut, mid_cut=mid_cut)

    def _upd(d: Dict[str, object]) -> None:
        d["count_maps"] = int(d["count_maps"]) + 1
        d["sum_mean"] = float(d["sum_mean"]) + mean
        d["sum_std"] = float(d["sum_std"]) + std
        d["sum_p95"] = float(d["sum_p95"]) + p95
        d["sum_p99"] = float(d["sum_p99"]) + p99
        d["global_min"] = min(float(d["global_min"]), amin)
        d["global_max"] = max(float(d["global_max"]), amax)
        d["sum_bulk_1_2"] = float(d["sum_bulk_1_2"]) + bulk_1_2
        d["sum_bulk_1_5"] = float(d["sum_bulk_1_5"]) + bulk_1_5
        d["sum_bulk_2_0"] = float(d["sum_bulk_2_0"]) + bulk_2_0
        d["sum_low_frac"] = float(d["sum_low_frac"]) + float(freq["low"])
        d["sum_mid_frac"] = float(d["sum_mid_frac"]) + float(freq["mid"])
        d["sum_high_frac"] = float(d["sum_high_frac"]) + float(freq["high"])

    _upd(acc)
    _upd(acc["by_channel"][str(ch_idx)])


def _finalize_acc(acc: Dict[str, object]) -> Dict[str, object]:
    def _fin(d: Dict[str, object]) -> Dict[str, float]:
        n = max(int(d["count_maps"]), 1)
        low = float(d["sum_low_frac"]) / n
        mid = float(d["sum_mid_frac"]) / n
        high = float(d["sum_high_frac"]) / n
        out = {
            "count_maps": int(d["count_maps"]),
            "mean_weight": float(d["sum_mean"]) / n,
            "std_weight": float(d["sum_std"]) / n,
            "p95_weight": float(d["sum_p95"]) / n,
            "p99_weight": float(d["sum_p99"]) / n,
            "global_min": float(d["global_min"]) if np.isfinite(d["global_min"]) else 0.0,
            "global_max": float(d["global_max"]) if np.isfinite(d["global_max"]) else 0.0,
            "bulk_frac_gt_1_2": float(d["sum_bulk_1_2"]) / n,
            "bulk_frac_gt_1_5": float(d["sum_bulk_1_5"]) / n,
            "bulk_frac_gt_2_0": float(d["sum_bulk_2_0"]) / n,
            "freq_low_frac": low,
            "freq_mid_frac": mid,
            "freq_high_frac": high,
            "freq_low_high_ratio": float(low / max(high, 1e-12)),
            "lowfreq_priority_score": float(low - 0.25 * high - 0.20 * (float(d["sum_bulk_1_2"]) / n)),
        }
        return out

    by_channel = {}
    for k, dch in acc["by_channel"].items():
        by_channel[k] = _fin(dch)
    result = _fin(acc)
    result["by_channel"] = by_channel
    return result


def _strategy_to_precompute_cmd(s: Strategy, h5_path: Path, out_stub: str, channels: List[int], wave: str, mode: str) -> str:
    base = [
        "python -m models.datapipes.precompute_wavelet_weights",
        f"--h5 {h5_path}",
        f"--out {out_stub}_{s.name}.h5",
        "--target-channels " + " ".join(str(c) for c in channels),
        f"--method {s.method}",
        f"--J {int(s.params.get('J', 1))}",
        f"--wave {wave}",
        f"--mode {mode}",
        f"--beta-w {float(s.params.get('beta_w', 50.0))}",
        "--batch-size 4",
        "--device cuda",
    ]
    if s.method == "quantile":
        base.append(f"--theta {float(s.params.get('theta', 0.85))}")
        base.append(f"--alpha {float(s.params.get('alpha', 1.1))}")
    elif s.method == "bandpass":
        base.extend(
            [
                f"--bp-sigma-low {float(s.params.get('sigma_low', 1.5))}",
                f"--bp-sigma-high {float(s.params.get('sigma_high', 10.0))}",
                f"--bp-band-sigma {float(s.params.get('band_sigma', 16.0))}",
                f"--bp-mask-quantile {float(s.params.get('mask_quantile', 0.95))}",
                f"--bp-power {float(s.params.get('power', 1.5))}",
                f"--bp-norm-quantile {float(s.params.get('norm_quantile', 0.99))}",
                "--bp-normalize-mean",
                f"--bp-clip-max {float(s.params.get('clip_max', 250.0))}",
            ]
        )
    elif s.method == "multiband":
        lw = ",".join(str(float(x)) for x in s.params.get("level_weights", [1.0, 1.0, 1.0]))
        base.extend(
            [
                f"--mb-level-weights {lw}",
                f"--mb-lowpass-weight {float(s.params.get('lowpass_weight', 1.0))}",
                f"--mb-power {float(s.params.get('power', 1.3))}",
                f"--mb-norm-quantile {float(s.params.get('norm_quantile', 0.99))}",
                "--mb-normalize-mean",
                "--mb-combine-norm",
                f"--mb-clip-max {float(s.params.get('clip_max', 250.0))}",
            ]
        )
    return " ".join(base)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare AE wavelet weighting strategies (compact, low-bulk outputs).")
    ap.add_argument("--h5", type=Path, required=True, help="AE dataset HDF5 (train or val).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-channels", type=str, default="0,1", help="Comma-separated target channels.")
    ap.add_argument("--sim-ids", type=str, default=None, help="Optional comma-separated sim IDs.")
    ap.add_argument("--max-sims", type=int, default=12, help="Cap number of simulations for quick comparison.")
    ap.add_argument("--frames-per-sim", type=int, default=4, help="Linspace frame samples if --frame-indices not set.")
    ap.add_argument("--frame-indices", type=str, default=None, help="Explicit frame indices (comma-separated).")
    ap.add_argument("--J", type=int, default=3, help="Wavelet levels (all-frequency comparison prefers >=3).")
    ap.add_argument("--wave", type=str, default="haar")
    ap.add_argument("--mode", type=str, default="zero")
    ap.add_argument("--beta-w", type=float, default=40.0)
    ap.add_argument("--strategies", type=str, default=None, help="Optional comma list of strategy names to keep.")
    ap.add_argument("--freq-size", type=int, default=256, help="Downsample size for frequency-fraction diagnostics.")
    ap.add_argument("--low-cut", type=float, default=0.10, help="Low-freq radial cutoff (normalized).")
    ap.add_argument("--mid-cut", type=float, default=0.30, help="Mid-freq radial cutoff (normalized).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = [int(x) for x in args.target_channels.split(",") if x.strip()]
    if not channels:
        raise ValueError("No target channels specified.")
    explicit_frames = _parse_int_list(args.frame_indices)
    sim_filter = set(_parse_str_list(args.sim_ids)) if args.sim_ids else None

    strategies = _default_strategies(J=int(args.J), beta_w=float(args.beta_w))
    keep = set(_parse_str_list(args.strategies)) if args.strategies else None
    if keep is not None:
        strategies = [s for s in strategies if s.name in keep]
    if not strategies:
        raise ValueError("No strategies selected.")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    strategy_acc = {s.name: _empty_acc(channels) for s in strategies}

    processed = {"num_sims": 0, "num_frames": 0, "items": []}
    with h5py.File(args.h5, "r") as hf:
        gids = sorted(hf.keys())
        if sim_filter is not None:
            gids = [g for g in gids if g in sim_filter]
        if args.max_sims is not None and args.max_sims > 0:
            gids = gids[: int(args.max_sims)]

        for gid in gids:
            if "images" not in hf[gid]:
                continue
            ds = hf[gid]["images"]
            num_frames = int(ds.shape[0])
            frames = _select_frames(num_frames=num_frames, frames_per_sim=int(args.frames_per_sim), explicit=explicit_frames)
            if not frames:
                continue
            processed["num_sims"] += 1
            for fi in frames:
                arr = ds[fi, channels, :, :].astype(np.float32)  # [C,H,W]
                y = torch.from_numpy(arr).unsqueeze(0).to(device=dev, dtype=torch.float32)
                processed["num_frames"] += 1
                processed["items"].append({"gid": gid, "frame": int(fi)})
                with torch.no_grad():
                    for s in strategies:
                        w = _compute_weights(y, s, wave=str(args.wave), mode=str(args.mode))
                        w_np = w[0].detach().cpu().numpy()
                        for ci, c in enumerate(channels):
                            _update_acc(
                                strategy_acc[s.name],
                                w_map=w_np[ci],
                                ch_idx=int(c),
                                low_cut=float(args.low_cut),
                                mid_cut=float(args.mid_cut),
                                freq_size=int(args.freq_size),
                            )

    results = {}
    for s in strategies:
        final = _finalize_acc(strategy_acc[s.name])
        final["method"] = s.method
        final["params"] = s.params
        results[s.name] = final

    # Ranking: prioritize low frequency, then avoid bulk.
    ranking = sorted(
        (
            {
                "name": name,
                "lowfreq_priority_score": float(m["lowfreq_priority_score"]),
                "freq_low_frac": float(m["freq_low_frac"]),
                "freq_high_frac": float(m["freq_high_frac"]),
                "bulk_frac_gt_1_2": float(m["bulk_frac_gt_1_2"]),
                "mean_weight": float(m["mean_weight"]),
                "method": str(m["method"]),
            }
            for name, m in results.items()
        ),
        key=lambda x: (-x["lowfreq_priority_score"], -x["freq_low_frac"], x["bulk_frac_gt_1_2"]),
    )

    # Recommend multiband first if available (all-frequency objective).
    multiband_rank = [r for r in ranking if r["method"] == "multiband"]
    recommended = multiband_rank[0] if multiband_rank else ranking[0]

    out_json = out_dir / "wavelet_strategy_compare_summary.json"
    summary = {
        "h5": str(args.h5.resolve()),
        "device": str(dev),
        "channels": channels,
        "processed": processed,
        "freq_bands": {"low_cut": float(args.low_cut), "mid_cut": float(args.mid_cut), "freq_size": int(args.freq_size)},
        "strategies": results,
        "ranking": ranking,
        "recommended": recommended,
        "notes": [
            "Ranking prioritizes low-frequency emphasis and penalizes overly bulk weighting.",
            "For AE all-frequency weighting, prefer multiband strategies with explicit lowpass_weight > 1.",
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    out_csv = out_dir / "wavelet_strategy_compare_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "name",
                "method",
                "lowfreq_priority_score",
                "freq_low_frac",
                "freq_mid_frac",
                "freq_high_frac",
                "bulk_frac_gt_1_2",
                "bulk_frac_gt_1_5",
                "mean_weight",
                "p99_weight",
                "global_max",
            ]
        )
        for i, r in enumerate(ranking, start=1):
            m = results[r["name"]]
            w.writerow(
                [
                    i,
                    r["name"],
                    r["method"],
                    m["lowfreq_priority_score"],
                    m["freq_low_frac"],
                    m["freq_mid_frac"],
                    m["freq_high_frac"],
                    m["bulk_frac_gt_1_2"],
                    m["bulk_frac_gt_1_5"],
                    m["mean_weight"],
                    m["p99_weight"],
                    m["global_max"],
                ]
            )

    cmd_txt = out_dir / "precompute_recommended_commands.sh"
    out_stub = str((out_dir / "wavelet_weights").resolve())
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Recommended strategy first: {recommended['name']}",
    ]
    for r in ranking[:3]:
        strat = next(s for s in strategies if s.name == r["name"])
        lines.append(_strategy_to_precompute_cmd(strat, args.h5.resolve(), out_stub, channels, str(args.wave), str(args.mode)))
    cmd_txt.write_text("\n".join(lines) + "\n")

    print(str(out_json), flush=True)
    print(str(out_csv), flush=True)
    print(str(cmd_txt), flush=True)


if __name__ == "__main__":
    main()

