#!/usr/bin/env python3
"""
Frequency diagnostics for PF surrogate dataset.

Computes average 2D power spectrum and radial PSD for selected sims/frames.
Also reports low/mid/high frequency energy fractions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _select_frames(num_frames: int, sample_frames: int, explicit: List[int] | None) -> List[int]:
    if explicit:
        return [i for i in explicit if 0 <= i < num_frames]
    if sample_frames <= 0:
        return []
    if sample_frames == 1:
        return [0]
    idx = np.linspace(0, num_frames - 1, sample_frames).round().astype(int)
    return sorted(set(idx.tolist()))


def _downsample_block_mean(x: np.ndarray, target_size: int) -> np.ndarray:
    # x shape: (H, W)
    h, w = x.shape
    if h == target_size and w == target_size:
        return x
    if h % target_size != 0 or w % target_size != 0:
        raise ValueError(f"Cannot downsample {x.shape} to {target_size} by block mean.")
    sh = h // target_size
    sw = w // target_size
    x = x.reshape(target_size, sh, target_size, sw)
    return x.mean(axis=(1, 3))


def _radial_bins(n: int) -> Tuple[np.ndarray, np.ndarray]:
    # Radius map and bin indices for an n x n grid.
    yy, xx = np.indices((n, n))
    center = (n - 1) / 2.0
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    rmax = rr.max()
    # integer bins from 0..floor(rmax)
    rbin = rr.astype(np.int32)
    return rr, rbin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--channels", default="0,1", type=str)
    ap.add_argument("--sim-ids", default=None, type=str, help="Comma-separated sim ids like sim_0001,sim_0002")
    ap.add_argument("--sample-frames", default=6, type=int, help="Frames per sim (linspace)")
    ap.add_argument("--frame-indices", default=None, type=str, help="Explicit comma-separated frame indices")
    ap.add_argument("--target-size", default=256, type=int, help="Downsample to this size for FFT")
    ap.add_argument("--low-cut", default=0.10, type=float, help="Low freq cutoff as fraction of Nyquist radius")
    ap.add_argument("--mid-cut", default=0.30, type=float, help="Mid freq cutoff as fraction of Nyquist radius")
    ap.add_argument("--max-sims", default=None, type=int, help="Optional cap on number of sims")
    ap.add_argument("--dpi", default=200, type=int)
    ap.add_argument(
        "--per-frame",
        action="store_true",
        help="If set, compute diagnostics separately for each frame index (requires --frame-indices).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [int(x) for x in args.channels.split(",") if x.strip()]
    frame_indices = _parse_int_list(args.frame_indices) if args.frame_indices else None

    with h5py.File(args.h5_path, "r") as hf:
        sim_ids = list(hf.keys())
        if args.sim_ids:
            sim_ids = [s for s in sim_ids if s in set(args.sim_ids.split(","))]
        if args.max_sims:
            sim_ids = sim_ids[: args.max_sims]

        # prepare radial bins for target size
        n = args.target_size
        rr, rbin = _radial_bins(n)
        rmax = rbin.max()
        nbins = rmax + 1
        rvals = np.arange(nbins)
        # frequency radius normalized to Nyquist
        r_norm = rvals / rmax if rmax > 0 else rvals
        r_norm_map = rr / rr.max() if rr.max() > 0 else rr
        low_mask = r_norm_map <= args.low_cut
        mid_mask = (r_norm_map > args.low_cut) & (r_norm_map <= args.mid_cut)
        high_mask = r_norm_map > args.mid_cut

        if args.per_frame:
            if not frame_indices:
                raise ValueError("--per-frame requires --frame-indices")
            per_frame_meta = {}
            for t in frame_indices:
                psd_sum = {c: np.zeros((n, n), dtype=np.float64) for c in channels}
                rad_sum = {c: np.zeros(nbins, dtype=np.float64) for c in channels}
                band_energy = {c: {"low": 0.0, "mid": 0.0, "high": 0.0, "total": 0.0} for c in channels}
                total_frames = 0

                for sim_id in sim_ids:
                    ds = hf[f"{sim_id}/images"]
                    num_frames = ds.shape[0]
                    if t < 0 or t >= num_frames:
                        continue
                    total_frames += 1
                    frame = ds[t]
                    for c in channels:
                        img = frame[c].astype(np.float32)
                        if args.target_size:
                            img = _downsample_block_mean(img, args.target_size)
                        f = np.fft.fft2(img)
                        f = np.fft.fftshift(f)
                        p = (np.abs(f) ** 2).astype(np.float64)
                        psd_sum[c] += p
                        for r in range(nbins):
                            mask = rbin == r
                            if mask.any():
                                rad_sum[c][r] += p[mask].sum()
                        total_e = p.sum()
                        band_energy[c]["total"] += total_e
                        band_energy[c]["low"] += p[low_mask].sum()
                        band_energy[c]["mid"] += p[mid_mask].sum()
                        band_energy[c]["high"] += p[high_mask].sum()

                # radial counts
                rad_cnt = np.array([(rbin == r).sum() for r in range(nbins)], dtype=np.float64)
                frame_dir = out_dir / f"frame{t:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)

                for c in channels:
                    avg_psd = psd_sum[c] / max(total_frames, 1)
                    avg_psd_log = np.log10(avg_psd + 1e-12)
                    plt.figure(figsize=(6, 5))
                    plt.imshow(avg_psd_log, cmap="magma")
                    plt.colorbar(label="log10 power")
                    plt.title(f"Avg PSD (ch{c}), frame {t}, {total_frames} sims")
                    plt.tight_layout()
                    plt.savefig(frame_dir / f"avg_psd_ch{c}.png", dpi=args.dpi)
                    plt.close()

                    rad_mean = rad_sum[c] / np.maximum(rad_cnt, 1.0) / max(total_frames, 1)
                    plt.figure(figsize=(6, 4))
                    plt.plot(r_norm, rad_mean, lw=2)
                    plt.yscale("log")
                    plt.xlabel("Normalized radius (0..1)")
                    plt.ylabel("Mean power")
                    plt.title(f"Radial PSD (ch{c}) frame {t}")
                    plt.tight_layout()
                    plt.savefig(frame_dir / f"radial_psd_ch{c}.png", dpi=args.dpi)
                    plt.close()

                # band fractions
                band_fractions = {}
                for c in channels:
                    total = band_energy[c]["total"] if band_energy[c]["total"] > 0 else 1.0
                    band_fractions[c] = {
                        "low": band_energy[c]["low"] / total,
                        "mid": band_energy[c]["mid"] / total,
                        "high": band_energy[c]["high"] / total,
                    }
                per_frame_meta[str(t)] = {
                    "total_sims": total_frames,
                    "band_fractions": band_fractions,
                }

            meta = {
                "h5_path": str(args.h5_path),
                "num_sims": len(sim_ids),
                "channels": channels,
                "target_size": args.target_size,
                "frame_indices": frame_indices,
                "low_cut": args.low_cut,
                "mid_cut": args.mid_cut,
                "per_frame": True,
                "per_frame_band_fractions": per_frame_meta,
            }
            (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        else:
            # accumulators per channel
            psd_sum = {c: np.zeros((n, n), dtype=np.float64) for c in channels}
            rad_sum = {c: np.zeros(nbins, dtype=np.float64) for c in channels}
            rad_cnt = np.zeros(nbins, dtype=np.float64)
            band_energy = {c: {"low": 0.0, "mid": 0.0, "high": 0.0, "total": 0.0} for c in channels}

            total_frames = 0

            for sim_id in sim_ids:
                ds = hf[f"{sim_id}/images"]
                num_frames = ds.shape[0]
                frames = _select_frames(num_frames, args.sample_frames, frame_indices)
                for t in frames:
                    total_frames += 1
                    # load frame
                    frame = ds[t]  # shape (C, H, W)
                    for c in channels:
                        img = frame[c].astype(np.float32)
                        if args.target_size:
                            img = _downsample_block_mean(img, args.target_size)
                        # compute power spectrum
                        f = np.fft.fft2(img)
                        f = np.fft.fftshift(f)
                        p = (np.abs(f) ** 2).astype(np.float64)
                        psd_sum[c] += p
                        # radial mean
                        for r in range(nbins):
                            mask = rbin == r
                            if mask.any():
                                rad_sum[c][r] += p[mask].sum()
                        # energy bands
                        total_e = p.sum()
                        band_energy[c]["total"] += total_e
                        band_energy[c]["low"] += p[low_mask].sum()
                        band_energy[c]["mid"] += p[mid_mask].sum()
                        band_energy[c]["high"] += p[high_mask].sum()

            # finalize radial counts
            for r in range(nbins):
                rad_cnt[r] = (rbin == r).sum()

            # compute averages and plots
            for c in channels:
                avg_psd = psd_sum[c] / max(total_frames, 1)
                avg_psd_log = np.log10(avg_psd + 1e-12)
                plt.figure(figsize=(6, 5))
                plt.imshow(avg_psd_log, cmap="magma")
                plt.colorbar(label="log10 power")
                plt.title(f"Avg PSD (ch{c}), {total_frames} frames")
                plt.tight_layout()
                plt.savefig(out_dir / f"avg_psd_ch{c}.png", dpi=args.dpi)
                plt.close()

                # radial PSD
                rad_mean = rad_sum[c] / np.maximum(rad_cnt, 1.0) / max(total_frames, 1)
                plt.figure(figsize=(6, 4))
                plt.plot(r_norm, rad_mean, lw=2)
                plt.yscale("log")
                plt.xlabel("Normalized radius (0..1)")
                plt.ylabel("Mean power")
                plt.title(f"Radial PSD (ch{c})")
                plt.tight_layout()
                plt.savefig(out_dir / f"radial_psd_ch{c}.png", dpi=args.dpi)
                plt.close()

            # save metadata
            band_fractions = {}
            for c in channels:
                total = band_energy[c]["total"] if band_energy[c]["total"] > 0 else 1.0
                band_fractions[c] = {
                    "low": band_energy[c]["low"] / total,
                    "mid": band_energy[c]["mid"] / total,
                    "high": band_energy[c]["high"] / total,
                }

            meta = {
                "h5_path": str(args.h5_path),
                "num_sims": len(sim_ids),
                "total_frames": total_frames,
                "channels": channels,
                "target_size": args.target_size,
                "sample_frames": args.sample_frames,
                "frame_indices": frame_indices,
                "low_cut": args.low_cut,
                "mid_cut": args.mid_cut,
                "band_fractions": band_fractions,
            }
            (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
