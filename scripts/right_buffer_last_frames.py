#!/usr/bin/env python3
"""Plot last frame (both channels) for every sim with right-edge width marker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _width_for_frame(frame: np.ndarray, tol: float) -> int:
    """Count contiguous rightmost columns identical to last column (all channels)."""
    last = frame[..., -1]
    diffs = np.abs(frame - last[..., None])
    eq = np.all(diffs <= tol, axis=(0, 1))
    w = 0
    for v in eq[::-1]:
        if v:
            w += 1
        else:
            break
    return int(w)


def _plot_last_frame(
    frame: np.ndarray,
    t: int,
    gid: str,
    width_last: int,
    out_path: Path,
    label_prefix: str = "",
) -> None:
    c_len = frame.shape[0]
    fig, axes = plt.subplots(1, c_len, figsize=(3.6 * c_len, 3.6), dpi=170)
    if c_len == 1:
        axes = [axes]

    vmin = frame.min(axis=(1, 2))
    vmax = frame.max(axis=(1, 2))
    cut_col = frame.shape[-1] - width_last

    for c in range(c_len):
        ax = axes[c]
        ax.imshow(frame[c], cmap="viridis", vmin=vmin[c], vmax=vmax[c], origin="lower")
        ax.axvline(cut_col, color="red", linestyle="--", linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        prefix = f"{label_prefix} " if label_prefix else ""
        ax.set_title(f"{prefix}{gid} ch{c} t={t} w={width_last}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot last frames for all sims in an H5.")
    ap.add_argument("--h5", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--cutoff-csv",
        type=Path,
        default=None,
        help="CSV with cutoff_frame per gid; if provided, plot t=cutoff_frame-1.",
    )
    ap.add_argument(
        "--cutoff-offset",
        type=int,
        default=-1,
        help="Offset applied to cutoff_frame to select plot time (default: -1).",
    )
    ap.add_argument(
        "--line-mode",
        type=str,
        default="tail",
        choices=("tail", "frame", "csv"),
        help="Use tail median width (tail), width at plotted frame (frame), or cut_width from CSV (csv).",
    )
    ap.add_argument(
        "--fixed-cut-width",
        type=int,
        default=None,
        help="If set, use this fixed cut width for all sims (overrides line-mode).",
    )
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    manifest_meta = {}
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        for name, meta in manifest.items():
            gid = meta.get("group_id")
            if not gid:
                continue
            grad = meta.get("physical_params", {}).get("thermal_gradient")
            seed = None
            if "/seed_" in name:
                seed = name.split("/seed_")[-1]
            manifest_meta[gid] = {
                "original_name": name,
                "thermal_gradient": grad,
                "seed": seed,
            }

    cutoff_map = {}
    tail_width_map = {}
    cut_width_map = {}
    if args.cutoff_csv and args.cutoff_csv.exists():
        with args.cutoff_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                gid = row.get("gid")
                cutoff = row.get("cutoff_frame")
                tail_w = row.get("median_tail_width")
                cut_w = row.get("cut_width")
                if gid and cutoff not in (None, "", " "):
                    cutoff_map[gid] = int(float(cutoff))
                if gid and tail_w not in (None, "", " "):
                    tail_width_map[gid] = int(float(tail_w))
                if gid and cut_w not in (None, "", " "):
                    cut_width_map[gid] = int(float(cut_w))

    with h5py.File(args.h5, "r") as h5:
        for gid in h5.keys():
            images = h5[gid]["images"]
            if gid in cutoff_map:
                t = max(0, cutoff_map[gid] + int(args.cutoff_offset))
                t = min(t, images.shape[0] - 1)
            else:
                t = images.shape[0] - 1
            frame = images[t]
            width_last = _width_for_frame(frame, args.tol)
            if args.fixed_cut_width is not None:
                width_cut = int(args.fixed_cut_width)
            elif args.line_mode == "csv":
                width_cut = cut_width_map.get(gid, width_last)
            elif args.line_mode == "tail":
                width_cut = tail_width_map.get(gid, width_last)
            else:
                width_cut = width_last
            meta = manifest_meta.get(gid, {})
            grad = meta.get("thermal_gradient")
            seed = meta.get("seed")
            if grad is not None and seed is not None:
                grad_tag = f"G{grad/1e6:.1f}e6"
                name = f"{grad_tag}_seed{seed}_{gid}_last_frame.png"
                label_prefix = f"{grad_tag} seed{seed}"
            else:
                name = f"{gid}_last_frame.png"
                label_prefix = ""
            out_path = args.outdir / name
            _plot_last_frame(frame, t, gid, width_cut, out_path, label_prefix=label_prefix)


if __name__ == "__main__":
    main()
