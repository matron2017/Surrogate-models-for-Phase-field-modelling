#!/usr/bin/env python3
"""Generate per-sim onset visuals (t=onset-1 and t=onset) with cut line.

Uses the same onset rule as right_buffer_filter.py (median tail width, K frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from right_buffer_filter import compute_onset_and_median


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_pair(onset: Optional[int], t_len: int) -> Tuple[int, int]:
    if onset is None:
        # fallback: show last two frames
        if t_len <= 1:
            return 0, 0
        return max(0, t_len - 2), t_len - 1
    t1 = max(0, onset - 1)
    t2 = min(t_len - 1, onset)
    return t1, t2


def _plot_onset_pair(
    images: h5py.Dataset,
    t1: int,
    t2: int,
    cut_col: int,
    out_path: Path,
) -> None:
    frame1 = images[t1]
    frame2 = images[t2]
    c_len = frame1.shape[0]

    fig, axes = plt.subplots(c_len, 2, figsize=(5.2, 2.6 * c_len), dpi=170)
    if c_len == 1:
        axes = np.expand_dims(axes, axis=0)

    # shared color scale per channel across the two frames
    vmin = np.minimum(frame1.min(axis=(1, 2)), frame2.min(axis=(1, 2)))
    vmax = np.maximum(frame1.max(axis=(1, 2)), frame2.max(axis=(1, 2)))

    for c in range(c_len):
        for j, (t, frame) in enumerate([(t1, frame1), (t2, frame2)]):
            ax = axes[c, j]
            ax.imshow(frame[c], cmap="viridis", vmin=vmin[c], vmax=vmax[c], origin="lower")
            ax.axvline(cut_col, color="red", linestyle="--", linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"t={t}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot onset frames for right-buffer touching.")
    ap.add_argument("--h5", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-width", type=int, default=10)
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    with h5py.File(args.h5, "r") as h5:
        for gid in h5.keys():
            images = h5[gid]["images"]
            onset, median_tail, _ = compute_onset_and_median(images, args.tol, args.tail, args.k, args.min_width)
            cut_col = images.shape[-1] - median_tail
            t1, t2 = _select_pair(onset, images.shape[0])
            out_path = args.outdir / f"{gid}_onset_t{t2:04d}.png"
            _plot_onset_pair(images, t1, t2, cut_col, out_path)


if __name__ == "__main__":
    main()
