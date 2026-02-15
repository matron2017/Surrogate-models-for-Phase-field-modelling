#!/usr/bin/env python3
"""Inspect right-boundary buffer onset for selected simulations.

Produces a width-vs-time plot and a frame series around onset for each gid.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_H5_PATH = os.environ.get(
    "PF_SURROGATE_DATA_H5",
    "/scratch/project_2008261/solidification_modelling/data/rapid_solidification/simulation_train.h5",
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _width_for_frame(frame: np.ndarray, tol: float) -> int:
    """Count contiguous rightmost columns identical to last column.

    frame shape: (C, H, W)
    """
    last = frame[..., -1]
    diffs = np.abs(frame - last[..., None])
    eq = np.all(diffs <= tol, axis=(0, 1))  # (W,)
    w = 0
    for v in eq[::-1]:
        if v:
            w += 1
        else:
            break
    return int(w)


def _compute_widths(images: h5py.Dataset, tol: float) -> np.ndarray:
    t_len = images.shape[0]
    widths = np.zeros(t_len, dtype=np.int32)
    for t in range(t_len):
        widths[t] = _width_for_frame(images[t], tol)
    return widths


def _find_onset(widths: np.ndarray, tail: int, k: int) -> Tuple[Optional[int], int]:
    if widths.size == 0:
        return None, 0
    tail = min(tail, widths.size)
    median_tail = int(np.median(widths[-tail:]))
    low, high = median_tail - 2, median_tail + 2
    onset = None
    for t in range(0, widths.size - k + 1):
        window = widths[t : t + k]
        if np.all((window >= low) & (window <= high)):
            onset = t
            break
    return onset, median_tail


def _select_frames(onset: Optional[int], t_len: int, window: int, step: int) -> List[int]:
    if onset is None:
        frames = [300, 350, 400, t_len - 1]
        return [t for t in frames if 0 <= t < t_len]
    start = max(0, onset - window)
    end = min(t_len - 1, onset + window)
    return list(range(start, end + 1, step))


def _plot_widths(
    widths: np.ndarray,
    onset: Optional[int],
    median_tail: int,
    cut_col: int,
    out_path: Path,
) -> None:
    t = np.arange(widths.size)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t, widths, color="black", linewidth=1)
    ax.axhline(median_tail, color="tab:blue", linestyle="--", linewidth=1)
    ax.fill_between(
        t,
        median_tail - 2,
        median_tail + 2,
        color="tab:blue",
        alpha=0.15,
        label="tail median ±2",
    )
    if onset is not None:
        ax.axvline(onset, color="tab:red", linestyle="--", linewidth=1, label="onset")
    ax.set_title(f"Right-buffer width vs time (cut col={cut_col})")
    ax.set_xlabel("frame")
    ax.set_ylabel("right buffer width (cols)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_frames(
    images: h5py.Dataset,
    frames: List[int],
    cut_col: int,
    out_path: Path,
) -> None:
    if not frames:
        return
    sample = images[frames[0]]
    c_len = sample.shape[0]
    ncols = len(frames)
    fig, axes = plt.subplots(c_len, ncols, figsize=(2.2 * ncols, 2.2 * c_len), dpi=150)
    if c_len == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Shared color scale per channel across the selected frames.
    vmin = np.full(c_len, np.inf)
    vmax = np.full(c_len, -np.inf)
    for t in frames:
        frame = images[t]
        vmin = np.minimum(vmin, frame.min(axis=(1, 2)))
        vmax = np.maximum(vmax, frame.max(axis=(1, 2)))

    for j, t in enumerate(frames):
        frame = images[t]
        for c in range(c_len):
            ax = axes[c, j]
            ax.imshow(frame[c], cmap="viridis", vmin=vmin[c], vmax=vmax[c], origin="lower")
            ax.axvline(cut_col, color="red", linestyle="--", linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"t={t}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect right-buffer onset for selected sims.")
    ap.add_argument(
        "--h5",
        type=str,
        default=DEFAULT_H5_PATH,
        help="Path to HDF5 file (or set PF_SURROGATE_DATA_H5).",
    )
    ap.add_argument(
        "--gids",
        type=str,
        nargs="+",
        required=True,
        help="Group IDs (e.g., sim_0001 sim_0008).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="results/right_buffer_diagnostics/inspect",
        help="Output directory for plots.",
    )
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    with h5py.File(args.h5, "r") as h5:
        for gid in args.gids:
            if gid not in h5:
                raise KeyError(f"Group {gid!r} not found in {args.h5}")
            grp = h5[gid]
            images = grp["images"]

            widths = _compute_widths(images, args.tol)
            onset, median_tail = _find_onset(widths, args.tail, args.k)
            cut_col = images.shape[-1] - median_tail

            width_path = outdir / f"{gid}_width_plot.png"
            frames_path = outdir / f"{gid}_frames_around_onset.png"
            _plot_widths(widths, onset, median_tail, cut_col, width_path)

            frames = _select_frames(onset, images.shape[0], args.window, args.step)
            _plot_frames(images, frames, cut_col, frames_path)

            print(
                f"{gid}: onset={onset} median_tail={median_tail} cut_col={cut_col} "
                f"width_plot={width_path} frames_plot={frames_path}"
            )


if __name__ == "__main__":
    main()
