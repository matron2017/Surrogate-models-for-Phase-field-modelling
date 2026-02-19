#!/usr/bin/env python3
"""Compare right-buffer onset across gradients/seeds (stochastic dataset)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _width_for_frame(frame: np.ndarray, tol: float) -> int:
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


def _plot_widths_side_by_side(
    widths_a: np.ndarray,
    onset_a: Optional[int],
    median_a: int,
    label_a: str,
    widths_b: np.ndarray,
    onset_b: Optional[int],
    median_b: int,
    label_b: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150, sharey=True)
    for ax, widths, onset, median, label in [
        (axes[0], widths_a, onset_a, median_a, label_a),
        (axes[1], widths_b, onset_b, median_b, label_b),
    ]:
        t = np.arange(widths.size)
        ax.plot(t, widths, color="black", linewidth=1)
        ax.axhline(median, color="tab:blue", linestyle="--", linewidth=1)
        ax.fill_between(t, median - 2, median + 2, color="tab:blue", alpha=0.15)
        if onset is not None:
            ax.axvline(onset, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("frame")
    axes[0].set_ylabel("right buffer width (cols)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_frames_side_by_side(
    images_a: h5py.Dataset,
    frames_a: List[int],
    cut_col_a: int,
    label_a: str,
    images_b: h5py.Dataset,
    frames_b: List[int],
    cut_col_b: int,
    label_b: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    ncols = max(len(frames_a), len(frames_b))
    c_len = images_a.shape[1]
    fig, axes = plt.subplots(2 * c_len, ncols, figsize=(3.2 * ncols, 6.0 * c_len), dpi=dpi)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    # shared color scale per channel for each row
    def _minmax(images: h5py.Dataset, frames: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        vmin = np.full(c_len, np.inf)
        vmax = np.full(c_len, -np.inf)
        for t in frames:
            frame = images[t]
            vmin = np.minimum(vmin, frame.min(axis=(1, 2)))
            vmax = np.maximum(vmax, frame.max(axis=(1, 2)))
        return vmin, vmax

    vmin_a, vmax_a = _minmax(images_a, frames_a)
    vmin_b, vmax_b = _minmax(images_b, frames_b)

    for row_idx, (images, frames, cut_col, label, vmin, vmax) in enumerate(
        [
            (images_a, frames_a, cut_col_a, label_a, vmin_a, vmax_a),
            (images_b, frames_b, cut_col_b, label_b, vmin_b, vmax_b),
        ]
    ):
        for j in range(ncols):
            t = frames[j] if j < len(frames) else frames[-1]
            frame = images[t]
            for c in range(c_len):
                ax = axes[row_idx * c_len + c, j]
                ax.imshow(frame[c], cmap="viridis", vmin=vmin[c], vmax=vmax[c], origin="lower")
                ax.axvline(cut_col, color="red", linestyle="--", linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_title(f"{label} t={t}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_seed_widths(
    seed_data: List[Tuple[str, np.ndarray, Optional[int], int]],
    out_path: Path,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=dpi)
    for seed_label, widths, onset, median in seed_data:
        t = np.arange(widths.size)
        line = ax.plot(t, widths, linewidth=1, label=seed_label)[0]
        if onset is not None:
            ax.axvline(onset, color=line.get_color(), linestyle="--", linewidth=1)
    ax.set_title("G2.3e6 seeds: right-buffer width vs time")
    ax.set_xlabel("frame")
    ax.set_ylabel("right buffer width (cols)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_seed_onset_frames(
    seed_frames: List[Tuple[str, np.ndarray]],
    cut_col: int,
    out_path: Path,
    dpi: int = 150,
) -> None:
    # seed_frames: list of (seed_label, frame[C,H,W])
    if not seed_frames:
        return
    c_len = seed_frames[0][1].shape[0]
    ncols = len(seed_frames)
    fig, axes = plt.subplots(c_len, ncols, figsize=(3.2 * ncols, 3.2 * c_len), dpi=dpi)
    if c_len == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    vmin = np.full(c_len, np.inf)
    vmax = np.full(c_len, -np.inf)
    for _, frame in seed_frames:
        vmin = np.minimum(vmin, frame.min(axis=(1, 2)))
        vmax = np.maximum(vmax, frame.max(axis=(1, 2)))

    for j, (seed_label, frame) in enumerate(seed_frames):
        for c in range(c_len):
            ax = axes[c, j]
            ax.imshow(frame[c], cmap="viridis", vmin=vmin[c], vmax=vmax[c], origin="lower")
            ax.axvline(cut_col, color="red", linestyle="--", linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(seed_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _load_meta(meta_path: Path) -> Dict[str, dict]:
    with meta_path.open("r") as f:
        return json.load(f)


def _select_three_frames(anchor: int, t_len: int, offset: int) -> List[int]:
    frames = [anchor - offset, anchor, anchor + offset]
    return [t for t in frames if 0 <= t < t_len]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare right-buffer onset across gradients/seeds.")
    ap.add_argument(
        "--h5",
        type=str,
        default="/scratch/project_2008261/pf_surrogate_modelling/data/stochastic/simulation_train.h5",
        help="Path to stochastic HDF5 file.",
    )
    ap.add_argument(
        "--meta",
        type=str,
        default="/scratch/project_2008261/pf_surrogate_modelling/data/stochastic/sim_meta.json",
        help="Path to sim_meta.json for stochastic dataset.",
    )
    ap.add_argument("--gid-12", type=str, default="sim_0001", help="G1.2e6 seed gid.")
    ap.add_argument("--gid-23", type=str, default="sim_0036", help="G2.3e6 seed gid.")
    ap.add_argument("--outdir", type=str, default="results/right_buffer_diagnostics/seed_compare")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--hq-dpi", type=int, default=300)
    ap.add_argument("--hq-only-3", action="store_true")
    ap.add_argument("--hq-offset", type=int, default=20)
    ap.add_argument("--include-seeds", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    meta = _load_meta(Path(args.meta))
    seeds_23 = []
    if args.include_seeds or not args.hq_only_3:
        seeds_23 = [
            (gid, info["original_name"])
            for gid, info in meta.items()
            if info.get("physical_params", {}).get("thermal_gradient") == 2300000.0
        ]
        seeds_23 = sorted(seeds_23, key=lambda x: x[0])

    with h5py.File(args.h5, "r") as h5:
        if args.gid_12 not in h5 or args.gid_23 not in h5:
            raise KeyError("Requested gid not found in H5 file.")

        g12 = h5[args.gid_12]
        g23 = h5[args.gid_23]

        widths_12 = _compute_widths(g12["images"], args.tol)
        onset_12, median_12 = _find_onset(widths_12, args.tail, args.k)
        widths_23 = _compute_widths(g23["images"], args.tol)
        onset_23, median_23 = _find_onset(widths_23, args.tail, args.k)

        label_12 = meta[args.gid_12]["original_name"]
        label_23 = meta[args.gid_23]["original_name"]

        _plot_widths_side_by_side(
            widths_12,
            onset_12,
            median_12,
            f"G1.2e6 {label_12}",
            widths_23,
            onset_23,
            median_23,
            f"G2.3e6 {label_23}",
            outdir / "g12_g23_widths_side_by_side.png",
        )

        if args.hq_only_3 and onset_23 is not None:
            frames_23 = _select_three_frames(onset_23, g23["images"].shape[0], args.hq_offset)
            frames_12 = _select_three_frames(onset_23, g12["images"].shape[0], args.hq_offset)
        else:
            frames_12 = _select_frames(onset_12, g12["images"].shape[0], args.window, args.step)
            frames_23 = _select_frames(onset_23, g23["images"].shape[0], args.window, args.step)
        cut_12 = g12["images"].shape[-1] - median_12
        cut_23 = g23["images"].shape[-1] - median_23
        _plot_frames_side_by_side(
            g12["images"],
            frames_12,
            cut_12,
            "G1.2e6",
            g23["images"],
            frames_23,
            cut_23,
            "G2.3e6",
            outdir / "g12_g23_frames_side_by_side.png",
            dpi=args.hq_dpi,
        )

        if seeds_23:
            # G2.3 seeds: width curves + onset markers.
            seed_widths = []
            seed_frames = []
            cut_cols = []
            for gid, name in seeds_23:
                if gid not in h5:
                    continue
                images = h5[gid]["images"]
                widths = _compute_widths(images, args.tol)
                onset, median = _find_onset(widths, args.tail, args.k)
                seed_label = name.split("/")[-1]
                seed_widths.append((seed_label, widths, onset, median))
                if onset is not None:
                    seed_frames.append((seed_label, images[onset]))
                cut_cols.append(images.shape[-1] - median)

            _plot_seed_widths(seed_widths, outdir / "g23_seeds_widths_onset.png", dpi=args.hq_dpi)

            if cut_cols:
                cut_col = int(np.median(cut_cols))
                _plot_seed_onset_frames(
                    seed_frames,
                    cut_col,
                    outdir / "g23_seeds_onset_frames.png",
                    dpi=args.hq_dpi,
                )

    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
