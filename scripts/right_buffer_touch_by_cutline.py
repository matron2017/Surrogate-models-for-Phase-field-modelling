#!/usr/bin/env python3
"""Compute per-sim touch time for a shared cut line (per gradient).

Uses grad_cutoffs CSV (thermal_gradient -> cut_width) and finds the first
frame where width(t) >= cut_width for each sim.
Outputs a CSV with touch_frame and cut_width per gid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from right_buffer_filter import _width_for_frame


def _first_touch_frame_fixed(images: h5py.Dataset, cut_width: int, tol: float, block: int = 8) -> int | None:
    """Find first frame where the rightmost cut_width columns are NOT all equal to the last column.

    This corresponds to the interface reaching (or crossing) the cut line at x = W - cut_width.
    Uses block-wise reads of only the rightmost cut_width columns to avoid scanning full frames.
    """
    t_len = images.shape[0]
    w = int(cut_width)
    if w <= 0:
        return 0
    if w > images.shape[-1]:
        return None
    for t0 in range(0, t_len, block):
        t1 = min(t_len, t0 + block)
        block_data = images[t0:t1, :, :, -w:]
        last = images[t0:t1, :, :, -1]
        diff = np.abs(block_data - last[..., None]) <= tol
        all_equal = np.all(diff, axis=(1, 2, 3))
        touch = ~all_equal
        if np.any(touch):
            return int(t0 + np.argmax(touch))
    return None


def _compute_widths(images: h5py.Dataset, tol: float) -> np.ndarray:
    t_len = images.shape[0]
    widths = np.zeros(t_len, dtype=np.int32)
    for t in range(t_len):
        widths[t] = _width_for_frame(images[t], tol)
    return widths


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-sim touch frame using gradient cut line.")
    ap.add_argument("--h5", required=True, type=Path)
    ap.add_argument("--grad-cut-csv", required=False, type=Path, default=None)
    ap.add_argument("--fixed-cut-width", type=int, default=None, help="Use a fixed cut width for all sims.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    grad_cut = {}
    if args.fixed_cut_width is None:
        if args.grad_cut_csv is None or not args.grad_cut_csv.exists():
            raise FileNotFoundError("grad-cut-csv is required unless --fixed-cut-width is set.")
        with args.grad_cut_csv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                grad = row.get("thermal_gradient")
                cut_width = row.get("cut_width")
                if grad and cut_width not in (None, "", " "):
                    grad_cut[grad] = int(float(cut_width))

    manifest = json.loads(args.manifest.read_text())
    gid_meta: Dict[str, Dict[str, str]] = {}
    for name, meta in manifest.items():
        gid = meta.get("group_id")
        if not gid:
            continue
        grad = meta.get("physical_params", {}).get("thermal_gradient")
        seed = None
        if "/seed_" in name:
            seed = name.split("/seed_")[-1]
        gid_meta[gid] = {
            "original_name": name,
            "thermal_gradient": str(grad) if grad is not None else "",
            "seed": seed or "",
        }

    rows = []
    with h5py.File(args.h5, "r") as h5:
        for gid in h5.keys():
            meta = gid_meta.get(gid, {})
            grad = meta.get("thermal_gradient", "")
            if args.fixed_cut_width is not None:
                cut_width = int(args.fixed_cut_width)
            else:
                cut_width = grad_cut.get(grad)
            images = h5[gid]["images"]
            if cut_width is None:
                touch = None
            elif args.fixed_cut_width is not None:
                touch = _first_touch_frame_fixed(images, cut_width, args.tol, block=128)
            else:
                widths = _compute_widths(images, args.tol)
                idx = np.where(widths >= cut_width)[0]
                touch = int(idx[0]) if idx.size else None

            rows.append({
                "gid": gid,
                "original_name": meta.get("original_name", ""),
                "thermal_gradient": grad,
                "seed": meta.get("seed", ""),
                "cut_width": cut_width if cut_width is not None else "",
                "touch_frame": touch if touch is not None else "",
                "cutoff_frame": touch if touch is not None else "",
                "total_frames": images.shape[0],
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gid", "original_name", "thermal_gradient", "seed", "cut_width", "touch_frame", "cutoff_frame", "total_frames"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    main()
