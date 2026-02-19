#!/usr/bin/env python3
"""Diagnose right-boundary buffer onset and optionally truncate frames after onset.

Creates a CSV summary per input H5 and (optionally) a cleaned H5 with
frames truncated at the detected onset for each simulation group.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


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


def _compute_widths(images: h5py.Dataset, tol: float) -> np.ndarray:
    t_len = images.shape[0]
    widths = np.zeros(t_len, dtype=np.int32)
    for t in range(t_len):
        widths[t] = _width_for_frame(images[t], tol)
    return widths


def _find_onset(widths: np.ndarray, tail: int, k: int, min_width: int) -> Tuple[Optional[int], int]:
    if widths.size == 0:
        return None, 0
    tail = min(tail, widths.size)
    median_tail = int(np.median(widths[-tail:]))
    if median_tail < min_width:
        return None, median_tail
    low, high = median_tail - 2, median_tail + 2
    onset = None
    for t in range(0, widths.size - k + 1):
        window = widths[t : t + k]
        if np.all((window >= low) & (window <= high)) and np.all(window >= min_width):
            onset = t
            break
    return onset, median_tail


def compute_onset_and_median(images: h5py.Dataset, tol: float, tail: int, k: int, min_width: int) -> Tuple[Optional[int], int, np.ndarray]:
    widths = _compute_widths(images, tol)
    onset, median_tail = _find_onset(widths, tail, k, min_width)
    return onset, median_tail, widths


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _safe_chunks(chunks: Optional[Tuple[int, ...]], shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if chunks is None:
        return None
    new_chunks = list(chunks)
    for i in range(len(new_chunks)):
        if new_chunks[i] > shape[i]:
            new_chunks[i] = max(1, shape[i])
    return tuple(new_chunks)


def _copy_time_series_dataset(src: h5py.Dataset, dst: h5py.Dataset, keep_len: int, chunk: int) -> None:
    for start in range(0, keep_len, chunk):
        end = min(keep_len, start + chunk)
        dst[start:end] = src[start:end]


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose right buffer onset and truncate frames.")
    ap.add_argument("--input-h5", required=True, type=Path)
    ap.add_argument("--output-h5", type=Path, default=None)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="sim_manifest.json for grad/seed info")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-width", type=int, default=10)
    ap.add_argument(
        "--buffer-steps",
        type=int,
        default=0,
        help="Remove this many steps before onset (e.g., 5 means keep frames < onset-5).",
    )
    ap.add_argument("--min-keep", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2, help="time-chunk size when copying arrays")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_h5 is not None and args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_h5} exists; use --overwrite to replace.")

    manifest_meta: Dict[str, Dict[str, str]] = {}
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
                "thermal_gradient": str(grad) if grad is not None else "",
                "seed": seed or "",
            }

    out_h5 = None
    if args.output_h5 is not None:
        out_h5 = h5py.File(args.output_h5, "w")

    rows = []
    with h5py.File(args.input_h5, "r") as h5:
        if out_h5 is not None:
            _copy_attrs(h5, out_h5)
        for gid in h5.keys():
            grp = h5[gid]
            images = grp["images"]
            t_len = images.shape[0]

            widths = _compute_widths(images, args.tol)
            onset, median_tail = _find_onset(widths, args.tail, args.k, args.min_width)
            if onset is None:
                cutoff = None
                keep_len = t_len
            else:
                cutoff = max(0, int(onset) - int(args.buffer_steps))
                keep_len = max(args.min_keep, cutoff)
            removed = t_len - keep_len

            row = {
                "gid": gid,
                "onset_frame": onset if onset is not None else "",
                "cutoff_frame": cutoff if onset is not None else "",
                "median_tail_width": median_tail,
                "total_frames": t_len,
                "kept_frames": keep_len,
                "removed_frames": removed,
                "width_t300": widths[300] if t_len > 300 else "",
                "width_t350": widths[350] if t_len > 350 else "",
                "width_t400": widths[400] if t_len > 400 else "",
                "width_tlast": widths[-1] if t_len > 0 else "",
            }
            row.update(manifest_meta.get(gid, {}))
            rows.append(row)

            if out_h5 is None:
                continue

            out_grp = out_h5.create_group(gid)
            _copy_attrs(grp, out_grp)

            # copy datasets
            for name, ds in grp.items():
                if name in {"pairs_idx", "pairs_time", "pairs_stride"}:
                    continue
                if ds.ndim >= 1 and ds.shape[0] == t_len:
                    new_shape = (keep_len,) + ds.shape[1:]
                    chunks = _safe_chunks(ds.chunks, new_shape)
                    out_ds = out_grp.create_dataset(
                        name,
                        shape=new_shape,
                        dtype=ds.dtype,
                        compression=ds.compression,
                        compression_opts=ds.compression_opts,
                        chunks=chunks,
                    )
                    _copy_time_series_dataset(ds, out_ds, keep_len, args.chunk)
                    _copy_attrs(ds, out_ds)
                else:
                    # non-time dataset: copy whole
                    out_ds = out_grp.create_dataset(
                        name,
                        data=ds[...],
                        dtype=ds.dtype,
                        compression=ds.compression,
                        compression_opts=ds.compression_opts,
                        chunks=ds.chunks,
                    )
                    _copy_attrs(ds, out_ds)

            # pairs datasets: filter by keep_len
            if "pairs_idx" in grp:
                pidx = grp["pairs_idx"][...]
                mask = pidx[:, 1] < keep_len
                pidx_new = pidx[mask]
                out_grp.create_dataset(
                    "pairs_idx",
                    data=pidx_new,
                    dtype=pidx_new.dtype,
                    compression=grp["pairs_idx"].compression,
                    compression_opts=grp["pairs_idx"].compression_opts,
                    chunks=_safe_chunks(grp["pairs_idx"].chunks, pidx_new.shape),
                )
                if "pairs_time" in grp:
                    ptime = grp["pairs_time"][...][mask]
                    out_grp.create_dataset(
                        "pairs_time",
                        data=ptime,
                        dtype=ptime.dtype,
                        compression=grp["pairs_time"].compression,
                        compression_opts=grp["pairs_time"].compression_opts,
                        chunks=_safe_chunks(grp["pairs_time"].chunks, ptime.shape),
                    )
                if "pairs_stride" in grp:
                    pstride = grp["pairs_stride"][...][mask]
                    out_grp.create_dataset(
                        "pairs_stride",
                        data=pstride,
                        dtype=pstride.dtype,
                        compression=grp["pairs_stride"].compression,
                        compression_opts=grp["pairs_stride"].compression_opts,
                        chunks=_safe_chunks(grp["pairs_stride"].chunks, pstride.shape),
                    )

    if out_h5 is not None:
        out_h5.close()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gid",
        "original_name",
        "thermal_gradient",
        "seed",
        "onset_frame",
        "cutoff_frame",
        "median_tail_width",
        "total_frames",
        "kept_frames",
        "removed_frames",
        "width_t300",
        "width_t350",
        "width_t400",
        "width_tlast",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
