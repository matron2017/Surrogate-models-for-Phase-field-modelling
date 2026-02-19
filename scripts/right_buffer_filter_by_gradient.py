#!/usr/bin/env python3
"""Truncate frames using a gradient-level cutoff based on earliest touching seed.

For each thermal gradient within a split:
  - compute onset per sim using median tail width rule
  - pick earliest onset across seeds
  - cutoff = onset - buffer_steps
  - apply cutoff to ALL sims of that gradient
Also exports per-sim CSV including cut_width from earliest seed at cutoff time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from right_buffer_filter import _width_for_frame, _compute_widths, _find_onset


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
    ap = argparse.ArgumentParser(description="Gradient-level right-buffer truncation.")
    ap.add_argument("--input-h5", required=True, type=Path)
    ap.add_argument("--output-h5", required=False, type=Path, default=None)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--grad-csv", required=True, type=Path, help="Per-gradient cutoff summary.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-width", type=int, default=10)
    ap.add_argument("--buffer-steps", type=int, default=10)
    ap.add_argument("--min-keep", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_h5 is not None and args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_h5} exists; use --overwrite to replace.")

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

    # pass 1: compute onset per sim and collect by gradient
    sim_info = {}
    grad_onsets: Dict[str, Dict[str, object]] = {}

    with h5py.File(args.input_h5, "r") as h5:
        for gid in h5.keys():
            images = h5[gid]["images"]
            t_len = images.shape[0]
            widths = _compute_widths(images, args.tol)
            onset, median_tail = _find_onset(widths, args.tail, args.k, args.min_width)
            meta = gid_meta.get(gid, {})
            grad = meta.get("thermal_gradient", "")
            sim_info[gid] = {
                "gid": gid,
                "grad": grad,
                "seed": meta.get("seed", ""),
                "original_name": meta.get("original_name", ""),
                "onset": onset,
                "median_tail": median_tail,
                "widths": widths,
                "total_frames": t_len,
            }
            if grad:
                entry = grad_onsets.setdefault(grad, {"best_onset": None, "best_gid": None})
                if onset is not None:
                    if entry["best_onset"] is None or onset < entry["best_onset"]:
                        entry["best_onset"] = onset
                        entry["best_gid"] = gid

    # compute gradient-level cutoff and cut_width
    grad_cut = {}
    for grad, entry in grad_onsets.items():
        onset = entry["best_onset"]
        gid = entry["best_gid"]
        if onset is None or gid is None:
            grad_cut[grad] = {"cutoff": None, "cut_width": None, "best_gid": None, "best_onset": None}
            continue
        cutoff = max(0, int(onset) - int(args.buffer_steps))
        widths = sim_info[gid]["widths"]
        if cutoff < len(widths):
            cut_width = int(widths[cutoff])
        else:
            cut_width = int(widths[-1])
        grad_cut[grad] = {
            "cutoff": cutoff,
            "cut_width": cut_width,
            "best_gid": gid,
            "best_onset": onset,
        }

    # write output H5 and per-sim CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.grad_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.output_h5 is not None:
        out_h5 = h5py.File(args.output_h5, "w")
    else:
        out_h5 = None

    with h5py.File(args.input_h5, "r") as h5:
        if out_h5 is not None:
            _copy_attrs(h5, out_h5)
        for gid, info in sim_info.items():
            grad = info["grad"]
            cut_info = grad_cut.get(grad, {})
            cutoff = cut_info.get("cutoff")
            cut_width = cut_info.get("cut_width")
            if cutoff is None:
                keep_len = info["total_frames"]
            else:
                keep_len = max(args.min_keep, int(cutoff))
            removed = info["total_frames"] - keep_len

            rows.append({
                "gid": gid,
                "original_name": info["original_name"],
                "thermal_gradient": grad,
                "seed": info["seed"],
                "onset_frame": info["onset"] if info["onset"] is not None else "",
                "cutoff_frame": cutoff if cutoff is not None else "",
                "cut_width": cut_width if cut_width is not None else "",
                "median_tail_width": info["median_tail"],
                "total_frames": info["total_frames"],
                "kept_frames": keep_len,
                "removed_frames": removed,
            })

            if out_h5 is not None:
                # copy group with truncation
                grp = h5[gid]
                out_grp = out_h5.create_group(gid)
                _copy_attrs(grp, out_grp)

                for name, ds in grp.items():
                    if name in {"pairs_idx", "pairs_time", "pairs_stride"}:
                        continue
                    if ds.ndim >= 1 and ds.shape[0] == info["total_frames"]:
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
                        out_ds = out_grp.create_dataset(
                            name,
                            data=ds[...],
                            dtype=ds.dtype,
                            compression=ds.compression,
                            compression_opts=ds.compression_opts,
                            chunks=ds.chunks,
                        )
                        _copy_attrs(ds, out_ds)

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

    # write CSVs
    fieldnames = [
        "gid",
        "original_name",
        "thermal_gradient",
        "seed",
        "onset_frame",
        "cutoff_frame",
        "cut_width",
        "median_tail_width",
        "total_frames",
        "kept_frames",
        "removed_frames",
    ]
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    with args.grad_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["thermal_gradient", "best_gid", "best_onset", "cutoff_frame", "cut_width"])
        w.writeheader()
        for grad, info in grad_cut.items():
            w.writerow({
                "thermal_gradient": grad,
                "best_gid": info.get("best_gid") or "",
                "best_onset": info.get("best_onset") if info.get("best_onset") is not None else "",
                "cutoff_frame": info.get("cutoff") if info.get("cutoff") is not None else "",
                "cut_width": info.get("cut_width") if info.get("cut_width") is not None else "",
            })


if __name__ == "__main__":
    main()
