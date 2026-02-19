#!/usr/bin/env python3
"""Truncate H5 time series using per-sim cutoff_frame from a CSV.

cutoff_frame is interpreted as the LAST frame to keep (inclusive).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py


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


def _to_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Truncate H5 using cutoff_frame from CSV (inclusive).")
    ap.add_argument("--input-h5", required=True, type=Path)
    ap.add_argument("--output-h5", required=True, type=Path)
    ap.add_argument("--cutoff-csv", required=True, type=Path)
    ap.add_argument("--summary-csv", type=Path, default=None)
    ap.add_argument("--min-keep", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_h5} exists; use --overwrite to replace.")

    cutoff_map: Dict[str, int] = {}
    with args.cutoff_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            gid = row.get("gid")
            cutoff = _to_int(row.get("cutoff_frame"))
            if gid and cutoff is not None:
                cutoff_map[gid] = cutoff

    summary_rows = []
    with h5py.File(args.input_h5, "r") as h5, h5py.File(args.output_h5, "w") as out_h5:
        _copy_attrs(h5, out_h5)
        for gid in h5.keys():
            grp = h5[gid]
            t_len = grp["images"].shape[0]
            cutoff = cutoff_map.get(gid)
            if cutoff is None:
                keep_len = t_len
            else:
                keep_len = max(args.min_keep, min(t_len, cutoff + 1))
            removed = t_len - keep_len

            summary_rows.append(
                {
                    "gid": gid,
                    "cutoff_frame": cutoff if cutoff is not None else "",
                    "total_frames": t_len,
                    "kept_frames": keep_len,
                    "removed_frames": removed,
                }
            )

            out_grp = out_h5.create_group(gid)
            _copy_attrs(grp, out_grp)

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

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["gid", "cutoff_frame", "total_frames", "kept_frames", "removed_frames"],
            )
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)


if __name__ == "__main__":
    main()
