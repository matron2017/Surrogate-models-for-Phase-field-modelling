#!/usr/bin/env python3
"""
Merge shard wavelet weight files into a single HDF5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import h5py


def _parse_args():
    ap = argparse.ArgumentParser(description="Merge wavelet weight shards")
    ap.add_argument("--out", type=str, required=True, help="Output HDF5 path to create")
    ap.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base output path used for shards (shards are base.partXX)",
    )
    ap.add_argument("--shard-count", type=int, required=True, help="Total number of shards")
    return ap.parse_args()


def _copy_attrs(src: h5py.File, dst: h5py.File):
    for key, val in src.attrs.items():
        dst.attrs[key] = val
    merged = {
        "base": str(src.filename),
        "shards": int(dst.attrs.get("shard_count", 0)),
    }
    dst.attrs["merged_from_parts"] = json.dumps(merged)


def main():
    args = _parse_args()
    out_path = Path(args.out).resolve()
    base_path = Path(args.base).resolve()
    shard_count = int(args.shard_count)
    if shard_count <= 0:
        raise ValueError("--shard-count must be > 0")

    if out_path.exists():
        out_path.unlink()

    part_paths = [Path(str(base_path) + f".part{rank:02d}") for rank in range(shard_count)]
    for p in part_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing shard {p}")

    with h5py.File(str(out_path), "w") as h5_out:
        with h5py.File(str(part_paths[0]), "r") as h5_first:
            _copy_attrs(h5_first, h5_out)

        for part in part_paths:
            with h5py.File(str(part), "r") as h5_part:
                for gid in h5_part.keys():
                    if gid in h5_out:
                        raise RuntimeError(f"Duplicate group {gid} in {part}")
                    h5_out.copy(h5_part[gid], gid)

    print(f"Merged {shard_count} shards into {out_path}")


if __name__ == "__main__":
    main()
