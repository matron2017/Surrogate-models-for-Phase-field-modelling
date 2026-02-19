#!/usr/bin/env python3
"""
Validate experimental latent HDF5 splits against source splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def _as_path_map(entries: List[str], expect_suffix: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for e in entries:
        if "=" not in e:
            raise ValueError(f"Expected split=path, got: {e}")
        split, path = e.split("=", 1)
        split = split.strip().lower()
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        if split in out:
            raise ValueError(f"Duplicate split entry: {split}")
        out[split] = p
    required = {"train", "val", "test"}
    missing = sorted(required - set(out.keys()))
    if missing:
        raise ValueError(f"Missing split mappings for: {missing}")
    return out


def _validate_one_split(
    split: str,
    src_path: Path,
    lat_path: Path,
    images_key: str,
    latent_key: str,
    expected_channels: int | None,
    allow_subset: bool,
) -> Dict[str, object]:
    errors: List[str] = []
    n_groups = 0
    total_frames = 0
    latent_shapes = set()
    bad_pairs = 0

    with h5py.File(src_path, "r") as src_h5, h5py.File(lat_path, "r") as lat_h5:
        src_gids = sorted(src_h5.keys())
        lat_gids = sorted(lat_h5.keys())
        if allow_subset:
            extra = [g for g in lat_gids if g not in src_gids]
            if extra:
                errors.append(
                    f"Latent has groups not present in source for {split}: {extra[:5]}"
                )
        else:
            if src_gids != lat_gids:
                errors.append(
                    f"Group mismatch for {split}: source={len(src_gids)} latent={len(lat_gids)}"
                )
        if lat_h5.attrs.get("source_split", None) != split:
            errors.append(
                f"source_split attr mismatch for {split}: got {lat_h5.attrs.get('source_split')!r}"
            )
        if "EXPERIMENTAL" not in str(lat_h5.attrs.get("export_note", "")):
            errors.append(f"Missing experimental marker attr in {lat_path.name}")

        gids = [g for g in src_gids if g in lat_h5]
        for gid in gids:
            n_groups += 1
            src_g = src_h5[gid]
            lat_g = lat_h5[gid]
            if images_key not in src_g:
                errors.append(f"{split}:{gid} missing source dataset '{images_key}'")
                continue
            if latent_key not in lat_g:
                errors.append(f"{split}:{gid} missing latent dataset '{latent_key}'")
                continue

            src_n = int(src_g[images_key].shape[0])
            lat_arr = lat_g[latent_key]
            lat_n = int(lat_arr.shape[0])
            if lat_n != src_n:
                errors.append(f"{split}:{gid} frame count mismatch source={src_n} latent={lat_n}")
            total_frames += lat_n

            if lat_arr.ndim != 4:
                errors.append(f"{split}:{gid} latent rank must be 4, got {lat_arr.shape}")
            else:
                latent_shapes.add(tuple(int(x) for x in lat_arr.shape[1:]))
                if expected_channels is not None and int(lat_arr.shape[1]) != int(expected_channels):
                    errors.append(
                        f"{split}:{gid} latent channels mismatch expected={expected_channels} got={lat_arr.shape[1]}"
                    )

            if "pairs_idx" in lat_g:
                pidx = lat_g["pairs_idx"][:]
                if pidx.size > 0:
                    if np.max(pidx) >= lat_n or np.min(pidx) < 0:
                        bad_pairs += 1
                        errors.append(f"{split}:{gid} pairs_idx out of bounds for latent length {lat_n}")

    return {
        "split": split,
        "source_h5": str(src_path),
        "latent_h5": str(lat_path),
        "groups": n_groups,
        "frames": total_frames,
        "latent_shapes_chw": sorted(list(latent_shapes)),
        "bad_pairs_groups": bad_pairs,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate latent split HDF5 files.")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="Split/source mapping, e.g. --source train=/path/train.h5",
    )
    ap.add_argument(
        "--latent",
        action="append",
        required=True,
        help="Split/latent mapping, e.g. --latent train=/path/train_latent.h5",
    )
    ap.add_argument("--images-key", type=str, default="images")
    ap.add_argument("--latent-key", type=str, default="latent_images")
    ap.add_argument("--expected-channels", type=int, default=32)
    ap.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow latent files to contain only a subset of source groups (e.g. quick exports).",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    src = _as_path_map(args.source, "source")
    lat = _as_path_map(args.latent, "latent")

    summary = {"ok": True, "splits": {}}
    for split in ("train", "val", "test"):
        res = _validate_one_split(
            split=split,
            src_path=src[split],
            lat_path=lat[split],
            images_key=args.images_key,
            latent_key=args.latent_key,
            expected_channels=(args.expected_channels if args.expected_channels > 0 else None),
            allow_subset=bool(args.allow_subset),
        )
        summary["splits"][split] = res
        if res["errors"]:
            summary["ok"] = False

    text = json.dumps(summary, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text)
    print(text)
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
