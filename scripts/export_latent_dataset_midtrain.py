#!/usr/bin/env python3
"""
Export latent HDF5 datasets from an in-progress AE checkpoint.

This is intended for development usage only while AE training is still running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core.utils import _load_symbol


def _parse_source(entry: str) -> Tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"--source must be split=path, got: {entry}")
    split, path = entry.split("=", 1)
    split = split.strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split '{split}' in --source '{entry}'.")
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Source H5 not found: {p}")
    return split, p


def _load_encoder(checkpoint: Path, cfg_path: Path | None, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config", None)
    if not cfg:
        if cfg_path is None:
            raise KeyError("Checkpoint missing config; provide --config.")
        cfg = yaml.safe_load(cfg_path.read_text())
    model_cfg = cfg.get("model", {})
    if "file" not in model_cfg or "class" not in model_cfg:
        raise KeyError("Model config must contain model.file and model.class for AE export.")
    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    model = ModelClass(**(model_cfg.get("params", {}) or {}))
    if "model" not in state:
        raise KeyError("Checkpoint missing 'model' state dict.")
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()

    encoder_owner = model.autoencoder if hasattr(model, "autoencoder") else model
    if not hasattr(encoder_owner, "encode"):
        raise TypeError("Loaded model does not expose encode() (or autoencoder.encode()).")

    meta = {
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "checkpoint_best_metric": float(state.get("best_metric", float("nan"))),
        "trainer_out_dir": str(cfg.get("trainer", {}).get("out_dir", "")),
        "model_class": str(model_cfg.get("class", "")),
    }
    return encoder_owner, meta


def _copy_attrs(src_attrs: h5py.AttributeManager, dst_attrs: h5py.AttributeManager) -> None:
    for k, v in src_attrs.items():
        dst_attrs[k] = v


def _copy_non_image_datasets(src_g: h5py.Group, dst_g: h5py.Group, images_key: str) -> List[str]:
    copied = []
    for name, obj in src_g.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        if name == images_key:
            continue
        src_g.copy(name, dst_g, name=name)
        copied.append(name)
    return copied


def _encode_chunk(
    encoder,
    x_np: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> torch.Tensor:
    x = torch.from_numpy(x_np).to(device=device, non_blocking=True)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda"), dtype=amp_dtype):
            z = encoder.encode(x)
    return z.detach().cpu()


def _export_one_split(
    split: str,
    src_path: Path,
    dst_path: Path,
    encoder,
    meta: Dict[str, Any],
    *,
    images_key: str,
    latent_key: str,
    batch_size: int,
    dtype: str,
    compression: str | None,
    compression_level: int | None,
    max_groups: int | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, Any]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_groups = 0
    total_frames = 0
    latent_shape = None
    copied_dataset_names: Dict[str, List[str]] = {}

    out_dtype = np.float16 if dtype == "float16" else np.float32

    with h5py.File(src_path, "r") as src_h5, h5py.File(dst_path, "w") as dst_h5:
        _copy_attrs(src_h5.attrs, dst_h5.attrs)
        dst_h5.attrs["dataset_kind"] = "latent_images"
        dst_h5.attrs["latent_key"] = latent_key
        dst_h5.attrs["source_split"] = split
        dst_h5.attrs["source_h5_path"] = str(src_path)
        dst_h5.attrs["source_images_key"] = images_key
        dst_h5.attrs["export_note"] = "EXPERIMENTAL_MIDTRAINING_NOT_FINAL"
        dst_h5.attrs["checkpoint_epoch"] = int(meta["checkpoint_epoch"])
        dst_h5.attrs["checkpoint_best_metric"] = float(meta["checkpoint_best_metric"])
        dst_h5.attrs["checkpoint_model_class"] = str(meta["model_class"])
        dst_h5.attrs["created_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        dst_h5.attrs["created_host"] = socket.gethostname()

        gids = sorted(src_h5.keys())
        if max_groups is not None:
            gids = gids[: max(0, int(max_groups))]

        for gid in gids:
            src_g = src_h5[gid]
            if images_key not in src_g:
                raise KeyError(f"{src_path}: group '{gid}' missing dataset '{images_key}'.")
            images = src_g[images_key]
            if images.ndim != 4:
                raise ValueError(f"{src_path}:{gid}/{images_key} expected 4D [N,C,H,W], got {images.shape}")
            n_frames = int(images.shape[0])
            if n_frames <= 0:
                continue

            dst_g = dst_h5.create_group(gid)
            _copy_attrs(src_g.attrs, dst_g.attrs)
            copied_dataset_names[gid] = _copy_non_image_datasets(src_g, dst_g, images_key=images_key)

            sample_np = np.asarray(images[0:1], dtype=np.float32)
            sample_z = _encode_chunk(encoder, sample_np, device=device, amp_dtype=amp_dtype, use_amp=use_amp)
            if sample_z.ndim != 4:
                raise ValueError(f"Encoded latent must be 4D [N,C,H,W], got {tuple(sample_z.shape)}")
            zc, zh, zw = int(sample_z.shape[1]), int(sample_z.shape[2]), int(sample_z.shape[3])
            latent_shape = (zc, zh, zw)

            chunks = (1, zc, zh, zw)
            kwargs = {}
            if compression:
                kwargs["compression"] = compression
                if compression_level is not None:
                    kwargs["compression_opts"] = int(compression_level)
            lat_ds = dst_g.create_dataset(
                latent_key,
                shape=(n_frames, zc, zh, zw),
                dtype=out_dtype,
                chunks=chunks,
                **kwargs,
            )
            lat_ds[0:1] = sample_z.numpy().astype(out_dtype, copy=False)

            for start in range(1, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                x_np = np.asarray(images[start:end], dtype=np.float32)
                z = _encode_chunk(encoder, x_np, device=device, amp_dtype=amp_dtype, use_amp=use_amp)
                lat_ds[start:end] = z.numpy().astype(out_dtype, copy=False)

            encoded_groups += 1
            total_frames += n_frames

    return {
        "split": split,
        "source_h5": str(src_path),
        "output_h5": str(dst_path),
        "encoded_groups": encoded_groups,
        "total_frames": total_frames,
        "latent_shape_chw": latent_shape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export latent HDF5 datasets from a mid-training AE checkpoint.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None, help="Optional config fallback if checkpoint has no config.")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="Split/source mapping, e.g. --source train=/path/train.h5 (repeat for val/test).",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--images-key", type=str, default="images")
    ap.add_argument("--latent-key", type=str, default="latent_images")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    ap.add_argument("--compression-level", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=0, help="Optional dev cap per split (0 means all groups).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.config is not None and not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    sources = [_parse_source(s) for s in args.source]
    by_split: Dict[str, Path] = {}
    for split, path in sources:
        if split in by_split:
            raise ValueError(f"Duplicate split in --source: {split}")
        by_split[split] = path

    seen_gids: Dict[str, str] = {}
    for split, path in sorted(by_split.items()):
        with h5py.File(path, "r") as h5:
            for gid in h5.keys():
                if gid in seen_gids:
                    raise RuntimeError(
                        f"Group overlap detected across splits: gid '{gid}' in both '{seen_gids[gid]}' and '{split}'."
                    )
                seen_gids[gid] = split

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    use_amp = True

    encoder, meta = _load_encoder(checkpoint, args.config, device=device)
    compression = None if args.compression == "none" else args.compression
    compression_level = None if compression is None else int(args.compression_level)
    max_groups = int(args.max_groups) if args.max_groups and args.max_groups > 0 else None

    summary = {
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "host": socket.gethostname(),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(meta["checkpoint_epoch"]),
        "checkpoint_best_metric": float(meta["checkpoint_best_metric"]),
        "experimental": True,
        "note": "MIDTRAINING_CHECKPOINT_NOT_FINAL",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        if split not in by_split:
            continue
        src_path = by_split[split]
        out_name = f"{split}_latent_experimental_midtrain.h5"
        out_path = out_dir / out_name
        split_summary = _export_one_split(
            split=split,
            src_path=src_path,
            dst_path=out_path,
            encoder=encoder,
            meta=meta,
            images_key=args.images_key,
            latent_key=args.latent_key,
            batch_size=int(args.batch_size),
            dtype=args.dtype,
            compression=compression,
            compression_level=compression_level,
            max_groups=max_groups,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        summary["splits"][split] = split_summary
        print(
            f"[export] split={split} groups={split_summary['encoded_groups']} "
            f"frames={split_summary['total_frames']} latent={split_summary['latent_shape_chw']} out={out_path}",
            flush=True,
        )

    summary_path = out_dir / "README_EXPERIMENTAL.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
