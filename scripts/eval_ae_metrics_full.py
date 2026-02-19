#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train.core.utils import _load_symbol, _collate
from models.train.core.pf_dataloader import PFPairDataset


def _resolve_h5_path(cfg: Dict[str, Any], split: str) -> str:
    entry = cfg.get("paths", {}).get("h5", {}).get(split)
    if entry is None:
        raise KeyError(f"Missing paths.h5.{split} in config.")
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        if "h5_path" in entry:
            return entry["h5_path"]
        if len(entry) == 1:
            k, v = next(iter(entry.items()))
            if isinstance(v, str) and v:
                return v
            return str(k)
    raise ValueError(f"Unsupported paths.h5.{split} entry: {entry!r}")


def _filter_pf_args(args: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "h5_path",
        "input_channels",
        "target_channels",
        "limit_per_group",
        "max_items",
        "weight_h5",
        "weight_key",
        "identity_pairs",
        "use_pairs_idx",
        "data_key",
        "return_cond",
        "add_thermal",
        "thermal_axis",
        "thermal_use_x0",
        "thermal_T0",
        "thermal_on_target",
        "thermal_debug",
        "thermal_debug_prob",
        "augment",
        "augment_flip",
        "augment_flip_prob",
        "augment_roll",
        "augment_roll_prob",
        "augment_roll_max",
        "augment_swap",
        "augment_swap_prob",
        "augment_rotate",
        "augment_rotate_prob",
        "normalize_images",
        "normalize_force",
        "normalize_source",
    }
    return {k: v for k, v in args.items() if k in allowed}


def _build_dataset(cfg: Dict[str, Any], split: str, h5_override: str | None = None) -> PFPairDataset:
    dl_cfg = cfg.get("dataloader", {}) or {}
    args = dict(dl_cfg.get("args", {}) or {})
    split_args = dict(dl_cfg.get(f"{split}_args", {}) or {})
    args.update(split_args)
    args["h5_path"] = h5_override or _resolve_h5_path(cfg, split)
    return PFPairDataset(**_filter_pf_args(args))


def _load_model(ckpt_path: Path, cfg_fallback: Dict[str, Any]) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) or cfg_fallback
    model_cfg = cfg.get("model") or cfg_fallback.get("model")
    if not model_cfg:
        raise KeyError("Model config missing from checkpoint and config file.")
    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    model = ModelClass(**(model_cfg.get("params", {}) or {}))
    if "model" not in state:
        raise KeyError("Checkpoint missing 'model' state.")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--pct-eps", type=float, default=1e-3)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--h5-path", type=str, default="")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, cfg).to(device)

    h5_override = args.h5_path.strip() or None
    ds = _build_dataset(cfg, args.split, h5_override=h5_override)
    if args.max_items and args.max_items > 0:
        # shallow truncation for quick tests
        ds.items = ds.items[: args.max_items]
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=False,
    )

    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    sum_sq_c: Optional[torch.Tensor] = None
    sum_abs_c: Optional[torch.Tensor] = None
    sum_mape_c: Optional[torch.Tensor] = None
    count_c = 0
    mape_images = []

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        cond = batch.get("cond")
        cond = cond.to(device, non_blocking=True) if cond is not None else None
        with torch.no_grad():
            yhat = model(x, cond) if cond is not None else model(x)

        err = yhat - y
        abs_err = err.abs()
        sq = err.square()

        sum_sq += sq.sum().item()
        sum_abs += abs_err.sum().item()
        count += sq.numel()

        # per-channel accumulation
        ch_sum_sq = sq.sum(dim=(0, 2, 3))
        ch_sum_abs = abs_err.sum(dim=(0, 2, 3))
        denom = y.abs() + float(args.pct_eps)
        ch_sum_mape = (abs_err / denom).sum(dim=(0, 2, 3))
        if sum_sq_c is None:
            sum_sq_c = ch_sum_sq.detach().cpu()
            sum_abs_c = ch_sum_abs.detach().cpu()
            sum_mape_c = ch_sum_mape.detach().cpu()
        else:
            sum_sq_c += ch_sum_sq.detach().cpu()
            sum_abs_c += ch_sum_abs.detach().cpu()
            sum_mape_c += ch_sum_mape.detach().cpu()
        count_c += y.shape[0] * y.shape[2] * y.shape[3]

        # per-image MAPE
        mape_img = (abs_err / denom).mean(dim=(1, 2, 3)) * 100.0
        mape_images.extend([float(v) for v in mape_img.detach().cpu().tolist()])

    overall_rmse = math.sqrt(sum_sq / max(count, 1))
    overall_mae = sum_abs / max(count, 1)
    if sum_sq_c is None:
        raise RuntimeError("No data processed.")
    per_channel_rmse = (sum_sq_c / max(count_c, 1)).sqrt().tolist()
    per_channel_mae = (sum_abs_c / max(count_c, 1)).tolist()
    per_channel_mape = (sum_mape_c / max(count_c, 1)).tolist()

    mape_arr = np.array(mape_images, dtype=float)
    mape_summary = {
        "mean": float(np.mean(mape_arr)) if mape_arr.size else None,
        "median": float(np.median(mape_arr)) if mape_arr.size else None,
        "p90": float(np.percentile(mape_arr, 90)) if mape_arr.size else None,
        "p95": float(np.percentile(mape_arr, 95)) if mape_arr.size else None,
        "max": float(np.max(mape_arr)) if mape_arr.size else None,
        "count": int(mape_arr.size),
    }

    out = {
        "checkpoint": str(ckpt_path),
        "config": str(args.config),
        "split": args.split,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "per_channel_rmse": per_channel_rmse,
        "per_channel_mae": per_channel_mae,
        "per_channel_mape_percent": per_channel_mape,
        "mape_image_summary_percent": mape_summary,
        "mape_images_percent": mape_images,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
