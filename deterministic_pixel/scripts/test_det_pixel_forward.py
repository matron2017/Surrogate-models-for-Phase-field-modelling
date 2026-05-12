#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path("/scratch/project_2008261/pf_surrogate_modelling")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_field_surrogates.models.unet_film_bottleneck import UNetFiLMAttn
from Phase_field_surrogates.utils.pf_dataloader import PFPairDataset
from Phase_field_surrogates.utils.wavelet_weight import wavelet_multiband_importance_per_channel


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str, max_items: int = 1):
    ds_args = dict(cfg["dataloader"].get("args", {}))
    split_args = dict(cfg["dataloader"].get(f"{split}_args", {}))
    ds_args.update(split_args)
    ds_args["max_items"] = max_items
    return PFPairDataset(cfg["paths"]["h5"][split], **ds_args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ds = build_dataset(cfg, args.split, max_items=1)
    sample = ds[0]

    x_full = sample["input"].unsqueeze(0).float()
    y = sample["target"].unsqueeze(0).float()
    x = x_full[:, :2]
    theta = x_full[:, 2:3]
    cond_vec = torch.zeros((1, int(cfg["model"]["params"].get("cond_dim", 2))), dtype=torch.float32)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = UNetFiLMAttn(**dict(cfg["model"]["params"]))
    model = model.to(device).eval()
    x = x.to(device)
    y = y.to(device)
    theta = theta.to(device)
    cond_vec = cond_vec.to(device)

    with torch.inference_mode():
        out = model(x, cond_vec, theta=theta)

    wavelet_cfg = dict(cfg["loss"].get("wavelet_weight", {}))
    wavelet_enabled = float(cfg["loss"].get("weight_wavelet_loss", 0.0)) > 0.0
    wavelet_compute_ok = False
    wavelet_shape = None
    wavelet_error = None
    if wavelet_enabled:
        try:
            wmap, _ = wavelet_multiband_importance_per_channel(
                y,
                J=int(wavelet_cfg.get("J", 3)),
                wave=str(wavelet_cfg.get("wave", "haar")),
                mode=str(wavelet_cfg.get("mode", "zero")),
                level_weights=list(wavelet_cfg.get("level_weights", [2.0, 1.4, 0.7])),
                lowpass_weight=float(wavelet_cfg.get("lowpass_weight", 1.2)),
                beta_w=float(wavelet_cfg.get("beta_w", 120.0)),
                power=float(wavelet_cfg.get("power", 1.8)),
                norm_quantile=float(wavelet_cfg.get("norm_quantile", 0.95)),
                normalize_mean=bool(wavelet_cfg.get("normalize", True)),
                rescale_max=bool(wavelet_cfg.get("rescale_max", False)),
                clip_max=float(wavelet_cfg.get("clip_max", 16.0)),
                combine_norm=bool(wavelet_cfg.get("combine_norm", True)),
            )
            wavelet_compute_ok = True
            wavelet_shape = list(wmap.shape)
        except Exception as e:
            wavelet_error = repr(e)

    report = {
        "config": str(Path(args.config).resolve()),
        "gid": sample["gid"],
        "pair_index": int(sample["pair_index"]),
        "input_shape": list(sample["input"].shape),
        "target_shape": list(sample["target"].shape),
        "model_x_shape": list(x.shape),
        "theta_shape": list(theta.shape),
        "cond_vec_shape": list(cond_vec.shape),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_device": str(out.device),
        "wavelet_enabled": wavelet_enabled,
        "wavelet_compute_ok": wavelet_compute_ok,
        "wavelet_shape": wavelet_shape,
        "wavelet_error": wavelet_error,
        "mse_to_target": float(torch.mean((out - y) ** 2).item()),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
