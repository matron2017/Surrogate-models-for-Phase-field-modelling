#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from Phase_field_surrogates.utils.config import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_yaml, wavelet_flags
from Phase_field_surrogates.utils.data import dataset_from_config
from Phase_field_surrogates.utils.model import build_model_from_config, make_model_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Shape and forward smoke test for the deterministic Puhti AFNO bridge extraction.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--full-model", action="store_true", help="Instantiate the full 452M parameter config. Default uses a tiny shape-compatible model.")
    parser.add_argument("--require-wavelet", action="store_true", help="Fail if trainer.use_wavelet_weights is false or loss.weight_wavelet_loss <= 0.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    flags = wavelet_flags(cfg)
    ds = dataset_from_config(cfg, split=args.split, max_items=1)
    sample = ds[0]
    model_x, cond_vec, timestep, thermal = make_model_inputs(sample)

    assert tuple(sample["input"].shape) == (33, 64, 64), tuple(sample["input"].shape)
    assert tuple(sample["target"].shape) == (32, 64, 64), tuple(sample["target"].shape)
    assert tuple(model_x.shape) == (1, 64, 64, 64), tuple(model_x.shape)
    assert tuple(cond_vec.shape) == (1, 1), tuple(cond_vec.shape)
    assert tuple(timestep.shape) == (1, 1), tuple(timestep.shape)
    assert tuple(thermal.shape) == (1, 1, 64, 64), tuple(thermal.shape)

    model = build_model_from_config(cfg, tiny=not args.full_model).eval()
    with torch.inference_mode():
        out = model(model_x, cond_vec, timestep, theta=thermal)
    assert tuple(out.shape) == (1, 32, 64, 64), tuple(out.shape)

    wavelet_enabled = flags["trainer.use_wavelet_weights"] and flags["loss.weight_wavelet_loss"] > 0.0
    if args.require_wavelet and not wavelet_enabled:
        raise AssertionError(f"wavelet weighting is not enabled: {flags}")

    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_exists": DEFAULT_CHECKPOINT.exists(),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "dataset_len_checked": len(ds),
        "input_shape": list(sample["input"].shape),
        "target_shape": list(sample["target"].shape),
        "model_input_shape": list(model_x.shape),
        "cond_vec_shape": list(cond_vec.shape),
        "timestep_shape": list(timestep.shape),
        "thermal_shape": list(thermal.shape),
        "output_shape": list(out.shape),
        "used_full_model": bool(args.full_model),
        "wavelet_flags": flags,
        "wavelet_enabled": wavelet_enabled,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
