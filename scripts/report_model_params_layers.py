#!/usr/bin/env python3
"""
Report model parameter counts and layer mapping for one or more train configs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

from models.backbones.registry import build_model as registry_build_model


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _group_param_counts(model: torch.nn.Module, depth: int = 2) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else name
        out[key] = out.get(key, 0) + int(p.numel())
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _layer_map(model: torch.nn.Module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        item: Dict[str, Any] = {"name": name, "type": mod.__class__.__name__}
        if isinstance(mod, torch.nn.Conv2d):
            item.update(
                {
                    "in_channels": int(mod.in_channels),
                    "out_channels": int(mod.out_channels),
                    "kernel_size": tuple(int(x) for x in mod.kernel_size),
                    "stride": tuple(int(x) for x in mod.stride),
                }
            )
        elif isinstance(mod, torch.nn.Linear):
            item.update(
                {
                    "in_features": int(mod.in_features),
                    "out_features": int(mod.out_features),
                }
            )
        elif isinstance(mod, torch.nn.GroupNorm):
            item.update(
                {
                    "num_groups": int(mod.num_groups),
                    "num_channels": int(mod.num_channels),
                }
            )
        else:
            continue
        rows.append(item)
    return rows


def _build_from_cfg(cfg_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text())
    train = cfg.get("train", {})
    model_cfg = dict(cfg.get("model", {}))
    model_family = str(train.get("model_family", "surrogate")).lower()
    backbone = str(model_cfg.get("backbone", "")).lower()
    if not backbone:
        raise ValueError(f"{cfg_path}: missing model.backbone")

    model = registry_build_model(model_family, backbone, model_cfg)
    total, trainable = _count_params(model)
    depth1 = _group_param_counts(model, depth=1)
    depth2 = _group_param_counts(model, depth=2)

    return {
        "config": str(cfg_path),
        "model_family": model_family,
        "backbone": backbone,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": int(total - trainable),
        "param_groups_depth1": depth1,
        "param_groups_depth2_top20": dict(list(depth2.items())[:20]),
        "layer_map": _layer_map(model),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Report model params and layer mapping for configs.")
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to YAML config (repeatable).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    args = ap.parse_args()

    reports = []
    for c in args.config:
        reports.append(_build_from_cfg(Path(c).expanduser().resolve()))

    text = json.dumps({"reports": reports}, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
