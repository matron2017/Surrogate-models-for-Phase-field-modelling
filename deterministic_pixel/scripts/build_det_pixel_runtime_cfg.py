#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a runtime config for deterministic pixel DDP runs.")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--steps-per-epoch", type=int, required=True)
    ap.add_argument("--batch-per-rank", type=int, required=True)
    ap.add_argument("--accumulation-steps", type=int, required=True)
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--use-val", type=int, default=1)
    ap.add_argument("--limit-per-group", type=int, default=None)
    ap.add_argument("--max-items", default=None)
    args = ap.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("trainer", {})
    cfg.setdefault("loader", {})
    cfg.setdefault("dataloader", {})
    cfg["dataloader"].setdefault("train_args", {})
    cfg["dataloader"].setdefault("val_args", {})

    cfg["trainer"]["out_dir"] = str(Path(args.out_dir).resolve())
    cfg["trainer"]["epochs"] = int(args.epochs)
    cfg["trainer"]["steps_per_epoch"] = int(args.steps_per_epoch)
    cfg["trainer"]["accumulation_steps"] = int(args.accumulation_steps)
    cfg["trainer"]["use_val"] = bool(int(args.use_val))
    cfg["trainer"]["resume"] = False

    cfg["loader"]["batch_size"] = int(args.batch_per_rank)
    cfg["loader"]["num_workers"] = int(args.num_workers)

    if args.limit_per_group is not None:
        cfg["dataloader"]["train_args"]["limit_per_group"] = int(args.limit_per_group)
        cfg["dataloader"]["val_args"]["limit_per_group"] = int(args.limit_per_group)

    if args.max_items is not None:
        try:
            max_items = float(args.max_items)
            if max_items.is_integer():
                max_items = int(max_items)
        except ValueError:
            max_items = args.max_items
        cfg["dataloader"]["train_args"]["max_items"] = max_items
        cfg["dataloader"]["val_args"]["max_items"] = max_items

    out_path = Path(args.out_config).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
