from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_VAL_H5, load_yaml
from .pf_dataloader import PFPairDataset


def dataset_from_config(cfg: Dict[str, Any] | None = None, split: str = "val", max_items: int | None = 1) -> PFPairDataset:
    cfg = cfg or load_yaml()
    h5_cfg = cfg["paths"]["h5"]
    h5_path = Path(h5_cfg[split]) if split in h5_cfg else DEFAULT_VAL_H5
    ds_args = dict(cfg["dataloader"].get("args", {}))
    split_args = dict(cfg["dataloader"].get(f"{split}_args", {}))
    ds_args.update(split_args)
    if max_items is not None:
        ds_args["max_items"] = max_items
    return PFPairDataset(str(h5_path), **ds_args)
