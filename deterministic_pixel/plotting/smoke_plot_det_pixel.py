#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader, Subset
import sys

ROOT = Path("/scratch/project_462001338/pf_surrogate_modelling")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _prepare_batch
from models.train.core.loops import _forward_surrogate
from plot_pair_style import render_gt_pred_rows, default_field_specs


def _to_tensor_pred(pred):
    if isinstance(pred, dict):
        if "recon" in pred:
            return pred["recon"]
        if "pred" in pred:
            return pred["pred"]
        raise KeyError("Model output dict missing recon/pred.")
    return pred


def find_dataset_index_by_pair(ds, pair_index: int):
    p = int(pair_index)
    for i in range(len(ds)):
        s = ds[i]
        if int(s.get("pair_index", -1)) == p:
            return i, str(s.get("gid", ""))
    raise ValueError(f"pair_index={p} not found in dataset (len={len(ds)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--pair-index", type=int, default=None)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    model_cfg = cfg["model"]
    backbone = str(model_cfg.get("backbone", "")).strip().lower()
    model = registry_build_model("surrogate", backbone, model_cfg)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev).eval()

    dcfg = cfg["dataloader"]
    base = dict(dcfg.get("args", {}) or {})
    sp = dict(dcfg.get(f"{args.split}_args", {}) or {})
    if args.pair_index is not None:
        sp.pop("max_items", None)
    h5 = cfg["paths"]["h5"][args.split]
    ds = PFPairDataset(h5_path=h5, **{**base, **sp})

    chosen_idx = int(args.index)
    chosen_pair = None
    chosen_gid = ""
    if args.pair_index is not None:
        chosen_idx, chosen_gid = find_dataset_index_by_pair(ds, int(args.pair_index))
        chosen_pair = int(args.pair_index)

    dl = DataLoader(Subset(ds, [int(chosen_idx)]), batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    if chosen_pair is None:
        chosen_pair = int(batch.get("pair_index", torch.tensor([-1]))[0].item()) if isinstance(batch.get("pair_index", None), torch.Tensor) else int(batch.get("pair_index", -1))
        chosen_gid = str(batch.get("gid", [""])[0]) if isinstance(batch.get("gid", None), list) else str(batch.get("gid", ""))

    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    x, y, cond, theta = _prepare_batch(batch, dev, cond_cfg=cond_cfg, use_chlast=False)

    with torch.inference_mode():
        pred = _to_tensor_pred(_forward_surrogate(model, x, cond, theta))

    xin = x[0].detach().cpu().numpy()
    ygt = y[0].detach().cpu().numpy()
    ypr = pred[0].detach().cpu().numpy()

    rmse_pred = float(np.sqrt(np.mean((ypr - ygt) ** 2)))
    rmse_copy = float(np.sqrt(np.mean((xin - ygt) ** 2)))
    ratio = rmse_pred / max(rmse_copy, 1e-12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_gt_pred_rows(
        args.out,
        gt_arrays=[ygt[c] for c in range(min(2, ygt.shape[0]))],
        pred_arrays=[ypr[c] for c in range(min(2, ypr.shape[0]))],
        field_specs=default_field_specs(min(2, ygt.shape[0])),
        title=f"pair={chosen_pair}; gid={chosen_gid}",
        footer_lines=[
            f"RMSE(pred,target)={rmse_pred:.4e}",
            f"RMSE(copy,target)={rmse_copy:.4e}",
            f"ratio={ratio:.3f}",
        ],
    )

    print(
        {
            "rmse_pred": rmse_pred,
            "rmse_copy": rmse_copy,
            "ratio_pred_over_copy": ratio,
            "plot": str(args.out),
            "pair_index": int(chosen_pair),
            "gid": chosen_gid,
            "dataset_index": int(chosen_idx),
        }
    )


if __name__ == "__main__":
    main()
