#!/usr/bin/env python3
# number_params_models.py

import os, sys, json, argparse
from pathlib import Path
from itertools import product
from typing import Dict, Any, Iterable, Tuple, List

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- models ------------------------------------------------------------------
from models.backbones.uafno_cond import UAFNO_PreSkip_Full  # type: ignore

# Try UNet from preferred file, then fallback
try:
    from models.backbones.unet_ssa_preskip_full import UNet_SSA_PreSkip_Full  # type: ignore
except ModuleNotFoundError:
    from models.backbones.unet_conv_att_cond import UNet_SSA_PreSkip_Full  # type: ignore

# --- utils -------------------------------------------------------------------
def count_params(m: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def fmt_int(n: int) -> str:
    return f"{n:,d} ({n/1e6:.3f} M)"

def cartesian(grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [list(grid[k]) for k in keys]
    from itertools import product as _prod
    for combo in _prod(*vals):
        yield dict(zip(keys, combo))

# --- grids (2×1024×1024, identical conditioning) -----------------------------
def grid_uafno(H: int, W: int, C: int) -> Iterable[Dict[str, Any]]:
    base = dict(
        n_channels=C, n_classes=C, cond_dim=2,
        afno_inp_shape=(H // 16, W // 16),
        afno_mlp_ratio=8.0,
    )
    grid = {"in_factor": [40], "afno_depth": [8]}
    for kw in cartesian(grid):
        cfg = dict(base); cfg.update(kw); yield cfg

def grid_unet_ssa(C: int) -> Iterable[Dict[str, Any]]:
    grid = {
        "n_channels": [C], "n_classes": [C], "cond_dim": [2],
        "in_factor": [75],
        "ssa_qk_ratio": [1/8],
        "ssa_heads": [4],
    }
    for kw in cartesian(grid):
        yield kw

# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    assert args.H % 16 == 0 and args.W % 16 == 0

    torch.set_grad_enabled(False)
    dev = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    rows: List[Dict[str, Any]] = []

    for cfg in grid_uafno(args.H, args.W, args.channels):
        try:
            m = UAFNO_PreSkip_Full(**cfg).to(dev)
            total, trainable = count_params(m)
            rows.append({"model": "UAFNO_PreSkip_Full", "total_params": total, "trainable_params": trainable, "config": cfg})
        except Exception as e:
            rows.append({"model": "UAFNO_PreSkip_Full", "error": f"{type(e).__name__}: {e}", "config": cfg})

    for cfg in grid_unet_ssa(args.channels):
        try:
            m = UNet_SSA_PreSkip_Full(**cfg).to(dev)
            total, trainable = count_params(m)
            rows.append({"model": "UNet_SSA_PreSkip_Full", "total_params": total, "trainable_params": trainable, "config": cfg})
        except Exception as e:
            rows.append({"model": "UNet_SSA_PreSkip_Full", "error": f"{type(e).__name__}: {e}", "config": cfg})

    if args.json:
        for r in rows: print(json.dumps(r, separators=(",", ":"), sort_keys=False))
    else:
        hdr = f"{'model':26s} | {'trainable':>18s} | {'total':>18s} | config"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            if "error" in r:
                print(f"{r['model']:26s} | {'-':>18s} | {'-':>18s} | error={r['error']} cfg={json.dumps(r['config'], separators=(',', ':'))}")
            else:
                print(f"{r['model']:26s} | {fmt_int(r['trainable_params']):>18s} | {fmt_int(r['total_params']):>18s} | {json.dumps(r['config'], separators=(',', ':'))}")

if __name__ == "__main__":
    main()
