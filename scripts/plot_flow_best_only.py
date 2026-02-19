#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_latent_flow_bridge_fields import (
    _decoder_call,
    _denorm_if_needed,
    _extract_norm_stats,
    _load_ae_model,
    _load_train_checkpoint,
    _predict_flow_source_anchored,
    _rescale_to_unit,
    _resolve_dataset_cfg,
    _split_theta,
)

CANONICAL_FLOW_CKPT = ROOT / (
    "runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/"
    "UNetFiLMAttn/checkpoint.best.pth"
)
CANONICAL_AE_CKPT = ROOT / (
    "runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_"
    "rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth"
)
CANONICAL_VAL_H5 = ROOT / "data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5"


def _parse_indices(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _panel_stats(arr: np.ndarray, cbar_min: float, cbar_max: float) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "cbar_min": float(cbar_min),
        "cbar_max": float(cbar_max),
    }


def _render_sample(
    out_png: Path,
    best_epoch: int,
    idx: int,
    gid: str,
    pair_idx: int,
    x_dec: np.ndarray,
    y_dec: np.ndarray,
    p_dec: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    ph_src_min = float(np.nanmin([np.nanmin(x_dec[0]), np.nanmin(y_dec[0]), np.nanmin(p_dec[0])]))
    ph_src_max = float(np.nanmax([np.nanmax(x_dec[0]), np.nanmax(y_dec[0]), np.nanmax(p_dec[0])]))
    if not np.isfinite(ph_src_min) or not np.isfinite(ph_src_max) or ph_src_max <= ph_src_min:
        ph_src_min, ph_src_max = -1.0, 1.0

    x0 = _rescale_to_unit(x_dec[0], ph_src_min, ph_src_max)
    y0 = _rescale_to_unit(y_dec[0], ph_src_min, ph_src_max)
    p0 = _rescale_to_unit(p_dec[0], ph_src_min, ph_src_max)
    d0 = y0 - x0
    r0 = p0 - y0

    x1 = x_dec[1]
    y1 = y_dec[1]
    p1 = p_dec[1]
    d1 = y1 - x1
    r1 = p1 - y1

    rows = [
        ("phase", [x0, y0, d0, p0, r0], "RdBu_r"),
        ("conc", [x1, y1, d1, p1, r1], "viridis"),
    ]
    col_titles = ["In", "GT", "d(G-I)", "Pred", "Res"]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for r, (name, arrs, cmap_val) in enumerate(rows):
        vmin = float(np.nanmin([np.nanmin(arrs[0]), np.nanmin(arrs[1]), np.nanmin(arrs[3])]))
        vmax = float(np.nanmax([np.nanmax(arrs[0]), np.nanmax(arrs[1]), np.nanmax(arrs[3])]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = -1.0, 1.0
        dres_abs = float(max(np.nanmax(np.abs(arrs[2])), np.nanmax(np.abs(arrs[4]))))
        if not np.isfinite(dres_abs) or dres_abs <= 0:
            dres_abs = 1.0

        row_stats: Dict[str, Dict[str, float]] = {}
        for c, arr in enumerate(arrs):
            ax = axes[r, c]
            ax.set_title(f"{name} {col_titles[c]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            if c in (2, 4):
                lo, hi, cmap = -dres_abs, dres_abs, "seismic"
            else:
                lo, hi, cmap = vmin, vmax, cmap_val

            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_ticks([lo, hi])
            cb.ax.tick_params(labelsize=8)
            row_stats[col_titles[c]] = _panel_stats(arr, lo, hi)
        stats[name] = row_stats

    fig.suptitle(f"FLOW best e{best_epoch} | idx={idx} gid={gid} pair={pair_idx}", fontsize=12)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Flow-only best-checkpoint visuals with compact labels.")
    ap.add_argument("--flow-ckpt", type=Path, default=CANONICAL_FLOW_CKPT)
    ap.add_argument("--ae-ckpt", type=Path, default=CANONICAL_AE_CKPT)
    ap.add_argument("--h5-override", type=Path, default=CANONICAL_VAL_H5)
    ap.add_argument("--indices", type=str, default="0,255,300")
    ap.add_argument("--flow-nfe", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/visuals/flow_best_only")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scale-batch-size", type=int, default=16)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    flow_ckpt = args.flow_ckpt.expanduser().resolve()
    ae_ckpt = args.ae_ckpt.expanduser().resolve()
    h5_override = args.h5_override.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    indices = _parse_indices(args.indices)

    dev = args.device.strip().lower()
    if dev != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    flow_model, flow_cfg = _load_train_checkpoint(flow_ckpt, device)
    ae_model = _load_ae_model(ae_ckpt, device)
    ds = _resolve_dataset_cfg(flow_cfg, split="val", h5_override=h5_override)

    h5_path = Path(ds.h5_path)
    mean, std, schema = _extract_norm_stats(h5_path, device=device)
    cond_cfg = dict(flow_cfg.get("conditioning", {}) or {})
    flow_train_cfg = dict(flow_cfg.get("flow_matching", {}) or {})
    flow_objective = str(
        flow_cfg.get("train", {}).get("objective", "rectified_flow_source_anchored_concat")
    ).lower()
    flow_noise_std = float(flow_train_cfg.get("noise_stochastic_std", 0.0))
    flow_noise_mode = str(flow_train_cfg.get("noise_stochastic_mode", "scalar")).strip().lower()
    flow_noise_perturb_source = bool(flow_train_cfg.get("noise_stochastic_perturb_source", True))

    ck = torch.load(flow_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck.get("epoch", -1))
    best_metric = float(ck.get("best_metric", float("nan")))

    rendered = []
    for idx in indices:
        if idx < 0 or idx >= len(ds):
            continue
        s = ds[idx]
        gid = str(s["gid"])
        pair_idx = int(s["pair_index"])
        x = s["input"].to(device).float()
        y = s["target"].to(device).float()
        x_state, theta = _split_theta(x, cond_cfg)
        y_batch = y.unsqueeze(0)

        with torch.inference_mode():
            y_flow = _predict_flow_source_anchored(
                model=flow_model,
                x=x_state,
                theta=theta,
                nfe=max(1, int(args.flow_nfe)),
                flow_objective=flow_objective,
                flow_noise_std=flow_noise_std,
                flow_noise_mode=flow_noise_mode,
                flow_noise_perturb_source=flow_noise_perturb_source,
            )
            x_dec = _decoder_call(ae_model, x_state)
            y_dec = _decoder_call(ae_model, y_batch)
            p_dec = _decoder_call(ae_model, y_flow)
            x_dec = _denorm_if_needed(x_dec, mean, std, schema)[0].detach().cpu().numpy()
            y_dec = _denorm_if_needed(y_dec, mean, std, schema)[0].detach().cpu().numpy()
            p_dec = _denorm_if_needed(p_dec, mean, std, schema)[0].detach().cpu().numpy()

        png = out_dir / f"flow_best_idx{idx:04d}_{gid}_pair{pair_idx:04d}.png"
        stats = _render_sample(
            out_png=png,
            best_epoch=best_epoch,
            idx=int(idx),
            gid=gid,
            pair_idx=pair_idx,
            x_dec=x_dec,
            y_dec=y_dec,
            p_dec=p_dec,
        )
        rendered.append(
            {
                "index": int(idx),
                "gid": gid,
                "pair_index": pair_idx,
                "png": str(png),
                "stats": stats,
            }
        )

    manifest = {
        "flow_ckpt": str(flow_ckpt),
        "flow_best_epoch": best_epoch,
        "flow_best_metric": best_metric,
        "flow_nfe": int(args.flow_nfe),
        "ae_ckpt": str(ae_ckpt),
        "dataset_h5": str(h5_path),
        "train_h5": str((((flow_cfg.get("paths", {}) or {}).get("h5", {}) or {}).get("train", ""))),
        "val_h5": str((((flow_cfg.get("paths", {}) or {}).get("h5", {}) or {}).get("val", ""))),
        "normalization_schema": schema,
        "indices_requested": [int(i) for i in indices],
        "indices_rendered": [int(r["index"]) for r in rendered],
        "value_colorbar_policy": "shared_in_gt_pred_per_figure",
        "value_scales_dataset_global": None,
        "colorbar_ticks": "min_max_only_actual_values_shown",
        "samples": rendered,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[done] {out_dir}", flush=True)
    print(f"[best_epoch] {best_epoch}", flush=True)
    print(f"[best_metric] {best_metric}", flush=True)
    print(f"[train_h5] {manifest['train_h5']}", flush=True)
    print(f"[val_h5] {manifest['val_h5']}", flush=True)
    for r in rendered:
        print(f"[png] {r['png']}", flush=True)


if __name__ == "__main__":
    main()
