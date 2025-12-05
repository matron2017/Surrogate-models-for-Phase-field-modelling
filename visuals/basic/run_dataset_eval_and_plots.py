#!/usr/bin/env python3
# pf_eval_and_panels.py — Evaluate TEST split on de-standardised scale,
# compute metrics and descriptive means, save per-field triangular panels,
# and perform explicit sanity checks.
# Difference-map scaling:
#   - Absolute-difference panels: dataset-global signed min/max per map and channel.
#   - Relative-error panels: dataset-global min/max percent per map and channel.
# No second-person phrasing in comments.

import os, json, math, argparse, yaml, importlib, importlib.util, csv
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utilities ----------

def _load_symbol(py_path: str, symbol: str):
    p = Path(py_path).resolve()
    def _guess_root(q: Path):
        for a in [q.parent, *q.parents]:
            if (a/"models").is_dir() or (a/"scripts").is_dir():
                return a
        return p.parent
    root = _guess_root(p)
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))
    try:
        rel = p.relative_to(root).with_suffix("")
        mod_name = ".".join(rel.parts)
        mod = importlib.import_module(mod_name)
    except Exception:
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, symbol):
        raise AttributeError(f"Symbol '{symbol}' not found in {p}")
    return getattr(mod, symbol)

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for k in ("input","target","cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "gid" in batch[0]: out["gid"] = [b["gid"] for b in batch]
    if "pair_index" in batch[0]: out["pair_index"] = [int(b["pair_index"]) for b in batch]
    return out

def _extract_cond_from_field(batch_cond: torch.Tensor, expected_dim: int) -> torch.Tensor:
    assert batch_cond.dim() == 2 and batch_cond.size(1) == expected_dim
    return batch_cond

def _to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _percentiles(arrs: List[np.ndarray], lo=2.0, hi=98.0) -> Tuple[float, float]:
    vec = np.concatenate([a.ravel() for a in arrs]) if arrs else np.array([0.0])
    return float(np.percentile(vec, lo)), float(np.percentile(vec, hi))

def _norm_gid(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    t = str(s).strip()
    if t.startswith("sim_"): return t
    if t.isdigit(): return f"sim_{int(t):04d}"
    return t

def _in_k_range(k: int, kr: Optional[Tuple[int,int]]) -> bool:
    if kr is None: return True
    a, b = kr
    return (a <= k <= b)

# ---------- metrics helpers (de-standardised) ----------

def _accumulate_metrics(acc: Dict[str, Any], y: np.ndarray, yhat: np.ndarray):
    diff = yhat - y
    acc["se"] += float((diff * diff).sum())
    acc["ye"] += float((y * y).sum())
    acc["l1"] += float(np.abs(diff).sum())
    acc["n"]  += int(y.size)
    C = y.shape[1]
    for c in range(C):
        dc = diff[:, c]
        acc["se_c"][c]  += float((dc * dc).sum())
        acc["l1_c"][c]  += float(np.abs(dc).sum())
        acc["n_c"][c]   += int(dc.size)
        acc["bias_sum_c"][c] += float(dc.mean())
    for b in range(y.shape[0]):
        num = float(((yhat[b] - y[b])**2).sum())
        den = float((y[b]**2).sum())
        acc["rel_l2_samples"].append(num / max(den, 1e-30))

def _accumulate_means(acc: Dict[str, Any], x: np.ndarray, y: np.ndarray, yhat: np.ndarray):
    for name, arr in (("x", x), ("y", y), ("yhat", yhat)):
        acc[f"{name}_per_channel"] += arr.mean(axis=(0,2,3))
    acc["delta_gt_per_channel"]   += (y - x).mean(axis=(0,2,3))
    acc["delta_pred_per_channel"] += (yhat - x).mean(axis=(0,2,3))
    acc["residual_per_channel"]   += (yhat - y).mean(axis=(0,2,3))
    acc["count_batches"]          += 1

def _finalise_metrics(acc: Dict[str, Any]) -> Dict[str, Any]:
    mse = acc["se"] / max(acc["n"], 1)
    rmse = math.sqrt(max(mse, 0.0))
    mae  = acc["l1"] / max(acc["n"], 1)
    C = len(acc["se_c"])
    per_channel = []
    for c in range(C):
        mse_c = acc["se_c"][c] / max(acc["n_c"][c], 1)
        per_channel.append({
            "channel": int(c),
            "mse": mse_c,
            "rmse": math.sqrt(max(mse_c, 0.0)),
            "mae":  acc["l1_c"][c] / max(acc["n_c"][c], 1),
            "bias": acc["bias_sum_c"][c],
        })
    rel_l2_global = acc["se"] / max(acc["ye"], 1e-30)
    rel_l2_mean   = float(np.mean(acc["rel_l2_samples"])) if acc["rel_l2_samples"] else float("nan")
    count = max(acc["count_batches"], 1)
    desc = {
        "x_per_channel":          (acc["x_per_channel"] / count).tolist(),
        "y_per_channel":          (acc["y_per_channel"] / count).tolist(),
        "yhat_per_channel":       (acc["yhat_per_channel"] / count).tolist(),
        "delta_gt_per_channel":   (acc["delta_gt_per_channel"] / count).tolist(),
        "delta_pred_per_channel": (acc["delta_pred_per_channel"] / count).tolist(),
        "residual_per_channel":   (acc["residual_per_channel"] / count).tolist(),
    }
    return {
        "overall": {"mse": mse, "rmse": rmse, "mae": mae, "psnr": (-10.0*math.log10(mse) if mse>0 else float("inf"))},
        "per_channel": per_channel,
        "relative_l2": {"global": rel_l2_global, "mean_over_samples": rel_l2_mean},
        "counts": {"elements": int(acc["n"]), "channels": C},
        "descriptive_means": desc,
    }

# ---------- plotting helpers ----------

def _tile_stats(ax, arr: np.ndarray):
    txt = f"μ={arr.mean():.3e}\nmin={arr.min():.3e}\nmax={arr.max():.3e}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, lw=0))

def _render_panel_absdiff(sample_pack: Dict[str, Any],
                          phys_vlims: Dict[str, List[Tuple[float,float]]],
                          absdiff_vlims: Dict[str, List[Tuple[float,float]]],
                          out_path_base: Path,
                          ch: int,
                          dpi: int,
                          tile_size: float):
    # Physical fields
    x   = sample_pack["x"][ch]
    y   = sample_pack["y"][ch]
    yh  = sample_pack["yhat"][ch]
    # Differences (signed)
    res = yh - y
    dgt = y - x
    dpr = yh - x

    fig = plt.figure(figsize=(tile_size*3, tile_size*3))
    gs = fig.add_gridspec(nrows=3, ncols=3)
    ax_x    = fig.add_subplot(gs[0,1])
    ax_y    = fig.add_subplot(gs[1,0])
    ax_yh   = fig.add_subplot(gs[1,2])
    ax_res  = fig.add_subplot(gs[2,0])
    ax_dgt  = fig.add_subplot(gs[2,1])
    ax_dpr  = fig.add_subplot(gs[2,2])

    # Physical tiles with their own vlims (unchanged behaviour)
    for ax, arr, key, title in [
        (ax_x,  x,  "x",    "Input x"),
        (ax_y,  y,  "y",    "Ground truth y"),
        (ax_yh, yh, "yhat", "Prediction ŷ"),
    ]:
        vmin, vmax = phys_vlims[key][ch]
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        _tile_stats(ax, arr)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Difference tiles with dataset-global signed vlims
    for ax, arr, key, title in [
        (ax_res, res, "res",  "Residual (ŷ−y)"),
        (ax_dgt, dgt, "dgt",  "ΔGT (y−x)"),
        (ax_dpr, dpr, "dpred","ΔPred (ŷ−x)"),
    ]:
        vmin, vmax = absdiff_vlims[key][ch]
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        _tile_stats(ax, arr)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Physical units")

    meta = sample_pack["meta"]
    fig.suptitle(
        f"ABS-DIFF • ch={ch}  gid={meta['gid']}  pair_index={meta['pair_index']}  stride={meta['stride']}  "
        f"Δt_phys={meta['dt_phys']:.3e}  z_Δt={meta['z_dt']:.3f}  G={meta['G_raw']:.3e}  z_G={meta['z_G']:.3f}",
        fontsize=10
    )
    fig.savefig(out_path_base.with_suffix(f".ch{ch}.absdiff.png"), dpi=dpi)
    plt.close(fig)

def _render_panel_reldiff(sample_pack: Dict[str, Any],
                          phys_vlims: Dict[str, List[Tuple[float,float]]],
                          reldiff_vlims_pct: Dict[str, List[Tuple[float,float]]],
                          out_path_base: Path,
                          ch: int,
                          dpi: int,
                          tile_size: float):
    # Physical fields
    x   = sample_pack["x"][ch]
    y   = sample_pack["y"][ch]
    yh  = sample_pack["yhat"][ch]
    # Relative errors in percent
    eps = 1e-12
    res_pct = 100.0 * (np.abs(yh - y)  / np.maximum.reduce([np.abs(y),  np.abs(yh), np.full_like(y, eps)]))
    dgt_pct = 100.0 * (np.abs(y - x)   / np.maximum.reduce([np.abs(y),  np.abs(x),  np.full_like(y, eps)]))
    dpr_pct = 100.0 * (np.abs(yh - x)  / np.maximum.reduce([np.abs(yh), np.abs(x),  np.full_like(y, eps)]))

    fig = plt.figure(figsize=(tile_size*3, tile_size*3))
    gs = fig.add_gridspec(nrows=3, ncols=3)
    ax_x    = fig.add_subplot(gs[0,1])
    ax_y    = fig.add_subplot(gs[1,0])
    ax_yh   = fig.add_subplot(gs[1,2])
    ax_res  = fig.add_subplot(gs[2,0])
    ax_dgt  = fig.add_subplot(gs[2,1])
    ax_dpr  = fig.add_subplot(gs[2,2])

    # Physical tiles with their own vlims (unchanged behaviour)
    for ax, arr, key, title in [
        (ax_x,  x,  "x",    "Input x"),
        (ax_y,  y,  "y",    "Ground truth y"),
        (ax_yh, yh, "yhat", "Prediction ŷ"),
    ]:
        vmin, vmax = phys_vlims[key][ch]
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        _tile_stats(ax, arr)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Difference tiles with dataset-global relative-error percent vlims
    for ax, arr, key, title in [
        (ax_res, res_pct, "res",  "Residual rel. error (%)"),
        (ax_dgt, dgt_pct, "dgt",  "ΔGT rel. error (%)"),
        (ax_dpr, dpr_pct, "dpred","ΔPred rel. error (%)"),
    ]:
        vmin, vmax = reldiff_vlims_pct[key][ch]
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        _tile_stats(ax, arr)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Relative error (%)")

    meta = sample_pack["meta"]
    fig.suptitle(
        f"REL-DIFF • ch={ch}  gid={meta['gid']}  pair_index={meta['pair_index']}  stride={meta['stride']}  "
        f"Δt_phys={meta['dt_phys']:.3e}  z_Δt={meta['z_dt']:.3f}  G={meta['G_raw']:.3e}  z_G={meta['z_G']:.3f}",
        fontsize=10
    )
    fig.savefig(out_path_base.with_suffix(f".ch{ch}.reldiff.png"), dpi=dpi)
    plt.close(fig)

def _compute_phys_vlims_from_packs(packs: List[Dict[str, Any]], C: int) -> Dict[str, List[Tuple[float,float]]]:
    # Percentile vlims for physical tiles only (x, y, yhat). Differences ignored here.
    pools = {k: [[] for _ in range(C)] for k in ["x","y","yhat"]}
    for p in packs:
        x, y, yh = p["x"], p["y"], p["yhat"]
        for c in range(C):
            pools["x"][c].append(x[c])
            pools["y"][c].append(y[c])
            pools["yhat"][c].append(yh[c])
    vlims = {k: [(0.0,0.0) for _ in range(C)] for k in pools.keys()}
    for key in vlims.keys():
        for c in range(C):
            lo, hi = _percentiles([arr for arr in pools[key][c]], lo=2.0, hi=98.0)
            vlims[key][c] = (lo, hi)
    return vlims

# ---------- main evaluation ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True, help="Path to training YAML or config_snapshot.yaml")
    ap.add_argument("-k","--ckpt",   required=True, help="Path to checkpoint .pth")
    ap.add_argument("-o","--outdir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=str, default="270:320", help="Spatial window 'y0:y1' equals 'x0:x1'")
    ap.add_argument("--n-panels", type=int, default=5)
    # deterministic selection
    ap.add_argument("--select-gid", type=str, default=None, help="gid like 'sim_0012' or integer '12'")
    ap.add_argument("--k-range",    type=str, default=None, help="inclusive pair_index range 'start:end'")
    # figure controls
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--tile-size", type=float, default=3.6, help="inches per grid step")
    # relative error safety floor
    ap.add_argument("--eps", type=float, default=1e-12, help="denominator floor for relative errors")

    args = ap.parse_args()
    sel_gid = _norm_gid(args.select_gid)
    sel_kr  = None
    if args.k_range:
        a, b = [int(z) for z in args.k_range.split(":")]
        sel_kr = (a, b)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = _ensure_dir(Path(args.outdir))
    panels_dir = _ensure_dir(out_dir / "panels")

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    torch.set_grad_enabled(False)

    # dataset and model
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})
    test_path = cfg["paths"]["h5"]["test"] if isinstance(cfg["paths"]["h5"]["test"], str) else cfg["paths"]["h5"]["test"]["path"]
    test_ds = DSClass(test_path, **ds_args)

    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # H5 root attrs for de-standardisation
    h5 = h5py.File(test_path, "r")
    ch_mean = np.array(h5.attrs["channel_mean"], dtype=np.float32)
    ch_std  = np.array(h5.attrs["channel_std"],  dtype=np.float32)

    # loader
    dl = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=int(cfg["loader"].get("num_workers", 4)),
        pin_memory=bool(cfg["loader"].get("pin_memory", True)),
        persistent_workers=(int(cfg["loader"].get("num_workers", 4)) > 0),
        collate_fn=_collate,
    )

    # conditioning
    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    assert cond_enabled and cond_dim == 2, "Expected field-based 2D conditioning vector"

    # spatial window
    y0, y1 = [int(t) for t in args.window.strip().split(":")]
    x0, x1 = y0, y1

    # accumulators
    C = len(ch_mean)
    m_acc = {
        "se": 0.0, "ye": 0.0, "l1": 0.0, "n": 0,
        "se_c": [0.0 for _ in range(C)], "l1_c": [0.0 for _ in range(C)], "n_c": [0 for _ in range(C)],
        "bias_sum_c": [0.0 for _ in range(C)],
        "rel_l2_samples": []
    }
    d_acc = {
        "x_per_channel": np.zeros(C, np.float64),
        "y_per_channel": np.zeros(C, np.float64),
        "yhat_per_channel": np.zeros(C, np.float64),
        "delta_gt_per_channel": np.zeros(C, np.float64),
        "delta_pred_per_channel": np.zeros(C, np.float64),
        "residual_per_channel": np.zeros(C, np.float64),
        "count_batches": 0
    }

    # dataset-global min/max trackers for difference maps (absolute and relative %)
    def _init_mm():
        return [(np.inf, -np.inf) for _ in range(C)]
    abs_mm = {"res": _init_mm(), "dgt": _init_mm(), "dpred": _init_mm()}
    rel_mm = {"res": _init_mm(), "dgt": _init_mm(), "dpred": _init_mm()}  # in percent

    # per-sample windowed means table
    csv_header = ["global_idx","gid","pair_index","stride","delta_t_phys","z_delta_t","thermal_raw","z_thermal",
                  "x_ch0","x_ch1","y_ch0","y_ch1","yhat_ch0","yhat_ch1",
                  "res_ch0","res_ch1","dgt_ch0","dgt_ch1","dpred_ch0","dpred_ch1","dres_ch0","dres_ch1"]
    csv_rows: List[List[str]] = []

    # selection stats (full frame + window) when selection is provided
    sel_rows: List[List[str]] = []
    sel_header = ["gid","pair_index","stride","delta_t_phys","z_delta_t","thermal_raw","z_thermal",
                  "x0_mean","x0_min","x0_max","x1_mean","x1_min","x1_max",
                  "y0_mean","y0_min","y0_max","y1_mean","y1_min","y1_max",
                  "yhat0_mean","yhat0_min","yhat0_max","yhat1_mean","yhat1_min","yhat1_max",
                  "res0_mean","res0_min","res0_max","res1_mean","res1_min","res1_max",
                  "dgt0_mean","dgt0_min","dgt0_max","dgt1_mean","dgt1_min","dgt1_max",
                  "dpred0_mean","dpred0_min","dpred0_max","dpred1_mean","dpred1_min","dpred1_max",
                  "dres0_mean","dres0_min","dres0_max","dres1_mean","dres1_min","dres1_max",
                  "x0_wmean","x0_wmin","x0_wmax","x1_wmean","x1_wmin","x1_wmax",
                  "y0_wmean","y0_wmin","y0_wmax","y1_wmean","y1_wmin","y1_wmax",
                  "yhat0_wmean","yhat0_wmin","yhat0_wmax","yhat1_wmean","yhat1_wmin","yhat1_wmax",
                  "res0_wmean","res0_wmin","res0_wmax","res1_wmean","res1_wmin","res1_wmax",
                  "dgt0_wmean","dgt0_wmin","dgt0_wmax","dgt1_wmean","dgt1_wmin","dgt1_wmax",
                  "dpred0_wmean","dpred0_wmin","dpred0_wmax","dpred1_wmean","dpred1_wmin","dpred1_wmax",
                  "dres0_wmean","dres0_wmin","dres0_wmax","dres1_wmean","dres1_wmin","dres1_wmax"]

    candidates: List[Dict[str, Any]] = []
    sanity_lines: List[str] = []

    eps = float(args.eps)
    g_index = 0
    for batch in dl:
        x = _to_device(batch["input"], device)
        y = _to_device(batch["target"], device)
        cond = _extract_cond_from_field(_to_device(batch["cond"], device), cond_dim) if cond_enabled else None

        with torch.inference_mode():
            pred = model(x, cond) if cond is not None else model(x)

        x = x.cpu().numpy().astype(np.float32)
        y = y.cpu().numpy().astype(np.float32)
        yh= pred.detach().cpu().numpy().astype(np.float32)

        ch_mean_b = ch_mean[None, :, None, None]
        ch_std_b  = ch_std [None, :, None, None]
        x_phys  = x  * ch_std_b + ch_mean_b
        y_phys  = y  * ch_std_b + ch_mean_b
        yh_phys = yh * ch_std_b + ch_mean_b

        _accumulate_metrics(m_acc, y_phys, yh_phys)
        _accumulate_means(d_acc, x_phys, y_phys, yh_phys)

        B = x.shape[0]
        for i in range(B):
            gid = batch["gid"][i]; k = int(batch["pair_index"][i])
            grp = h5[gid]
            stride = int(grp["pairs_stride"][k]) if "pairs_stride" in grp else int(grp["pairs_dt_euler"][k])
            dt_phys = float(grp["pairs_dt"][k])
            z_dt    = float(grp["pairs_dt_norm"][k])
            G_raw   = float(grp.attrs["thermal_gradient_raw"])
            muG, sdG = float(grp.attrs["thermal_mean"]), float(grp.attrs["thermal_std"])
            z_G     = (G_raw - muG) / (sdG if sdG > 0 else 1.0)

            if cond is not None:
                cvec = cond[i].detach().cpu().numpy().tolist()
                ok_g_dt  = (abs(cvec[0] - z_G) < 5e-3 and abs(cvec[1] - z_dt) < 5e-3)
                ok_dt_g  = (abs(cvec[0] - z_dt) < 5e-3 and abs(cvec[1] - z_G) < 5e-3)
                if not (ok_g_dt or ok_dt_g):
                    sanity_lines.append(f"[COND MISMATCH] idx={g_index} gid={gid} k={k} cvec={cvec} vs (z_G={z_G:.4f}, z_dt={z_dt:.4f})")
                elif ok_dt_g:
                    sanity_lines.append(f"[COND ORDER] Dataset emits [Δt′, G′]; consider standardising to [G′, Δt′].")

            xf   = x_phys [i]
            yf   = y_phys [i]
            yhf  = yh_phys[i]

            # global min/max trackers for difference maps
            resf = yhf - yf
            dgtf = yf  - xf
            dprf = yhf - xf

            # update absolute signed min/max per channel
            for c in range(C):
                mn, mx = abs_mm["res"][c]; abs_mm["res"][c]   = (min(mn, resf[c].min()),  max(mx, resf[c].max()))
                mn, mx = abs_mm["dgt"][c]; abs_mm["dgt"][c]   = (min(mn, dgtf[c].min()),  max(mx, dgtf[c].max()))
                mn, mx = abs_mm["dpred"][c]; abs_mm["dpred"][c] = (min(mn, dprf[c].min()),  max(mx, dprf[c].max()))

            # relative error denominators and percent values per channel
            den_res = np.maximum.reduce([np.abs(yf),  np.abs(yhf), np.full_like(yf, eps)])
            den_dgt = np.maximum.reduce([np.abs(yf),  np.abs(xf),  np.full_like(yf, eps)])
            den_dpr = np.maximum.reduce([np.abs(yhf), np.abs(xf),  np.full_like(yf, eps)])

            res_pct = 100.0 * np.abs(resf) / den_res
            dgt_pct = 100.0 * np.abs(dgtf) / den_dgt
            dpr_pct = 100.0 * np.abs(dprf) / den_dpr

            for c in range(C):
                mn, mx = rel_mm["res"][c]; rel_mm["res"][c]   = (min(mn, res_pct[c].min()),  max(mx, res_pct[c].max()))
                mn, mx = rel_mm["dgt"][c]; rel_mm["dgt"][c]   = (min(mn, dgt_pct[c].min()),  max(mx, dgt_pct[c].max()))
                mn, mx = rel_mm["dpred"][c]; rel_mm["dpred"][c] = (min(mn, dpr_pct[c].min()),  max(mx, dpr_pct[c].max()))

            # window stats and candidate selection (unchanged)
            xw   = xf[:, y0:y1, x0:x1]
            yw   = yf[:, y0:y1, x0:x1]
            yhw  = yhf[:, y0:y1, x0:x1]
            resw = yhw - yw
            dgtw = yw  - xw
            dprw = yhw - xw
            drsw = dprw - dgtw

            row = [str(g_index), gid, str(k), str(stride),
                   f"{dt_phys:.6e}", f"{z_dt:.6f}", f"{G_raw:.6e}", f"{z_G:.6f}"]
            def _means(v):
                m = v.mean(axis=(1,2))
                return [f"{float(m[0]):.6e}", f"{float(m[1]):.6e}"]
            row += _means(xw) + _means(yw) + _means(yhw) + _means(resw) + _means(dgtw) + _means(dprw) + _means(drsw)
            csv_rows.append(row)

            is_sel = (sel_gid is not None and gid == sel_gid and _in_k_range(k, sel_kr))
            if is_sel:
                def mm3(a):
                    m = a.mean(axis=(1,2)); mn = a.min(axis=(1,2)); mx = a.max(axis=(1,2))
                    return [float(m[0]), float(mn[0]), float(mx[0]), float(m[1]), float(mn[1]), float(mx[1])]
                def mm3w(a):
                    w = a[:, y0:y1, x0:x1]
                    m = w.mean(axis=(1,2)); mn = w.min(axis=(1,2)); mx = w.max(axis=(1,2))
                    return [float(m[0]), float(mn[0]), float(mx[0]), float(m[1]), float(mn[1]), float(mx[1])]
                sel_rows.append(
                    [gid, str(k), str(stride), f"{dt_phys:.6e}", f"{z_dt:.6f}", f"{G_raw:.6e}", f"{z_G:.6f}"] +
                    mm3(xf) + mm3(yf) + mm3(yhf) + mm3(resf) + mm3(dgtf) + mm3(dprf) + mm3(dprf - dgtf) +
                    mm3w(xf) + mm3w(yf) + mm3w(yhf) + mm3w(resf) + mm3w(dgtf) + mm3w(dprf) + mm3w(dprf - dgtf)
                )

            if stride == 1 and (sel_gid is None or is_sel):
                rmse_i = math.sqrt(float(((yhf - yf)**2).mean()))
                candidates.append({
                    "x": xf, "y": yf, "yhat": yhf, "rmse": rmse_i, "global_idx": g_index,
                    "meta": {"gid": gid, "pair_index": k, "stride": stride,
                             "dt_phys": dt_phys, "z_dt": z_dt, "G_raw": G_raw, "z_G": z_G}
                })

            if not np.allclose((yhf - xf) - (yf - xf), (yhf - yf), rtol=1e-6, atol=1e-6):
                sanity_lines.append(f"[DELTA IDENTITY FAIL] idx={g_index} gid={gid} k={k}")

            g_index += 1

    # final metrics
    metrics = _finalise_metrics({
        **m_acc,
        **{
            "x_per_channel": d_acc["x_per_channel"],
            "y_per_channel": d_acc["y_per_channel"],
            "yhat_per_channel": d_acc["yhat_per_channel"],
            "delta_gt_per_channel": d_acc["delta_gt_per_channel"],
            "delta_pred_per_channel": d_acc["delta_pred_per_channel"],
            "residual_per_channel": d_acc["residual_per_channel"],
            "count_batches": d_acc["count_batches"]
        }
    })

    metrics_explained = {
        "scale": "physical (de-standardised with HDF5 root channel_mean/std)",
        "overall": "Pixel-weighted MSE/RMSE/MAE across all samples and both channels",
        "per_channel": "Metrics computed independently per channel",
        "bias": "Dataset mean of (ŷ − y) per channel",
        "relative_l2": {
            "global": "Sum of squared errors / sum of squared ground truth over TEST",
            "mean_over_samples": "Arithmetic mean of per-sample SSE/||y||^2"
        },
        "descriptive_means": "Dataset-level per-channel means of x, y, ŷ, ΔGT=y−x, ΔPred=ŷ−x, Residual=ŷ−y"
    }

    info_blob = {
        "config_paths": {"config": str(Path(args.config).resolve()),
                         "checkpoint": str(Path(args.ckpt).resolve()),
                         "h5_test": str(Path(test_path).resolve())},
        "conditioning": cfg.get("conditioning", {}),
        "window": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "panel_selection_summary": {"requested": int(args.n_panels)},
        "metrics": metrics,
        "metrics_explained": metrics_explained,
        "selection": {"gid": sel_gid, "k_range": sel_kr}
    }
    (out_dir / "metrics.json").write_text(json.dumps(info_blob, indent=2))

    with open(out_dir / "means_window.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(csv_header); w.writerows(csv_rows)

    if not candidates:
        raise RuntimeError("No eligible samples found for plotting (require stride==1 and optional selection).")
    if sel_gid is not None:
        chosen = sorted(candidates, key=lambda p: p["meta"]["pair_index"])
    else:
        cand_sorted = sorted(candidates, key=lambda p: p["rmse"])
        idxs = [0, len(cand_sorted)//4, len(cand_sorted)//2, 3*len(cand_sorted)//4, len(cand_sorted)-1]
        chosen = [cand_sorted[i] for i in idxs[:args.n_panels]]

    # vlims for physical tiles only
    phys_vlims = _compute_phys_vlims_from_packs(chosen, C)

    # dataset-global vlims for difference maps (absolute signed, and relative percent)
    absdiff_vlims = {k: [(mn, mx) for (mn, mx) in abs_mm[k]] for k in ["res","dgt","dpred"]}
    # ensure sensible ordering if data were constant
    for k in absdiff_vlims:
        absdiff_vlims[k] = [(float(mn), float(mx) if mx > mn else float(mn)) for (mn, mx) in absdiff_vlims[k]]

    reldiff_vlims_pct = {k: [(mn, mx) for (mn, mx) in rel_mm[k]] for k in ["res","dgt","dpred"]}
    # floor min at 0 if tiny negatives appear due to numerical noise
    for k in reldiff_vlims_pct:
        reldiff_vlims_pct[k] = [(max(0.0, float(mn)), float(mx) if mx > mn else max(0.0, float(mn)))
                                 for (mn, mx) in reldiff_vlims_pct[k]]

    panels_dir.mkdir(parents=True, exist_ok=True)
    for j, pack in enumerate(chosen):
        base = panels_dir / f"panel_{j+1:02d}_gid_{pack['meta']['gid']}_k_{pack['meta']['pair_index']}"
        for ch in range(C):
            _render_panel_absdiff(pack, phys_vlims, absdiff_vlims, base, ch, dpi=args.dpi, tile_size=args.tile_size)
            _render_panel_reldiff(pack, phys_vlims, reldiff_vlims_pct, base, ch, dpi=args.dpi, tile_size=args.tile_size)

    for pack in chosen:
        gid = pack["meta"]["gid"]; k = pack["meta"]["pair_index"]
        grp = h5[gid]
        stride = int(grp["pairs_stride"][k]) if "pairs_stride" in grp else int(grp["pairs_dt_euler"][k])
        eff_dt = float(grp.attrs["effective_dt"])
        dt_phys = float(grp["pairs_dt"][k])
        if stride != 1:
            sanity_lines.append(f"[STRIDE] panel uses stride={stride} (expected 1) gid={gid} k={k}")
        if not (abs(dt_phys - eff_dt) <= 1e-10 or math.isclose(dt_phys, eff_dt, rel_tol=1e-6, abs_tol=1e-10)):
            sanity_lines.append(f"[DELTA_T] dt_phys={dt_phys:.6e} != effective_dt={eff_dt:.6e} for gid={gid} k={k}")

    sanity_lines.append("[INVERT CHECK] de-standardisation applied via file-level channel_mean/std; used consistently.")
    (out_dir / "sanity_checks.txt").write_text("\n".join(sanity_lines) if sanity_lines else "No issues detected.")

    if sel_rows:
        with open(out_dir / "selection_stats.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(sel_header); w.writerows(sel_rows)

    sel_meta = [{"gid": p["meta"]["gid"], "pair_index": int(p["meta"]["pair_index"]),
                 "stride": int(p["meta"]["stride"]), "dt_phys": float(p["meta"]["dt_phys"]),
                 "z_dt": float(p["meta"]["z_dt"]), "G_raw": float(p["meta"]["G_raw"]), "z_G": float(p["meta"]["z_G"]),
                 "rmse": float(p["rmse"])} for p in chosen]
    summary = {
        "selected": sel_meta,
        "phys_vlims": phys_vlims,
        "absdiff_vlims": absdiff_vlims,
        "reldiff_vlims_pct": reldiff_vlims_pct
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    info_blob["panel_selection_summary"]["saved"] = len(chosen) * C * 2  # two images per channel (absdiff + reldiff)
    (out_dir / "metrics.json").write_text(json.dumps(info_blob, indent=2))

if __name__ == "__main__":
    main()
