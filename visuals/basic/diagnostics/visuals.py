#!/usr/bin/env python3
# visuals.py  (pf_eval_single_pair_with_errors.py)
#
# Per-channel PNGs for a single pair: x, ŷ, |ŷ−y|, rel.% |ŷ−y|, rel.% |y−x|.
# Adds a compact bar plot of per-channel mean relative errors (residual vs ΔGT).
# Selection defaults: gid=sim_0012, transition 201->202, stride==1 required.
# No second-person phrasing in comments.

from __future__ import annotations
import argparse, importlib, importlib.util, json, os, sys, gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch, yaml, h5py
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter

# ---------- dynamic import helpers ----------

def _guess_repo_root(p: Path) -> Optional[Path]:
    for a in [p.parent, *p.parents]:
        if (a / "models").is_dir() or (a / "scripts").is_dir() or (a / "src").is_dir():
            return a
    return None

def _load_symbol(py_path: str, symbol: str):
    p = Path(py_path)
    if p.exists():
        root = _guess_repo_root(p)
        if root and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load python file: {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    else:
        mod = importlib.import_module(py_path)
    if not hasattr(mod, symbol):
        raise AttributeError(f"Symbol '{symbol}' not found in {py_path}")
    return getattr(mod, symbol)

# ---------- utils ----------

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("input", "target", "cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in ("gid", "pair_index", "stride"):
        if k in batch[0]:
            out[k] = [b[k] for b in batch]
    return out

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pair_stride(grp, idx: int) -> int:
    if "pairs_stride" in grp:
        return int(grp["pairs_stride"][idx])
    if "pairs_dt_euler" in grp:
        return int(grp["pairs_dt_euler"][idx])
    i, j = grp["pairs_idx"][idx]
    return int(j - i)


def _pair_time_values(grp, idx: int) -> Tuple[float, float]:
    i, j = grp["pairs_idx"][idx]
    eff_dt = float(grp.attrs.get("effective_dt", grp.file.attrs.get("effective_dt", 1.0)))
    eps_time = float(grp.attrs.get("zscore_eps_time", grp.file.attrs.get("zscore_eps_time", 1e-12)))

    if "time_phys" in grp:
        t_abs = float(grp["time_phys"][j])
        t_mean = float(grp.attrs.get("time_mean", grp.file.attrs.get("time_mean", t_abs)))
        t_std = float(grp.attrs.get("time_std", grp.file.attrs.get("time_std", 1.0)))
    elif "pairs_time" in grp:
        arr = grp["pairs_time"]
        t_abs = float(arr[idx][1] * eff_dt)
        if "times" in grp:
            t_series = grp["times"][:].astype("float64") * eff_dt
            t_mean = float(t_series.mean())
            t_std = float(t_series.std())
        else:
            t_mean = float(grp.attrs.get("time_mean", grp.file.attrs.get("time_mean", t_abs)))
            t_std = float(grp.attrs.get("time_std", grp.file.attrs.get("time_std", 1.0)))
    elif "times" in grp:
        t_series = grp["times"][:].astype("float64") * eff_dt
        t_abs = float(t_series[j])
        t_mean = float(t_series.mean())
        t_std = float(t_series.std())
    else:
        raise KeyError("Expected absolute time datasets ('time_phys', 'pairs_time', or 'times').")

    denom = t_std if isinstance(t_std, (int, float)) and t_std > 0 else eps_time
    denom = denom if denom and denom > 0 else 1.0
    z_t = float((t_abs - t_mean) / denom)
    return t_abs, z_t

def _percentiles(arrs: List[np.ndarray], lo=2.0, hi=98.0) -> Tuple[float, float]:
    vec = np.concatenate([a.ravel() for a in arrs]) if arrs else np.array([0.0], dtype=np.float64)
    return float(np.percentile(vec, lo)), float(np.percentile(vec, hi))

# ---------- rendering: single-tile saver (fixed size) ----------

def _save_single_tile(
    arr: np.ndarray,                   # [H,W]
    vlims: Tuple[float,float],
    title: str,
    meta_title: str,
    out_path: Path,
    *,
    dpi: int,
    tile_w_in: float,
    tile_h_in: float,
    cbar_label: str,
    cbar_fontsize: int,
    cbar_max_ticks: int,
    tick_format: Optional[str] = None,  # None | 'percent' | 'sci'
) -> None:
    """Save one image tile at fixed figure size with its own colourbar."""
    fig = plt.figure(figsize=(tile_w_in, tile_h_in))
    ax = fig.add_axes([0.12, 0.12, 0.68, 0.76])
    cax = fig.add_axes([0.83, 0.14, 0.06, 0.72])

    im = ax.imshow(arr, vmin=vlims[0], vmax=vlims[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)

    cb = fig.colorbar(im, cax=cax)
    cb.locator = MaxNLocator(nbins=cbar_max_ticks, min_n_ticks=3, steps=[1, 2, 5, 10])
    if tick_format == 'percent':
        cb.formatter = FuncFormatter(lambda v, _: f"{v:.0f}%")
    elif tick_format == 'sci':
        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-3, 3))
        cb.formatter = sf
    cb.update_ticks()
    cb.ax.tick_params(labelsize=max(8, cbar_fontsize-2), width=0.8)
    cb.set_label(cbar_label, fontsize=cbar_fontsize, labelpad=6)

    fig.suptitle(meta_title, fontsize=9)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def _save_barplot_rel_means(
    rel_mean_res: List[float],
    rel_mean_dgt: List[float],
    out_path: Path,
    *,
    dpi: int,
    fig_w_in: float,
    fig_h_in: float,
) -> None:
    """Grouped bar plot of mean relative errors per channel (residual vs ΔGT)."""
    C = len(rel_mean_res)
    xs = np.arange(C)
    width = 0.38

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.76])

    ax.bar(xs - width/2, rel_mean_res, width, label="Residual |ŷ−y| (%)")
    ax.bar(xs + width/2, rel_mean_dgt, width, label="ΔGT |y−x| (%)")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in xs])
    ax.set_ylim(0, 400)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("-k","--ckpt", required=True)
    ap.add_argument("-o","--outdir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch", type=int, default=1)
    # Fixed selection defaults: sim_0012 and k=201
    ap.add_argument("--select-gid", type=str, default="sim_0012")
    ap.add_argument("--k", type=int, default=201)
    ap.add_argument("--require-stride1", action="store_true", default=True)
    # Visuals and numerical stability
    ap.add_argument("--phys-lo", type=float, default=2.0)
    ap.add_argument("--phys-hi", type=float, default=98.0)
    ap.add_argument("--rel-eps", type=float, default=1e-8)
    # Defaults aligned with curvature overlay script
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--tile-w-in", type=float, default=13.0)
    ap.add_argument("--tile-h-in", type=float, default=8.0)
    ap.add_argument("--bar-w-in", type=float, default=13.0)
    ap.add_argument("--bar-h-in", type=float, default=4.0)
    ap.add_argument("--cbar-fontsize", type=int, default=11)
    ap.add_argument("--cbar-max-ticks", type=int, default=5)
    args = ap.parse_args()

    print("ARGV:", " ".join(sys.argv), flush=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths and symbols
    h5_info = cfg.get("paths",{}).get("h5",{}).get("test")
    h5_test_path = h5_info.get("path") if isinstance(h5_info,dict) else h5_info
    if not h5_test_path: raise ValueError("paths.h5.test not found in config")

    ds_file = cfg.get("dataloader",{}).get("file")
    ds_class = cfg.get("dataloader",{}).get("class","PFPairDataset")
    model_file = cfg.get("model",{}).get("file")
    model_class = cfg.get("model",{}).get("class")
    if not ds_file or not model_file or not model_class:
        raise ValueError("dataloader.file, model.file, and model.class must be present in config")

    DSClass = _load_symbol(ds_file, ds_class)
    ModelClass = _load_symbol(model_file, model_class)

    cond_cfg = cfg.get("conditioning",{}) or {}
    cond_dim_cfg = cond_cfg.get("cond_dim", None)
    model_cond_dim = cfg.get("model",{}).get("params",{}).get("cond_dim", cond_dim_cfg)

    # Device and loader
    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_grad_enabled(False)

    ds_args = dict(cfg.get("dataloader",{}).get("args", {}))
    test_ds = DSClass(h5_test_path, **ds_args)

    state = torch.load(args.ckpt, map_location=device)
    state_dict = state.get("model", state)
    model = ModelClass(**cfg.get("model",{}).get("params", {})).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    env_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    cfg_workers = int(cfg.get("loader",{}).get("num_workers", 0))
    num_workers = min(cfg_workers, env_cpus, 4)
    pin_memory = bool(cfg.get("loader",{}).get("pin_memory", True))
    dl_kwargs: Dict[str, Any] = dict(
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_fn=_collate,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1
    dl = DataLoader(test_ds, **dl_kwargs)

    out_dir = _ensure_dir(Path(args.outdir))
    imgs_dir = _ensure_dir(out_dir / "images")

    # Search for the single required example
    example: Optional[Dict[str, Any]] = None
    with h5py.File(h5_test_path, "r") as h5:
        ch_mean = np.array(h5.attrs["channel_mean"], dtype=np.float32)
        ch_std  = np.array(h5.attrs["channel_std"], dtype=np.float32)
        ch_mean_b = ch_mean[None, :, None, None]
        ch_std_b  = ch_std [None, :, None, None]

        target_gid = str(args.select_gid)
        target_k = int(args.k)

        for batch in dl:
            gids = batch.get("gid", [None]*len(batch["input"]))
            if target_gid not in {str(g) for g in gids}:
                continue

            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            cond = batch.get("cond")
            if cond is not None:
                cond = cond.to(device, non_blocking=True)
                if cond.dim() != 2:
                    raise RuntimeError(f"Expected cond with shape [B, D], got {tuple(cond.shape)}")
                D = int(cond.shape[1])
                if (cond_dim_cfg is not None and D != int(cond_dim_cfg)) or \
                   (model_cond_dim is not None and D != int(model_cond_dim)):
                    raise RuntimeError(f"conditioning/model cond_dim mismatch with dataset D={D}")

            with torch.inference_mode():
                yhat = model(x, cond) if cond is not None else model(x)

            # CPU and de-standardise
            x_np    = x.detach().cpu().numpy().astype(np.float32)
            y_np    = y.detach().cpu().numpy().astype(np.float32)
            yhat_np = yhat.detach().cpu().numpy().astype(np.float32)

            x_phys    = x_np    * ch_std_b + ch_mean_b
            y_phys    = y_np    * ch_std_b + ch_mean_b
            yhat_phys = yhat_np * ch_std_b + ch_mean_b

            B = y_phys.shape[0]
            for i in range(B):
                gid_i = str(gids[i])
                if gid_i != target_gid:
                    continue
                k_i = int(batch.get("pair_index", [i])[i]) if "pair_index" in batch else i
                if k_i != target_k:
                    continue

                grp = h5[gid_i]
                stride_i = _pair_stride(grp, k_i)
                if args.require_stride1 and stride_i != 1:
                    continue

                t_abs, z_t = _pair_time_values(grp, k_i)
                G_raw = float(grp.attrs.get("thermal_gradient_raw", float("nan")))
                muG = float(grp.attrs.get("thermal_mean", float("nan")))
                sdG = float(grp.attrs.get("thermal_std", float("nan")))
                z_G = (G_raw - muG) / sdG if (sdG and sdG > 0) else float("nan")

                example = {
                    "x": x_phys[i], "y": y_phys[i], "yhat": yhat_phys[i],
                    "meta": {
                        "gid": gid_i, "pair_index": k_i,
                        "t_from": k_i, "t_to": k_i + stride_i, "stride": stride_i,
                        "time_phys": t_abs, "z_time": z_t, "G_raw": G_raw, "z_G": z_G,
                    },
                }
                break

            del x, y, yhat, x_np, y_np, yhat_np
            gc.collect()
            if example is not None:
                break

    if example is None:
        raise RuntimeError("Requested example not found with the given constraints.")

    C = int(example["x"].shape[0])

    # Value limits for x/ŷ tiles via percentiles per channel
    vlims_per_ch: List[Tuple[float,float]] = []
    for c in range(C):
        lo, hi = _percentiles([example["x"][c], example["yhat"][c]], lo=args.phys_lo, hi=args.phys_hi)
        vlims_per_ch.append((float(lo), float(hi)))

    # Errors
    diff = example["yhat"] - example["y"]
    abs_err = np.abs(diff)

    denom_res = np.maximum(np.abs(example["y"]), float(args.rel_eps))
    rel_pct_res = 100.0 * abs_err / denom_res
    rel_pct_res = np.clip(rel_pct_res, 0.0, 400.0)

    dgt = np.abs(example["y"] - example["x"])
    denom_dgt = np.maximum.reduce([np.abs(example["y"]), np.abs(example["x"]), np.full_like(example["y"], float(args.rel_eps))])
    rel_pct_dgt = 100.0 * dgt / denom_dgt
    rel_pct_dgt = np.clip(rel_pct_dgt, 0.0, 400.0)

    # Absolute error vlims per channel by robust percentiles
    abs_vlims_per_ch: List[Tuple[float,float]] = []
    for c in range(C):
        lo, hi = _percentiles([abs_err[c]], lo=2.0, hi=98.0)
        if not (hi > lo):
            hi = lo + 1e-12
        abs_vlims_per_ch.append((float(lo), float(hi)))

    gid = example["meta"]["gid"]; tf = example["meta"]["t_from"]; tt = example["meta"]["t_to"]
    t_abs = example["meta"]["time_phys"]; zt = example["meta"]["z_time"]; G = example["meta"]["G_raw"]; zG = example["meta"]["z_G"]
    meta_str = f"gid={gid}  k={example['meta']['pair_index']}  t{tf}->{tt}  stride={example['meta']['stride']}  t_abs={t_abs:.3e}  zt={zt:.3f}  G={G:.3e}  zG={zG:.3f}"

    imgs_dir = _ensure_dir(imgs_dir)
    saved_paths: List[str] = []

    # Per-channel exports: x, yhat, |ŷ−y|, relative % residual, relative % ΔGT
    for c in range(C):
        base = f"gid_{gid}__t{tf}_to_t{tt}__ch{c}"

        p_x = imgs_dir / f"io_x__{base}.png"
        _save_single_tile(
            example["x"][c], vlims_per_ch[c], f"Input x • ch={c}", meta_str, p_x,
            dpi=args.dpi, tile_w_in=args.tile_w_in, tile_h_in=args.tile_h_in,
            cbar_label="Physical value", cbar_fontsize=args.cbar_fontsize,
            cbar_max_ticks=args.cbar_max_ticks, tick_format='sci'
        ); saved_paths.append(str(p_x))

        p_yh = imgs_dir / f"io_yhat__{base}.png"
        _save_single_tile(
            example["yhat"][c], vlims_per_ch[c], f"Prediction ŷ • ch={c}", meta_str, p_yh,
            dpi=args.dpi, tile_w_in=args.tile_w_in, tile_h_in=args.tile_h_in,
            cbar_label="Physical value", cbar_fontsize=args.cbar_fontsize,
            cbar_max_ticks=args.cbar_max_ticks, tick_format='sci'
        ); saved_paths.append(str(p_yh))

        p_abs = imgs_dir / f"error_abs__{base}.png"
        _save_single_tile(
            abs_err[c], abs_vlims_per_ch[c], f"|ŷ − y| • ch={c}", meta_str, p_abs,
            dpi=args.dpi, tile_w_in=args.tile_w_in, tile_h_in=args.tile_h_in,
            cbar_label="Absolute error (units)", cbar_fontsize=args.cbar_fontsize,
            cbar_max_ticks=args.cbar_max_ticks, tick_format='sci'
        ); saved_paths.append(str(p_abs))

        p_rel_res = imgs_dir / f"error_relpct_res__{base}.png"
        _save_single_tile(
            rel_pct_res[c], (0.0, 400.0), f"Rel. error |ŷ − y| (%) • ch={c}", meta_str, p_rel_res,
            dpi=args.dpi, tile_w_in=args.tile_w_in, tile_h_in=args.tile_h_in,
            cbar_label="Relative error (%)", cbar_fontsize=args.cbar_fontsize,
            cbar_max_ticks=args.cbar_max_ticks, tick_format='percent'
        ); saved_paths.append(str(p_rel_res))

        p_rel_dgt = imgs_dir / f"error_relpct_dgt__{base}.png"
        _save_single_tile(
            rel_pct_dgt[c], (0.0, 400.0), f"Rel. error |y − x| (%) • ch={c}", meta_str, p_rel_dgt,
            dpi=args.dpi, tile_w_in=args.tile_w_in, tile_h_in=args.tile_h_in,
            cbar_label="Relative error (%)", cbar_fontsize=args.cbar_fontsize,
            cbar_max_ticks=args.cbar_max_ticks, tick_format='percent'
        ); saved_paths.append(str(p_rel_dgt))

    # Bar plot: mean relative errors per channel
    rel_mean_res = [float(rel_pct_res[c].mean()) for c in range(C)]
    rel_mean_dgt = [float(rel_pct_dgt[c].mean()) for c in range(C)]
    p_bar = imgs_dir / f"barplot_relpct__gid_{gid}__t{tf}_to_t{tt}.png"
    _save_barplot_rel_means(rel_mean_res, rel_mean_dgt, p_bar,
                            dpi=args.dpi, fig_w_in=args.bar_w_in, fig_h_in=args.bar_h_in)
    saved_paths.append(str(p_bar))

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "h5_test": str(Path(h5_test_path).resolve()),
        "images_dir": str(imgs_dir.resolve()),
        "saved_pngs": len(saved_paths),
        "channels": C,
        "phys_vlims_per_channel": vlims_per_ch,
        "abs_error_vlims_per_channel": abs_vlims_per_ch,
        "rel_error_clip_percent": [0.0, 400.0],
        "mean_rel_error_residual_per_channel": rel_mean_res,
        "mean_rel_error_dgt_per_channel": rel_mean_dgt,
        "example_meta": example["meta"],
        "defaults": {
            "tile_inches": [args.tile_w_in, args.tile_h_in],
            "bar_inches": [args.bar_w_in, args.bar_h_in],
            "dpi": args.dpi,
            "tile_pixels": [int(args.tile_w_in * args.dpi), int(args.tile_h_in * args.dpi)],
            "bar_pixels": [int(args.bar_w_in * args.dpi), int(args.bar_h_in * args.dpi)],
        },
        "files": saved_paths,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
