#!/usr/bin/env python3
# pf_eval_grids_individual_minmetrics.py
#
# Evaluate on physical scale and export INDIVIDUAL high-DPI grids for selected
# time transitions only. Whole-test metrics are optional and stream-safe.
# Error metrics available:
#   rmse_global, mean_phase_error, mean_concentration_error, relative_l2_percent.
# Supports both absolute difference maps and per-pixel relative percentage error maps.
# No second-person phrasing in comments.

import argparse, importlib, importlib.util, json, sys, os, gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml, numpy as np, torch, h5py
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
# Debug: log argv and parsed args
import sys, json, hashlib, yaml, torch
from pathlib import Path

print("ARGV:", " ".join(sys.argv), flush=True)

# Replace with the real Namespace variable
ns = args

# Compute a stable “effective config” to a file next to outputs
eff = {}

# 1) start from YAML
yaml_cfg = {}
if ns.config:  # -c / --config
    p = Path(ns.config)
    yaml_cfg = yaml.safe_load(p.read_text())
    eff["__yaml_path__"] = str(p)
    eff["__yaml_sha256__"] = hashlib.sha256(p.read_bytes()).hexdigest()

# 2) overlay checkpoint config if present
ckpt_cfg = {}
if getattr(ns, "ckpt", None):
    ckpt = torch.load(ns.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config", ckpt.get("cfg", {})) or {}
    eff["__ckpt_path__"] = ns.ckpt

merged = {**yaml_cfg, **ckpt_cfg}

# 3) finally overlay explicit CLI options used for evaluation
cli_overrides = {
    "transitions": getattr(ns, "transitions", None),
    "select_gid": getattr(ns, "select_gid", None),
    "error_map": getattr(ns, "error_map", None),
    "cbar_fontsize": getattr(ns, "cbar_fontsize", None),
    "cbar_max_ticks": getattr(ns, "cbar_max_ticks", None),
}
for k, v in list(cli_overrides.items()):
    if v is None:
        cli_overrides.pop(k)

merged.update(cli_overrides)

eff["effective"] = merged
out_dir = Path(ns.out if getattr(ns, "out", None) else ".")
out_dir.mkdir(parents=True, exist_ok=True)
(Path(out_dir) / "effective_config.yaml").write_text(yaml.safe_dump(eff, sort_keys=True))
print("Wrote effective_config.yaml to", out_dir, flush=True)
print("Parsed args:", json.dumps(vars(ns), indent=2, default=str), flush=True)

# --------------------------- path-aware import helper ---------------------------

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

# --------------------------- utils ---------------------------

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("input", "target", "cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in ("gid", "pair_index", "stride"):
        if k in batch[0]:
            out[k] = [b[k] for b in batch]
    return out

def _percentiles(arrs: List[np.ndarray], lo=2.0, hi=98.0) -> Tuple[float, float]:
    vec = np.concatenate([a.ravel() for a in arrs]) if arrs else np.array([0.0], dtype=np.float64)
    return float(np.percentile(vec, lo)), float(np.percentile(vec, hi))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _norm_gid(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    t = str(s).strip()
    if t.startswith("sim_"): return t
    if t.isdigit(): return f"sim_{int(t):04d}"
    return t

# --------------------------- image export ---------------------------

def _save_single_grid(
    arr: np.ndarray,
    vlims: Tuple[float, float],
    title: str,
    out_path: Path,
    dpi: int,
    px_multiplier: float,
    *,
    cbar_label: str,
    cbar_fontsize: int = 16,
    cbar_max_ticks: int = 6,
    tick_format: Optional[str] = None,
) -> None:
    """Render a single grid to PNG with a readable colourbar.

    tick_format: None | 'percent' | 'sci'
    """
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    width_px = max(1, int(px_multiplier * w))
    height_px = max(1, int(px_multiplier * h))
    fig_w_in = width_px / dpi
    fig_h_in = height_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    # Slightly wider colourbar for larger tick labels
    ax = fig.add_axes([0.05, 0.05, 0.78, 0.90])
    im = ax.imshow(arr, vmin=vlims[0], vmax=vlims[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    cax = fig.add_axes([0.85, 0.10, 0.05, 0.80])
    cb = fig.colorbar(im, cax=cax)

    # Ticks: few, readable, consistent
    cb.locator = MaxNLocator(nbins=cbar_max_ticks, min_n_ticks=3, steps=[1, 2, 5, 10])
    if tick_format == 'percent':
        cb.formatter = FuncFormatter(lambda v, _: f"{v:.0f}%")
    elif tick_format == 'sci':
        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-3, 3))
        cb.formatter = sf
    cb.update_ticks()

    cb.ax.tick_params(labelsize=cbar_fontsize, width=0.8)
    cb.set_label(cbar_label, fontsize=cbar_fontsize, labelpad=8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("-k","--ckpt", required=True)
    ap.add_argument("-o","--outdir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--px-multiplier", type=float, default=10.0)
    ap.add_argument("--phys_lo", type=float, default=2.0)
    ap.add_argument("--phys_hi", type=float, default=98.0)
    # Selection of transitions
    ap.add_argument("--select-gid", type=str, default=None, help="gid like 'sim_0012' or integer '12'")
    ap.add_argument("--k-list", type=str, default=None, help="Comma-separated pair_index values to plot")
    ap.add_argument("--transitions", type=str, default=None,
                    help="Comma-separated 'a:b' transitions, unit stride only (e.g., '201:202,202:203')")
    ap.add_argument("--require-stride1", action="store_true", default=True)
    # Metrics scope
    ap.add_argument("--metrics-scope", choices=["all","selected","none"], default="selected")
    # Loader controls
    ap.add_argument("--loader-workers", type=int, default=None)
    ap.add_argument("--no-pin-memory", action="store_true", default=False)
    # Visual controls
    ap.add_argument(
        "--error-map",
        choices=["difference", "relative_percent", "both"],
        default="both",
        help="Error maps to export in the fourth panel(s). 'both' writes both absolute difference and percentage maps.",
    )
    ap.add_argument("--rel-eps", type=float, default=1e-8, help="Denominator clamp for relative percent error")
    ap.add_argument("--cbar-fontsize", type=int, default=16)
    ap.add_argument("--cbar-max-ticks", type=int, default=6)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Paths from YAML
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

    # Device
    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_grad_enabled(False)

    # Dataset + model
    ds_args = dict(cfg.get("dataloader",{}).get("args", {}))
    test_ds = DSClass(h5_test_path, **ds_args)

    model = ModelClass(**cfg.get("model",{}).get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()

    # Loader bounds
    env_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    cfg_workers = int(cfg.get("loader",{}).get("num_workers", 0))
    num_workers = args.loader_workers if args.loader_workers is not None else min(cfg_workers, env_cpus, 4)
    pin_memory = False if args.no_pin_memory else bool(cfg.get("loader",{}).get("pin_memory", True))

    # DataLoader kwargs: only set prefetch_factor when multiprocessing is enabled
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
        dl_kwargs["prefetch_factor"] = 1  # small prefetch to limit resident memory
    dl = DataLoader(test_ds, **dl_kwargs)

    out_dir = _ensure_dir(Path(args.outdir))
    imgs_dir = _ensure_dir(out_dir / "images")

    sel_gid = _norm_gid(args.select_gid)

    # Build k_set from --transitions or --k-list
    if args.transitions:
        k_set = set()
        for tok in args.transitions.split(","):
            a,b = [int(t) for t in tok.split(":")]
            if b != a + 1:
                raise SystemExit(f"Only unit-stride transitions supported, got {tok}")
            k_set.add(a)
        args.require_stride1 = True
    elif args.k_list:
        k_set = {int(k.strip()) for k in args.k_list.split(",") if k.strip()}
    else:
        raise SystemExit("Provide --transitions or --k-list.")

    with h5py.File(h5_test_path, "r") as h5:
        ch_mean = np.array(h5.attrs["channel_mean"], dtype=np.float32)
        ch_std = np.array(h5.attrs["channel_std"], dtype=np.float32)

        # Metrics accumulators
        do_metrics_all = args.metrics_scope == "all"
        do_metrics_selected = args.metrics_scope == "selected"

        se_total = 0.0; ye_total = 0.0; n_total = 0
        mae_per_channel: Optional[np.ndarray] = None

        # Selected examples only
        examples: List[Dict[str, Any]] = []
        C_detected: Optional[int] = None

        have_pairs = set()

        for batch in dl:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            cond = batch.get("cond")
            if cond is not None:
                cond = cond.to(device, non_blocking=True)

            if cond is not None:
                if cond.dim() != 2:
                    raise RuntimeError(f"Expected cond with shape [B, D], got {tuple(cond.shape)}")
                D = int(cond.shape[1])
                if (cond_dim_cfg is not None and D != int(cond_dim_cfg)) or \
                   (model_cond_dim is not None and D != int(model_cond_dim)):
                    raise RuntimeError(f"conditioning/model cond_dim mismatch with dataset D={D}")

            with torch.inference_mode():
                yhat = model(x, cond) if cond is not None else model(x)

            # CPU + de-standardise
            x_np = x.cpu().numpy().astype(np.float32, copy=False)
            y_np = y.cpu().numpy().astype(np.float32, copy=False)
            yhat_np = yhat.detach().cpu().numpy().astype(np.float32, copy=False)

            if C_detected is None:
                C_detected = int(x_np.shape[1])
                mae_per_channel = np.zeros((C_detected, 2), dtype=np.float64)

            ch_mean_b = ch_mean[None, :, None, None]
            ch_std_b = ch_std[None, :, None, None]
            x_phys = x_np * ch_std_b + ch_mean_b
            y_phys = y_np * ch_std_b + ch_mean_b
            yhat_phys = yhat_np * ch_std_b + ch_mean_b

            # Metrics scope
            if do_metrics_all:
                diff_all = yhat_phys - y_phys
                se_total += float(np.sum(diff_all * diff_all))
                ye_total += float(np.sum(y_phys * y_phys))
                n_total += int(y_phys.size)
                abs_diff_all = np.abs(diff_all)
                for c in range(C_detected):
                    mae_per_channel[c, 0] += float(abs_diff_all[:, c, :, :].sum())
                    mae_per_channel[c, 1] += int(abs_diff_all[:, c, :, :].size)

            # Selection capture
            B = y_phys.shape[0]
            for i in range(B):
                gid = batch.get("gid", [None])[i]
                if sel_gid is not None and str(gid) != sel_gid:
                    continue
                k = int(batch.get("pair_index", [i])[i]) if "pair_index" in batch else i

                grp = h5[str(gid)]
                stride_i = int(grp["pairs_stride"][k]) if "pairs_stride" in grp else int(grp.get("pairs_dt_euler",[1])[k])
                if args.require_stride1 and stride_i != 1:
                    continue
                if k not in k_set:
                    continue

                # Per-pair metadata
                dt_phys = float(grp["pairs_dt"][k]) if "pairs_dt" in grp else float("nan")
                z_dt = float(grp["pairs_dt_norm"][k]) if "pairs_dt_norm" in grp else float("nan")
                G_raw = float(grp.attrs.get("thermal_gradient_raw", float("nan")))
                muG = float(grp.attrs.get("thermal_mean", float("nan")))
                sdG = float(grp.attrs.get("thermal_std", float("nan")))
                z_G = (G_raw - muG) / sdG if (sdG and sdG > 0) else float("nan")

                rec = {
                    "x": x_phys[i], "y": y_phys[i], "yhat": yhat_phys[i],
                    "meta": {
                        "gid": gid, "pair_index": k,
                        "t_from": k, "t_to": k + stride_i, "stride": stride_i,
                        "dt_phys": dt_phys, "z_dt": z_dt, "G_raw": G_raw, "z_G": z_G,
                    },
                }
                examples.append(rec)
                have_pairs.add(k)

                # Metrics on selected only
                if do_metrics_selected:
                    dsel = rec["yhat"] - rec["y"]
                    se_total += float(np.sum(dsel * dsel))
                    ye_total += float(np.sum(rec["y"] * rec["y"]))
                    n_total += int(rec["y"].size)
                    abs_dsel = np.abs(dsel)
                    for c in range(C_detected):
                        mae_per_channel[c, 0] += float(abs_dsel[c].sum())
                        mae_per_channel[c, 1] += int(abs_dsel[c].size)

                # Early stop when all requested k are captured and metrics do not require full pass
                if have_pairs >= k_set and args.metrics_scope != "all":
                    break
            if have_pairs >= k_set and args.metrics_scope != "all":
                break

            # Release batch tensors early
            del x, y, yhat, x_np, y_np, yhat_np
            gc.collect()

    if not examples:
        raise RuntimeError("No examples matched the selection.")

    C = C_detected if C_detected is not None else 1

    # vlims from selected examples
    phys_vlims: List[Tuple[float, float]] = []
    diff_vlims: List[Tuple[float, float]] = []
    relpct_vlims: List[Tuple[float, float]] = []

    for c in range(C):
        phys_list, diff_list, relpct_list = [], [], []
        for rec in examples:
            phys_list.extend([rec["x"][c], rec["y"][c], rec["yhat"][c]])
            d = rec["y"][c] - rec["yhat"][c]
            diff_list.append(d)
            denom = np.maximum(np.abs(rec["y"][c]), float(args.rel_eps))
            relpct_list.append(100.0 * np.abs(d) / denom)
        lo, hi = _percentiles(phys_list, lo=args.phys_lo, hi=args.phys_hi)
        d_lo = min([a.min() for a in diff_list]); d_hi = max([a.max() for a in diff_list])
        if d_hi <= d_lo: d_hi = d_lo + 1e-12
        phys_vlims.append((lo, hi))
        diff_vlims.append((float(d_lo), float(d_hi)))
        rp_hi = max([a.max() for a in relpct_list]) if relpct_list else 1.0
        relpct_vlims.append((0.0, float(rp_hi)))

    # Metrics
    def _mae_from_acc(acc_row: np.ndarray) -> float:
        num, den = float(acc_row[0]), float(acc_row[1])
        return float(num / max(den, 1.0))

    rmse_global = float(np.sqrt(se_total / max(n_total, 1)))
    relative_l2_percent = float(np.sqrt(se_total / max(ye_total, 1e-30)) * 100.0)
    mean_phase_error = float("nan"); mean_concentration_error = float("nan")
    if "mae_per_channel" in locals() and mae_per_channel is not None:
        if C >= 1: mean_phase_error = _mae_from_acc(mae_per_channel[0])
        if C >= 2: mean_concentration_error = _mae_from_acc(mae_per_channel[1])

    # Export images
    imgs_dir = _ensure_dir(imgs_dir)
    for idx, rec in enumerate(examples):
        gid = rec["meta"]["gid"]; tf = rec["meta"]["t_from"]; tt = rec["meta"]["t_to"]
        k = rec["meta"]["pair_index"]; s = rec["meta"]["stride"]
        dt = rec["meta"]["dt_phys"]; zdt = rec["meta"]["z_dt"]; G = rec["meta"]["G_raw"]; zG = rec["meta"]["z_G"]
        meta_str = f"gid={gid}  k={k}  t{tf}->{tt}  stride={s}  dt={dt:.3e}  zdt={zdt:.3f}  G={G:.3e}  zG={zG:.3f}"
        for c in range(C):
            _save_single_grid(
                rec["x"][c], phys_vlims[c],
                title=f"Input x  |  {meta_str}  ch={c}",
                out_path=imgs_dir / f"ex_{idx:02d}__gid_{gid}__t{tf}_to_t{tt}__ch{c}__x_input.png",
                dpi=args.dpi, px_multiplier=args.px_multiplier,
                cbar_label="Physical value",
                cbar_fontsize=args.cbar_fontsize,
                cbar_max_ticks=args.cbar_max_ticks,
                tick_format='sci',
            )
            _save_single_grid(
                rec["y"][c], phys_vlims[c],
                title=f"Ground truth y  |  {meta_str}  ch={c}",
                out_path=imgs_dir / f"ex_{idx:02d}__gid_{gid}__t{tf}_to_t{tt}__ch{c}__y_ground_truth.png",
                dpi=args.dpi, px_multiplier=args.px_multiplier,
                cbar_label="Physical value",
                cbar_fontsize=args.cbar_fontsize,
                cbar_max_ticks=args.cbar_max_ticks,
                tick_format='sci',
            )
            _save_single_grid(
                rec["yhat"][c], phys_vlims[c],
                title=f"Prediction y_hat  |  {meta_str}  ch={c}",
                out_path=imgs_dir / f"ex_{idx:02d}__gid_{gid}__t{tf}_to_t{tt}__ch{c}__yhat_prediction.png",
                dpi=args.dpi, px_multiplier=args.px_multiplier,
                cbar_label="Physical value",
                cbar_fontsize=args.cbar_fontsize,
                cbar_max_ticks=args.cbar_max_ticks,
                tick_format='sci',
            )

            # Error maps: export according to selection
            if args.error_map in ("relative_percent", "both"):
                denom = np.maximum(np.abs(rec["y"][c]), float(args.rel_eps))
                relpct = 100.0 * np.abs(rec["y"][c] - rec["yhat"][c]) / denom
                _save_single_grid(
                    relpct, relpct_vlims[c],
                    title=f"Relative error (%)  |  {meta_str}  ch={c}",
                    out_path=imgs_dir / f"ex_{idx:02d}__gid_{gid}__t{tf}_to_t{tt}__ch{c}__relerr_percent.png",
                    dpi=args.dpi, px_multiplier=args.px_multiplier,
                    cbar_label="Relative error (%)",
                    cbar_fontsize=args.cbar_fontsize,
                    cbar_max_ticks=args.cbar_max_ticks,
                    tick_format='percent',
                )
            if args.error_map in ("difference", "both"):
                _save_single_grid(
                    rec["y"][c] - rec["yhat"][c], diff_vlims[c],
                    title=f"Difference (y - y_hat)  |  {meta_str}  ch={c}",
                    out_path=imgs_dir / f"ex_{idx:02d}__gid_{gid}__t{tf}_to_t{tt}__ch{c}__diff_y_minus_yhat.png",
                    dpi=args.dpi, px_multiplier=args.px_multiplier,
                    cbar_label="Difference (units)",
                    cbar_fontsize=args.cbar_fontsize,
                    cbar_max_ticks=args.cbar_max_ticks,
                    tick_format='sci',
                )

    # Outputs
    metrics = {
        "scope": args.metrics_scope,
        "rmse_global": rmse_global,
        "mean_phase_error": mean_phase_error,
        "mean_concentration_error": mean_concentration_error,
        "relative_l2_percent": relative_l2_percent,
        "plotted_transitions": [rec["meta"] for rec in examples],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "h5_test": str(Path(h5_test_path).resolve()),
        "examples_saved": len(examples),
        "channels": C,
        "images_dir": str(imgs_dir.resolve()),
        "dpi": args.dpi,
        "px_multiplier": args.px_multiplier,
        "phys_vlims": phys_vlims,
        "diff_vlims": diff_vlims,
        "relpct_vlims": relpct_vlims,
        "error_map": args.error_map,
        "rel_eps": args.rel_eps,
        "cbar_fontsize": args.cbar_fontsize,
        "cbar_max_ticks": args.cbar_max_ticks,
        "select_gid": sel_gid,
        "k_set": sorted(list(k_set)),
        "require_stride1": args.require_stride1,
        "loader": {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": (1 if num_workers > 0 else None),
            "batch": args.batch,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
