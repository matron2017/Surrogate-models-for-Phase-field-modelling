#!/usr/bin/env python3
# pf_eval_standardised_with_split_overview.py
# Standardised-scale metrics over the TEST split plus a dataset/split overview.
# Reports both energy-ratio (SSE / target energy) and operator-learning relative L2 NORM errors.
# Also reports relative L2 NORM error in percent with 5% pass/fail flags.
# Memory-lean: on-device computation, scalar accumulators, optional AMP.
# No second-person phrasing in comments.

import os, argparse, json, importlib, importlib.util, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
import yaml
import h5py

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
    out: Dict[str, Any] = {}
    for k in ("input","target","cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "gid" in batch[0]: out["gid"] = [b["gid"] for b in batch]
    if "pair_index" in batch[0]: out["pair_index"] = [int(b["pair_index"]) for b in batch]
    return out

def _to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x

def _get_h5_path(entry: Any) -> Optional[str]:
    if entry is None: return None
    if isinstance(entry, str): return entry
    if isinstance(entry, dict) and "path" in entry: return entry["path"]
    return None

def _summarise_split(h5_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not h5_path: return None
    h5p = Path(h5_path)
    if not h5p.exists():
        return {"path": str(h5p), "error": "file_not_found"}
    pairs_total = 0
    sims = 0
    stride_counts: Dict[int,int] = defaultdict(int)
    dt_values: Counter = Counter()
    gradient_counts: Dict[float,int] = defaultdict(int)
    pairs_per_sim: List[int] = []

    with h5py.File(h5p, "r") as f:
        for gid in f.keys():
            grp = f[gid]
            sims += 1
            n_pairs = int(len(grp["pairs_dt"])) if "pairs_dt" in grp else 0
            pairs_total += n_pairs
            pairs_per_sim.append(n_pairs)

            if "pairs_stride" in grp:
                strides = grp["pairs_stride"][:]
            elif "pairs_dt_euler" in grp:
                strides = grp["pairs_dt_euler"][:]
            else:
                strides = None
            if strides is not None and n_pairs > 0:
                for s in strides[:n_pairs]:
                    stride_counts[int(s)] += 1

            if "pairs_dt" in grp and n_pairs > 0:
                for dt in grp["pairs_dt"][:]:
                    try:
                        dt_values[float(dt)] += 1
                    except Exception:
                        pass

            tg = None
            for key in ("thermal_gradient_raw", "thermal_gradient"):
                if key in grp.attrs:
                    try:
                        tg = float(grp.attrs[key])
                    except Exception:
                        tg = None
                    break
            if tg is not None:
                gradient_counts[tg] += n_pairs

    def _stats(v: List[int]) -> Dict[str, Any]:
        if not v: return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        v_sorted = sorted(v)
        m = len(v_sorted)
        median = float(v_sorted[m//2] if m % 2 else 0.5*(v_sorted[m//2-1]+v_sorted[m//2]))
        return {
            "min": int(min(v_sorted)),
            "max": int(max(v_sorted)),
            "mean": float(sum(v_sorted)/len(v_sorted)),
            "median": float(median)
        }

    strides_sorted = {int(k): int(v) for k, v in sorted(stride_counts.items(), key=lambda kv: kv[0])}
    grads_sorted = {f"{g:.6g}": int(cnt) for g, cnt in sorted(gradient_counts.items(), key=lambda kv: kv[0])}
    dt_sorted = [float(k) for k, _ in sorted(dt_values.items(), key=lambda kv: kv[0])]

    return {
        "path": str(h5p.resolve()),
        "pairs_total": int(pairs_total),
        "sim_trajectories": int(sims),
        "pairs_per_sim_stats": _stats(pairs_per_sim),
        "strides_pair_counts": strides_sorted,
        "unique_delta_t_values": dt_sorted,
        "num_unique_delta_t": len(dt_sorted),
        "pairs_per_thermal_gradient": grads_sorted
    }

def _shape_str(t: torch.Tensor) -> str:
    return "[" + ", ".join(str(int(s)) for s in t.shape) + "]"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("-k","--ckpt",   required=True)
    ap.add_argument("-o","--outdir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conc-ch", type=int, default=0, help="channel index for concentration")
    ap.add_argument("--phase-ch", type=int, default=1, help="channel index for phase field")
    ap.add_argument("--eps", type=float, default=1e-30)
    ap.add_argument("--amp", action="store_true", help="enable autocast for model forward on CUDA")
    ap.add_argument("--num-workers", type=int, default=None, help="override DataLoader workers")
    ap.add_argument("--pin-memory", type=int, choices=[0,1], default=None, help="override pin_memory")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    # Dataset and model
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = dict(cfg.get("dataloader", {}).get("args", {}))
    h5_paths = cfg.get("paths", {}).get("h5", {})
    test_h5  = _get_h5_path(h5_paths.get("test"))
    train_h5 = _get_h5_path(h5_paths.get("train"))
    val_h5   = _get_h5_path(h5_paths.get("val"))
    if not test_h5:
        raise RuntimeError("Missing TEST HDF5 path in config at paths.h5.test")

    test_ds = DSClass(test_h5, **ds_args)

    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # Conditioning passthrough if present
    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))

    # Loader
    loader_cfg = dict(cfg.get("loader", {}))
    num_workers = args.num_workers if args.num_workers is not None else int(loader_cfg.get("num_workers", 4))
    pin_memory = bool(args.pin_memory) if args.pin_memory is not None else bool(loader_cfg.get("pin_memory", True))
    dl = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=_collate,
    )

    # Probe one batch to report tensor shapes
    example_shapes: Dict[str, str] = {}
    for batch in dl:
        xin = batch["input"]
        ygt = batch["target"]
        example_shapes["input_bchw"]  = _shape_str(xin[0])  # per-item
        example_shapes["target_bchw"] = _shape_str(ygt[0])  # per-item
        del xin, ygt
        break

    # ---------- metrics on STANDARDISED scale ----------
    # Energy-ratio quantities (SSE / target energy) and corresponding NORM errors (sqrt of ratios).
    # Operator-learning convention commonly reports the per-sample relative L2 NORM, mean over samples.
    eps = float(args.eps)
    se_total = 0.0                      # sum of squared error over all elements
    ye_total = 0.0                      # sum of squared target over all elements
    sum_rel_l2_ratio = 0.0              # mean of per-sample energy ratios
    sum_rel_l2_norm  = 0.0              # mean of per-sample relative L2 NORMs
    n_samples = 0

    # Zero-guard counters
    used_eps_samples = 0                # count of samples where ||y||^2 <= eps
    used_eps_samples_c: List[int] = []  # per-channel counts, allocated on first batch

    # Per-channel accumulators
    l1_c: List[float] = []
    n_c: List[int] = []
    sse_c: List[float] = []
    ye_c:  List[float] = []
    sum_rel_l2_ratio_c: List[float] = []   # mean of per-sample energy ratios per channel
    sum_rel_l2_norm_c:  List[float] = []   # mean of per-sample relative L2 NORMs per channel
    n_samples_c: List[int] = []

    autocast_cm = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_cuda and args.amp))
    C_known = None

    for batch in dl:
        y = _to_device(batch["target"], device).to(torch.float32)
        x = _to_device(batch["input"], device)
        cond = _to_device(batch["cond"], device) if (cond_enabled and "cond" in batch) else None

        if C_known is None:
            C_known = int(y.shape[1])
            l1_c = [0.0] * C_known
            n_c = [0] * C_known
            sse_c = [0.0] * C_known
            ye_c  = [0.0] * C_known
            sum_rel_l2_ratio_c = [0.0] * C_known
            sum_rel_l2_norm_c  = [0.0] * C_known
            n_samples_c = [0] * C_known
            used_eps_samples_c = [0] * C_known

        with torch.inference_mode(), autocast_cm:
            pred = model(x, cond) if cond is not None else model(x)
        pred = pred.to(torch.float32)

        B = y.shape[0]
        for i in range(B):
            yi  = y[i]               # [C,H,W]
            yhi = pred[i]            # [C,H,W]
            diff = yhi - yi

            sse_i = (diff * diff).sum().item()
            ye_i  = (yi * yi).sum().item()
            se_total += sse_i
            ye_total += ye_i

            # per-item MAE by channel
            l1_per_ch = diff.abs().sum(dim=(1,2)).tolist()
            hw = diff.shape[1] * diff.shape[2]
            for c in range(C_known):
                l1_c[c] += float(l1_per_ch[c])
                n_c[c]  += int(hw)

            # per-item energy ratio and norm (all channels) with explicit zero guard
            denom = ye_i if ye_i > eps else eps
            if ye_i <= eps:
                used_eps_samples += 1
            ratio_i = sse_i / denom
            sum_rel_l2_ratio += ratio_i
            sum_rel_l2_norm  += math.sqrt(ratio_i)
            n_samples += 1

            # per-channel accumulators for energy and norms with zero guards
            for c in range(C_known):
                sse_i_c = (diff[c] * diff[c]).sum().item()
                ye_i_c  = (yi[c] * yi[c]).sum().item()
                sse_c[c] += sse_i_c
                ye_c[c]  += ye_i_c
                denom_c = ye_i_c if ye_i_c > eps else eps
                if ye_i_c <= eps:
                    used_eps_samples_c[c] += 1
                ratio_i_c = sse_i_c / denom_c
                sum_rel_l2_ratio_c[c] += ratio_i_c
                sum_rel_l2_norm_c[c]  += math.sqrt(ratio_i_c)
                n_samples_c[c] += 1

            del yi, yhi, diff

        del x, y, pred
        if use_cuda:
            torch.cuda.empty_cache()

    if C_known is None:
        raise RuntimeError("Empty TEST dataset.")
    elements_total = int(sum(n_c))
    if n_samples == 0 or elements_total == 0:
        raise RuntimeError("No TEST samples; metrics cannot be computed.")
    if not (0 <= args.conc_ch < C_known and 0 <= args.phase_ch < C_known):
        raise ValueError(f"Channel indices must be in [0, {C_known-1}].")

    # Scalar errors on standardised scale
    mean_concentration_error = l1_c[args.conc_ch] / max(n_c[args.conc_ch], 1)
    mean_phasefield_error    = l1_c[args.phase_ch] / max(n_c[args.phase_ch], 1)
    mse                      = se_total / elements_total
    rmse                     = math.sqrt(mse)

    # Energy ratios
    rel_l2_ratio_global      = se_total / max(ye_total, eps)
    rel_l2_ratio_mean        = sum_rel_l2_ratio / n_samples

    # Relative L2 NORM errors (operator-learning convention)
    rel_l2_norm_global       = math.sqrt(rel_l2_ratio_global)
    rel_l2_norm_mean         = sum_rel_l2_norm / n_samples

    # Per-channel: energy ratios and norms
    rel_l2_ratio_global_c = [(sse_c[c] / max(ye_c[c], eps)) for c in range(C_known)]
    rel_l2_ratio_mean_c   = [(sum_rel_l2_ratio_c[c] / max(n_samples_c[c], 1)) for c in range(C_known)]
    rel_l2_norm_global_c  = [math.sqrt(rel_l2_ratio_global_c[c]) for c in range(C_known)]
    rel_l2_norm_mean_c    = [(sum_rel_l2_norm_c[c] / max(n_samples_c[c], 1)) for c in range(C_known)]

    # Relative L2 NORM error in percent (non-negative)
    rel_l2_pct_global   = 100.0 * rel_l2_norm_global
    rel_l2_pct_mean     = 100.0 * rel_l2_norm_mean
    rel_l2_pct_global_c = [100.0 * v for v in rel_l2_norm_global_c]
    rel_l2_pct_mean_c   = [100.0 * v for v in rel_l2_norm_mean_c]

    # 5% pass/fail flags
    passes_5pct = {
        "global": bool(rel_l2_pct_global <= 5.0),
        "mean_over_samples": bool(rel_l2_pct_mean <= 5.0),
        "per_channel_global": [bool(v <= 5.0) for v in rel_l2_pct_global_c],
        "per_channel_mean_over_samples": [bool(v <= 5.0) for v in rel_l2_pct_mean_c],
    }

    # Zero-guard summary
    zero_guard_info = {
        "epsilon": eps,
        "samples_with_guard": int(used_eps_samples),
        "per_channel_samples_with_guard": [int(z) for z in used_eps_samples_c],
        "global_denominator_used_guard": bool(ye_total <= eps),
    }

    # ---------- split summaries from HDF5 ----------
    split_summary = {
        "train": _summarise_split(train_h5),
        "val":   _summarise_split(val_h5),
        "test":  _summarise_split(test_h5),
    }

    # ---------- compose dataset overview Markdown ----------
    def _fmt_dt_list(dts: Optional[List[float]]) -> str:
        if not dts: return "[]"
        return "[" + ", ".join(f"{d:.6g}" for d in dts) + "]"

    test_info = split_summary["test"] or {}
    strides = test_info.get("strides_pair_counts") or {}
    stride_keys = sorted(strides.keys())
    dt_vals = test_info.get("unique_delta_t_values") or []
    num_dt = test_info.get("num_unique_delta_t", 0)
    n_pairs_train = (split_summary["train"] or {}).get("pairs_total", 0)
    n_pairs_val   = (split_summary["val"] or {}).get("pairs_total", 0)
    n_pairs_test  = (split_summary["test"] or {}).get("pairs_total", 0)
    sims_train    = (split_summary["train"] or {}).get("sim_trajectories", 0)
    sims_val      = (split_summary["val"] or {}).get("sim_trajectories", 0)
    sims_test     = (split_summary["test"] or {}).get("sim_trajectories", 0)

    per_item_input = example_shapes.get("input_bchw", "?")
    per_item_target = example_shapes.get("target_bchw", "?")

    md_lines = []
    md_lines.append("## Dataset structure and split (phase-field evolution with thermal gradients)")
    md_lines.append("")
    md_lines.append("**Temporal pairing used.**")
    md_lines.append(f"`pair_strides_observed = {stride_keys}`")
    md_lines.append(f"`unique_delta_t_values = {_fmt_dt_list(dt_vals)}`  (count = {num_dt})")
    md_lines.append("")
    md_lines.append("**Per-item tensors (channel-first).**")
    md_lines.append(f"- **Input** `{per_item_input}` (standardised)")
    md_lines.append(f"- **Target** `{per_item_target}` (standardised)")
    md_lines.append("")
    md_lines.append("**Models and pairing.**")
    md_lines.append(f"- **Test trajectories:** `{sims_test}`  "
                    f"pairs/sim stats: {json.dumps((split_summary['test'] or {}).get('pairs_per_sim_stats', {}))}")
    md_lines.append(f"- **Thermal gradients (pairs per gradient, TEST):** "
                    f"{json.dumps((split_summary['test'] or {}).get('pairs_per_thermal_gradient', {}))}")
    md_lines.append("")
    md_lines.append("**Split sizes implied by the pairing.**")
    md_lines.append(f"- **Train:** `{sims_train} sims`, `{n_pairs_train}` pairs")
    md_lines.append(f"- **Val:** `{sims_val} sims`, `{n_pairs_val}` pairs")
    md_lines.append(f"- **Test:** `{sims_test} sims`, `{n_pairs_test}` pairs")
    md_lines.append(f"**Totals:** `{sims_train + sims_val + sims_test} sims`, "
                    f"`{n_pairs_train + n_pairs_val + n_pairs_test}` pairs")
    md_lines.append("")
    md_lines.append("**How the split is defined.**")
    md_lines.append(f"- Files: train=`{train_h5}`, val=`{val_h5}`, test=`{test_h5}`")

    overview_md = "\n".join(md_lines)

    # ---------- output ----------
    report = {
        "scale": "standardised",
        "metrics_test": {
            "mae": {
                "concentration": float(mean_concentration_error),
                "phasefield":    float(mean_phasefield_error)
            },
            "mse": float(mse),
            "rmse": float(rmse),

            # Energy-ratio form (squared relative error)
            "relative_l2_energy_ratio": {
                "global": float(rel_l2_ratio_global),
                "mean_over_samples": float(rel_l2_ratio_mean),
                "per_channel": {
                    "global": [float(v) for v in rel_l2_ratio_global_c],
                    "mean_over_samples": [float(v) for v in rel_l2_ratio_mean_c]
                }
            },

            # Operator-learning convention: relative L2 NORM error (ratio's square root)
            "relative_l2_norm_error": {
                "global": float(rel_l2_norm_global),
                "mean_over_samples": float(rel_l2_norm_mean),
                "per_channel": {
                    "global": [float(v) for v in rel_l2_norm_global_c],
                    "mean_over_samples": [float(v) for v in rel_l2_norm_mean_c]
                }
            },

            # Relative L2 NORM error in percent
            "relative_l2_norm_error_pct": {
                "global": float(rel_l2_pct_global),
                "mean_over_samples": float(rel_l2_pct_mean),
                "per_channel": {
                    "global": [float(v) for v in rel_l2_pct_global_c],
                    "mean_over_samples": [float(v) for v in rel_l2_pct_mean_c]
                }
            },

            # Threshold checks (5% preferred)
            "thresholds": {
                "five_percent": passes_5pct
            },

            # Division-by-zero guards summary
            "denominator_zero_guards": zero_guard_info,

            "counts": {
                "channels": C_known,
                "samples_used": n_samples,
                "elements": elements_total
            },
            "channels": {
                "concentration_index": args.conc_ch,
                "phasefield_index": args.phase_ch
            },
            "by_field": {
                "concentration": {
                    "mae": float(mean_concentration_error),
                    "relative_l2_energy_ratio": {
                        "global": float(rel_l2_ratio_global_c[args.conc_ch]),
                        "mean_over_samples": float(rel_l2_ratio_mean_c[args.conc_ch])
                    },
                    "relative_l2_norm_error": {
                        "global": float(rel_l2_norm_global_c[args.conc_ch]),
                        "mean_over_samples": float(rel_l2_norm_mean_c[args.conc_ch])
                    },
                    "relative_l2_norm_error_pct": {
                        "global": float(rel_l2_pct_global_c[args.conc_ch]),
                        "mean_over_samples": float(rel_l2_pct_mean_c[args.conc_ch])
                    }
                },
                "phasefield": {
                    "mae": float(mean_phasefield_error),
                    "relative_l2_energy_ratio": {
                        "global": float(rel_l2_ratio_global_c[args.phase_ch]),
                        "mean_over_samples": float(rel_l2_ratio_mean_c[args.phase_ch])
                    },
                    "relative_l2_norm_error": {
                        "global": float(rel_l2_norm_global_c[args.phase_ch]),
                        "mean_over_samples": float(rel_l2_norm_mean_c[args.phase_ch])
                    },
                    "relative_l2_norm_error_pct": {
                        "global": float(rel_l2_pct_global_c[args.phase_ch]),
                        "mean_over_samples": float(rel_l2_pct_mean_c[args.phase_ch])
                    }
                }
            }
        },
        "split_summary": split_summary,
        "overview_markdown": overview_md,
        "config_paths": {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.ckpt).resolve()),
            "h5_train": str(Path(train_h5).resolve()) if train_h5 else None,
            "h5_val":   str(Path(val_h5).resolve()) if val_h5 else None,
            "h5_test":  str(Path(test_h5).resolve()) if test_h5 else None
        },
        "amp_enabled": bool(use_cuda and args.amp)
    }

    (out_dir / "metrics_standardised.json").write_text(json.dumps(report, indent=2))
    (out_dir / "dataset_overview.md").write_text(overview_md)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
