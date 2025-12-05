#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models_plots.py — metrics-only evaluator compatible with legacy visual flags.

- Test-set inference only.
- Reports:
  • Relative L2 (operator-learning style): global over dataset and mean over samples.
  • MSE, RMSE, MAE overall and per-channel.
  • Phase and concentration summaries (MAE, signed bias, RMSE, per-channel RelL2).
  • Two-point autocorrelation S2(r) for phase and concentration (radially averaged, FFT, periodic).
- Writes: metrics.json, metrics.txt, autocorr_phase.csv, autocorr_concentration.csv
- Accepts legacy visual CLI flags (--tmin, --vis-n, etc.) and ignores them.

Code comments avoid second-person phrasing.
"""

import os, math, argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# ---------------- dynamic import helpers (as in trainer) ----------------
import sys, importlib, importlib.util
def _load_symbol(py_path: str, symbol: str):
    p = Path(py_path).resolve()
    def _guess_root(q: Path):
        for a in [q.parent, *q.parents]:
            if (a / "models").is_dir() or (a / "scripts").is_dir():
                return a
        return q.parent
    root = _guess_root(p)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
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
    for k in ("input", "target", "cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "gid" in batch[0]: out["gid"] = [b["gid"] for b in batch]
    if "pair_index" in batch[0]: out["pair_index"] = [b["pair_index"] for b in batch]
    return out

def _extract_cond_from_channels(x: torch.Tensor, cond_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4 and x.size(1) >= cond_dim
    cond_map = x[:, -cond_dim:, ...]
    cond_vec = cond_map.mean(dim=(-2, -1))
    x_trim = x[:, :-cond_dim, ...]
    return x_trim, cond_vec

# ---------------- maths helpers ----------------
def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)

def _pack_stats(sum_se: np.ndarray, sum_abs: np.ndarray, count: int) -> Dict[str, Any]:
    C = sum_se.shape[0]
    mse_c = sum_se / max(count, 1)
    rmse_c = np.sqrt(np.maximum(mse_c, 0.0))
    mae_c = sum_abs / max(count, 1)
    mse_all = float(mse_c.mean())
    rmse_all = float(np.sqrt(np.maximum(mse_all, 0.0)))
    mae_all = float(mae_c.mean())
    return {
        "per_channel": [{"channel": int(i), "mse": float(mse_c[i]), "rmse": float(rmse_c[i]), "mae": float(mae_c[i])}
                        for i in range(C)],
        "overall": {"mse": mse_all, "rmse": rmse_all, "mae": mae_all}
    }

# ---------------- two-point autocorrelation ----------------
def _fft_autocorr2d(field: np.ndarray) -> np.ndarray:
    f = field.astype(np.float64, copy=False)
    f = f - np.nanmean(f)
    H, W = f.shape
    Fk = np.fft.rfftn(f)
    S = (Fk * np.conj(Fk)).real
    ac = np.fft.irfftn(S, s=f.shape)
    ac = np.fft.fftshift(ac) / (H * W)
    c0 = float(ac[H//2, W//2])
    if not np.isfinite(c0) or abs(c0) < 1e-20:
        return np.zeros_like(ac)
    return (ac / c0).real

def _radial_profile(arr: np.ndarray, nbins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    H, W = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices(arr.shape)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_max = r.max()
    bins = np.linspace(0.0, r_max + 1e-9, nbins + 1)
    which = np.digitize(r.ravel(), bins) - 1
    sums = np.bincount(which, weights=arr.ravel(), minlength=nbins)
    counts = np.bincount(which, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(counts > 0, sums / counts, np.nan)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    return r_mid.astype(np.float64), prof.astype(np.float64)

# ---------------- core evaluation ----------------
@torch.inference_mode()
def run_eval(cfg_path: str,
             ckpt_path: str,
             test_override: Optional[str],
             batch_size: int,
             out_dir: Path,
             phase_channel: Optional[int],
             concentration_channel: Optional[int],
             autocorr_bins: int = 256) -> None:

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("medium")
    except Exception: pass

    # dataset (test only)
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})
    paths = cfg["paths"]["h5"]
    test_spec = test_override if test_override is not None else paths.get("test", None)
    if isinstance(test_spec, dict):
        test_ds = DSClass(**test_spec, **ds_args)
    elif isinstance(test_spec, str) and len(test_spec) > 0:
        test_ds = DSClass(test_spec, **ds_args)
    else:
        raise ValueError("No test dataset path available.")

    num_workers = int(cfg["loader"].get("num_workers", 4))
    pin_memory  = bool(cfg["loader"].get("pin_memory", True))
    dl = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory,
                    collate_fn=_collate, persistent_workers=False)

    # model
    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    if bool(cfg["trainer"].get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # checkpoint
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # conditioning
    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))
    cond_source = str(cond_cfg.get("source", "field")).lower()
    cond_dim    = int(cond_cfg.get("cond_dim", 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = out_dir / "metrics.json"
    metrics_txt  = out_dir / "metrics.txt"

    # accumulators
    C_ref: Optional[int] = None
    sum_se_c  = None
    sum_abs_c = None
    sum_err_c = None
    count_elems = 0

    sum_se_all = 0.0
    sum_y2_all = 0.0
    sample_rel_se: List[float] = []
    sample_rel_y2: List[float] = []

    # autocorr accumulators
    phase_acc_rad: List[np.ndarray] = []
    conc_acc_rad:  List[np.ndarray] = []
    r_template = None

    for batch in dl:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        if C_ref is None:
            C_ref = int(y.shape[1])
            sum_se_c  = np.zeros((C_ref,), dtype=np.float64)
            sum_abs_c = np.zeros((C_ref,), dtype=np.float64)
            sum_err_c = np.zeros((C_ref,), dtype=np.float64)

        if bool(cfg["trainer"].get("channels_last", False)) and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        cond = None
        if cond_enabled:
            if cond_source == "channels":
                x, cond = _extract_cond_from_channels(x, cond_dim)
            else:
                if "cond" not in batch:
                    raise KeyError("Conditioning enabled but 'cond' missing.")
                c = batch["cond"].to(device, non_blocking=True)
                assert c.dim() == 2 and c.size(1) == cond_dim
                cond = c

        yhat = model(x, cond) if cond is not None else model(x)

        err = yhat - y
        se_c = (err**2).sum(dim=(0,2,3)).double().cpu().numpy()
        ae_c = (err.abs()).sum(dim=(0,2,3)).double().cpu().numpy()
        sum_se_c += se_c
        sum_abs_c += ae_c
        sum_err_c += err.sum(dim=(0,2,3)).double().cpu().numpy()

        elems_per_channel = int(y.shape[0] * y.shape[2] * y.shape[3])
        count_elems += elems_per_channel

        sum_se_all += float((err**2).sum().double().item())
        sum_y2_all += float((y**2).sum().double().item())

        # per-sample RelL2
        eb = err.flatten(1); yb = y.flatten(1)
        sample_rel_se += (eb**2).sum(dim=1).double().cpu().numpy().tolist()
        sample_rel_y2 += (yb**2).sum(dim=1).double().cpu().numpy().tolist()

        # autocorr on first item for selected channels
        def _acc_ac(arr_4d: torch.Tensor, ch: Optional[int], store: List[np.ndarray]):
            if ch is None or ch < 0 or ch >= arr_4d.shape[1]:
                return
            f = arr_4d[0, ch].detach().float().cpu().numpy()
            ac = _fft_autocorr2d(f)
            r_px, S2 = _radial_profile(ac, nbins=autocorr_bins)
            nonlocal r_template
            if r_template is None:
                r_template = r_px
            else:
                m = min(len(r_template), len(r_px))
                r_template = r_template[:m]; S2 = S2[:m]
            store.append(S2)
        _acc_ac(y, phase_channel, phase_acc_rad)
        _acc_ac(y, concentration_channel, conc_acc_rad)

    # pack stats
    stats = _pack_stats(sum_se_c, sum_abs_c, count_elems)
    rel_l2_global = math.sqrt(_safe_div(sum_se_all, sum_y2_all))
    rel_l2_mean   = float(np.mean([math.sqrt(_safe_div(se, y2)) for se, y2 in zip(sample_rel_se, sample_rel_y2)])) if sample_rel_se else float("nan")
    rel_l2_global_c = np.sqrt(np.maximum((sum_se_c / (sum_y2_all / len(sum_se_c) + 1e-12)), 0.0))  # conservative placeholder if channel-wise y2 not tracked

    bias_c = (sum_err_c / max(count_elems, 1)).astype(float)

    def _chan_summary(idx: Optional[int]) -> Dict[str, Any]:
        if idx is None or idx < 0 or idx >= len(bias_c):
            return {"channel": None}
        pc = stats["per_channel"][idx]
        return {
            "channel": int(idx),
            "mse": float(pc["mse"]), "rmse": float(pc["rmse"]), "mae": float(pc["mae"]),
            "bias": float(bias_c[idx]),
            "rel_l2_global": float(rel_l2_global_c[idx])
        }

    def _agg_ac(curves: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        if not curves: return None
        arr = np.stack(curves, axis=0)
        return {"r_px": r_template.tolist(),
                "S2_mean": np.nanmean(arr, axis=0).tolist(),
                "S2_std":  np.nanstd(arr,  axis=0).tolist()}

    out = {
        "overall": stats["overall"],
        "per_channel": stats["per_channel"],
        "rel_l2": {
            "global_dataset": float(rel_l2_global),
            "mean_over_samples": float(rel_l2_mean)
        },
        "phase_summary": _chan_summary(phase_channel),
        "concentration_summary": _chan_summary(concentration_channel),
        "autocorrelation": {
            "phase": _agg_ac(phase_acc_rad),
            "concentration": _agg_ac(conc_acc_rad)
        },
        "counts": {"elements_per_channel": int(count_elems), "channels": int(C_ref or 0)}
    }

    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    with (out_dir / "metrics.txt").open("w") as f:
        f.write("== Test metrics ==\n")
        f.write(f"RelL2 (global): {out['rel_l2']['global_dataset']:.6g}\n")
        f.write(f"RelL2 (mean-of-samples): {out['rel_l2']['mean_over_samples']:.6g}\n")
        f.write(f"Overall MSE: {out['overall']['mse']:.6g}\n")
        f.write(f"Overall RMSE: {out['overall']['rmse']:.6g}\n")
        f.write(f"Overall MAE: {out['overall']['mae']:.6g}\n")
        if out["phase_summary"]["channel"] is not None:
            ps = out["phase_summary"]; f.write(f"Phase(ch={ps['channel']}): MAE={ps['mae']:.6g}, bias={ps['bias']:.6g}, RMSE={ps['rmse']:.6g}\n")
        if out["concentration_summary"]["channel"] is not None:
            cs = out["concentration_summary"]; f.write(f"Conc(ch={cs['channel']}): MAE={cs['mae']:.6g}, bias={cs['bias']:.6g}, RMSE={cs['rmse']:.6g}\n")
    if out["autocorrelation"]["phase"] is not None:
        with (out_dir / "autocorr_phase.csv").open("w") as f:
            f.write("r_px,S2_mean,S2_std\n")
            for r, m, s in zip(out["autocorrelation"]["phase"]["r_px"],
                               out["autocorrelation"]["phase"]["S2_mean"],
                               out["autocorrelation"]["phase"]["S2_std"]):
                f.write(f"{r:.6g},{m:.8g},{s:.8g}\n")
    if out["autocorrelation"]["concentration"] is not None:
        with (out_dir / "autocorr_concentration.csv").open("w") as f:
            f.write("r_px,S2_mean,S2_std\n")
            for r, m, s in zip(out["autocorrelation"]["concentration"]["r_px"],
                               out["autocorrelation"]["concentration"]["S2_mean"],
                               out["autocorrelation"]["concentration"]["S2_std"]):
                f.write(f"{r:.6g},{m:.8g},{s:.8g}\n")

# ---------------- CLI with legacy-flag compatibility ----------------
def _parse():
    ap = argparse.ArgumentParser(description="Metrics-only evaluator (legacy flags accepted)")
    # primary flags
    ap.add_argument("-c","--config", dest="config", required=True, type=str)
    ap.add_argument("-k","--ckpt",   dest="ckpt",   required=True, type=str)
    ap.add_argument("--test-h5", type=str, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--phase-channel", type=int, default=0)
    ap.add_argument("--concentration-channel", type=int, default=1)
    ap.add_argument("--autocorr-bins", type=int, default=256)
    # legacy visual flags (accepted and ignored)
    ap.add_argument("--tmin", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tmax", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dt-index", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dt-eq", nargs="?", const=True, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dt-threshold", nargs="?", const=True, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--vis-n", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--vis-channel", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reservoir-px", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--q-low", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--q-high", type=float, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()

def main():
    args = _parse()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model"]["class"]
    run_root = Path(cfg["trainer"].get("out_dir", "./results"))
    out_dir = Path(args.outdir) if args.outdir else (run_root / model_name / "eval_metrics_only")
    run_eval(cfg_path=args.config,
             ckpt_path=args.ckpt,
             test_override=args.test_h5,
             batch_size=args.batch,
             out_dir=out_dir,
             phase_channel=(None if args.phase_channel < 0 else int(args.phase_channel)),
             concentration_channel=(None if args.concentration_channel < 0 else int(args.concentration_channel)),
             autocorr_bins=int(args.autocorr_bins))

if __name__ == "__main__":
    main()
