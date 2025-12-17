#!/usr/bin/env python3
# pf_autoreg_rollout.py — Autoregressive multi-step predictor and visualiser.
#
# For each requested starting pair_index and a chosen channel, this script:
#
#   • Builds an autoregressive roll-out of up to N steps:
#         x_0  → ŷ_1 → ŷ_2 → … → ŷ_N
#     where x_0 is the normalised "input" of the starting sample, and at each
#     step the model output is fed back as the next model input.
#
#   • Uses ground-truth targets from consecutive pair_index values:
#         pair_index = p, p+1, ..., p+N-1
#     assuming each dataset item (p) encodes x_p → y_{p+1}.
#
#   • Keeps the conditioning vector cond fixed to its initial value
#     (thermal gradient and time step assumed constant).
#
#   • For every step, computes the same metrics as pf_hqviz:
#         - mae_ch_norm, phase_acorr_rel, curvature_rel, ligament_align_rel
#         - relL2_phase, relL2_conc, relACorr_phase
#     for both:
#         (i)  copy-baseline: x_baseline vs y_true
#         (ii) model:        ŷ vs y_true
#
#   • Writes a CSV table with per-step metrics:
#         autoregressive_metrics.csv
#
#   • Writes high-quality PNGs per step:
#         - Combined 5-panel figure: [x_baseline | y_true | ŷ | (x − y) | (y − ŷ)]
#         - Single-panel figures for x, y, ŷ, and the two differences
#           (x − y and y − ŷ share an identical symmetric colour scale).
#
#   • Additionally, produces one plot per metric showing metric value versus
#     autoregressive step, for all starting pair_index values and both
#     comparisons (copy-baseline and model):
#         metric_vs_step_<metric>.png
#
# All field panels are plotted in physical units, using channel-wise z-score
# denormalisation from normalization_config.json for the image channels.

import os
import argparse
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
import csv

import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- dynamic import helpers ----------


def _import_module_from_path(py_path: str, name: str = "extmod"):
    p = Path(py_path).resolve()
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------- repo import helper (models, datasets) ----------


def _load_symbol(py_path: str, symbol: str):
    p = Path(py_path).resolve()

    def _guess_root(q: Path):
        for a in [q.parent, *q.parents]:
            if (a / "models").is_dir() or (a / "scripts").is_dir():
                return a
        return p.parent

    root = _guess_root(p)
    if str(root) not in os.sys.path:
        os.sys.path.insert(0, str(root))
    try:
        rel = p.relative_to(root).with_suffix("")
        mod_name = ".".join(rel.parts)
        mod = __import__(mod_name, fromlist=["*"])
    except Exception:
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, symbol):
        raise AttributeError(f"Symbol '{symbol}' not found in {p}")
    return getattr(mod, symbol)


# ---------- dataset utilities ----------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_pair_index(ds) -> Dict[int, int]:
    """
    Build a mapping from pair_index → dataset index.

    Prefers ds.items if present, falling back to __getitem__ if necessary.
    """
    mapping: Dict[int, int] = {}

    if hasattr(ds, "items"):
        for idx, it in enumerate(ds.items):
            k = None
            if isinstance(it, tuple) and len(it) >= 2:
                k = int(it[1])
            elif isinstance(it, dict) and "pair_index" in it:
                k = int(it["pair_index"])
            if k is not None and k not in mapping:
                mapping[k] = idx
        return mapping

    try:
        n = len(ds)  # type: ignore[arg-type]
    except Exception:
        return mapping

    for idx in range(n):
        try:
            sample = ds[idx]
        except Exception:
            continue
        if isinstance(sample, dict) and "pair_index" in sample:
            k = int(sample["pair_index"])
            if k not in mapping:
                mapping[k] = idx
    return mapping


def _to_tensor_4d(field: Any, device: torch.device) -> torch.Tensor:
    """
    Convert a field to a torch.Tensor of shape [1, C, H, W] on the given device.
    """
    if isinstance(field, torch.Tensor):
        t = field.to(device=device, dtype=torch.float32)
    else:
        arr = np.asarray(field, dtype=np.float32)
        t = torch.from_numpy(arr).to(device=device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def _prepare_cond_tensor(cond_val: Any, device: torch.device) -> Optional[torch.Tensor]:
    """
    Prepare conditioning tensor on device with shape [1, cond_dim] if present.
    """
    if cond_val is None:
        return None
    if isinstance(cond_val, torch.Tensor):
        c = cond_val.to(device=device, dtype=torch.float32)
    else:
        c = torch.from_numpy(np.asarray(cond_val, dtype=np.float32)).to(device=device)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    return c


# ---------- normalisation utilities ----------

NORM_CFG_PATH = Path(
    "/scratch/project_2008261/rapid_solidification/data/"
    "rapid_solidification/normalization_config.json"
)


def _load_normalisation_config(path: Path = NORM_CFG_PATH) -> Dict[str, Any]:
    """Load normalisation configuration from JSON."""
    with path.open("r") as f:
        cfg = json.load(f)
    return cfg


def _denormalise_channel_zscore(
    x_ch_norm: np.ndarray,
    mean_ch: float,
    std_ch: float,
) -> np.ndarray:
    """
    Inverse of z-score normalisation for a single channel:

        x_norm = (x_real - mean) / std
        => x_real = x_norm * std + mean

    Args:
        x_ch_norm: Normalised channel [H, W].
        mean_ch:   Channel mean in physical units.
        std_ch:    Channel standard deviation in physical units.

    Returns:
        Channel in physical units [H, W].
    """
    return x_ch_norm * std_ch + mean_ch


def _relative_l2_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 error ||a - b||_2 / ||b||_2 for a single 2D field.

    Computed on normalised fields; the ratio is invariant to a global scale.
    """
    diff = a - b
    num = float(np.sqrt(np.sum(diff * diff)))
    den = float(np.sqrt(np.sum(b * b)))
    if not np.isfinite(den) or den < eps:
        if np.isfinite(num) and num < eps:
            return 0.0
        return float("nan")
    return num / den


# ---------- plotting: fields and differences ----------


def _save_hq_sample_png(
    out_path: Path,
    x_ch_real: np.ndarray,           # [H,W], baseline previous state (physical units)
    y_ch_real: np.ndarray,           # [H,W], GT next step (physical units)
    pred_ch_real: np.ndarray,        # [H,W], prediction (physical units)
    diff_copy_real: np.ndarray,      # [H,W], (x − y) in physical units
    gid: Any,
    start_pair: int,
    pair_index: int,
    step_index: int,
    channel: int,
    figsize: Tuple[float, float],
    dpi: int,
    interpolation: str = "nearest",
    optimize_png: bool = False,
):
    """
    Create a single-row, five-panel high-quality figure in physical units:
        [ x | y | ŷ | (x − y) | (y − ŷ) ].

    Panels:
        1) x:           baseline state at time t.
        2) y:           ground-truth at time t+1.
        3) ŷ:           model prediction at t+1.
        4) x − y:       signed copy-baseline difference.
        5) y − ŷ:       signed model difference.

    All three field panels (x, y, ŷ) use a shared colour scale.
    The two difference panels (x − y and y − ŷ) share a single symmetric
    scale based on the maximum absolute difference across both.
    """

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=5)
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_diff_copy = fig.add_subplot(gs[0, 3])
    ax_diff_model = fig.add_subplot(gs[0, 4])

    for ax in (ax_x, ax_y, ax_pred, ax_diff_copy, ax_diff_model):
        ax.set_aspect("equal")

    vmin_val = float(
        min(
            x_ch_real.min(),
            y_ch_real.min(),
            pred_ch_real.min(),
        )
    )
    vmax_val = float(
        max(
            x_ch_real.max(),
            y_ch_real.max(),
            pred_ch_real.max(),
        )
    )
    if not np.isfinite(vmin_val):
        vmin_val = 0.0
    if not np.isfinite(vmax_val) or vmax_val <= vmin_val:
        vmax_val = vmin_val + 1.0

    diff_model_real = y_ch_real - pred_ch_real

    diffs_finite: List[np.ndarray] = []
    dc_finite = diff_copy_real[np.isfinite(diff_copy_real)]
    if dc_finite.size > 0:
        diffs_finite.append(dc_finite)
    dm_finite = diff_model_real[np.isfinite(diff_model_real)]
    if dm_finite.size > 0:
        diffs_finite.append(dm_finite)

    if diffs_finite:
        all_diffs = np.concatenate([d.ravel() for d in diffs_finite])
        max_abs_diff = float(np.max(np.abs(all_diffs)))
    else:
        max_abs_diff = 1.0

    if not np.isfinite(max_abs_diff) or max_abs_diff <= 0.0:
        max_abs_diff = 1.0

    im_x = ax_x.imshow(
        x_ch_real,
        vmin=vmin_val,
        vmax=vmax_val,
        interpolation=interpolation,
    )
    ax_x.set_title("x (baseline at t)")
    ax_x.set_xticks([])
    ax_x.set_yticks([])
    fig.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.02)

    im_y = ax_y.imshow(
        y_ch_real,
        vmin=vmin_val,
        vmax=vmax_val,
        interpolation=interpolation,
    )
    ax_y.set_title("y (GT at t+1)")
    ax_y.set_xticks([])
    ax_y.set_yticks([])
    fig.colorbar(im_y, ax=ax_y, fraction=0.046, pad=0.02)

    im_pred = ax_pred.imshow(
        pred_ch_real,
        vmin=vmin_val,
        vmax=vmax_val,
        interpolation=interpolation,
    )
    ax_pred.set_title("ŷ (prediction at t+1)")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.02)

    im_diff_copy = ax_diff_copy.imshow(
        diff_copy_real,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        interpolation=interpolation,
    )
    ax_diff_copy.set_title("x − y (Δ copy baseline)")
    ax_diff_copy.set_xticks([])
    ax_diff_copy.set_yticks([])
    fig.colorbar(im_diff_copy, ax=ax_diff_copy, fraction=0.046, pad=0.02)

    im_diff_model = ax_diff_model.imshow(
        diff_model_real,
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        interpolation=interpolation,
    )
    ax_diff_model.set_title("y − ŷ (Δ model)")
    ax_diff_model.set_xticks([])
    ax_diff_model.set_yticks([])
    fig.colorbar(im_diff_model, ax=ax_diff_model, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"gid={gid}, start_pair={start_pair}, pair_index={pair_index}, "
        f"step={step_index}, ch={channel}",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if optimize_png:
        try:
            save_kwargs["pil_kwargs"] = {"optimize": True}
        except TypeError:
            pass

    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def _save_single_panel_png(
    out_path: Path,
    field_real: np.ndarray,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    interpolation: str = "nearest",
    optimize_png: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Save a single-panel high-quality image with colour bar.
    Optional vmin/vmax enable a shared colour scale across images.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")

    im = ax.imshow(
        field_real,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()

    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if optimize_png:
        try:
            save_kwargs["pil_kwargs"] = {"optimize": True}
        except TypeError:
            pass

    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


# ---------- plotting: metrics vs autoregressive step ----------


def _plot_metric_evolution(
    metrics_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    For each metric present in metrics_rows, create a plot of metric value
    versus autoregressive step, with separate curves for each
    (start_pair, comparison) combination.
    """
    if not metrics_rows:
        return

    base_cols = ["gid", "start_pair", "step", "pair_index", "channel", "comparison"]

    metric_keys: Set[str] = set()
    for row in metrics_rows:
        for k in row.keys():
            if k not in base_cols:
                metric_keys.add(k)

    for metric in sorted(metric_keys):
        series: Dict[Tuple[int, str], List[Tuple[int, float]]] = {}

        for row in metrics_rows:
            if metric not in row:
                continue
            v = row[metric]
            if v is None:
                continue
            try:
                v_float = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v_float):
                continue

            step = int(row["step"])
            start_pair = int(row["start_pair"])
            comparison = str(row["comparison"])
            key = (start_pair, comparison)
            series.setdefault(key, []).append((step, v_float))

        if not series:
            continue

        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)
        for (start_pair, comparison), pts in sorted(series.items()):
            pts_sorted = sorted(pts, key=lambda t: t[0])
            steps = [p[0] for p in pts_sorted]
            values = [p[1] for p in pts_sorted]
            label = f"start={start_pair}, {comparison}"
            ax.plot(
                steps,
                values,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=3,
                label=label,
            )

        ax.set_xlabel("Autoregressive step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs step")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()

        fig.savefig(out_dir / f"metric_vs_step_{metric}.png", bbox_inches="tight")
        plt.close(fig)


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to training YAML or config_snapshot.yaml")
    ap.add_argument("-k", "--ckpt", required=True, help="Path to checkpoint .pth")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    ap.add_argument(
        "--start-pairs",
        type=int,
        nargs="+",
        required=True,
        help="List of starting pair_index values for autoregressive roll-outs",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Maximum number of autoregressive steps to predict from each starting pair_index",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to visualise (e.g. 0 → Field_0, 1 → Field_1)",
    )

    ap.add_argument("--figwidth", type=float, default=16.0, help="Figure width in inches")
    ap.add_argument("--figheight", type=float, default=4.0, help="Figure height in inches")
    ap.add_argument("--dpi", type=int, default=600, help="Base DPI for PNG export (higher → sharper images)")
    ap.add_argument(
        "--interp",
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation mode passed to imshow",
    )
    ap.add_argument(
        "--optimize-png",
        action="store_true",
        help="Enable Pillow optimisation for PNG output (may reduce file size)",
    )

    args = ap.parse_args()

    out_dir = _ensure_dir(Path(args.outdir))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_grad_enabled(False)

    figsize: Tuple[float, float] = (float(args.figwidth), float(args.figheight))
    dpi: int = int(args.dpi)
    interpolation: str = str(args.interp)
    optimize_png: bool = bool(args.optimize_png)

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    channel: int = int(args.channel)
    start_pairs: List[int] = [int(p) for p in args.start_pairs]
    max_steps: int = int(args.steps)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    LOSS_PATH = "/scratch/project_2008261/rapid_solidification/models/train/core/loss_functions.py"
    ops = _import_module_from_path(LOSS_PATH, name="loss_functions_ext")

    metric_specs: List[Tuple[str, str]] = [
        ("mae_ch_norm", "mae"),
        ("phase_acorr_rel", "phase_error_metric"),
        ("curvature_rel", "curvature_error_metric"),
        ("ligament_align_rel", "ligament_alignment_metric"),
    ]
    metric_fns: Dict[str, Any] = {}
    for label, attr in metric_specs:
        fn = getattr(ops, attr, None)
        if fn is not None:
            metric_fns[label] = fn

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    norm_cfg = _load_normalisation_config()
    channel_means = norm_cfg["channel_mean"]
    channel_stds = norm_cfg["channel_std"]

    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})
    test_path = (
        cfg["paths"]["h5"]["test"]
        if isinstance(cfg["paths"]["h5"]["test"], str)
        else cfg["paths"]["h5"]["test"]["path"]
    )
    ds = DSClass(test_path, **ds_args)

    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    assert cond_enabled and cond_dim == 2, "Expected field-based 2D conditioning vector"

    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16

    if channel < 0 or channel >= len(channel_means):
        raise ValueError(
            f"Requested channel {channel} has no normalisation stats "
            f"(available: 0..{len(channel_means) - 1})"
        )
    mean_ch = float(channel_means[channel])
    std_ch = float(channel_stds[channel])

    pair_to_idx = _build_pair_index(ds)

    metrics_rows: List[Dict[str, Any]] = []
    seq_summaries: List[Dict[str, Any]] = []

    with torch.inference_mode(), autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        for start_pair in start_pairs:
            if start_pair not in pair_to_idx:
                continue

            seq_samples: List[Dict[str, Any]] = []
            seq_pairs: List[int] = []
            for s in range(max_steps):
                p = start_pair + s
                idx = pair_to_idx.get(p, None)
                if idx is None:
                    break
                sample = ds[idx]
                if not isinstance(sample, dict):
                    break
                seq_samples.append(sample)
                seq_pairs.append(p)

            if not seq_samples:
                continue

            cond0_val = seq_samples[0].get("cond", None)
            cond_t = _prepare_cond_tensor(cond0_val, device=device)

            gid = seq_samples[0].get("gid", "unknown")

            x0_field = seq_samples[0]["input"]
            x_curr = _to_tensor_4d(x0_field, device=device)

            _, C_total, _, _ = x_curr.shape
            if channel < 0 or channel >= C_total:
                raise ValueError(f"Requested channel {channel} is out of range for tensor with C={C_total}")

            seq_dir = _ensure_dir(out_dir / f"start_{start_pair}_ch_{channel}")

            prev_true_target_t: Optional[torch.Tensor] = None
            steps_realised = 0

            for step_idx, (sample_s, pair_s) in enumerate(zip(seq_samples, seq_pairs), start=1):
                y_true_t = _to_tensor_4d(sample_s["target"], device=device)

                if step_idx == 1:
                    x_baseline = x_curr.clone()
                else:
                    assert prev_true_target_t is not None
                    x_baseline = prev_true_target_t.clone()

                if cond_t is not None:
                    pred_t = model(x_curr, cond_t)
                else:
                    pred_t = model(x_curr)

                prev_true_target_t = y_true_t
                x_curr = pred_t

                x_baseline_np = x_baseline[0].detach().cpu().numpy().astype(np.float32)
                y_true_np = y_true_t[0].detach().cpu().numpy().astype(np.float32)
                pred_np = pred_t[0].detach().cpu().numpy().astype(np.float32)

                x_ch_norm = x_baseline_np[channel]
                y_ch_norm = y_true_np[channel]
                pred_ch_norm = pred_np[channel]

                metrics_baseline: Dict[str, float] = {}
                metrics_model: Dict[str, float] = {}

                if metric_fns:
                    def _compute_metrics(a_norm: np.ndarray, b_norm: np.ndarray) -> Dict[str, float]:
                        a_t = torch.from_numpy(a_norm[None, None, :, :]).to(
                            device=device, dtype=torch.float32
                        )
                        b_t = torch.from_numpy(b_norm[None, None, :, :]).to(
                            device=device, dtype=torch.float32
                        )
                        out: Dict[str, float] = {}
                        with torch.inference_mode():
                            for label, fn in metric_fns.items():
                                try:
                                    val_t = fn(a_t, b_t)
                                    if hasattr(val_t, "detach"):
                                        val = float(val_t.detach().cpu().item())
                                    else:
                                        val = float(val_t)
                                    out[label] = val
                                except Exception:
                                    continue
                        return out

                    metrics_baseline.update(_compute_metrics(x_ch_norm, y_ch_norm))
                    metrics_model.update(_compute_metrics(pred_ch_norm, y_ch_norm))
                else:
                    mae_xy = float(np.mean(np.abs(x_ch_norm - y_ch_norm)))
                    mae_predy = float(np.mean(np.abs(pred_ch_norm - y_ch_norm)))
                    metrics_baseline["mae_ch_norm"] = mae_xy
                    metrics_model["mae_ch_norm"] = mae_predy

                phase_idx = 0
                if phase_idx < C_total:
                    phase_x = x_baseline_np[phase_idx]
                    phase_y = y_true_np[phase_idx]
                    phase_pred = pred_np[phase_idx]

                    metrics_baseline["relL2_phase"] = _relative_l2_np(phase_x, phase_y)
                    metrics_model["relL2_phase"] = _relative_l2_np(phase_pred, phase_y)

                    if hasattr(ops, "autocorr_rel_error"):
                        phase_x_t = torch.from_numpy(phase_x[None, :, :]).to(
                            device=device, dtype=torch.float32
                        )
                        phase_y_t = torch.from_numpy(phase_y[None, :, :]).to(
                            device=device, dtype=torch.float32
                        )
                        phase_pred_t = torch.from_numpy(phase_pred[None, :, :]).to(
                            device=device, dtype=torch.float32
                        )
                        with torch.inference_mode():
                            try:
                                e_ac_base = ops.autocorr_rel_error(phase_x_t, phase_y_t)
                                e_ac_model = ops.autocorr_rel_error(phase_pred_t, phase_y_t)
                                metrics_baseline["relACorr_phase"] = float(
                                    e_ac_base.mean().detach().cpu().item()
                                )
                                metrics_model["relACorr_phase"] = float(
                                    e_ac_model.mean().detach().cpu().item()
                                )
                            except Exception:
                                pass

                conc_idx = 1
                if conc_idx < C_total:
                    conc_x = x_baseline_np[conc_idx]
                    conc_y = y_true_np[conc_idx]
                    conc_pred = pred_np[conc_idx]

                    metrics_baseline["relL2_conc"] = _relative_l2_np(conc_x, conc_y)
                    metrics_model["relL2_conc"] = _relative_l2_np(conc_pred, conc_y)

                row_base: Dict[str, Any] = {
                    "gid": gid,
                    "start_pair": start_pair,
                    "step": step_idx,
                    "pair_index": pair_s,
                    "channel": channel,
                    "comparison": "x_vs_y",
                }
                for k, v in metrics_baseline.items():
                    row_base[k] = v
                metrics_rows.append(row_base)

                row_model: Dict[str, Any] = {
                    "gid": gid,
                    "start_pair": start_pair,
                    "step": step_idx,
                    "pair_index": pair_s,
                    "channel": channel,
                    "comparison": "yhat_vs_y",
                }
                for k, v in metrics_model.items():
                    row_model[k] = v
                metrics_rows.append(row_model)

                x_ch_real = _denormalise_channel_zscore(x_ch_norm, mean_ch, std_ch)
                y_ch_real = _denormalise_channel_zscore(y_ch_norm, mean_ch, std_ch)
                pred_ch_real = _denormalise_channel_zscore(pred_ch_norm, mean_ch, std_ch)

                diff_copy_real = (x_ch_real - y_ch_real).astype(np.float32)
                diff_model_real = (y_ch_real - pred_ch_real).astype(np.float32)

                diffs_finite: List[np.ndarray] = []
                dc_finite = diff_copy_real[np.isfinite(diff_copy_real)]
                if dc_finite.size > 0:
                    diffs_finite.append(dc_finite)
                dm_finite = diff_model_real[np.isfinite(diff_model_real)]
                if dm_finite.size > 0:
                    diffs_finite.append(dm_finite)

                if diffs_finite:
                    all_diffs = np.concatenate([d.ravel() for d in diffs_finite])
                    max_abs_diff = float(np.max(np.abs(all_diffs)))
                else:
                    max_abs_diff = 1.0

                if not np.isfinite(max_abs_diff) or max_abs_diff <= 0.0:
                    max_abs_diff = 1.0

                out_png = seq_dir / f"hq_step_{step_idx:03d}_pair_{pair_s}_gid_{gid}.png"
                _save_hq_sample_png(
                    out_path=out_png,
                    x_ch_real=x_ch_real.astype(np.float32),
                    y_ch_real=y_ch_real.astype(np.float32),
                    pred_ch_real=pred_ch_real.astype(np.float32),
                    diff_copy_real=diff_copy_real,
                    gid=gid,
                    start_pair=start_pair,
                    pair_index=pair_s,
                    step_index=step_idx,
                    channel=channel,
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                )

                step_dir = _ensure_dir(
                    seq_dir / f"step_{step_idx:03d}_pair_{pair_s}"
                )

                _save_single_panel_png(
                    out_path=step_dir / "x_t_baseline.png",
                    field_real=x_ch_real.astype(np.float32),
                    title="x (baseline at t)",
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                )
                _save_single_panel_png(
                    out_path=step_dir / "y_tplus1.png",
                    field_real=y_ch_real.astype(np.float32),
                    title="y (GT at t+1)",
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                )
                _save_single_panel_png(
                    out_path=step_dir / "y_pred_tplus1.png",
                    field_real=pred_ch_real.astype(np.float32),
                    title="ŷ (prediction at t+1)",
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                )
                _save_single_panel_png(
                    out_path=step_dir / "diff_copy_x_minus_y.png",
                    field_real=diff_copy_real,
                    title="x − y (Δ copy baseline)",
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                    vmin=-max_abs_diff,
                    vmax=max_abs_diff,
                )
                _save_single_panel_png(
                    out_path=step_dir / "diff_model_y_minus_yhat.png",
                    field_real=diff_model_real,
                    title="y − ŷ (Δ model)",
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                    vmin=-max_abs_diff,
                    vmax=max_abs_diff,
                )

                steps_realised += 1

            seq_summaries.append(
                {
                    "gid": str(gid),
                    "start_pair": int(start_pair),
                    "steps_realised": int(steps_realised),
                    "requested_steps": int(max_steps),
                    "sequence_dir": str(seq_dir),
                }
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if metrics_rows:
        base_cols = ["gid", "start_pair", "step", "pair_index", "channel", "comparison"]
        metric_keys: Set[str] = set()
        for row in metrics_rows:
            for k in row.keys():
                if k not in base_cols:
                    metric_keys.add(k)
        fieldnames = base_cols + sorted(metric_keys)
        metrics_csv_path = out_dir / "autoregressive_metrics.csv"
        with metrics_csv_path.open("w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)

        _plot_metric_evolution(metrics_rows, out_dir)

    summary = {
        "start_pairs_requested": sorted(list(set(start_pairs))),
        "steps_requested": int(max_steps),
        "sequences": seq_summaries,
        "device": str(device),
        "figsize": {"width_in": figsize[0], "height_in": figsize[1]},
        "dpi": dpi,
        "interpolation": interpolation,
        "optimize_png": optimize_png,
        "channel": channel,
        "normalisation_config": str(NORM_CFG_PATH),
        "metrics_available": sorted(list(metric_fns.keys())),
        "metrics_table": "autoregressive_metrics.csv" if metrics_rows else None,
    }
    (out_dir / "autoregressive_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
