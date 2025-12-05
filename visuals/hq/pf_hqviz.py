#!/usr/bin/env python3
# pf_hqviz.py — High-quality visualiser for selected pair_index values.
#
# For each requested pair_index and a chosen channel, produces one PNG:
#   [ x (input last step) | y (GT next step) | ŷ (prediction) | (x − y) | (y − ŷ) ]
#
# Additionally:
#   • Writes a metrics table (CSV) with x–y and ŷ–y metrics.
#   • Creates per-sample subfolders with single-panel high-quality PNGs
#     for x(t), y(t+1), ŷ(t+1), and now also:
#         - diff_copy_x_minus_y.png   (x − y)
#         - diff_model_y_minus_yhat.png (y − ŷ)
#     The two difference images share an identical symmetric colour scale
#     so that differences are directly comparable.
#
# Metrics included in the CSV (dimensionless relative unless otherwise stated):
#   • mae_ch_norm          (absolute MAE on the visualised channel, normalised units)
#   • phase_acorr_rel      (relative phase-structure error via auto-correlation)
#   • curvature_rel        (relative curvature error)
#   • ligament_align_rel   (relative ligament alignment error)
#   • relL2_phase          (relative L2 error of phase field, channel 0)
#   • relL2_conc           (relative L2 error of concentration field, channel 1)
#   • relACorr_phase       (relative auto-correlation error of phase field)
#
# The script assumes that:
#   • "input" in the dataset corresponds to the last available state x.
#   • "target" corresponds to the next-step ground truth y.
#   • Model takes (input, cond) as in the training setup.
#
# All panels are plotted in physical units, using channel-wise z-score
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
from torch.utils.data import DataLoader, Subset
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


# ---------- dataloader utilities ----------


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("input", "target", "cond"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "gid" in batch[0]:
        out["gid"] = [b["gid"] for b in batch]
    if "pair_index" in batch[0]:
        out["pair_index"] = [int(b["pair_index"]) for b in batch]
    return out


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _subset_by_pairs(ds, wanted: Set[int]):
    """Return a Subset restricted to wanted pair_index values, if accessible."""
    try:
        if hasattr(ds, "items"):
            idxs = []
            for i, it in enumerate(ds.items):
                k = None
                if isinstance(it, tuple) and len(it) >= 2:
                    k = int(it[1])
                elif isinstance(it, dict) and "pair_index" in it:
                    k = int(it["pair_index"])
                if k is not None and k in wanted:
                    idxs.append(i)
            if idxs:
                return Subset(ds, idxs)
    except Exception:
        pass
    return ds


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
            # Both fields essentially zero.
            return 0.0
        # Target nearly zero but prediction not; mark as undefined.
        return float("nan")
    return num / den


# ---------- plotting ----------


def _save_hq_sample_png(
    out_path: Path,
    x_ch_real: np.ndarray,           # [H,W], input last step (physical units)
    y_ch_real: np.ndarray,           # [H,W], GT next step (physical units)
    pred_ch_real: np.ndarray,        # [H,W], prediction (physical units)
    diff_copy_real: np.ndarray,      # [H,W], (x − y) in physical units
    gid: Any,
    pair_index: int,
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
        1) x:           input at time t.
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

    # Shared colour scale for x, y, ŷ
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

    # Signed model difference in physical units: y − ŷ
    diff_model_real = y_ch_real - pred_ch_real

    # Shared symmetric scale for both difference panels
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

    # Panel 1: x_t
    im_x = ax_x.imshow(
        x_ch_real,
        vmin=vmin_val,
        vmax=vmax_val,
        interpolation=interpolation,
    )
    ax_x.set_title("x (input at t)")
    ax_x.set_xticks([])
    ax_x.set_yticks([])
    fig.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.02)

    # Panel 2: y_{t+1} true
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

    # Panel 3: ŷ
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

    # Panel 4: copy-baseline difference x − y
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

    # Panel 5: model difference y − ŷ
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
        f"gid={gid}, pair_index={pair_index}, ch={channel}",
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


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to training YAML or config_snapshot.yaml")
    ap.add_argument("-k", "--ckpt", required=True, help="Path to checkpoint .pth")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[200, 207, 217, 350],
        help="List of pair_index values to visualise",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to visualise (e.g. 0 → Field_0, 1 → Field_1)",
    )

    # Figure and image export controls.
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

    # Increase global Matplotlib DPI settings for sharper output.
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi

    channel: int = int(args.channel)
    wanted_pairs: Set[int] = {int(p) for p in args.pairs}

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # Shared operators and metrics from loss_functions.py
    LOSS_PATH = "/scratch/project_2008261/rapid_solidification/training/core/loss_functions.py"
    ops = _import_module_from_path(LOSS_PATH, name="loss_functions_ext")

    # Metric functions present in loss_functions.py.
    # Absolute vs relative:
    #   - mae_ch_norm:        absolute (normalised units)
    #   - phase_acorr_rel:    relative (dimensionless)
    #   - curvature_rel:      relative (dimensionless)
    #   - ligament_align_rel: relative (dimensionless)
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

    # Load image normalisation configuration for denormalisation.
    norm_cfg = _load_normalisation_config()
    channel_means = norm_cfg["channel_mean"]
    channel_stds = norm_cfg["channel_std"]

    # Dataset
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})
    test_path = (
        cfg["paths"]["h5"]["test"]
        if isinstance(cfg["paths"]["h5"]["test"], str)
        else cfg["paths"]["h5"]["test"]["path"]
    )
    base_ds = DSClass(test_path, **ds_args)
    test_ds = _subset_by_pairs(base_ds, wanted_pairs)

    # Model
    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=_collate,
    )

    # Conditioning configuration (as in training).
    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    assert cond_enabled and cond_dim == 2, "Expected field-based 2D conditioning vector"

    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16  # V100 friendly

    remaining = set(wanted_pairs)
    figures_saved = 0

    # Channel-wise denormalisation parameters for the selected channel.
    if channel < 0 or channel >= len(channel_means):
        raise ValueError(
            f"Requested channel {channel} has no normalisation stats "
            f"(available: 0..{len(channel_means) - 1})"
        )
    mean_ch = float(channel_means[channel])
    std_ch = float(channel_stds[channel])

    # Accumulate metrics for CSV table.
    metrics_rows: List[Dict[str, Any]] = []

    for batch in dl:
        if "pair_index" not in batch:
            continue

        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True) if "cond" in batch else None

        with torch.inference_mode(), autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            pred = model(x, cond) if cond is not None else model(x)

        B, C, H, W = y.shape
        if channel < 0 or channel >= C:
            raise ValueError(f"Requested channel {channel} is out of range for tensor with C={C}")

        for i in range(B):
            k_idx = int(batch["pair_index"][i])
            if k_idx not in remaining:
                continue
            gid = batch.get("gid", ["unknown"] * B)[i]

            # Normalised tensors → numpy for metrics and visualisation
            x_i_norm = x[i].detach().cpu().numpy().astype(np.float32)       # [C,H,W]
            y_i_norm = y[i].detach().cpu().numpy().astype(np.float32)       # [C,H,W]
            pred_i_norm = pred[i].detach().cpu().numpy().astype(np.float32) # [C,H,W]

            x_ch_norm = x_i_norm[channel]
            y_ch_norm = y_i_norm[channel]
            pred_ch_norm = pred_i_norm[channel]

            # ------------------------------------------------------------------
            # Metrics for tables:
            #   - Per-channel metrics on the visualised channel:
            #       mae_ch_norm           (absolute)
            #       phase_acorr_rel       (relative)
            #       curvature_rel         (relative)
            #       ligament_align_rel    (relative)
            #   - Field-wise metrics using phase and concentration channels:
            #       relL2_phase           (relative L2, phase, channel 0)
            #       relL2_conc            (relative L2, conc, channel 1)
            #       relACorr_phase        (relative auto-correlation error, phase)
            # All relative metrics are dimensionless fractions (0.03 = 3%).
            # ------------------------------------------------------------------

            metrics_baseline: Dict[str, float] = {}
            metrics_model: Dict[str, float] = {}

            # Channel-wise metrics on the currently visualised channel.
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
                                # Metric with incompatible signature is skipped.
                                continue
                    return out

                metrics_baseline.update(_compute_metrics(x_ch_norm, y_ch_norm))
                metrics_model.update(_compute_metrics(pred_ch_norm, y_ch_norm))
            else:
                # Fallback: at least provide MAE on the visualised channel.
                mae_xy = float(np.mean(np.abs(x_ch_norm - y_ch_norm)))
                mae_predy = float(np.mean(np.abs(pred_ch_norm - y_ch_norm)))
                metrics_baseline["mae_ch_norm"] = mae_xy
                metrics_model["mae_ch_norm"] = mae_predy

            # ------------------------------------------------------------------
            # Relative L2 errors and phase auto-correlation errors for fields.
            # Phase field assumed in channel 0; concentration in channel 1.
            # ------------------------------------------------------------------

            C_total = x_i_norm.shape[0]

            # Phase field: channel 0
            phase_idx = 0
            if phase_idx < C_total:
                phase_x = x_i_norm[phase_idx]
                phase_y = y_i_norm[phase_idx]
                phase_pred = pred_i_norm[phase_idx]

                metrics_baseline["relL2_phase"] = _relative_l2_np(phase_x, phase_y)
                metrics_model["relL2_phase"] = _relative_l2_np(phase_pred, phase_y)

                # Relative auto-correlation error on the phase field.
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
                            # If auto-correlation fails, leave unset.
                            pass

            # Concentration field: channel 1, if present.
            conc_idx = 1
            if conc_idx < C_total:
                conc_x = x_i_norm[conc_idx]
                conc_y = y_i_norm[conc_idx]
                conc_pred = pred_i_norm[conc_idx]

                metrics_baseline["relL2_conc"] = _relative_l2_np(conc_x, conc_y)
                metrics_model["relL2_conc"] = _relative_l2_np(conc_pred, conc_y)

            # Add rows to metrics table: x vs y and ŷ vs y.
            row_base: Dict[str, Any] = {
                "gid": gid,
                "pair_index": k_idx,
                "channel": channel,
                "comparison": "x_vs_y",
            }
            for k, v in metrics_baseline.items():
                row_base[k] = v
            metrics_rows.append(row_base)

            row_model: Dict[str, Any] = {
                "gid": gid,
                "pair_index": k_idx,
                "channel": channel,
                "comparison": "yhat_vs_y",
            }
            for k, v in metrics_model.items():
                row_model[k] = v
            metrics_rows.append(row_model)

            # Denormalise to physical units for plotting.
            x_ch_real = _denormalise_channel_zscore(x_ch_norm, mean_ch, std_ch)
            y_ch_real = _denormalise_channel_zscore(y_ch_norm, mean_ch, std_ch)
            pred_ch_real = _denormalise_channel_zscore(pred_ch_norm, mean_ch, std_ch)

            # Differences in physical units.
            diff_copy_real = (x_ch_real - y_ch_real).astype(np.float32)      # x − y
            diff_model_real = (y_ch_real - pred_ch_real).astype(np.float32)  # y − ŷ

            # Shared symmetric colour scale for difference images (x − y and y − ŷ).
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

            # Combined 5-panel HQ figure.
            out_png = out_dir / f"hq_gid_{gid}_k_{k_idx}_ch_{channel}.png"
            _save_hq_sample_png(
                out_path=out_png,
                x_ch_real=x_ch_real.astype(np.float32),
                y_ch_real=y_ch_real.astype(np.float32),
                pred_ch_real=pred_ch_real.astype(np.float32),
                diff_copy_real=diff_copy_real,
                gid=gid,
                pair_index=k_idx,
                channel=channel,
                figsize=figsize,
                dpi=dpi,
                interpolation=interpolation,
                optimize_png=optimize_png,
            )

            # Additional single-panel HQ plots in per-sample subfolder.
            sample_dir = _ensure_dir(
                out_dir / f"gid_{gid}_k_{k_idx}_ch_{channel}"
            )

            # Fields x, y, ŷ with automatic scaling.
            _save_single_panel_png(
                out_path=sample_dir / "x_t.png",
                field_real=x_ch_real.astype(np.float32),
                title="x (input at t)",
                figsize=figsize,
                dpi=dpi,
                interpolation=interpolation,
                optimize_png=optimize_png,
            )
            _save_single_panel_png(
                out_path=sample_dir / "y_tplus1.png",
                field_real=y_ch_real.astype(np.float32),
                title="y (GT at t+1)",
                figsize=figsize,
                dpi=dpi,
                interpolation=interpolation,
                optimize_png=optimize_png,
            )
            _save_single_panel_png(
                out_path=sample_dir / "y_pred_tplus1.png",
                field_real=pred_ch_real.astype(np.float32),
                title="ŷ (prediction at t+1)",
                figsize=figsize,
                dpi=dpi,
                interpolation=interpolation,
                optimize_png=optimize_png,
            )

            # New: high-quality difference images with shared colour scale.
            _save_single_panel_png(
                out_path=sample_dir / "diff_copy_x_minus_y.png",
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
                out_path=sample_dir / "diff_model_y_minus_yhat.png",
                field_real=diff_model_real,
                title="y − ŷ (Δ model)",
                figsize=figsize,
                dpi=dpi,
                interpolation=interpolation,
                optimize_png=optimize_png,
                vmin=-max_abs_diff,
                vmax=max_abs_diff,
            )

            remaining.discard(k_idx)
            figures_saved += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not remaining:
            break

    # Write metrics table CSV for x−y and ŷ−y comparisons.
    if metrics_rows:
        base_cols = ["gid", "pair_index", "channel", "comparison"]
        metric_keys: Set[str] = set()
        for row in metrics_rows:
            for k in row.keys():
                if k not in base_cols:
                    metric_keys.add(k)
        fieldnames = base_cols + sorted(metric_keys)
        metrics_csv_path = out_dir / "metrics_table.csv"
        with metrics_csv_path.open("w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)

    # Summary JSON
    summary = {
        "pairs_requested": sorted(list(wanted_pairs)),
        "pairs_missing": sorted(list(remaining)),
        "figures_saved": int(figures_saved),
        "device": str(device),
        "figsize": {"width_in": figsize[0], "height_in": figsize[1]},
        "dpi": dpi,
        "interpolation": interpolation,
        "optimize_png": optimize_png,
        "channel": channel,
        "normalisation_config": str(NORM_CFG_PATH),
        "metrics_available": sorted(list(metric_fns.keys())),
        "metrics_table": "metrics_table.csv" if metrics_rows else None,
    }
    (out_dir / "hqviz_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
