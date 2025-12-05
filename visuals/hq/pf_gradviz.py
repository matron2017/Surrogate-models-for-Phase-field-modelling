#!/usr/bin/env python3
# pf_gradviz.py — Pixel-error and wavelet-weight visualiser for selected pair_index values.
# Produces per-channel PNGs per sample:
#   • Value row: y, ŷ, Δy
#   • Optional wavelet-based importance and weighted error² per channel

import os
import argparse
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

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


# ---------- plotting ----------


def _save_row_png(
    out_path: Path,
    title_left: str,
    title_mid: str,
    title_right: str,
    arr_left: np.ndarray,    # [H,W]
    arr_mid: np.ndarray,     # [H,W]
    arr_diff: np.ndarray,    # [H,W]
    vmin_shared: float,
    vmax_shared: float,
    vabs_diff: float,
    figsize: Tuple[float, float],
    dpi: int,
    interpolation: str = "nearest",
    optimize_png: bool = False,
):
    # Single-row, three-column figure with explicit size and DPI.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=3)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])

    im_l = ax_l.imshow(arr_left, vmin=vmin_shared, vmax=vmax_shared, interpolation=interpolation)
    ax_l.set_title(title_left)
    im_m = ax_m.imshow(arr_mid, vmin=vmin_shared, vmax=vmax_shared, interpolation=interpolation)
    ax_m.set_title(title_mid)
    im_r = ax_r.imshow(arr_diff, vmin=-vabs_diff, vmax=vabs_diff, interpolation=interpolation)
    ax_r.set_title(title_right)

    for ax, im in ((ax_l, im_l), (ax_m, im_m), (ax_r, im_r)):
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if optimize_png:
        try:
            save_kwargs["pil_kwargs"] = {"optimize": True}
        except TypeError:
            pass

    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def _save_wavelet_weight_png(
    out_path: Path,
    weight: np.ndarray,       # [H,W], wavelet importance a
    figsize: Tuple[float, float],
    dpi: int,
    interpolation: str = "nearest",
    optimize_png: bool = False,
):
    # Single-panel figure: wavelet weight a.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    w_vmin = float(np.min(weight))
    w_vmax = float(np.max(weight))
    if not np.isfinite(w_vmin) or not np.isfinite(w_vmax) or w_vmax <= w_vmin:
        w_vmin, w_vmax = 1.0, 1.0

    im = ax.imshow(weight, vmin=w_vmin, vmax=w_vmax, interpolation=interpolation)
    ax.set_title("wavelet weight a")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

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
    ap.add_argument("--batch", type=int, default=1, help="Overrides loader batch size")

    # Figure and image export controls.
    ap.add_argument("--figwidth", type=float, default=12.0, help="Figure width in inches")
    ap.add_argument("--figheight", type=float, default=4.0, help="Figure height in inches")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG export")
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

    # Wavelet-based importance weights and visualisation (per channel).
    ap.add_argument(
        "--wavelet-vis",
        action="store_true",
        help="Enable wavelet-based importance weight visualisations",
    )
    ap.add_argument(
        "--wavelet-theta",
        type=float,
        default=0.8,
        help="Quantile in (0,1) for wavelet importance weighting",
    )
    ap.add_argument(
        "--wavelet-alpha",
        type=float,
        default=1.25,
        help="Minimum importance weight for high-frequency region",
    )
    ap.add_argument(
        "--wavelet-beta",
        type=float,
        default=6.0,
        help="Maximum importance weight for high-frequency region",
    )
    ap.add_argument(
        "--wavelet-J",
        type=int,
        default=1,
        help="Number of DWT levels for wavelet importance",
    )
    ap.add_argument(
        "--wavelet-name",
        type=str,
        default="haar",
        help="Wavelet name for DWT (e.g. 'haar', 'db2')",
    )
    ap.add_argument(
        "--wavelet-mode",
        type=str,
        default="zero",
        choices=["zero", "symmetric", "reflect", "periodization"],
        help="Padding mode for DWTForward inside wavelet_importance_per_channel",
    )

    args = ap.parse_args()

    out_dir = _ensure_dir(Path(args.outdir))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_grad_enabled(False)

    figsize: Tuple[float, float] = (float(args.figwidth), float(args.figheight))
    dpi: int = int(args.dpi)
    interpolation: str = str(args.interp)
    optimize_png: bool = bool(args.optimize_png)

    wavelet_vis: bool = bool(args.wavelet_vis)
    wavelet_theta: float = float(args.wavelet_theta)
    wavelet_alpha: float = float(args.wavelet_alpha)
    wavelet_beta: float = float(args.wavelet_beta)
    wavelet_J: int = int(args.wavelet_J)
    wavelet_name: str = str(args.wavelet_name)
    wavelet_mode: str = str(args.wavelet_mode)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # Shared operators from loss_functions.py
    LOSS_PATH = "/scratch/project_2008261/rapid_solidification/training/core/loss_functions.py"
    ops = _import_module_from_path(LOSS_PATH, name="loss_functions_ext")

    if not hasattr(ops, "rmse"):
        raise AttributeError(f"Missing 'rmse' in {LOSS_PATH}")

    rmse_fn = getattr(ops, "rmse")
    wavelet_importance = getattr(ops, "wavelet_importance_per_channel", None)

    if wavelet_vis and wavelet_importance is None:
        raise AttributeError(
            "wavelet_vis requested but 'wavelet_importance_per_channel' "
            "is not defined in loss_functions."
        )

    # Load config, dataset, model
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})
    test_path = (
        cfg["paths"]["h5"]["test"]
        if isinstance(cfg["paths"]["h5"]["test"], str)
        else cfg["paths"]["h5"]["test"]["path"]
    )
    base_ds = DSClass(test_path, **ds_args)

    # Pair indices of interest
    wanted_pairs = {2, 40, 209, 210, 211, 212}
    test_ds = _subset_by_pairs(base_ds, wanted_pairs)

    ModelClass = _load_symbol(cfg["model"]["file"], cfg["model"]["class"])
    model = ModelClass(**cfg["model"].get("params", {})).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    dl = DataLoader(
        test_ds,
        batch_size=max(1, int(args.batch)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=_collate,
    )

    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = bool(cond_cfg.get("enabled", True))
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    assert cond_enabled and cond_dim == 2, "Expected field-based 2D conditioning vector"

    remaining = set(wanted_pairs)
    saved = 0
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16  # V100 friendly

    for batch in dl:
        if "pair_index" not in batch:
            continue

        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True) if "cond" in batch else None

        with torch.inference_mode(), autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            pred = model(x, cond) if cond is not None else model(x)

        # Wavelet-based importance weights for this batch (per sample, per channel).
        wavelet_weights = None
        if wavelet_vis and wavelet_importance is not None:
            with torch.inference_mode():
                a_w, F_up = wavelet_importance(
                    y,
                    J=wavelet_J,
                    wave=wavelet_name,
                    mode=wavelet_mode,
                    theta=wavelet_theta,
                    alpha=wavelet_alpha,
                    beta_w=wavelet_beta,
                )
            wavelet_weights = a_w  # [B,C,H,W]

        B, C, H, W = y.shape
        for i in range(B):
            k_idx = int(batch["pair_index"][i])
            if k_idx not in remaining:
                continue
            gid = batch.get("gid", ["unknown"] * B)[i]

            # Convert to numpy for plotting
            y_i = y[i].detach().cpu().numpy().astype(np.float32)       # [C,H,W]
            pred_i = pred[i].detach().cpu().numpy().astype(np.float32) # [C,H,W]

            # Per-channel RMSE (values only)
            rmse_val: List[float] = []
            for ch in range(C):
                rmse_val.append(float(rmse_fn(pred[i, ch], y[i, ch]).detach().cpu()))

            # Save rows per channel
            for ch in range(C):
                # --- Value row: y, ŷ, Δy ---
                vmin_val = float(min(y_i[ch].min(), pred_i[ch].min()))
                vmax_val = float(max(y_i[ch].max(), pred_i[ch].max()))
                d_val = pred_i[ch] - y_i[ch]
                vmax_abs_dval = float(np.max(np.abs(d_val)))

                out_val = out_dir / (
                    f"val_gid_{gid}_k_{k_idx}_ch_{ch}"
                    f"_rmse_val_{rmse_val[ch]:.4e}.png"
                )
                _save_row_png(
                    out_path=out_val,
                    title_left="y",
                    title_mid="ŷ",
                    title_right="Δy",
                    arr_left=y_i[ch],
                    arr_mid=pred_i[ch],
                    arr_diff=d_val,
                    vmin_shared=vmin_val,
                    vmax_shared=vmax_val,
                    vabs_diff=vmax_abs_dval,
                    figsize=figsize,
                    dpi=dpi,
                    interpolation=interpolation,
                    optimize_png=optimize_png,
                )

                # Wavelet-based importance weight visualisation (single panel).
                if wavelet_vis and wavelet_weights is not None:
                    weight_ch = wavelet_weights[i, ch].detach().cpu().numpy().astype(np.float32)  # [H,W]

                    out_wave = out_dir / (
                        f"waveletW_gid_{gid}_k_{k_idx}_ch_{ch}"
                        f"_theta_{wavelet_theta:.2f}_alpha_{wavelet_alpha:.2f}_beta_{wavelet_beta:.2f}.png"
                    )
                    _save_wavelet_weight_png(
                        out_path=out_wave,
                        weight=weight_ch,
                        figsize=figsize,
                        dpi=dpi,
                        interpolation=interpolation,
                        optimize_png=optimize_png,
                    )

            remaining.discard(k_idx)
            saved += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not remaining:
            break

    # Visualisation run summary.
    (out_dir / "gradviz_summary.json").write_text(json.dumps({
        "pairs_requested": sorted(list(wanted_pairs)),
        "pairs_missing": sorted(list(remaining)),
        "figures_saved": int(saved),
        "operators_path": str(LOSS_PATH),
        "amp": {"enabled": amp_enabled, "dtype": "fp16" if amp_dtype == torch.float16 else "bf16"},
        "device": str(device),
        "figsize": {"width_in": figsize[0], "height_in": figsize[1]},
        "dpi": dpi,
        "interpolation": interpolation,
        "optimize_png": optimize_png,
        "wavelet_vis": wavelet_vis,
        "wavelet_theta": wavelet_theta,
        "wavelet_alpha": wavelet_alpha,
        "wavelet_beta": wavelet_beta,
        "wavelet_J": wavelet_J,
        "wavelet_name": wavelet_name,
        "wavelet_mode": wavelet_mode,
    }, indent=2))


if __name__ == "__main__":
    main()
