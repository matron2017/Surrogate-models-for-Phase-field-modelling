#!/usr/bin/env python3
"""
Plot side-by-side decoded field comparisons for four model styles:
- old flow style
- diffusion bridge style
- flow style
- new style

For each selected sample index and channel, write one panel:
rows   = model styles
cols   = input | target | prediction | residual(pred-target)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")

from models.diffusion.scheduler_registry import get_noise_schedule
from scripts.eval_latent_flow_bridge_decoded import (
    _DIFFUSION_PREDICT_NEXT_OBJECTIVES,
    _assert_ae_matches_latent_source,
    _decoder_call,
    _denorm_if_needed,
    _extract_norm_stats,
    _load_ae_model,
    _load_train_checkpoint,
    _predict_bridge_rollout_dbim,
    _predict_flow_source_anchored,
    _resolve_dataset_cfg,
    _split_theta,
)

_CANONICAL_AE_CKPT = Path(
    "/scratch/project_2008261/pf_surrogate_modelling/runs/"
    "ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_"
    "rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth"
)


@dataclass
class ModelCtx:
    key: str
    label: str
    ckpt: Path
    model: torch.nn.Module
    cfg: Dict[str, Any]
    family: str
    flow_objective: str
    flow_noise_std: float
    flow_noise_mode: str
    flow_noise_perturb_source: bool
    bridge_schedule: Any
    bridge_predict_next: bool


def _parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _to_np(x: torch.Tensor) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy(), dtype=np.float32)


def _load_ctx(
    key: str,
    label: str,
    ckpt_path: Path,
    device: torch.device,
    flow_noise_std_override: float,
    flow_noise_mode_override: str,
    flow_noise_perturb_source_override: Optional[bool],
) -> ModelCtx:
    model, cfg, _ = _load_train_checkpoint(ckpt_path, device)
    family = str(cfg.get("train", {}).get("model_family", "surrogate")).lower()
    flow_cfg = dict(cfg.get("flow_matching", {}) or {})
    flow_noise_std = (
        float(flow_cfg.get("noise_stochastic_std", 0.0))
        if float(flow_noise_std_override) < 0.0
        else float(flow_noise_std_override)
    )
    flow_noise_mode = (
        str(flow_cfg.get("noise_stochastic_mode", "scalar")).strip().lower()
        if not str(flow_noise_mode_override).strip()
        else str(flow_noise_mode_override).strip().lower()
    )
    if flow_noise_mode not in {"scalar", "field"}:
        flow_noise_mode = "scalar"
    if flow_noise_perturb_source_override is None:
        flow_noise_perturb_source = bool(flow_cfg.get("noise_stochastic_perturb_source", True))
    else:
        flow_noise_perturb_source = bool(flow_noise_perturb_source_override)

    bridge_schedule = None
    bridge_predict_next = False
    if family == "diffusion":
        diff_cfg = dict(cfg.get("diffusion", {}) or {})
        bridge_schedule = get_noise_schedule(
            diff_cfg["noise_schedule"],
            **(diff_cfg.get("schedule_kwargs", {}) or {}),
        )
        diff_obj = str((cfg.get("loss", {}) or {}).get("diffusion_objective", "epsilon_mse")).lower()
        bridge_predict_next = diff_obj in _DIFFUSION_PREDICT_NEXT_OBJECTIVES

    return ModelCtx(
        key=key,
        label=label,
        ckpt=ckpt_path,
        model=model,
        cfg=cfg,
        family=family,
        flow_objective=str(cfg.get("train", {}).get("objective", "rectified_flow_source_anchored_concat")).lower(),
        flow_noise_std=flow_noise_std,
        flow_noise_mode=flow_noise_mode,
        flow_noise_perturb_source=flow_noise_perturb_source,
        bridge_schedule=bridge_schedule,
        bridge_predict_next=bridge_predict_next,
    )


def _predict_latent(
    ctx: ModelCtx,
    x_full: torch.Tensor,
    y_true: torch.Tensor,
    flow_nfe: int,
    flow_num_samples: int,
    bridge_nfe: int,
    bridge_eta: float,
) -> torch.Tensor:
    cond_cfg = dict(ctx.cfg.get("conditioning", {}) or {})
    x_state, theta = _split_theta(x_full, cond_cfg)
    if x_state.shape[1] != y_true.shape[1]:
        # Fallback for unexpected channel wiring.
        x_state = x_full[:, : y_true.shape[1], ...]
        theta = None

    if ctx.family == "flow_matching":
        ns = max(1, int(flow_num_samples))
        y_samples: List[torch.Tensor] = []
        for _ in range(ns):
            y_samples.append(
                _predict_flow_source_anchored(
                    model=ctx.model,
                    x=x_state,
                    theta=theta,
                    nfe=max(1, int(flow_nfe)),
                    flow_objective=ctx.flow_objective,
                    flow_noise_std=float(ctx.flow_noise_std),
                    flow_noise_mode=str(ctx.flow_noise_mode),
                    flow_noise_perturb_source=bool(ctx.flow_noise_perturb_source),
                )
            )
        if len(y_samples) == 1:
            return y_samples[0]
        return torch.stack(y_samples, dim=0).mean(dim=0)

    if ctx.family == "diffusion":
        if ctx.bridge_schedule is None:
            raise RuntimeError(f"Diffusion model {ctx.label} has no bridge schedule.")
        return _predict_bridge_rollout_dbim(
            model=ctx.model,
            schedule=ctx.bridge_schedule,
            x=x_state,
            theta=theta,
            nfe=max(1, int(bridge_nfe)),
            eta=float(bridge_eta),
            predict_next=bool(ctx.bridge_predict_next),
        )

    raise ValueError(f"Unsupported model family for {ctx.label}: {ctx.family}")


def _panel_plot(
    out_path: Path,
    title: str,
    model_order: List[ModelCtx],
    x_map: np.ndarray,
    y_map: np.ndarray,
    pred_by_key: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}

    stack_fields = [x_map, y_map] + [pred_by_key[m.key] for m in model_order]
    vmin = float(np.nanmin(np.stack(stack_fields, axis=0)))
    vmax = float(np.nanmax(np.stack(stack_fields, axis=0)))

    residuals = [pred_by_key[m.key] - y_map for m in model_order]
    rmax = float(max(np.nanmax(np.abs(r)) for r in residuals))
    if not np.isfinite(rmax) or rmax <= 0.0:
        rmax = 1e-6

    fig, axes = plt.subplots(
        nrows=len(model_order),
        ncols=4,
        figsize=(16.0, 3.2 * len(model_order)),
        squeeze=False,
    )
    col_names = ["input", "target", "prediction", "residual"]
    for cidx, cname in enumerate(col_names):
        axes[0, cidx].set_title(cname, fontsize=11)

    for ridx, ctx in enumerate(model_order):
        pred = pred_by_key[ctx.key]
        resid = pred - y_map
        rmse = float(np.sqrt(np.mean((pred - y_map) ** 2)))
        mae = float(np.mean(np.abs(pred - y_map)))
        metrics[ctx.label] = {"rmse": rmse, "mae": mae}

        images = [x_map, y_map, pred, resid]
        cmaps = ["viridis", "viridis", "viridis", "seismic"]
        limits = [(vmin, vmax), (vmin, vmax), (vmin, vmax), (-rmax, rmax)]
        for cidx in range(4):
            ax = axes[ridx, cidx]
            im = ax.imshow(
                images[cidx],
                origin="lower",
                cmap=cmaps[cidx],
                vmin=limits[cidx][0],
                vmax=limits[cidx][1],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if cidx == 0:
                ax.set_ylabel(f"{ctx.label}\nRMSE={rmse:.4f}\nMAE={mae:.4f}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot model-style comparison panels in decoded field space.")
    ap.add_argument("--old-ckpt", type=Path, required=True)
    ap.add_argument("--bridge-ckpt", type=Path, required=True)
    ap.add_argument("--flow-ckpt", type=Path, required=True)
    ap.add_argument("--new-ckpt", type=Path, required=True)
    ap.add_argument("--old-label", type=str, default="old_flow_style")
    ap.add_argument("--bridge-label", type=str, default="diffusion_bridge_style")
    ap.add_argument("--flow-label", type=str, default="flow_style")
    ap.add_argument("--new-label", type=str, default="new_sfm_style")
    ap.add_argument("--ae-ckpt", type=Path, default=_CANONICAL_AE_CKPT)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample-indices", type=str, default="0,1")
    ap.add_argument("--channels", type=str, default="0,1")
    ap.add_argument("--flow-nfe", type=int, default=20)
    ap.add_argument("--flow-num-samples", type=int, default=1)
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--bridge-eta", type=float, default=0.0)
    ap.add_argument("--flow-noise-std", type=float, default=-1.0)
    ap.add_argument("--flow-noise-mode", type=str, default="")
    ap.add_argument(
        "--flow-noise-perturb-source",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    ap.add_argument(
        "--dataset-from",
        type=str,
        default="flow",
        choices=["old", "bridge", "flow", "new"],
        help="Which checkpoint config to use for dataloader args/sample indexing.",
    )
    ap.add_argument("--h5-override", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    perturb_override: Optional[bool]
    if str(args.flow_noise_perturb_source) == "auto":
        perturb_override = None
    else:
        perturb_override = str(args.flow_noise_perturb_source) == "true"

    model_order = [
        _load_ctx(
            key="old",
            label=str(args.old_label),
            ckpt_path=args.old_ckpt.expanduser().resolve(),
            device=device,
            flow_noise_std_override=float(args.flow_noise_std),
            flow_noise_mode_override=str(args.flow_noise_mode),
            flow_noise_perturb_source_override=perturb_override,
        ),
        _load_ctx(
            key="bridge",
            label=str(args.bridge_label),
            ckpt_path=args.bridge_ckpt.expanduser().resolve(),
            device=device,
            flow_noise_std_override=float(args.flow_noise_std),
            flow_noise_mode_override=str(args.flow_noise_mode),
            flow_noise_perturb_source_override=perturb_override,
        ),
        _load_ctx(
            key="flow",
            label=str(args.flow_label),
            ckpt_path=args.flow_ckpt.expanduser().resolve(),
            device=device,
            flow_noise_std_override=float(args.flow_noise_std),
            flow_noise_mode_override=str(args.flow_noise_mode),
            flow_noise_perturb_source_override=perturb_override,
        ),
        _load_ctx(
            key="new",
            label=str(args.new_label),
            ckpt_path=args.new_ckpt.expanduser().resolve(),
            device=device,
            flow_noise_std_override=float(args.flow_noise_std),
            flow_noise_mode_override=str(args.flow_noise_mode),
            flow_noise_perturb_source_override=perturb_override,
        ),
    ]
    by_key = {m.key: m for m in model_order}
    ds_ctx = by_key[str(args.dataset_from)]
    ds, ds_meta = _resolve_dataset_cfg(
        ds_ctx.cfg,
        split=str(args.split),
        override_h5=(args.h5_override.expanduser().resolve() if args.h5_override else None),
    )

    h5_path = Path(ds_meta["h5_path"]).expanduser().resolve()
    ae_ckpt = args.ae_ckpt.expanduser().resolve()
    _assert_ae_matches_latent_source(h5_path, ae_ckpt)
    ae_model = _load_ae_model(ae_ckpt, device)
    mean, std, schema = _extract_norm_stats(h5_path, device=device)

    sample_indices = _parse_int_csv(args.sample_indices)
    channels = _parse_int_csv(args.channels)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "split": str(args.split),
        "dataset_h5": str(h5_path),
        "dataset_from": str(args.dataset_from),
        "sample_indices": sample_indices,
        "channels": channels,
        "flow_nfe": int(args.flow_nfe),
        "bridge_nfe": int(args.bridge_nfe),
        "bridge_eta": float(args.bridge_eta),
        "flow_num_samples": int(args.flow_num_samples),
        "flow_noise_std": float(args.flow_noise_std),
        "models": [
            {"key": m.key, "label": m.label, "family": m.family, "ckpt": str(m.ckpt), "flow_objective": m.flow_objective}
            for m in model_order
        ],
        "outputs": [],
    }

    with torch.inference_mode():
        for sidx in sample_indices:
            if sidx < 0 or sidx >= len(ds):
                raise IndexError(f"Sample index {sidx} out of range for dataset length {len(ds)}")
            sample = ds[int(sidx)]
            x_full = sample["input"].unsqueeze(0).to(device, non_blocking=True).float()
            y_lat_true = sample["target"].unsqueeze(0).to(device, non_blocking=True).float()
            gid = str(sample.get("gid", "unknown"))
            pair_index = int(sample.get("pair_index", -1))

            # Decode source state as first C_target channels from input (thermal is appended at the end).
            x_lat_state = x_full[:, : y_lat_true.shape[1], ...]
            x_dec = _decoder_call(ae_model, x_lat_state)
            y_dec_true = _decoder_call(ae_model, y_lat_true)
            x_dec = _denorm_if_needed(x_dec, mean, std, schema)
            y_dec_true = _denorm_if_needed(y_dec_true, mean, std, schema)
            x_dec_np = _to_np(x_dec[0])
            y_dec_np = _to_np(y_dec_true[0])

            pred_dec_by_key: Dict[str, np.ndarray] = {}
            for ctx in model_order:
                y_lat_pred = _predict_latent(
                    ctx=ctx,
                    x_full=x_full,
                    y_true=y_lat_true,
                    flow_nfe=int(args.flow_nfe),
                    flow_num_samples=int(args.flow_num_samples),
                    bridge_nfe=int(args.bridge_nfe),
                    bridge_eta=float(args.bridge_eta),
                )
                y_dec_pred = _decoder_call(ae_model, y_lat_pred)
                y_dec_pred = _denorm_if_needed(y_dec_pred, mean, std, schema)
                pred_dec_by_key[ctx.key] = _to_np(y_dec_pred[0])

            for c in channels:
                if c < 0 or c >= y_dec_np.shape[0]:
                    raise IndexError(f"Channel {c} out of range for decoded tensor with {y_dec_np.shape[0]} channels")
                pred_map_by_key = {ctx.key: pred_dec_by_key[ctx.key][c] for ctx in model_order}
                panel_name = f"style_compare_idx{sidx:05d}_{gid}_pair{pair_index:04d}_ch{c}.png"
                panel_path = out_dir / panel_name
                title = (
                    f"sample_idx={sidx} gid={gid} pair={pair_index} ch={c} "
                    "(decoded raw values; common color limits per panel)"
                )
                panel_metrics = _panel_plot(
                    out_path=panel_path,
                    title=title,
                    model_order=model_order,
                    x_map=x_dec_np[c],
                    y_map=y_dec_np[c],
                    pred_by_key=pred_map_by_key,
                )
                manifest["outputs"].append(
                    {
                        "sample_index": int(sidx),
                        "gid": gid,
                        "pair_index": int(pair_index),
                        "channel": int(c),
                        "panel_png": str(panel_path),
                        "metrics": panel_metrics,
                    }
                )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote panels to: {out_dir}")
    print(f"[done] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
