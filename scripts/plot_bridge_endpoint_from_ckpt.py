#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _load_symbol

_DIFFUSION_PREDICT_NEXT_OBJECTIVES = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
}


def _parse_indices(s: str) -> List[int]:
    out: List[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("No indices parsed from --indices.")
    return sorted(set(out))


def _decoder_call(ae_model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if hasattr(ae_model, "autoencoder") and hasattr(ae_model.autoencoder, "decode"):
        dec = ae_model.autoencoder.decode
    elif hasattr(ae_model, "decode"):
        dec = ae_model.decode
    else:
        raise RuntimeError("AE model has no decode method")
    try:
        sig = inspect.signature(dec)
        if "noisy" in sig.parameters:
            return dec(z, noisy=False)
    except Exception:
        pass
    return dec(z)


def _bridge_coeffs(schedule, t_long: torch.Tensor, ref: torch.Tensor):
    if getattr(schedule, "kind", "") == "unidb":
        b_t = schedule._m(t_long, ref)
        a_t = schedule._n(t_long, ref)
        c_t = schedule.f_sigma(t_long, ref)
        return a_t, b_t, c_t
    if hasattr(schedule, "a") and hasattr(schedule, "b") and hasattr(schedule, "c"):
        a_t = schedule.a.to(ref.device)[t_long].view(-1, 1, 1, 1)
        b_t = schedule.b.to(ref.device)[t_long].view(-1, 1, 1, 1)
        c_t = schedule.c.to(ref.device)[t_long].view(-1, 1, 1, 1)
        return a_t, b_t, c_t
    raise TypeError(f"Unsupported bridge schedule kind={getattr(schedule, 'kind', 'unknown')}")


def _build_desc_grid(max_t: int, min_t: int, nfe: int) -> List[int]:
    if max_t <= min_t:
        return [max_t]
    vals = torch.linspace(float(max_t), float(min_t), steps=max(1, int(nfe))).round().to(torch.long).tolist()
    out: List[int] = []
    for v in vals:
        iv = int(max(min_t, min(max_t, v)))
        if not out or iv != out[-1]:
            out.append(iv)
    if out[-1] != min_t:
        out.append(min_t)
    return out


def _predict_bridge_rollout_dbim(
    model: torch.nn.Module,
    schedule,
    x: torch.Tensor,
    theta: Optional[torch.Tensor],
    nfe: int,
    *,
    predict_next: bool,
) -> torch.Tensor:
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"Expected UniDB schedule, got kind={getattr(schedule, 'kind', '')}")
    bsz = int(x.shape[0])
    eval_ts = _build_desc_grid(max_t=max(int(schedule.timesteps) - 1, 1), min_t=1, nfe=max(1, int(nfe)))
    x_curr = x
    for i, s_idx in enumerate(eval_ts):
        s = torch.full((bsz,), int(s_idx), device=x.device, dtype=torch.long)
        s_model = s.view(-1, 1).to(dtype=x.dtype)
        x_in = torch.cat([x_curr, x], dim=1)
        pred = model(x_in, s_model, theta=theta) if theta is not None else model(x_in, s_model)

        a_s, b_s, c_s = _bridge_coeffs(schedule, s, ref=x_curr)
        y_hat = pred if predict_next else (x_curr - a_s * x - c_s * pred) / b_s.clamp_min(1e-6)

        if i == 0:
            x_curr = a_s * x + b_s * y_hat

        t_idx = int(eval_ts[i + 1]) if (i + 1) < len(eval_ts) else 0
        t = torch.full((bsz,), t_idx, device=x.device, dtype=torch.long)
        a_t, b_t, c_t = _bridge_coeffs(schedule, t, ref=x_curr)

        coeff_xs = c_t / c_s.clamp_min(1e-8)
        coeff_y = b_t - coeff_xs * b_s
        coeff_src = a_t - coeff_xs * a_s
        x_curr = coeff_xs * x_curr + coeff_y * y_hat + coeff_src * x
    return x_curr


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot bridge endpoint predictions for selected dataset indices.")
    ap.add_argument("--bridge-ckpt", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--indices", type=str, default="234,255,300")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--h5-override", type=Path, default=None)
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    indices = _parse_indices(args.indices)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_ckpt = args.bridge_ckpt.expanduser().resolve()
    ae_ckpt = args.ae_ckpt.expanduser().resolve()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(bridge_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model_family = str(cfg.get("train", {}).get("model_family", "surrogate")).lower()
    backbone = str(model_cfg.get("backbone", "")).strip().lower()
    model = registry_build_model(model_family, backbone, model_cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    ckpt_epoch = int(ckpt.get("epoch", -1))

    ckpt_ae = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    ae_cfg = ckpt_ae.get("config", {})
    ae_model_cfg = ae_cfg.get("model", {})
    ModelClass = _load_symbol(ae_model_cfg["file"], ae_model_cfg["class"])
    ae_model = ModelClass(**(ae_model_cfg.get("params", {}) or {}))
    ae_model.load_state_dict(ckpt_ae["model"], strict=True)
    ae_model = ae_model.to(device).eval()

    dcfg = cfg["dataloader"]
    base_args = dict((dcfg.get("args", {}) or {}))
    split_args = dict((dcfg.get(f"{args.split}_args", {}) or {}))
    if args.h5_override is not None:
        h5_path = args.h5_override.expanduser().resolve()
    else:
        h5_map = dict((cfg.get("paths", {}) or {}).get("h5", {}) or {})
        if args.split in h5_map:
            h5_path = Path(str(h5_map[args.split]))
        else:
            raise KeyError(f"Missing paths.h5.{args.split} in checkpoint config.")
    ds = PFPairDataset(h5_path=str(h5_path), **{**base_args, **split_args})

    mean = std = None
    schema = ""
    with h5py.File(h5_path, "r") as f:
        schema = str(f.attrs.get("normalization_schema", "")).lower()
        if "channel_mean" in f.attrs and "channel_std" in f.attrs:
            mean = torch.tensor(np.array(f.attrs["channel_mean"], dtype=np.float32), device=device).view(1, -1, 1, 1)
            std = torch.tensor(np.array(f.attrs["channel_std"], dtype=np.float32), device=device).view(1, -1, 1, 1)

    def denorm_if_needed(x: torch.Tensor) -> torch.Tensor:
        if mean is None or std is None:
            return x
        if schema == "zscore" and x.shape[1] == mean.shape[1]:
            return x * std + mean
        return x

    noise_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(noise_cfg["noise_schedule"], **noise_cfg.get("schedule_kwargs", {}))
    objective = str(
        (cfg.get("loss", {}) or {}).get("diffusion_objective", (cfg.get("diffusion", {}) or {}).get("objective", "eps_bridge"))
    ).lower()
    predict_next = objective in _DIFFUSION_PREDICT_NEXT_OBJECTIVES

    manifest: Dict[str, object] = {
        "bridge_ckpt": str(bridge_ckpt),
        "bridge_epoch": ckpt_epoch,
        "bridge_nfe": int(args.bridge_nfe),
        "objective": objective,
        "predict_next": bool(predict_next),
        "ae_ckpt": str(ae_ckpt),
        "dataset_h5": str(h5_path),
        "normalization_schema": schema,
        "indices": [int(i) for i in indices],
        "outputs": [],
    }

    with torch.inference_mode():
        for idx in indices:
            sample = ds[idx]
            gid = str(sample["gid"])
            pair_idx = int(sample["pair_index"])
            x = sample["input"].to(device).float().unsqueeze(0)
            y = sample["target"].to(device).float().unsqueeze(0)

            cond_cfg = dict(cfg.get("conditioning", {}) or {})
            if bool(cond_cfg.get("use_theta", False)):
                th = int(cond_cfg.get("theta_channels", 1))
                theta = x[:, -th:, ...]
                x_state = x[:, :-th, ...]
            else:
                theta = None
                x_state = x

            y_hat = _predict_bridge_rollout_dbim(
                model=model,
                schedule=schedule,
                x=x_state,
                theta=theta,
                nfe=int(args.bridge_nfe),
                predict_next=bool(predict_next),
            )

            x_dec = denorm_if_needed(_decoder_call(ae_model, x_state))[0].detach().cpu().numpy()
            y_dec = denorm_if_needed(_decoder_call(ae_model, y))[0].detach().cpu().numpy()
            p_dec = denorm_if_needed(_decoder_call(ae_model, y_hat))[0].detach().cpu().numpy()

            channels = [0, 1] if y_dec.shape[0] >= 2 else [0]
            fig, axes = plt.subplots(len(channels), 5, figsize=(20, 4.4 * len(channels)), constrained_layout=True)
            if len(channels) == 1:
                axes = np.expand_dims(axes, axis=0)

            stats: Dict[str, Dict[str, Dict[str, float]]] = {}
            for r, ch in enumerate(channels):
                field = "phase" if ch == 0 else ("concentration" if ch == 1 else f"ch{ch}")
                xin, ygt, ypr = x_dec[ch], y_dec[ch], p_dec[ch]
                res_pred_gt = ypr - ygt
                res_pred_in = ypr - xin

                vmin = float(np.nanmin(np.stack([xin, ygt, ypr], axis=0)))
                vmax = float(np.nanmax(np.stack([xin, ygt, ypr], axis=0)))
                rabs = float(np.nanmax(np.stack([np.abs(res_pred_gt), np.abs(res_pred_in)], axis=0)))
                if not np.isfinite(rabs) or rabs <= 0:
                    rabs = 1e-8

                cmap_val = "RdBu_r" if ch == 0 else "viridis"
                panels = [
                    ("Input (past)", xin, vmin, vmax, cmap_val),
                    ("GT endpoint (next)", ygt, vmin, vmax, cmap_val),
                    ("Pred endpoint", ypr, vmin, vmax, cmap_val),
                    ("Residual (pred-GT)", res_pred_gt, -rabs, rabs, "seismic"),
                    ("Residual (pred-input)", res_pred_in, -rabs, rabs, "seismic"),
                ]
                stats[field] = {}
                for c, (title, arr, lo, hi, cmap) in enumerate(panels):
                    ax = axes[r, c]
                    im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                    ax.set_title(f"{field} | {title}", fontsize=11)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    mn, mx, me = float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))
                    ax.text(
                        0.02,
                        0.02,
                        f"min {mn:.4g}  max {mx:.4g}  mean {me:.4g}",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
                    )
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=8)
                    stats[field][title] = {"min": mn, "max": mx, "mean": me}

            fig.suptitle(
                f"Bridge Endpoint Prediction | idx={idx} gid={gid} pair={pair_idx} | UniDB rollout NFE={int(args.bridge_nfe)}\n"
                f"checkpoint={bridge_ckpt.name} (epoch={ckpt_epoch})",
                fontsize=13,
            )
            out_png = out_dir / f"bridge_endpoint_pred_idx{idx:04d}_{gid}_pair{pair_idx:04d}.png"
            fig.savefig(out_png, dpi=int(args.dpi))
            plt.close(fig)

            manifest["outputs"].append(
                {
                    "index": int(idx),
                    "gid": gid,
                    "pair_index": pair_idx,
                    "png": str(out_png),
                    "stats": stats,
                }
            )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote visuals to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
