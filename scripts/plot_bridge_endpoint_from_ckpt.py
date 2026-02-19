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
from models.train.core.diffusion_forward import _build_diffusion_model_input
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _load_symbol, _prepare_batch

_DIFFUSION_PREDICT_NEXT_OBJECTIVES = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
}


def _to_float(value) -> float:
    try:
        v = float(value)
    except Exception:
        v = float("nan")
    return v


def _safe_float_isfinite(value) -> bool:
    v = _to_float(value)
    return float("nan") != v and float("inf") != v and -float("inf") != v


def _scale_phase_to_unit(arr: np.ndarray) -> np.ndarray:
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return arr.astype(np.float32, copy=False)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 0.0:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 0.0:
        return arr.astype(np.float32, copy=False)
    scaled = (arr.astype(np.float32, copy=False) - lo) / max(1e-12, (hi - lo))
    return np.clip(scaled, 0.0, 1.0) * 2.0 - 1.0


def _read_sample_metadata(
    h5_path: str | Path,
    gid: str,
    pair_index: int,
    *,
    use_pairs_idx: bool,
    bridge_nfe: int,
) -> Dict[str, object]:
    h5_path = Path(h5_path)
    out: Dict[str, object] = {
        "pair_index": int(pair_index),
        "pair_step_src": None,
        "pair_step_dst": None,
        "pair_dt": None,
        "euler_step": None,
        "thermal_gradient": float("nan"),
        "pulling_speed": float("nan"),
        "step_text": "Euler step unavailable",
    }

    with h5py.File(h5_path, "r") as f:
        if gid not in f:
            return out
        g = f[gid]

        def _frame_time(g: h5py.Group, idx: int, eff_dt: float) -> float:
            idx = int(max(0, idx))
            if "time_phys" in g and int(g["time_phys"].shape[0]) > idx:
                return float(np.asarray(g["time_phys"][idx]))
            if "times" in g and int(g["times"].shape[0]) > idx:
                return float(np.asarray(g["times"][idx]) * eff_dt)
            return float(idx * eff_dt)

        eff_dt = float(g.attrs.get("effective_dt", f.attrs.get("effective_dt", 1.0)))
        gidx = int(pair_index)
        if use_pairs_idx and "pairs_idx" in g and int(g["pairs_idx"].shape[0]) > gidx:
            p = np.asarray(g["pairs_idx"][gidx]).astype(int)
            if p.size >= 2:
                i0, i1 = int(p[0]), int(p[1])
                src_step = i0
                dst_step = i1
            else:
                src_step = gidx
                dst_step = gidx
            if "pairs_time" in g and int(g["pairs_time"].shape[0]) > gidx and np.asarray(g["pairs_time"][gidx]).size >= 2:
                t0 = float(np.asarray(g["pairs_time"][gidx][0]) * eff_dt)
                t1 = float(np.asarray(g["pairs_time"][gidx][1]) * eff_dt)
            else:
                t0 = _frame_time(g, src_step, eff_dt)
                t1 = _frame_time(g, dst_step, eff_dt)
        else:
            src_step = gidx
            dst_step = gidx
            t0 = _frame_time(g, src_step, eff_dt)
            t1 = _frame_time(g, dst_step, eff_dt)

        g_raw = g.attrs.get("thermal_gradient_raw", g.attrs.get("thermal_gradient", float("nan")))
        if not _safe_float_isfinite(g_raw) and "thermal_gradient_series" in g and int(g["thermal_gradient_series"].shape[0]) > gidx:
            g_raw = np.asarray(g["thermal_gradient_series"])[int(gidx)]
        v_pull = g.attrs.get("pulling_speed", f.attrs.get("pulling_speed", float("nan")))

        dt = float(t1 - t0)
        euler = dt / max(int(bridge_nfe), 1)
        out["pair_step_src"] = int(src_step)
        out["pair_step_dst"] = int(dst_step)
        out["pair_dt"] = float(dt)
        out["euler_step"] = float(euler)
        out["thermal_gradient"] = float(_to_float(g_raw))
        out["pulling_speed"] = float(_to_float(v_pull))
        out["step_text"] = f"Euler step src={src_step} -> dst={dst_step}, dt={dt:.6g}, euler={euler:.6g} (nfe={max(int(bridge_nfe), 1)})"

    return out


def _prepare_channel_fields(
    ch: int,
    xin: np.ndarray,
    ygt: np.ndarray,
    ypr: np.ndarray,
    *,
    concentration_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ch == 1:
        xin_c = xin * float(concentration_scale)
        ygt_c = ygt * float(concentration_scale)
        ypr_c = ypr * float(concentration_scale)
        return xin_c, ygt_c, ypr_c, (ypr_c - ygt_c), (ygt_c - xin_c)

    if ch == 0:
        stack = np.stack([xin, ygt, ypr], axis=0).astype(np.float32)
        finite = stack[np.isfinite(stack)]
        if finite.size == 0:
            stack_scaled = stack
        else:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 0.0:
                stack_scaled = stack
            else:
                stack_scaled = np.clip((stack - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
        xin_p, ygt_p, ypr_p = stack_scaled[0], stack_scaled[1], stack_scaled[2]
        return xin_p, ygt_p, ypr_p, (ypr_p - ygt_p), (ygt_p - xin_p)

    return xin, ygt, ypr, (ypr - ygt), (ygt - xin)


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


def _build_cond_for_model(
    model: torch.nn.Module, bsz: int, device: torch.device, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "backbone"):
        base = base.backbone
    cond_dim = int(getattr(base, "cond_dim", 0))
    use_time = bool(getattr(base, "use_time", False))
    if not use_time and cond_dim <= 0:
        return None
    cond_cols = max(cond_dim, 1)
    return torch.zeros((bsz, cond_cols), device=device, dtype=dtype)


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
    cond_vec = _build_cond_for_model(model, bsz=bsz, device=x.device, dtype=x.dtype)
    for i, s_idx in enumerate(eval_ts):
        s = torch.full((bsz,), int(s_idx), device=x.device, dtype=torch.long)
        s_model = s.view(-1, 1).to(dtype=x.dtype)
        x_in = _build_diffusion_model_input(
            x_noisy=x_curr,
            source=x,
            noise_schedule_obj=schedule,
            sched_kind=str(getattr(schedule, "kind", "")),
        )
        if theta is not None:
            if cond_vec is None:
                cond_vec = torch.zeros((bsz, 1), device=x.device, dtype=x.dtype)
            try:
                pred = model(x_in, cond_vec, s_model, hint=theta)
            except TypeError:
                pred = model(x_in, cond_vec, s_model, theta=theta)
            except ValueError as exc:
                if "Generative timestep is required" in str(exc):
                    # Some older forward signatures accept timestep as first positional arg.
                    pred = model(x_in, s_model, hint=theta)
                else:
                    raise
        else:
            try:
                pred = model(x_in, cond_vec, s_model)
            except ValueError as exc:
                if "Generative timestep is required" in str(exc):
                    pred = model(x_in, s_model)
                else:
                    raise

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
    ap.add_argument("--phase-ch", type=int, default=0, help="Phase channel index in decoded pixel space.")
    ap.add_argument("--concentration-ch", type=int, default=1, help="Concentration channel index in decoded pixel space.")
    ap.add_argument("--concentration-scale", type=float, default=3.0, help="Scale concentration by this factor before plotting.")
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

    dset_len = len(ds)
    valid_indices = [idx for idx in indices if 0 <= int(idx) < dset_len]
    dropped = [idx for idx in indices if idx not in valid_indices]
    if dropped:
        print(f"[warn] skipping out-of-range indices: {dropped}", flush=True)
    if len(valid_indices) == 0:
        raise IndexError(f"No valid indices in requested list {indices} for dataset size {dset_len}")
    cond_cfg = dict(cfg.get("conditioning", {}) or {})

    with torch.inference_mode():
        for idx in indices:
            if idx < 0 or idx >= dset_len:
                continue
            sample = ds[idx]
            gid = str(sample["gid"])
            pair_idx = int(sample["pair_index"])
            sample_batch = {"input": sample["input"].unsqueeze(0), "target": sample["target"].unsqueeze(0)}
            x_state, y, _, theta = _prepare_batch(sample_batch, device, cond_cfg, use_chlast=False)
            x_state = x_state.float()
            y = y.float()
            if theta is not None:
                theta = theta.float()
            meta = _read_sample_metadata(
                h5_path=h5_path,
                gid=gid,
                pair_index=pair_idx,
                use_pairs_idx=bool(getattr(ds, "use_pairs_idx", True)),
                bridge_nfe=int(args.bridge_nfe),
            )
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

            channels = []
            if y_dec.shape[0] > args.phase_ch:
                channels.append(int(args.phase_ch))
            if y_dec.shape[0] > args.concentration_ch and args.concentration_ch != args.phase_ch:
                channels.append(int(args.concentration_ch))
            if not channels:
                channels = [0] if y_dec.shape[0] >= 1 else []
            if not channels:
                continue

            fig, axes = plt.subplots(len(channels), 5, figsize=(20, 4.4 * len(channels)), constrained_layout=True)
            if len(channels) == 1:
                axes = np.expand_dims(axes, axis=0)

            stats: Dict[str, Dict[str, Dict[str, float]]] = {}
            for r, ch in enumerate(channels):
                field = "phase" if ch == 0 else ("concentration" if ch == 1 else f"ch{ch}")
                xin, ygt, ypr = x_dec[ch], y_dec[ch], p_dec[ch]
                xin, ygt, ypr, res_pred_gt, res_gt_in = _prepare_channel_fields(
                    ch=ch,
                    xin=xin,
                    ygt=ygt,
                    ypr=ypr,
                    concentration_scale=float(args.concentration_scale),
                )

                vmin = float(np.nanmin(np.stack([xin, ygt, ypr], axis=0)))
                vmax = float(np.nanmax(np.stack([xin, ygt, ypr], axis=0)))
                rabs = float(np.nanmax(np.stack([np.abs(res_pred_gt), np.abs(res_gt_in)], axis=0)))
                if not np.isfinite(rabs) or rabs <= 0:
                    rabs = 1e-8

                cmap_val = "RdBu_r" if ch == 0 else "viridis"
                panels = [
                    ("Input (past)", xin, vmin, vmax, cmap_val),
                    ("GT endpoint (next)", ygt, vmin, vmax, cmap_val),
                    ("Pred endpoint", ypr, vmin, vmax, cmap_val),
                    ("Residual (pred-GT)", res_pred_gt, -rabs, rabs, "seismic"),
                    ("Residual (GT-input)", res_gt_in, -rabs, rabs, "seismic"),
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

                residual_pred_text = f"residual_pred-GT mean={float(np.nanmean(res_pred_gt)):.4g}"
                residual_gt_text = f"residual_GT-input mean={float(np.nanmean(res_gt_in)):.4g}"
                meta_text = (
                    f"pair_steps: {meta['step_text']} | "
                    f"thermal_gradient: {meta['thermal_gradient']:.6g} | "
                    f"pulling_speed: {meta['pulling_speed']:.6g} | "
                    f"{residual_pred_text}; {residual_gt_text}"
                )
                axes[r, 0].set_title(f"{field} | {meta_text}", fontsize=9)

            fig.suptitle(
                f"Bridge Endpoint Prediction | idx={idx} gid={gid} pair={pair_idx} | UniDB rollout NFE={int(args.bridge_nfe)}\n"
                f"checkpoint={bridge_ckpt.name} (epoch={ckpt_epoch}) | phase->[-1,1], concentration*x{args.concentration_scale}",
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
