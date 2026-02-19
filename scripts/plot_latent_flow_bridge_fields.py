#!/usr/bin/env python3
"""
Create AE-style decoded field visuals for latent flow vs bridge models.

For each selected sample and per channel:
  - flow panel: past | true next | true_delta(next-past) | flow pred | flow residual
  - bridge panel: past | true next | true_delta(next-past) | bridge pred | bridge residual

Also writes:
  - per-sample/channel stats JSON
  - side-by-side summary table (flow vs bridge RMSE/MAE) as CSV + Markdown
  - manifest.json
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.diffusion_forward import _build_diffusion_model_input
from models.train.core.loops import _forward_diffusion, _forward_flow_concat, _forward_flow_nonconcat
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _load_symbol, _prepare_batch

_DIFFUSION_PREDICT_NEXT_OBJECTIVES = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
}

_CANONICAL_AE_CKPT = Path(
    "/scratch/project_2008261/pf_surrogate_modelling/runs/"
    "ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_"
    "rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth"
)


@dataclass
class SamplePack:
    dataset_index: int
    gid: str
    pair_index: int
    x_dec: np.ndarray  # [C,H,W]
    y_dec: np.ndarray  # [C,H,W]
    flow_dec: np.ndarray  # [C,H,W]
    bridge_dec: np.ndarray  # [C,H,W]


def _load_train_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", None)
    if not cfg:
        raise RuntimeError(f"Checkpoint missing embedded config: {ckpt_path}")
    model_cfg = cfg["model"]
    backbone = str(model_cfg.get("backbone", "")).strip().lower()
    if not backbone:
        raise RuntimeError(f"Config model.backbone missing in {ckpt_path}")
    model_family = str(cfg.get("train", {}).get("model_family", "surrogate")).lower()
    model = registry_build_model(model_family, backbone, model_cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    return model, cfg


def _load_ae_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    if "file" not in model_cfg or "class" not in model_cfg:
        raise RuntimeError(f"AE checkpoint has no model.file/class: {ckpt_path}")
    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    model = ModelClass(**(model_cfg.get("params", {}) or {}))
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    return model


def _decoder_call(ae_model: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    if hasattr(ae_model, "autoencoder") and hasattr(ae_model.autoencoder, "decode"):
        dec = ae_model.autoencoder.decode
    elif hasattr(ae_model, "decode"):
        dec = ae_model.decode
    else:
        raise RuntimeError("AE model has no decode method.")
    try:
        sig = inspect.signature(dec)
        if "noisy" in sig.parameters:
            return dec(z, noisy=False)
    except Exception:
        pass
    return dec(z)


def _resolve_dataset_cfg(
    train_cfg: Dict[str, Any], split: str, h5_override: Optional[Path]
) -> PFPairDataset:
    dcfg = train_cfg["dataloader"]
    args = dict(dcfg.get("args", {}) or {})
    split_key = f"{split}_args"
    if split_key in dcfg:
        split_args = dict((dcfg.get(split_key, {}) or {}))
    elif split == "test":
        split_args = dict((dcfg.get("val_args", {}) or {}))
    else:
        split_args = {}
    ds_args = {**args, **split_args}
    if h5_override is not None:
        ds_args["h5_path"] = str(h5_override)
    else:
        h5_map = dict((train_cfg.get("paths", {}) or {}).get("h5", {}) or {})
        if split in h5_map:
            ds_args["h5_path"] = str(h5_map[split])
        elif split == "test":
            # Many latent dev configs only declare train/val; infer test path by filename.
            cand_src = h5_map.get("val", h5_map.get("train", None))
            if cand_src is None:
                raise KeyError("Missing paths.h5.test and no train/val entry to infer from.")
            base = Path(str(cand_src))
            name = base.name
            if name.startswith("val_"):
                test_name = "test_" + name[len("val_") :]
            elif name.startswith("train_"):
                test_name = "test_" + name[len("train_") :]
            else:
                test_name = name.replace("val", "test")
            inferred = base.with_name(test_name)
            if not inferred.exists():
                raise KeyError(
                    f"Missing paths.h5.test and inferred test file not found: {inferred}"
                )
            ds_args["h5_path"] = str(inferred)
        else:
            raise KeyError(f"Missing paths.h5.{split} in config and no override provided.")
    return PFPairDataset(**ds_args)


def _split_theta(x: torch.Tensor, cond_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bool(cond_cfg.get("use_theta", False)):
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if x.size(0) <= theta_channels:
            raise ValueError(f"Input channels {x.size(0)} <= theta_channels {theta_channels}")
        theta = x[-theta_channels:, ...].unsqueeze(0)
        x_state = x[:-theta_channels, ...].unsqueeze(0)
        return x_state, theta
    return x.unsqueeze(0), None


def _extract_norm_stats(h5_path: Path, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    with h5py.File(h5_path, "r") as f:
        schema = str(f.attrs.get("normalization_schema", "")).lower()
        if "channel_mean" in f.attrs and "channel_std" in f.attrs:
            mean = torch.tensor(np.array(f.attrs["channel_mean"], dtype=np.float32), device=device).view(1, -1, 1, 1)
            std = torch.tensor(np.array(f.attrs["channel_std"], dtype=np.float32), device=device).view(1, -1, 1, 1)
            return mean, std, schema
    return None, None, ""


def _denorm_if_needed(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor], schema: str) -> torch.Tensor:
    if mean is None or std is None:
        return x
    if schema == "zscore" and x.shape[1] == mean.shape[1]:
        return x * std + mean
    return x


def _predict_flow_source_anchored(
    model: torch.nn.Module,
    x: torch.Tensor,  # [1,C,H,W]
    theta: Optional[torch.Tensor],
    nfe: int,
    flow_objective: str = "rectified_flow_source_anchored_concat",
    flow_noise_std: float = 0.0,
    flow_noise_mode: str = "scalar",
    flow_noise_perturb_source: bool = True,
) -> torch.Tensor:
    # ODE rollout from x_t(0)=x to x_t(1) using explicit Euler.
    flow_objective = str(flow_objective).strip().lower()
    sfm_objective = flow_objective in {
        "sfm_latent_source_denoise_concat",
        "sfm_latent_source_concat",
    }
    source_anchored = flow_objective.startswith("rectified_flow_source_anchored")
    noise_source_concat = flow_objective in {
        "rectified_flow_noise_source_concat",
        "rectified_flow_noise_cond_concat",
    }
    dt = 1.0 / float(nfe)
    if sfm_objective:
        sigma_eval = max(float(flow_noise_std), 0.0)
        if sigma_eval > 0.0 and flow_noise_perturb_source:
            z0 = x + torch.randn_like(x) * sigma_eval
        else:
            z0 = x
        sigma_scalar = torch.full((x.shape[0], 1), sigma_eval, device=x.device, dtype=x.dtype)
        x_t = z0
        for i in range(nfe):
            t_val = (i + 0.5) * dt
            t = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
            cond_t = torch.cat([t, sigma_scalar], dim=1)
            x_in = torch.cat([x_t, x], dim=1)
            x_hat = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
            denom = (1.0 - t).clamp_min(1e-4).view(-1, 1, 1, 1)
            v = (x_hat - x_t) / denom
            x_t = x_t + dt * v
        return x_t

    if source_anchored:
        x_t = x
        z0 = x
    else:
        z0 = torch.randn_like(x)
        x_t = z0
    use_concat = flow_objective.endswith("_concat") or source_anchored or noise_source_concat
    if flow_noise_std > 0:
        if str(flow_noise_mode).strip().lower() == "field":
            noise_field = torch.randn_like(x) * float(flow_noise_std)
            noise_scalar = noise_field.reshape(noise_field.shape[0], -1).mean(dim=1, keepdim=True)
        else:
            noise_scalar = torch.randn(x.shape[0], 1, device=x.device, dtype=x.dtype) * float(flow_noise_std)
            noise_field = noise_scalar.view(-1, 1, 1, 1).expand_as(x)
    else:
        noise_scalar = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        noise_field = torch.zeros_like(x)

    if (source_anchored or noise_source_concat) and flow_noise_perturb_source:
        z0 = z0 + noise_field
        if not source_anchored:
            x_t = z0
    for i in range(nfe):
        t_val = (i + 0.5) * dt
        t = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
        cond_t = torch.cat([t, noise_scalar], dim=1)
        if use_concat:
            source_ctx = x if noise_source_concat else z0
            x_in = torch.cat([x_t, source_ctx], dim=1)
            v = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
        else:
            v = _forward_flow_nonconcat(model=model, x_t=x_t, t_match=t, cond_t=cond_t, theta=theta)
        x_t = x_t + dt * v
    return x_t


def _predict_bridge_teacher_forced(
    model: torch.nn.Module,
    schedule,
    x: torch.Tensor,  # [1,C,H,W]
    y: torch.Tensor,  # [1,C,H,W]
    theta: Optional[torch.Tensor],
    t_index: int,
) -> torch.Tensor:
    t_index = int(max(0, min(t_index, schedule.timesteps - 1)))
    t_long = torch.full((1,), t_index, device=x.device, dtype=torch.long)
    t_model = t_long.view(-1, 1).to(dtype=x.dtype)
    a_t, b_t, c_t = _bridge_coeffs(schedule, t_long, ref=x)
    eps = torch.randn_like(y)
    x_t = a_t * x + b_t * y + c_t * eps
    x_in = _build_diffusion_model_input(
        x_noisy=x_t,
        source=x,
        noise_schedule_obj=schedule,
        sched_kind=str(getattr(schedule, "kind", "")),
    )
    y_hat = _forward_diffusion(
        model=model,
        x_in=x_in,
        t_model=t_model,
        cond=None,
        theta=theta,
        region_info=None,
    )
    return y_hat


def _predict_bridge_teacher_forced_eps(
    model: torch.nn.Module,
    schedule,
    x: torch.Tensor,  # [1,C,H,W]
    y: torch.Tensor,  # [1,C,H,W]
    theta: Optional[torch.Tensor],
    t_index: int,
) -> torch.Tensor:
    t_index = int(max(0, min(t_index, schedule.timesteps - 1)))
    t_long = torch.full((1,), t_index, device=x.device, dtype=torch.long)
    t_model = t_long.view(-1, 1).to(dtype=x.dtype)
    a_t, b_t, c_t = _bridge_coeffs(schedule, t_long, ref=x)
    eps = torch.randn_like(y)
    x_t = a_t * x + b_t * y + c_t * eps
    x_in = _build_diffusion_model_input(
        x_noisy=x_t,
        source=x,
        noise_schedule_obj=schedule,
        sched_kind=str(getattr(schedule, "kind", "")),
    )
    eps_hat = _forward_diffusion(
        model=model,
        x_in=x_in,
        t_model=t_model,
        cond=None,
        theta=theta,
        region_info=None,
    )
    y_hat = (x_t - a_t * x - c_t * eps_hat) / b_t.clamp_min(1e-6)
    return y_hat


def _bridge_coeffs(schedule, t_long: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if getattr(schedule, "kind", "") == "unidb":
        b_t = schedule._m(t_long, ref)  # target weight
        a_t = schedule._n(t_long, ref)  # source weight
        c_t = schedule.f_sigma(t_long, ref)
        return a_t, b_t, c_t
    if hasattr(schedule, "a") and hasattr(schedule, "b") and hasattr(schedule, "c"):
        a_t = schedule.a.to(ref.device)[t_long].view(-1, 1, 1, 1)
        b_t = schedule.b.to(ref.device)[t_long].view(-1, 1, 1, 1)
        c_t = schedule.c.to(ref.device)[t_long].view(-1, 1, 1, 1)
        return a_t, b_t, c_t
    raise TypeError(f"Unsupported bridge schedule type: kind={getattr(schedule, 'kind', 'unknown')}")


def _build_descending_time_grid(max_t: int, min_t: int, nfe: int) -> List[int]:
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
    x: torch.Tensor,  # [1,C,H,W]
    theta: Optional[torch.Tensor],
    nfe: int,
    eta: float,
    *,
    predict_next: bool = False,
) -> torch.Tensor:
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"bridge_rollout_dbim requires UniDB schedule, got {getattr(schedule, 'kind', '')}")
    bsz = int(x.shape[0])
    T = int(schedule.timesteps)
    eta = float(max(0.0, min(1.0, eta)))
    eval_ts = _build_descending_time_grid(max_t=max(T - 1, 1), min_t=1, nfe=max(1, int(nfe)))

    x_curr = x
    for i, s_idx in enumerate(eval_ts):
        s = torch.full((bsz,), int(s_idx), device=x.device, dtype=torch.long)
        s_model = s.view(-1, 1).to(dtype=x.dtype)
        x_in = _build_diffusion_model_input(
            x_noisy=x_curr,
            source=x,
            noise_schedule_obj=schedule,
            sched_kind=str(getattr(schedule, "kind", "")),
        )
        pred = _forward_diffusion(
            model=model,
            x_in=x_in,
            t_model=s_model,
            cond=None,
            theta=theta,
            region_info=None,
        )

        a_s, b_s, c_s = _bridge_coeffs(schedule, s, ref=x_curr)
        if predict_next:
            y_hat = pred
        else:
            y_hat = (x_curr - a_s * x - c_s * pred) / b_s.clamp_min(1e-6)

        if i == 0:
            if eta > 0.0:
                x_curr = a_s * x + b_s * y_hat + c_s * torch.randn_like(x_curr)
            else:
                x_curr = a_s * x + b_s * y_hat

        t_idx = int(eval_ts[i + 1]) if (i + 1) < len(eval_ts) else 0
        t = torch.full((bsz,), t_idx, device=x.device, dtype=torch.long)
        a_t, b_t, c_t = _bridge_coeffs(schedule, t, ref=x_curr)

        coeff_xs = c_t / c_s.clamp_min(1e-8)
        coeff_y = b_t - coeff_xs * b_s
        coeff_src = a_t - coeff_xs * a_s
        x_next = coeff_xs * x_curr + coeff_y * y_hat + coeff_src * x

        if eta > 0.0 and t_idx > 0:
            base_var = (c_t**2 - (coeff_xs * c_s) ** 2).clamp_min(0.0)
            sigma_add = eta * torch.sqrt(base_var)
            x_next = x_next + sigma_add * torch.randn_like(x_next)
        x_curr = x_next
    return x_curr


def _stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _overlay_stats(ax: plt.Axes, arr: np.ndarray, *, fontsize: int = 9) -> Dict[str, float]:
    s = _stats(arr)
    txt = f"min={s['min']:.6g}\nmax={s['max']:.6g}\nmean={s['mean']:.6g}"
    ax.text(
        0.99,
        0.99,
        txt,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65),
    )
    return s


def _rescale_to_unit(arr: np.ndarray, src_min: float, src_max: float) -> np.ndarray:
    if not np.isfinite(src_min) or not np.isfinite(src_max) or src_max <= src_min:
        return np.zeros_like(arr, dtype=np.float32)
    z = (arr - src_min) / (src_max - src_min)
    z = np.clip(z, 0.0, 1.0)
    return (z * 2.0 - 1.0).astype(np.float32, copy=False)


def _plot_panel(
    out_path: Path,
    maps: Dict[str, np.ndarray],
    *,
    value_vmin: float,
    value_vmax: float,
    residual_vabs: float,
    value_cmap: str,
    residual_cmap: str,
    title: str,
    dpi: int,
) -> Dict[str, Dict[str, float]]:
    labels = ["past", "true_next", "true_delta_next_minus_past", "pred", "residual_pred_minus_true"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), squeeze=False)
    out_stats: Dict[str, Dict[str, float]] = {}
    for j, key in enumerate(labels):
        ax = axes[0, j]
        arr = maps[key]
        is_res = key.startswith("residual") or key.startswith("true_delta")
        vmin = -residual_vabs if is_res else value_vmin
        vmax = residual_vabs if is_res else value_vmax
        cmap = residual_cmap if is_res else value_cmap
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Pin colorbar ticks so the visible labels always include plotted range endpoints.
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            cb.set_ticks(np.linspace(vmin, vmax, 5))
        cb.ax.tick_params(labelsize=9)
        out_stats[key] = _overlay_stats(ax, arr, fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_stats


def _choose_indices(n_total: int, n_samples: int, seed: int, skip_first: bool) -> List[int]:
    start = 1 if skip_first else 0
    if n_total <= start:
        return [0]
    valid = n_total - start
    n = max(1, min(n_samples, valid))
    if n == 1:
        return [start]
    rng = random.Random(seed)
    edges = np.linspace(0, valid, n + 1, dtype=int)
    out: List[int] = []
    for i in range(n):
        lo = int(edges[i])
        hi = max(lo + 1, int(edges[i + 1]))
        out.append(start + rng.randrange(lo, hi))
    return sorted(set(out))


def _parse_indices(indices: str) -> List[int]:
    out: List[int] = []
    for tok in str(indices).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def _indices_from_ae_metrics(path: Path) -> List[int]:
    obj = json.loads(path.read_text())
    rows = obj.get("results", [])
    out = []
    for r in rows:
        if "index" in r:
            out.append(int(r["index"]))
    return sorted(set(out))


def _expected_ae_from_latent_h5(h5_path: Path) -> Optional[Path]:
    meta = h5_path.parent / "README_EXPERIMENTAL.json"
    if not meta.exists():
        return None
    try:
        obj = json.loads(meta.read_text())
    except Exception:
        return None
    ckpt = obj.get("checkpoint", "")
    if not ckpt:
        return None
    return Path(str(ckpt)).expanduser().resolve()


def _assert_ae_matches_latent_source(h5_path: Path, ae_ckpt: Path) -> None:
    expected = _expected_ae_from_latent_h5(h5_path)
    if expected is None:
        return
    got = ae_ckpt.expanduser().resolve()
    if str(got) != str(expected):
        raise RuntimeError(
            "AE checkpoint does not match latent dataset provenance.\n"
            f"  latent_h5: {h5_path}\n"
            f"  expected_ae_ckpt: {expected}\n"
            f"  provided_ae_ckpt: {got}\n"
            "Use the expected checkpoint or regenerate the latent dataset with your AE."
        )


def collect_samples(
    *,
    flow_ckpt: Path,
    bridge_ckpt: Path,
    ae_ckpt: Path,
    split: str,
    max_samples: int,
    flow_nfe: int,
    bridge_mode: str,
    bridge_nfe: int,
    bridge_eta: float,
    bridge_t_index: int,
    device: torch.device,
    seed: int,
    h5_override: Optional[Path],
    indices: Optional[List[int]],
    ae_metrics_json: Optional[Path],
    skip_first: bool,
) -> Tuple[List[SamplePack], Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    flow_model, flow_cfg = _load_train_checkpoint(flow_ckpt, device)
    bridge_model, bridge_cfg = _load_train_checkpoint(bridge_ckpt, device)

    ds = _resolve_dataset_cfg(flow_cfg, split=split, h5_override=h5_override)
    h5_path = Path(ds.h5_path)
    _assert_ae_matches_latent_source(h5_path, ae_ckpt)
    ae_model = _load_ae_model(ae_ckpt, device)
    if indices:
        selected = [i for i in indices if 0 <= i < len(ds)]
    elif ae_metrics_json is not None:
        selected = [i for i in _indices_from_ae_metrics(ae_metrics_json) if 0 <= i < len(ds)]
    else:
        selected = _choose_indices(len(ds), max_samples, seed, skip_first=skip_first)
    if not selected:
        raise RuntimeError("No valid indices selected for visualization.")
    if len(selected) > max_samples:
        selected = selected[:max_samples]

    cond_cfg = dict(flow_cfg.get("conditioning", {}) or {})
    diff_cfg = dict(bridge_cfg.get("diffusion", {}) or {})
    diffusion_objective = str((bridge_cfg.get("loss", {}) or {}).get("diffusion_objective", "epsilon_mse")).lower()
    bridge_predict_next = diffusion_objective in _DIFFUSION_PREDICT_NEXT_OBJECTIVES
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    t_idx = int(max(0, min(bridge_t_index, schedule.timesteps - 1)))
    bridge_mode = str(bridge_mode).strip().lower()
    if bridge_mode not in {"teacher_forced", "rollout_dbim"}:
        raise ValueError(f"Unknown bridge_mode={bridge_mode}")

    mean, std, schema = _extract_norm_stats(h5_path, device=device)

    packs: List[SamplePack] = []
    for idx in selected:
        s = ds[idx]
        gid = str(s["gid"])
        pair_index = int(s["pair_index"])

        sample_batch = {"input": s["input"].unsqueeze(0), "target": s["target"].unsqueeze(0)}
        x_state, y_batch, _, theta = _prepare_batch(sample_batch, device, cond_cfg, use_chlast=False)
        x_state = x_state.float()
        y_batch = y_batch.float()
        if theta is not None:
            theta = theta.float()

        with torch.inference_mode():
            flow_train_cfg = dict((flow_cfg.get("flow_matching", {}) or {}))
            flow_objective = str(
                flow_cfg.get("train", {}).get("objective", "rectified_flow_source_anchored_concat")
            ).lower()
            flow_noise_std = float(
                flow_train_cfg.get("noise_stochastic_std", 0.0)
                if flow_train_cfg is not None
                else 0.0
            )
            flow_noise_mode = str(
                flow_train_cfg.get("noise_stochastic_mode", "scalar")
                if flow_train_cfg is not None
                else "scalar"
            ).strip().lower()
            flow_noise_perturb_source = bool(
                flow_train_cfg.get("noise_stochastic_perturb_source", True)
                if flow_train_cfg is not None
                else True
            )
            y_flow = _predict_flow_source_anchored(
                flow_model,
                x_state,
                theta,
                nfe=max(1, flow_nfe),
                flow_objective=flow_objective,
                flow_noise_std=flow_noise_std,
                flow_noise_mode=flow_noise_mode,
                flow_noise_perturb_source=flow_noise_perturb_source,
            )
            if bridge_mode == "teacher_forced":
                if bridge_predict_next:
                    y_bridge = _predict_bridge_teacher_forced(bridge_model, schedule, x_state, y_batch, theta, t_idx)
                else:
                    y_bridge = _predict_bridge_teacher_forced_eps(bridge_model, schedule, x_state, y_batch, theta, t_idx)
            else:
                y_bridge = _predict_bridge_rollout_dbim(
                    bridge_model,
                    schedule,
                    x_state,
                    theta,
                    nfe=max(1, bridge_nfe),
                    eta=float(bridge_eta),
                    predict_next=bridge_predict_next,
                )

            x_dec = _decoder_call(ae_model, x_state)
            y_dec = _decoder_call(ae_model, y_batch)
            flow_dec = _decoder_call(ae_model, y_flow)
            bridge_dec = _decoder_call(ae_model, y_bridge)

            x_dec = _denorm_if_needed(x_dec, mean, std, schema)
            y_dec = _denorm_if_needed(y_dec, mean, std, schema)
            flow_dec = _denorm_if_needed(flow_dec, mean, std, schema)
            bridge_dec = _denorm_if_needed(bridge_dec, mean, std, schema)

        packs.append(
            SamplePack(
                dataset_index=int(idx),
                gid=gid,
                pair_index=pair_index,
                x_dec=x_dec[0].detach().cpu().numpy(),
                y_dec=y_dec[0].detach().cpu().numpy(),
                flow_dec=flow_dec[0].detach().cpu().numpy(),
                bridge_dec=bridge_dec[0].detach().cpu().numpy(),
            )
        )

    meta = {
        "split": split,
        "n_selected": len(packs),
        "selected_indices": [int(i) for i in selected],
        "flow_ckpt": str(flow_ckpt),
        "bridge_ckpt": str(bridge_ckpt),
        "ae_ckpt": str(ae_ckpt),
        "flow_nfe": int(flow_nfe),
        "bridge_mode": bridge_mode,
        "diffusion_objective": diffusion_objective,
        "bridge_predict_next": bool(bridge_predict_next),
        "bridge_nfe": int(bridge_nfe),
        "bridge_eta": float(bridge_eta),
        "bridge_t_index": int(t_idx),
        "dataset_h5": str(h5_path),
        "normalization_schema": schema,
        "seed": int(seed),
        "selection_mode": (
            "explicit_indices" if indices else ("ae_metrics_json" if ae_metrics_json is not None else "auto")
        ),
        "ae_metrics_json": str(ae_metrics_json) if ae_metrics_json is not None else None,
        "skip_first": bool(skip_first),
    }
    return packs, meta


def _compute_scales(packs: Sequence[SamplePack]) -> Dict[str, Dict[str, float]]:
    if not packs:
        raise ValueError("No sample packs to scale.")
    scales: Dict[str, Dict[str, float]] = {
        "ch0": {"value_vmin": -1.0, "value_vmax": 1.0, "res_vabs": 1.0},
        "ch1": {"value_vmin": 0.0, "value_vmax": 1.0, "res_vabs": 1.0},
    }
    for ch in range(2):
        val_maps: List[np.ndarray] = []
        res_maps: List[np.ndarray] = []
        for p in packs:
            if p.y_dec.shape[0] <= ch:
                continue
            val_maps.extend([p.x_dec[ch], p.y_dec[ch], p.flow_dec[ch], p.bridge_dec[ch]])
            res_maps.extend([p.flow_dec[ch] - p.y_dec[ch], p.bridge_dec[ch] - p.y_dec[ch], p.y_dec[ch] - p.x_dec[ch]])

        if val_maps:
            vmin = float(min(np.nanmin(v) for v in val_maps))
            vmax = float(max(np.nanmax(v) for v in val_maps))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                scales[f"ch{ch}"]["value_vmin"] = vmin
                scales[f"ch{ch}"]["value_vmax"] = vmax
                if ch == 0:
                    # Keep raw phase bounds for pixel-space remap and plot phase in [-1, 1].
                    scales[f"ch{ch}"]["phase_src_vmin"] = vmin
                    scales[f"ch{ch}"]["phase_src_vmax"] = vmax
                    scales[f"ch{ch}"]["value_vmin"] = -1.0
                    scales[f"ch{ch}"]["value_vmax"] = 1.0

        if res_maps:
            rv = float(max(np.nanmax(np.abs(v)) for v in res_maps))
            if np.isfinite(rv) and rv > 0:
                scales[f"ch{ch}"]["res_vabs"] = rv
    return scales


def _write_summary_tables(rows: List[Dict[str, Any]], out_root: Path) -> Dict[str, str]:
    csv_path = out_root / "summary_flow_vs_bridge_per_sample_channel.csv"
    md_path = out_root / "summary_flow_vs_bridge_per_sample_channel.md"
    headers = [
        "sample_dir",
        "dataset_index",
        "gid",
        "pair_index",
        "channel",
        "flow_rmse",
        "bridge_rmse",
        "flow_mae",
        "bridge_mae",
        "true_delta_rmse",
        "flow_res_mean",
        "bridge_res_mean",
        "flow_rmse_over_true_delta",
        "bridge_rmse_over_true_delta",
        "better_model_by_rmse",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

    with md_path.open("w") as f:
        f.write(
            "|sample_dir|idx|gid|pair|ch|flow_rmse|bridge_rmse|true_delta_rmse|flow/true|bridge/true|winner|\n"
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|\n"
        )
        for r in rows:
            f.write(
                f"|{r['sample_dir']}|{r['dataset_index']}|{r['gid']}|{r['pair_index']}|{r['channel']}|"
                f"{r['flow_rmse']:.6g}|{r['bridge_rmse']:.6g}|{r['true_delta_rmse']:.6g}|"
                f"{r['flow_rmse_over_true_delta']:.6g}|{r['bridge_rmse_over_true_delta']:.6g}|"
                f"{r['better_model_by_rmse']}|\n"
            )
    return {"csv": str(csv_path.relative_to(out_root)), "md": str(md_path.relative_to(out_root))}


def write_visuals(
    packs: Sequence[SamplePack],
    out_root: Path,
    scales: Dict[str, Dict[str, float]],
    dpi: int,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"samples": []}
    table_rows: List[Dict[str, Any]] = []
    channel_names = {0: "phase", 1: "concentration"}

    for i, p in enumerate(packs):
        sample_dir = out_root / f"s{i:03d}_i{p.dataset_index:05d}_{p.gid}_p{p.pair_index:05d}"
        stats_obj: Dict[str, Any] = {
            "dataset_index": p.dataset_index,
            "gid": p.gid,
            "pair_index": p.pair_index,
            "channels": {},
        }

        for ch in range(min(2, p.y_dec.shape[0])):
            field = channel_names.get(ch, f"ch{ch}")
            ch_key = f"ch{ch}"
            val_vmin = float(scales[ch_key]["value_vmin"])
            val_vmax = float(scales[ch_key]["value_vmax"])
            res_vabs = float(scales[ch_key]["res_vabs"])
            ch_dir = sample_dir / f"{ch_key}_{field}"
            ch_dir.mkdir(parents=True, exist_ok=True)

            value_cmap = "RdBu_r" if ch == 0 else "viridis"
            residual_cmap = "seismic"

            if ch == 0:
                src_min = float(scales[ch_key].get("phase_src_vmin", np.nan))
                src_max = float(scales[ch_key].get("phase_src_vmax", np.nan))
                x_plot = _rescale_to_unit(p.x_dec[ch], src_min, src_max)
                y_plot = _rescale_to_unit(p.y_dec[ch], src_min, src_max)
                flow_plot = _rescale_to_unit(p.flow_dec[ch], src_min, src_max)
                bridge_plot = _rescale_to_unit(p.bridge_dec[ch], src_min, src_max)
            else:
                x_plot = p.x_dec[ch]
                y_plot = p.y_dec[ch]
                flow_plot = p.flow_dec[ch]
                bridge_plot = p.bridge_dec[ch]

            flow_maps = {
                "past": x_plot,
                "true_next": y_plot,
                "true_delta_next_minus_past": y_plot - x_plot,
                "pred": flow_plot,
                "residual_pred_minus_true": flow_plot - y_plot,
            }
            bridge_maps = {
                "past": x_plot,
                "true_next": y_plot,
                "true_delta_next_minus_past": y_plot - x_plot,
                "pred": bridge_plot,
                "residual_pred_minus_true": bridge_plot - y_plot,
            }

            flow_panel = ch_dir / "flow.png"
            bridge_panel = ch_dir / "bridge.png"

            flow_stats = _plot_panel(
                flow_panel,
                flow_maps,
                value_vmin=val_vmin,
                value_vmax=val_vmax,
                residual_vabs=res_vabs,
                value_cmap=value_cmap,
                residual_cmap=residual_cmap,
                title=f"{field} | FLOW | idx={p.dataset_index} gid={p.gid} pair={p.pair_index}",
                dpi=dpi,
            )
            bridge_stats = _plot_panel(
                bridge_panel,
                bridge_maps,
                value_vmin=val_vmin,
                value_vmax=val_vmax,
                residual_vabs=res_vabs,
                value_cmap=value_cmap,
                residual_cmap=residual_cmap,
                title=f"{field} | BRIDGE | idx={p.dataset_index} gid={p.gid} pair={p.pair_index}",
                dpi=dpi,
            )

            flow_res = flow_maps["residual_pred_minus_true"]
            bridge_res = bridge_maps["residual_pred_minus_true"]
            true_delta = flow_maps["true_delta_next_minus_past"]
            flow_rmse = float(np.sqrt(np.mean(flow_res**2)))
            bridge_rmse = float(np.sqrt(np.mean(bridge_res**2)))
            flow_mae = float(np.mean(np.abs(flow_res)))
            bridge_mae = float(np.mean(np.abs(bridge_res)))
            true_delta_rmse = float(np.sqrt(np.mean(true_delta**2)))
            denom = max(true_delta_rmse, 1e-12)

            ch_stats = {
                "scales": {
                    "value_vmin": val_vmin,
                    "value_vmax": val_vmax,
                    "residual_vmin": -res_vabs,
                    "residual_vmax": res_vabs,
                    "phase_src_vmin": float(scales[ch_key].get("phase_src_vmin")) if ch == 0 and "phase_src_vmin" in scales[ch_key] else None,
                    "phase_src_vmax": float(scales[ch_key].get("phase_src_vmax")) if ch == 0 and "phase_src_vmax" in scales[ch_key] else None,
                },
                "flow_panel": {"file": str(flow_panel.relative_to(out_root)), "maps": flow_stats},
                "bridge_panel": {"file": str(bridge_panel.relative_to(out_root)), "maps": bridge_stats},
                "metrics": {
                    "flow_rmse": flow_rmse,
                    "bridge_rmse": bridge_rmse,
                    "flow_mae": flow_mae,
                    "bridge_mae": bridge_mae,
                    "true_delta_rmse": true_delta_rmse,
                    "flow_res_mean": float(np.mean(flow_res)),
                    "bridge_res_mean": float(np.mean(bridge_res)),
                    "flow_rmse_over_true_delta": flow_rmse / denom,
                    "bridge_rmse_over_true_delta": bridge_rmse / denom,
                },
            }
            stats_obj["channels"][ch_key] = ch_stats
            table_rows.append(
                {
                    "sample_dir": str(sample_dir.relative_to(out_root)),
                    "dataset_index": int(p.dataset_index),
                    "gid": p.gid,
                    "pair_index": int(p.pair_index),
                    "channel": ch_key,
                    "flow_rmse": flow_rmse,
                    "bridge_rmse": bridge_rmse,
                    "flow_mae": flow_mae,
                    "bridge_mae": bridge_mae,
                    "true_delta_rmse": true_delta_rmse,
                    "flow_res_mean": float(np.mean(flow_res)),
                    "bridge_res_mean": float(np.mean(bridge_res)),
                    "flow_rmse_over_true_delta": flow_rmse / denom,
                    "bridge_rmse_over_true_delta": bridge_rmse / denom,
                    "better_model_by_rmse": "flow" if flow_rmse <= bridge_rmse else "bridge",
                }
            )

        (sample_dir / "stats.json").write_text(json.dumps(stats_obj, indent=2) + "\n")
        summary["samples"].append(
            {
                "dataset_index": p.dataset_index,
                "gid": p.gid,
                "pair_index": p.pair_index,
                "dir": str(sample_dir.relative_to(out_root)),
            }
        )

    summary["tables"] = _write_summary_tables(table_rows, out_root)
    return summary


def _default_flow_ckpt() -> Path:
    return Path(
        "/scratch/project_2008261/pf_surrogate_modelling/runs/"
        "flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth"
    )


def _default_bridge_ckpt() -> Path:
    return Path(
        "/scratch/project_2008261/pf_surrogate_modelling/runs/"
        "diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8/UNetFiLMAttn/checkpoint.best.pth"
    )


def _default_ae_ckpt() -> Path:
    return _CANONICAL_AE_CKPT


def main() -> None:
    ap = argparse.ArgumentParser(description="Create AE-style per-channel flow vs bridge decoded visuals.")
    ap.add_argument("--flow-ckpt", type=Path, default=_default_flow_ckpt())
    ap.add_argument("--bridge-ckpt", type=Path, default=_default_bridge_ckpt())
    ap.add_argument("--ae-ckpt", type=Path, default=_default_ae_ckpt())
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--indices", type=str, default="")
    ap.add_argument("--ae-metrics-json", type=Path, default=None)
    ap.add_argument("--skip-first", action="store_true")
    ap.add_argument("--flow-nfe", type=int, default=4)
    ap.add_argument("--bridge-mode", type=str, default="rollout_dbim", choices=["rollout_dbim", "teacher_forced"])
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--bridge-eta", type=float, default=0.0)
    ap.add_argument("--bridge-t-index", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--h5-override", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (ROOT / "results" / "visuals" / f"flow_bridge_compare_{args.split}_{stamp}")
    )

    explicit_indices = _parse_indices(args.indices) if args.indices.strip() else None
    ae_metrics_json = args.ae_metrics_json.expanduser().resolve() if args.ae_metrics_json else None
    if ae_metrics_json is not None and not ae_metrics_json.exists():
        raise FileNotFoundError(f"AE metrics json not found: {ae_metrics_json}")

    packs, meta = collect_samples(
        flow_ckpt=args.flow_ckpt.expanduser().resolve(),
        bridge_ckpt=args.bridge_ckpt.expanduser().resolve(),
        ae_ckpt=args.ae_ckpt.expanduser().resolve(),
        split=str(args.split),
        max_samples=int(args.max_samples),
        flow_nfe=int(args.flow_nfe),
        bridge_mode=str(args.bridge_mode),
        bridge_nfe=int(args.bridge_nfe),
        bridge_eta=float(args.bridge_eta),
        bridge_t_index=int(args.bridge_t_index),
        device=device,
        seed=int(args.seed),
        h5_override=(args.h5_override.expanduser().resolve() if args.h5_override else None),
        indices=explicit_indices,
        ae_metrics_json=ae_metrics_json,
        skip_first=bool(args.skip_first),
    )
    scales = _compute_scales(packs)
    summary = write_visuals(packs, out_root=out_dir, scales=scales, dpi=int(args.dpi))

    manifest = {
        "created_utc": stamp,
        "meta": meta,
        "scales": scales,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] visuals written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
