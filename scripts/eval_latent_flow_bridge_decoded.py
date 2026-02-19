#!/usr/bin/env python3
"""
Evaluate latent flow/bridge checkpoints in decoded PDE field space.

This script is intended for development diagnostics:
- Flow matching: rollout from source latent x -> predicted target latent y_hat.
- Diffusion bridge:
  - teacher-forced denoising diagnostic (x_t built from x, y),
  - rollout diagnostic via a UniDB-adapted DBIM-style reverse update.

Outputs aggregate metrics (latent RMSE, decoded RMSE/MAE, mass/interface metrics).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.diffusion_forward import _build_diffusion_model_input
from models.train.core.loss_functions import (
    curvature_levelset,
    edge_gradient_strength,
    interface_perimeter,
    relative_mass_error,
)
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
class EvalResult:
    count: int = 0
    latent_rmse_sum: float = 0.0
    latent_mae_sum: float = 0.0
    decoded_rmse_sum: float = 0.0
    decoded_mae_sum: float = 0.0
    mass_rel_ch0_sum: float = 0.0
    mass_rel_ch1_sum: float = 0.0
    curv_mean_rel_sum: float = 0.0
    perim_rel_sum: float = 0.0
    edge_strength_rel_sum: float = 0.0
    flow_ensemble_std_sum: float = 0.0
    flow_ensemble_count: int = 0
    flow_ensemble_sample_count: int = 0

    def update(
        self,
        y_lat_pred: torch.Tensor,
        y_lat_true: torch.Tensor,
        y_dec_pred: torch.Tensor,
        y_dec_true: torch.Tensor,
    ) -> None:
        bsz = int(y_lat_true.shape[0])
        self.count += bsz

        lat_mse = torch.mean((y_lat_pred - y_lat_true) ** 2, dim=(1, 2, 3)).sqrt()
        lat_mae = torch.mean(torch.abs(y_lat_pred - y_lat_true), dim=(1, 2, 3))
        dec_mse = torch.mean((y_dec_pred - y_dec_true) ** 2, dim=(1, 2, 3)).sqrt()
        dec_mae = torch.mean(torch.abs(y_dec_pred - y_dec_true), dim=(1, 2, 3))
        self.latent_rmse_sum += float(lat_mse.sum().item())
        self.latent_mae_sum += float(lat_mae.sum().item())
        self.decoded_rmse_sum += float(dec_mse.sum().item())
        self.decoded_mae_sum += float(dec_mae.sum().item())

        if y_dec_true.shape[1] >= 1:
            m0 = relative_mass_error(y_dec_pred[:, 0], y_dec_true[:, 0]).view(-1)
            self.mass_rel_ch0_sum += float(m0.sum().item())
        if y_dec_true.shape[1] >= 2:
            m1 = relative_mass_error(y_dec_pred[:, 1], y_dec_true[:, 1]).view(-1)
            self.mass_rel_ch1_sum += float(m1.sum().item())

        # Geometry metrics on channel 0 as phase-field proxy.
        if y_dec_true.shape[1] > 0:
            ph_pred = y_dec_pred[:, 0]
            ph_true = y_dec_true[:, 0]
            level = 0.5
            eps_band = 0.02
            eps_delta = 0.02
            px = 1.0
            py = 1.0

            k_pred, m_pred = curvature_levelset(ph_pred, level=level, eps_band=eps_band)
            k_true, m_true = curvature_levelset(ph_true, level=level, eps_band=eps_band)
            mu_k_pred = k_pred.abs()[m_pred].mean() if m_pred.any() else k_pred.abs().mean()
            mu_k_true = k_true.abs()[m_true].mean() if m_true.any() else k_true.abs().mean()
            rel_curv = (mu_k_pred - mu_k_true).abs() / mu_k_true.abs().clamp_min(1e-12)
            self.curv_mean_rel_sum += float(rel_curv.item()) * bsz

            p_pred = interface_perimeter(ph_pred, level=level, eps_delta=eps_delta, pixel_size=(px, py))
            p_true = interface_perimeter(ph_true, level=level, eps_delta=eps_delta, pixel_size=(px, py))
            rel_perim = (p_pred - p_true).abs() / p_true.abs().clamp_min(1e-12)
            self.perim_rel_sum += float(rel_perim.item()) * bsz

            es_pred = edge_gradient_strength(ph_pred, level=level, eps_band=eps_band)
            es_true = edge_gradient_strength(ph_true, level=level, eps_band=eps_band)
            rel_es = (es_pred - es_true).abs() / es_true.abs().clamp_min(1e-12)
            self.edge_strength_rel_sum += float(rel_es.item()) * bsz

    def as_dict(self) -> Dict[str, float]:
        n = max(self.count, 1)
        flow_ensemble_latent_std = float("nan")
        if self.flow_ensemble_count > 0:
            flow_ensemble_latent_std = self.flow_ensemble_std_sum / float(self.flow_ensemble_count)
        return {
            "samples": int(self.count),
            "latent_rmse": self.latent_rmse_sum / n,
            "latent_mae": self.latent_mae_sum / n,
            "decoded_rmse": self.decoded_rmse_sum / n,
            "decoded_mae": self.decoded_mae_sum / n,
            "mass_rel_ch0": self.mass_rel_ch0_sum / n,
            "mass_rel_ch1": self.mass_rel_ch1_sum / n if self.mass_rel_ch1_sum > 0 else float("nan"),
            "curv_mean_rel": self.curv_mean_rel_sum / n,
            "perim_rel": self.perim_rel_sum / n,
            "edge_strength_rel": self.edge_strength_rel_sum / n,
            "flow_ensemble_latent_std": flow_ensemble_latent_std,
            "flow_ensemble_num_samples": int(self.flow_ensemble_sample_count),
        }

    def update_flow_ensemble(self, y_lat_samples: List[torch.Tensor], y_lat_true: torch.Tensor) -> None:
        if not y_lat_samples or len(y_lat_samples) <= 1:
            return
        y_lat_stack = torch.stack(y_lat_samples, dim=0)
        sample_std = y_lat_stack.std(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        self.flow_ensemble_std_sum += float(sample_std.sum().item())
        self.flow_ensemble_count += int(y_lat_true.shape[0])
        self.flow_ensemble_sample_count = len(y_lat_samples)


def _load_train_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
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
    return model, cfg, ckpt


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


def _coerce_dataset_max_items(value: Any) -> Optional[int | float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"none", "null"}:
        return None
    if any(ch in s.lower() for ch in [".", "e"]):
        return float(s)
    return int(s)


def _resolve_dataset_cfg(
    train_cfg: Dict[str, Any],
    split: str,
    override_h5: Optional[Path],
    dataset_clear_caps: bool = False,
    dataset_limit_per_group: Optional[int] = None,
    dataset_max_items: Optional[int | float] = None,
) -> Tuple[PFPairDataset, Dict[str, Any]]:
    dcfg = train_cfg["dataloader"]
    args = dict(dcfg.get("args", {}) or {})
    split_args = dict((dcfg.get(f"{split}_args", {}) or {}))
    ds_args = {**args, **split_args}
    if dataset_clear_caps:
        ds_args["limit_per_group"] = None
        ds_args["max_items"] = None
    if dataset_limit_per_group is not None:
        ds_args["limit_per_group"] = int(dataset_limit_per_group)
    if dataset_max_items is not None:
        ds_args["max_items"] = dataset_max_items
    h5_path = str(override_h5) if override_h5 is not None else str(train_cfg["paths"]["h5"][split])
    ds = PFPairDataset(h5_path=h5_path, **ds_args)
    return ds, {"h5_path": h5_path, "dataset_args": ds_args}


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


def _split_theta(x: torch.Tensor, cond_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bool(cond_cfg.get("use_theta", False)):
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if x.size(1) <= theta_channels:
            raise ValueError(f"Input channels {x.size(1)} <= theta_channels {theta_channels}")
        theta = x[:, -theta_channels:, ...]
        x_state = x[:, :-theta_channels, ...]
        return x_state, theta
    return x, None


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
    # The source datasets are marked zscore; decode outputs live in that space.
    if schema == "zscore":
        if x.shape[1] != mean.shape[1]:
            return x
        return x * std + mean
    return x


def _build_flow_noise(
    ref_tensor: torch.Tensor,
    noise_std: float,
    mode: str,
    perturb_source: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(noise_std) <= 0:
        return torch.zeros_like(ref_tensor), torch.zeros(
            (ref_tensor.shape[0], 1), device=ref_tensor.device, dtype=ref_tensor.dtype
        )

    mode = str(mode).strip().lower()
    if mode not in {"scalar", "field"}:
        mode = "scalar"

    if mode == "field":
        noise_field = torch.randn_like(ref_tensor) * float(noise_std)
        noise_scalar = noise_field.reshape(noise_field.shape[0], -1).mean(dim=1, keepdim=True)
    else:
        noise_scalar = torch.randn(ref_tensor.shape[0], 1, device=ref_tensor.device, dtype=ref_tensor.dtype) * float(noise_std)
        noise_field = noise_scalar.view(-1, 1, 1, 1).expand_as(ref_tensor)

    if not perturb_source:
        noise_field = torch.zeros_like(ref_tensor)

    return noise_field, noise_scalar


def _predict_flow_source_anchored(
    model: torch.nn.Module,
    x: torch.Tensor,
    theta: Optional[torch.Tensor],
    nfe: int,
    flow_objective: str,
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
    flow_noise_field, flow_noise_scalar = _build_flow_noise(
        ref_tensor=x,
        noise_std=float(flow_noise_std),
        mode=flow_noise_mode,
        perturb_source=flow_noise_perturb_source,
    )
    if (source_anchored or noise_source_concat) and flow_noise_perturb_source:
        z0 = z0 + flow_noise_field
        if not source_anchored:
            x_t = z0
    for i in range(nfe):
        t_val = (i + 0.5) * dt
        t = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
        if flow_noise_scalar is not None:
            cond_t = torch.cat([t, flow_noise_scalar], dim=1)
        else:
            cond_t = t
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
    x: torch.Tensor,
    y: torch.Tensor,
    theta: Optional[torch.Tensor],
    t_index: int,
) -> torch.Tensor:
    # Teacher-forced bridge diagnostic for predict-next objective:
    # x_t = a_t*x + b_t*y + c_t*eps ; predict y_hat directly.
    bsz = x.shape[0]
    t_index = int(max(0, min(t_index, schedule.timesteps - 1)))
    t_long = torch.full((bsz,), t_index, device=x.device, dtype=torch.long)
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
    x: torch.Tensor,
    y: torch.Tensor,
    theta: Optional[torch.Tensor],
    t_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Teacher-forced bridge denoising diagnostic for epsilon/noise objective:
    # x_t = a_t*x + b_t*y + c_t*eps ; predict eps_hat ; solve y_hat.
    bsz = x.shape[0]
    t_index = int(max(0, min(t_index, schedule.timesteps - 1)))
    t_long = torch.full((bsz,), t_index, device=x.device, dtype=torch.long)
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
    return y_hat, eps_hat, eps


def _bridge_coeffs(schedule, t_long: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return bridge coefficients in x_t = a_t * source + b_t * target + c_t * eps form.

    Supports:
    - BridgeSchedule-like objects exposing a/b/c tensors.
    - UniDBSchedule where source weight is n(t), target weight is m(t), noise is f_sigma(t).
    """
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
    if getattr(schedule, "kind", "") == "vp":
        if not hasattr(schedule, "alpha_bar"):
            raise TypeError("vp schedule is missing alpha_bar for bridge-style coefficient reconstruction.")
        alpha_bar = schedule.alpha_bar.to(ref.device)
        alpha_t = alpha_bar.to(dtype=ref.dtype)[t_long]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        a_t = torch.zeros_like(alpha_t)
        b_t = torch.sqrt(alpha_t)
        c_t = torch.sqrt((1.0 - alpha_t).clamp_min(0.0))
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
    x: torch.Tensor,
    theta: Optional[torch.Tensor],
    nfe: int,
    eta: float,
    *,
    predict_next: bool = False,
) -> torch.Tensor:
    """
    UniDB-adapted DBIM-style rollout from source x -> predicted target y_hat.

    This uses one model evaluation per selected bridge time. The reverse update
    follows the deterministic DBIM coefficient structure; eta>0 injects extra
    stochasticity into intermediate steps.
    """
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"bridge_rollout_dbim currently requires UniDB schedule, got {getattr(schedule, 'kind', '')}")
    bsz = x.shape[0]
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


def _build_loader(ds: PFPairDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def evaluate(
    model_ckpt: Path,
    ae_ckpt: Path,
    split: str,
    mode: str,
    device: torch.device,
    max_batches: int,
    batch_size: int,
    num_workers: int,
    flow_nfe: int,
    bridge_nfe: int,
    bridge_eta: float,
    bridge_t_index: Optional[int],
    flow_num_samples: int,
    flow_noise_std: float,
    flow_noise_mode: str,
    flow_noise_perturb_source: bool,
    h5_override: Optional[Path],
    dataset_clear_caps: bool,
    dataset_limit_per_group: Optional[int],
    dataset_max_items: Optional[int | float],
) -> Dict[str, Any]:
    model, cfg, _ = _load_train_checkpoint(model_ckpt, device)
    diffusion_objective = str((cfg.get("loss", {}) or {}).get("diffusion_objective", "epsilon_mse")).lower()
    bridge_predict_next = diffusion_objective in _DIFFUSION_PREDICT_NEXT_OBJECTIVES
    flow_objective = str(cfg.get("train", {}).get("objective", "rectified_flow_source_anchored_concat")).lower()
    flow_cfg = dict(cfg.get("flow_matching", {}) or {})

    if flow_noise_std < 0:
        flow_noise_std = float(flow_cfg.get("noise_stochastic_std", 0.0))
    if not flow_noise_mode:
        flow_noise_mode = str(flow_cfg.get("noise_stochastic_mode", "scalar")).strip().lower()
    flow_noise_perturb_source = bool(flow_cfg.get("noise_stochastic_perturb_source", flow_noise_perturb_source))
    ds, ds_meta = _resolve_dataset_cfg(
        cfg,
        split=split,
        override_h5=h5_override,
        dataset_clear_caps=dataset_clear_caps,
        dataset_limit_per_group=dataset_limit_per_group,
        dataset_max_items=dataset_max_items,
    )
    h5_for_norm = Path(ds_meta["h5_path"])
    _assert_ae_matches_latent_source(h5_for_norm, ae_ckpt)
    ae_model = _load_ae_model(ae_ckpt, device)
    dl = _build_loader(ds, batch_size=batch_size, num_workers=num_workers)

    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    family = str(cfg.get("train", {}).get("model_family", "surrogate")).lower()
    if family == "flow_matching":
        sfm_objectives = {"sfm_latent_source_denoise_concat", "sfm_latent_source_concat"}
        model_params = dict((cfg.get("model", {}) or {}).get("params", {}) or {})
        cond_dim = int(model_params.get("cond_dim", 1))
        needs_noise_scalar = bool(flow_objective in sfm_objectives) or (float(flow_noise_std) > 0.0)
        if needs_noise_scalar and cond_dim < 2:
            raise ValueError(
                "Flow stochastic evaluation requested, but checkpoint/config uses cond_dim<2.\n"
                f"Got cond_dim={cond_dim}, expected >=2 to consume [t, noise] conditioning.\n"
                "Use a stochastic-trained flow config/checkpoint with model.params.cond_dim=2."
            )
    if mode == "auto":
        mode = "flow_rollout" if family == "flow_matching" else "bridge_rollout_dbim"

    schedule = None
    if mode in {"bridge_teacher_forced", "bridge_rollout_dbim"}:
        diff_cfg = dict(cfg.get("diffusion", {}) or {})
        schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
        if bridge_t_index is None:
            bridge_t_index = int(0.5 * max(schedule.timesteps - 1, 1))

    mean, std, schema = _extract_norm_stats(h5_for_norm, device=device)

    result = EvalResult()
    eps_rmse_sum = 0.0
    eps_rmse_count = 0

    with torch.inference_mode():
        for bidx, batch in enumerate(dl):
            if max_batches > 0 and bidx >= max_batches:
                break

            x_state, y, _, theta = _prepare_batch(batch, device, cond_cfg, use_chlast=False)
            x_state = x_state.float()
            y = y.float()
            if theta is not None:
                theta = theta.float()

            if mode == "flow_rollout":
                flow_ns = max(1, int(flow_num_samples))
                y_lat_samples: List[torch.Tensor] = []
                for _ in range(flow_ns):
                    y_lat_samples.append(
                        _predict_flow_source_anchored(
                            model=model,
                            x=x_state,
                            theta=theta,
                            nfe=max(1, flow_nfe),
                            flow_objective=flow_objective,
                            flow_noise_std=float(flow_noise_std),
                            flow_noise_mode=flow_noise_mode,
                            flow_noise_perturb_source=flow_noise_perturb_source,
                        )
                    )
                y_lat_pred = y_lat_samples[0]
                if flow_ns > 1:
                    y_lat_pred = torch.stack(y_lat_samples, dim=0).mean(dim=0)
            elif mode == "bridge_teacher_forced":
                if schedule is None or bridge_t_index is None:
                    raise RuntimeError("Bridge schedule not initialized.")
                if bridge_predict_next:
                    y_lat_pred = _predict_bridge_teacher_forced(
                        model=model,
                        schedule=schedule,
                        x=x_state,
                        y=y,
                        theta=theta,
                        t_index=bridge_t_index,
                    )
                else:
                    y_lat_pred, eps_hat, eps_true = _predict_bridge_teacher_forced_eps(
                        model=model,
                        schedule=schedule,
                        x=x_state,
                        y=y,
                        theta=theta,
                        t_index=bridge_t_index,
                    )
                    # Diagnostic: objective-space epsilon RMSE.
                    eps_rmse = torch.mean((eps_hat - eps_true) ** 2, dim=(1, 2, 3)).sqrt()
                    eps_rmse_sum += float(eps_rmse.sum().item())
                    eps_rmse_count += int(eps_rmse.shape[0])
            elif mode == "bridge_rollout_dbim":
                if schedule is None:
                    raise RuntimeError("Bridge schedule not initialized.")
                y_lat_pred = _predict_bridge_rollout_dbim(
                    model=model,
                    schedule=schedule,
                    x=x_state,
                    theta=theta,
                    nfe=max(1, bridge_nfe),
                    eta=float(bridge_eta),
                    predict_next=bridge_predict_next,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            y_dec_pred = _decoder_call(ae_model, y_lat_pred)
            y_dec_true = _decoder_call(ae_model, y)
            y_dec_pred = _denorm_if_needed(y_dec_pred, mean, std, schema)
            y_dec_true = _denorm_if_needed(y_dec_true, mean, std, schema)

            result.update(y_lat_pred, y, y_dec_pred, y_dec_true)
            if mode == "flow_rollout":
                result.update_flow_ensemble(y_lat_samples, y)

    out = {
        "model_ckpt": str(model_ckpt),
        "ae_ckpt": str(ae_ckpt),
        "mode": mode,
        "split": split,
        "family": family,
        "diffusion_objective": diffusion_objective,
        "bridge_predict_next": bool(bridge_predict_next),
        "max_batches": int(max_batches),
        "batch_size": int(batch_size),
        "flow_nfe": int(flow_nfe),
        "flow_num_samples": int(max(1, int(flow_num_samples))),
        "flow_noise_std": float(flow_noise_std),
        "flow_noise_mode": str(flow_noise_mode),
        "flow_noise_perturb_source": bool(flow_noise_perturb_source),
        "flow_objective": flow_objective,
        "bridge_nfe": int(bridge_nfe),
        "bridge_eta": float(bridge_eta),
        "bridge_t_index": int(bridge_t_index) if bridge_t_index is not None else None,
        "dataset": ds_meta,
        "metrics": result.as_dict(),
    }
    if eps_rmse_count > 0:
        out["metrics"]["bridge_eps_rmse_proxy"] = eps_rmse_sum / max(eps_rmse_count, 1)
    return out


def _default_ae_ckpt() -> Path:
    return _CANONICAL_AE_CKPT


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate latent flow/bridge checkpoints in decoded PDE space.")
    ap.add_argument("--model-ckpt", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, default=_default_ae_ckpt())
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "flow_rollout", "bridge_teacher_forced", "bridge_rollout_dbim"],
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--flow-nfe", type=int, default=8)
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--bridge-eta", type=float, default=0.0)
    ap.add_argument("--bridge-t-index", type=int, default=-1)
    ap.add_argument("--flow-num-samples", type=int, default=1)
    ap.add_argument("--flow-noise-std", type=float, default=-1.0)
    ap.add_argument("--flow-noise-mode", type=str, default="")
    ap.add_argument("--flow-noise-perturb-source", action="store_true")
    ap.add_argument("--h5-override", type=Path, default=None)
    ap.add_argument("--dataset-clear-caps", action="store_true")
    ap.add_argument("--dataset-limit-per-group", type=int, default=None)
    ap.add_argument("--dataset-max-items", type=str, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    dataset_max_items = _coerce_dataset_max_items(args.dataset_max_items)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    bridge_t_index = None if int(args.bridge_t_index) < 0 else int(args.bridge_t_index)
    out = evaluate(
        model_ckpt=args.model_ckpt.expanduser().resolve(),
        ae_ckpt=args.ae_ckpt.expanduser().resolve(),
        split=str(args.split),
        mode=str(args.mode),
        device=device,
        max_batches=int(args.max_batches),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        flow_nfe=int(args.flow_nfe),
        bridge_nfe=int(args.bridge_nfe),
        bridge_eta=float(args.bridge_eta),
        bridge_t_index=bridge_t_index,
        flow_num_samples=int(args.flow_num_samples),
        flow_noise_std=float(args.flow_noise_std),
        flow_noise_mode=str(args.flow_noise_mode),
        flow_noise_perturb_source=bool(args.flow_noise_perturb_source),
        h5_override=(args.h5_override.expanduser().resolve() if args.h5_override else None),
        dataset_clear_caps=bool(args.dataset_clear_caps),
        dataset_limit_per_group=args.dataset_limit_per_group,
        dataset_max_items=dataset_max_items,
    )
    text = json.dumps(out, indent=2)
    print(text, flush=True)
    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
