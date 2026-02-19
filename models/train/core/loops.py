"""
Train/validation loop helpers and per-epoch strategies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
import math
from typing import Any, Dict, Optional, Tuple

import os
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from models.train.core.loss_functions import spectral_rmse_bands
from models.train.core.diffusion_forward import _build_diffusion_model_input, _sample_diffusion_noisy_pair
from models.train.core.metric_stats import _compute_vrmse_from_stats, _init_channel_stats, _update_channel_stats
from models.train.core.utils import (
    _prepare_batch,
    _allreduce_sum_count,
    _allreduce_sum_tensor,
    _all_gather_object,
    _rank0,
)
from models.train.core.latent import encode_latent_pair, split_latent_fields, select_latent_fields


def _pairs_per_gid_from_dataset(ds):
    if hasattr(ds, "items"):
        return Counter(g for g, _ in ds.items)
    from torch.utils.data import Subset

    if isinstance(ds, Subset) and hasattr(ds.dataset, "items"):
        base_items = ds.dataset.items
        idxs = ds.indices
        return Counter(base_items[i][0] for i in idxs)
    return None


def _iter_train_batches(train_dl, steps_per_epoch: Optional[int]):
    if steps_per_epoch is None or steps_per_epoch <= 0:
        for batch in train_dl:
            yield batch
        return
    steps = int(steps_per_epoch)
    train_iter = iter(train_dl)
    for _ in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            try:
                batch = next(train_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Train DataLoader produced zero batches. "
                    "This usually means dataset_size < batch_size with loader.drop_last=true. "
                    "Use a smaller batch size, disable drop_last, or increase dataset size."
                ) from exc
        yield batch


def _first_nonfinite_named(named_params, use_grad: bool = False):
    for name, param in named_params:
        tensor = param.grad if use_grad else param
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            return name, tensor
    return None, None


def _find_named_param(model, needle: Optional[str]):
    if needle is None:
        return None, None
    needle = str(needle).strip()
    if not needle:
        return None, None
    for name, param in model.named_parameters():
        if name == needle or name.endswith(needle):
            return name, param
    return None, None


def _tensor_stats(t: torch.Tensor, name: str) -> Dict[str, Any]:
    t_det = t.detach()
    finite_mask = torch.isfinite(t_det)
    finite_cnt = int(finite_mask.sum().item())
    total_cnt = t_det.numel()
    if finite_cnt > 0:
        finite_min = float(t_det[finite_mask].min().item())
        finite_max = float(t_det[finite_mask].max().item())
    else:
        finite_min = float("nan")
        finite_max = float("nan")
    return {
        "name": name,
        "finite": f"{finite_cnt}/{total_cnt}",
        "min": finite_min,
        "max": finite_max,
        "has_nan": bool(torch.isnan(t_det).any().item()),
        "has_inf": bool(torch.isinf(t_det).any().item()),
    }


def _grad_norm_or_clip(params, grad_clip: float) -> torch.Tensor:
    grads = [p.grad for p in params if torch.is_tensor(p.grad)]
    if not grads:
        return torch.tensor(float("nan"), device=params[0].device)
    if grad_clip > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))


def _resize_theta_to_match(theta: Optional[torch.Tensor], ref: torch.Tensor, use_chlast: bool) -> Optional[torch.Tensor]:
    if theta is None:
        return None
    if theta.shape[-2:] != ref.shape[-2:]:
        src_h, src_w = theta.shape[-2], theta.shape[-1]
        dst_h, dst_w = ref.shape[-2], ref.shape[-1]
        mode = "area" if (src_h > dst_h or src_w > dst_w) else "bilinear"
        if mode == "area":
            theta = F.interpolate(theta, size=(dst_h, dst_w), mode=mode)
        else:
            theta = F.interpolate(theta, size=(dst_h, dst_w), mode=mode, align_corners=False)
    if use_chlast:
        theta = theta.contiguous(memory_format=torch.channels_last)
    return theta


def _maybe_select_latent_fields(x: torch.Tensor, y: torch.Tensor, latent_cfg, model_family: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        latent_cfg
        and bool(latent_cfg.get("split_fields", False))
        and bool(latent_cfg.get("drop_thermal_target", False))
        and model_family in {"diffusion", "flow_matching"}
    ):
        return x, y
    num_fields = int(latent_cfg.get("num_fields", 3))
    cond_fields = [int(i) for i in latent_cfg.get("cond_fields", list(range(num_fields)))]
    target_fields = [int(i) for i in latent_cfg.get("target_fields", [0, 1])]
    x_fields = split_latent_fields(x, num_fields)
    y_fields = split_latent_fields(y, num_fields)
    x = select_latent_fields(x_fields, cond_fields)
    y = select_latent_fields(y_fields, target_fields)
    return x, y


def _placeholder_channels(t: torch.Tensor, out_ch: int) -> torch.Tensor:
    if out_ch <= 0:
        raise ValueError(f"placeholder channels must be > 0 (got {out_ch}).")
    if t.shape[1] == out_ch:
        return t
    reps = (out_ch + t.shape[1] - 1) // t.shape[1]
    return t.repeat(1, reps, 1, 1)[:, :out_ch, ...]


def _apply_placeholder_latent(
    x: torch.Tensor,
    y: torch.Tensor,
    latent_cfg,
    model_family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = dict(latent_cfg or {})
    ph = cfg.get("placeholder", {}) if isinstance(cfg, dict) else {}
    if not isinstance(ph, dict) or not bool(ph.get("enabled", False)):
        return x, y
    if model_family not in {"diffusion", "flow_matching"}:
        return x, y

    out_ch = int(ph.get("channels", y.shape[1]))
    spatial_size = ph.get("spatial_size", None)
    mode = str(ph.get("mode", "bilinear")).lower()
    if mode not in {"nearest", "bilinear", "bicubic", "area"}:
        mode = "bilinear"

    target_hw = None
    if spatial_size is not None:
        if isinstance(spatial_size, int):
            target_hw = (int(spatial_size), int(spatial_size))
        elif isinstance(spatial_size, (list, tuple)) and len(spatial_size) == 2:
            target_hw = (int(spatial_size[0]), int(spatial_size[1]))
    if target_hw is not None and (x.shape[-2:] != target_hw or y.shape[-2:] != target_hw):
        interp_kwargs = {"mode": mode}
        if mode in {"bilinear", "bicubic"}:
            interp_kwargs["align_corners"] = False
        x = F.interpolate(x, size=target_hw, **interp_kwargs)
        y = F.interpolate(y, size=target_hw, **interp_kwargs)

    x = _placeholder_channels(x, out_ch)
    y = _placeholder_channels(y, out_ch)
    return x, y


def _apply_latent_pipeline(
    x: torch.Tensor,
    y: torch.Tensor,
    theta: Optional[torch.Tensor],
    autoencoder,
    autoencoder_trainable: bool,
    latent_amp_enabled: bool,
    latent_amp_dtype,
    latent_cfg,
    model_family: str,
    use_chlast: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if autoencoder is None:
        x, y = _apply_placeholder_latent(x, y, latent_cfg, model_family)
        if use_chlast:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
        theta = _resize_theta_to_match(theta, x, use_chlast=use_chlast)
        return x, y, theta
    x, y = encode_latent_pair(
        x, y, autoencoder, use_amp=latent_amp_enabled, amp_dtype=latent_amp_dtype, requires_grad=autoencoder_trainable
    )
    x, y = _maybe_select_latent_fields(x, y, latent_cfg, model_family)
    if use_chlast:
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    theta = _resize_theta_to_match(theta, x, use_chlast=use_chlast)
    return x, y, theta


def _build_flow_noise(
    ref_tensor: torch.Tensor,
    noise_std: float,
    mode: str,
    perturb_source: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (noise_field, noise_scalar) for flow matching stochastic conditioning.

    noise_scalar is Bx1 and can be concatenated onto conditioning vectors.
    """
    if float(noise_std) <= 0:
        return torch.zeros_like(ref_tensor), torch.zeros(
            (ref_tensor.shape[0], 1), device=ref_tensor.device, dtype=ref_tensor.dtype
        )

    mode = str(mode).strip().lower()
    if mode not in {"scalar", "field"}:
        mode = "scalar"

    if mode == "scalar":
        noise_scalar = torch.randn(ref_tensor.shape[0], 1, device=ref_tensor.device, dtype=ref_tensor.dtype) * float(
            noise_std
        )
        if perturb_source:
            noise_field = noise_scalar.view(-1, 1, 1, 1).expand_as(ref_tensor)
        else:
            noise_field = torch.zeros_like(ref_tensor)
        return noise_field, noise_scalar

    noise_field = torch.randn_like(ref_tensor) * float(noise_std)
    noise_scalar = noise_field.reshape(noise_field.shape[0], -1).mean(dim=1, keepdim=True)
    if not perturb_source:
        noise_field = torch.zeros_like(ref_tensor)
    return noise_field, noise_scalar


def _resolve_sfm_sigma_z(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    base_sigma_z: float,
    sigma_min: float,
    sigma_max: float,
    adaptive: bool,
    ema_beta: float,
    state: Optional[Dict[str, float]] = None,
    update_state: bool = True,
) -> float:
    sigma_floor = max(float(sigma_min), 1e-6)
    sigma_cap = max(float(sigma_max), sigma_floor)
    sigma_base = float(max(base_sigma_z, sigma_floor))

    if adaptive:
        diff = (source.detach().float() - target.detach().float()).reshape(source.shape[0], -1)
        batch_rmse = float(torch.sqrt(torch.mean(diff * diff) + 1e-12).detach().cpu())
        prev = sigma_base
        if state is not None and "sigma_z_ema" in state:
            try:
                prev = float(state["sigma_z_ema"])
            except Exception:
                prev = sigma_base
        beta = min(max(float(ema_beta), 0.0), 1.0)
        sigma_val = (1.0 - beta) * prev + beta * batch_rmse
        if update_state and state is not None:
            state["sigma_z_batch"] = batch_rmse
            state["sigma_z_ema"] = sigma_val
    else:
        sigma_val = sigma_base

    sigma_val = float(min(max(sigma_val, sigma_floor), sigma_cap))
    if update_state and state is not None and "sigma_z_ema" not in state:
        state["sigma_z_ema"] = sigma_val
    return sigma_val


def _forward_surrogate(model, x: torch.Tensor, cond: Optional[torch.Tensor], theta: Optional[torch.Tensor]):
    if theta is not None:
        try:
            return model(x, cond, theta=theta) if cond is not None else model(x, theta=theta)
        except TypeError:
            pass
    return model(x, cond) if cond is not None else model(x)


def _forward_diffusion(
    model,
    x_in: torch.Tensor,
    t_model: torch.Tensor,
    cond: Optional[torch.Tensor],
    theta: Optional[torch.Tensor],
    region_info,
):
    kwargs = {"region_info": region_info} if region_info is not None else {}
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "backbone"):
        base = base.backbone
    is_unet_film_attn = base.__class__.__name__ == "UNetFiLMAttn"
    if is_unet_film_attn and theta is not None:
        use_time = bool(getattr(base, "use_time", False))
        cond_dim = int(getattr(base, "cond_dim", 1))
        cond_for_model = cond
        if use_time and cond_for_model is None:
            cond_for_model = torch.zeros(
                (x_in.shape[0], max(cond_dim, 1)),
                device=x_in.device,
                dtype=x_in.dtype,
            )
        use_control_branch = bool(getattr(base, "use_control_branch", False))
        if use_control_branch and getattr(base, "control_model", None) is not None:
            expected_in = int(getattr(base, "in_channels", x_in.shape[1]))
            if x_in.shape[1] != expected_in:
                raise RuntimeError(
                    "UNet control-branch input channel mismatch: "
                    f"got x_in channels={x_in.shape[1]}, expected={expected_in}. "
                    "Check model.params.in_channels, state channel inference, and diffusion objective wiring."
                )
            if use_time:
                return (
                    model(x_in, cond_for_model, t_model, hint=theta, **kwargs)
                    if cond_for_model is not None
                    else model(x_in, t_model, hint=theta, **kwargs)
                )
            return model(x_in, t_model, cond, hint=theta, **kwargs) if cond is not None else model(x_in, t_model, hint=theta, **kwargs)
        # Legacy path: concatenate thermal map to model input.
        x_model = torch.cat([x_in, theta], dim=1)
        expected_in = int(getattr(base, "in_channels", x_model.shape[1]))
        if x_model.shape[1] != expected_in:
            raise RuntimeError(
                "UNet thermal-input channel mismatch: "
                f"got x_model channels={x_model.shape[1]}, expected={expected_in}. "
                "Check model.params.in_channels, conditioning.use_theta/theta_channels, "
                "and dataloader thermal settings."
            )
        if use_time:
            return (
                model(x_model, cond_for_model, t_model, **kwargs)
                if cond_for_model is not None
                else model(x_model, t_model, **kwargs)
            )
        return model(x_model, t_model, cond, **kwargs) if cond is not None else model(x_model, t_model, **kwargs)
    if theta is not None:
        try:
            return model(x_in, t_model, cond, theta=theta, **kwargs) if cond is not None else model(x_in, t_model, theta=theta, **kwargs)
        except TypeError:
            pass
        try:
            return model(x_in, t_model, cond, theta, **kwargs) if cond is not None else model(x_in, t_model, theta, **kwargs)
        except TypeError:
            pass
    return model(x_in, t_model, cond, **kwargs) if cond is not None else model(x_in, t_model, **kwargs)


def _forward_flow_concat(model, x_in: torch.Tensor, cond_t: torch.Tensor, theta: Optional[torch.Tensor]):
    if theta is not None:
        if cond_t is not None:
            try:
                return model(x_in, cond_t, theta=theta)
            except TypeError:
                pass
            try:
                return model(x_in, cond_t, theta)
            except TypeError:
                pass
        try:
            return model(x_in, cond_t, theta=theta)
        except TypeError:
            pass
        try:
            return model(x_in, cond_t, theta)
        except TypeError:
            pass
    return model(x_in, cond_t) if cond_t is not None else model(x_in)


def _forward_flow_nonconcat(
    model,
    x_t: torch.Tensor,
    t_match: torch.Tensor,
    theta: Optional[torch.Tensor],
    cond_t: Optional[torch.Tensor] = None,
):
    if theta is not None:
        if cond_t is not None:
            try:
                return model(x_t, cond_t, t_match, theta=theta)
            except TypeError:
                pass
            try:
                return model(x_t, cond_t, t_match, theta)
            except TypeError:
                pass
        try:
            return model(x_t, t_match, theta=theta)
        except TypeError:
            pass
        try:
            return model(x_t, t_match, theta)
        except TypeError:
            pass
    if cond_t is not None:
        return model(x_t, cond_t, t_match)
    return model(x_t, t_match)


def _forward_flow_dbfm(
    model,
    x_t: torch.Tensor,
    t_match: torch.Tensor,
    theta: Optional[torch.Tensor],
    cond_t: Optional[torch.Tensor] = None,
):
    if theta is not None:
        if cond_t is not None:
            try:
                return model(x_t, cond_t, t_match, theta=theta)
            except TypeError:
                pass
            try:
                return model(x_t, cond_t, t_match, theta)
            except TypeError:
                pass
        try:
            return model(x_t, t_match, theta=theta)
        except TypeError:
            pass
        try:
            return model(x_t, t_match, theta)
        except TypeError:
            pass
    if cond_t is not None:
        return model(x_t, cond_t, t_match)
    return model(x_t, t_match)


_DBFM_FLOW_OBJECTIVES = {
    "dbfm_source_anchored",
    "dbfm_rectified_flow",
    "dbfm_flow",
    "dbfm",
}

_RECTIFIED_FLOW_OBJECTIVES = {
    "rectified_flow",
    "rectified_flow_constant_displacement",
    "rectified_flow_constant_displacement_concat",
    "rectified_flow_source_anchored",
    "rectified_flow_source_anchored_concat",
    # PBFM-style: random-noise start, but concatenate source state as conditioner.
    "rectified_flow_noise_source_concat",
    "rectified_flow_noise_cond_concat",
}

_NOISE_SOURCE_CONCAT_FLOW_OBJECTIVES = {
    "rectified_flow_noise_source_concat",
    "rectified_flow_noise_cond_concat",
}

_SFM_FLOW_OBJECTIVES = {
    # SFM-style latent objective: encoder-anchored noisy start + denoising target.
    "sfm_latent_source_denoise_concat",
    # Backward-compatible alias.
    "sfm_latent_source_concat",
}

_DIFFUSION_PREDICT_NEXT_OBJECTIVES = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
}


def _predict_flow_rollout_single(
    model,
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    theta: Optional[torch.Tensor],
    flow_objective: str,
    nfe: int,
    flow_noise_std: float,
    flow_noise_mode: str,
    flow_noise_perturb_source: bool,
) -> torch.Tensor:
    """
    Source-only flow rollout from s=0 -> s=1 with explicit Euler integration.
    """
    objective = str(flow_objective or "default").strip().lower()
    source_anchored = objective.startswith("rectified_flow_source_anchored")
    noise_source_concat = objective in _NOISE_SOURCE_CONCAT_FLOW_OBJECTIVES
    is_dbfm = objective in _DBFM_FLOW_OBJECTIVES
    is_rectified = objective in _RECTIFIED_FLOW_OBJECTIVES
    is_sfm = objective in _SFM_FLOW_OBJECTIVES
    nfe = max(1, int(nfe))
    dt = 1.0 / float(nfe)

    if is_sfm:
        sigma_eval = max(float(flow_noise_std), 0.0)
        if sigma_eval > 0.0:
            z0 = x + torch.randn_like(x) * sigma_eval
        else:
            z0 = x
        x_t = z0
        sigma_scalar = torch.full((x.shape[0], 1), sigma_eval, device=x.device, dtype=x.dtype)
        for i in range(nfe):
            t_val = (i + 0.5) * dt
            t_match = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
            cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
            cond_t = torch.cat([cond_t, sigma_scalar], dim=1)
            x_in = torch.cat([x_t, x], dim=1)
            x_hat = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
            denom = (1.0 - t_match).clamp_min(1e-4).view(-1, 1, 1, 1)
            v = (x_hat - x_t) / denom
            x_t = x_t + dt * v
        return x_t

    if is_dbfm:
        z0 = x
    elif is_rectified:
        z0 = x if source_anchored else torch.randn_like(x)
    else:
        # torchcfm fallback path: source-conditioned random start.
        z0 = torch.randn_like(x)
    x_t = z0

    flow_noise_field, flow_noise_scalar = _build_flow_noise(
        ref_tensor=x,
        noise_std=float(flow_noise_std),
        mode=flow_noise_mode,
        perturb_source=flow_noise_perturb_source,
    )

    if (source_anchored or noise_source_concat or is_dbfm) and flow_noise_perturb_source and flow_noise_field is not None:
        z0 = z0 + flow_noise_field
        x_t = z0

    if is_dbfm:
        use_concat = False
        source_ctx_mode = "none"
    elif is_rectified:
        use_concat = objective.endswith("_concat") or source_anchored or noise_source_concat
        source_ctx_mode = "x" if noise_source_concat else "z0"
    else:
        # torchcfm fallback was trained with [x_t, source] concat.
        use_concat = True
        source_ctx_mode = "x"

    for i in range(nfe):
        t_val = (i + 0.5) * dt
        t_match = torch.full((x.shape[0], 1), t_val, device=x.device, dtype=x.dtype)
        cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
        cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t

        if use_concat:
            source_ctx = x if source_ctx_mode == "x" else z0
            x_in = torch.cat([x_t, source_ctx], dim=1)
            v = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
        elif is_dbfm:
            v = _forward_flow_dbfm(model=model, x_t=x_t, t_match=t_match, cond_t=cond_t, theta=theta)
        else:
            v = _forward_flow_nonconcat(model=model, x_t=x_t, t_match=t_match, cond_t=cond_t, theta=theta)
        x_t = x_t + dt * v
    return x_t


def _predict_flow_rollout_samples(
    model,
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    theta: Optional[torch.Tensor],
    flow_objective: str,
    nfe: int,
    flow_noise_std: float,
    flow_noise_mode: str,
    flow_noise_perturb_source: bool,
    num_samples: int = 1,
) -> torch.Tensor:
    ns = max(1, int(num_samples))
    samples = []
    for _ in range(ns):
        samples.append(
            _predict_flow_rollout_single(
                model=model,
                x=x,
                cond=cond,
                theta=theta,
                flow_objective=flow_objective,
                nfe=nfe,
                flow_noise_std=flow_noise_std,
                flow_noise_mode=flow_noise_mode,
                flow_noise_perturb_source=flow_noise_perturb_source,
            )
        )
    return torch.stack(samples, dim=0)


def _flow_ensemble_batch_sums(endpoint_samples: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, int]:
    """
    Returns CRPS term-1/term-2 sums and predictive-variance sum over all spatial elements.

    endpoint_samples: (S,B,C,H,W), target: (B,C,H,W)
    """
    if endpoint_samples.dim() != target.dim() + 1:
        raise ValueError(
            "Expected endpoint_samples with one leading ensemble dimension; "
            f"got samples={tuple(endpoint_samples.shape)} target={tuple(target.shape)}."
        )
    if endpoint_samples.shape[1:] != target.shape:
        raise ValueError(
            "Ensemble sample/target shape mismatch; "
            f"got samples={tuple(endpoint_samples.shape)} target={tuple(target.shape)}."
        )

    ens = endpoint_samples.float()
    y = target.float()
    ns = int(ens.shape[0])
    elem_count = int(y.numel())

    # CRPS term 1: E|X - y|.
    term1 = (ens - y.unsqueeze(0)).abs().mean(dim=0)
    term1_sum = float(term1.sum().detach().cpu())

    # CRPS term 2: 0.5 * E|X - X'| using all pair combinations.
    if ns > 1:
        pair_sum = ens.new_zeros(())
        for i in range(ns):
            for j in range(ns):
                pair_sum = pair_sum + (ens[i] - ens[j]).abs().sum()
        term2_sum = float((0.5 * pair_sum / float(ns * ns)).detach().cpu())
    else:
        term2_sum = 0.0

    # Spread for SSR: sqrt(E[var(X)]).
    var_sum = float(ens.var(dim=0, unbiased=False).sum().detach().cpu())
    return term1_sum, term2_sum, var_sum, elem_count


def _predict_flow_rollout(
    model,
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    theta: Optional[torch.Tensor],
    flow_objective: str,
    nfe: int,
    flow_noise_std: float,
    flow_noise_mode: str,
    flow_noise_perturb_source: bool,
    num_samples: int = 1,
) -> torch.Tensor:
    return _predict_flow_rollout_samples(
        model=model,
        x=x,
        cond=cond,
        theta=theta,
        flow_objective=flow_objective,
        nfe=nfe,
        flow_noise_std=flow_noise_std,
        flow_noise_mode=flow_noise_mode,
        flow_noise_perturb_source=flow_noise_perturb_source,
        num_samples=num_samples,
    ).mean(dim=0)




def _train_one_epoch(
    epoch: int,
    model,
    train_dl,
    device,
    cond_cfg,
    autoencoder,
    autoencoder_trainable: bool,
    latent_amp_enabled: bool,
    latent_amp_dtype,
    latent_cfg,
    use_chlast: bool,
    amp_enabled: bool,
    amp_dtype,
    scaler: GradScaler,
    optim,
    grad_clip: float,
    model_family: str,
    loss_fns: Dict[str, Any],
    flow_matcher,
    flow_objective: str,
    diffusion_objective: str,
    noise_schedule_obj,
    timestep_sampler_obj,
    region_selector_obj,
    use_weight_loss: bool,
    log_interval: int,
    want_mae: bool,
    want_vrmse: bool,
    vrmse_eps: float,
    metric_channels: int,
    want_spectral: bool,
    spectral_cfg: Dict[str, Any],
    spectral_train: bool,
    steps_per_epoch: Optional[int],
    total_pairs_by_gid: Optional[Counter],
    nan_debug: bool,
    nan_debug_steps: int,
    nan_debug_param: Optional[str],
    nan_debug_input_stats: bool,
    nan_tolerate_steps: int,
    detect_anomaly: bool,
    detect_anomaly_steps: int,
    lr_warmup_steps: int,
    lr_warmup_start_lr: Optional[float],
    lr_warmup_phases,
    accumulation_steps: int,
    flow_stochastic_std: float = 0.0,
    flow_stochastic_mode: str = "scalar",
    flow_stochastic_perturb_source: bool = True,
    sfm_sigma_z: float = 0.05,
    sfm_sigma_min: float = 1e-3,
    sfm_sigma_max: float = 0.25,
    sfm_adaptive_sigma: bool = True,
    sfm_sigma_ema_beta: float = 0.02,
    sfm_encoder_reg_lambda: float = 0.0,
    sfm_state: Optional[Dict[str, float]] = None,
):
    model.train()
    se_sum, elem_count = 0.0, 0
    mae_sum = 0.0 if want_mae else None
    obj_sum, obj_count = 0.0, 0
    log_se_sum, log_elem_count = 0.0, 0
    log_mae_sum = 0.0 if want_mae else None
    if steps_per_epoch is not None and steps_per_epoch > 0:
        num_steps = int(steps_per_epoch)
    else:
        num_steps = len(train_dl) if hasattr(train_dl, "__len__") else None

    gid_stats_local = defaultdict(lambda: [0.0, 0])
    seen_pairs_local = set()
    train_task_metric_sum: Dict[str, float] = {}
    train_task_metric_count: Dict[str, int] = {}
    channel_stats: Optional[Dict[str, torch.Tensor]] = _init_channel_stats(metric_channels, device) if want_vrmse else None
    spec_sum: Dict[str, torch.Tensor] = {}
    spec_count = 0
    accum_steps = max(int(accumulation_steps), 1)
    opt_step_idx = 0
    last_step_idx = 0
    startup_logged = False
    theta_logged = False
    warmup_phases = None
    warmup_total = 0
    if lr_warmup_phases:
        phases = []
        total = 0
        for phase in lr_warmup_phases:
            try:
                steps = int(phase.get("steps", 0))
                factor = float(phase.get("factor", 1.0))
            except AttributeError:
                continue
            if steps <= 0:
                continue
            total += steps
            phases.append((total, factor))
        if phases:
            warmup_phases = phases
            warmup_total = total

    for pg in optim.param_groups:
        pg.setdefault("base_lr", pg["lr"])

    if autoencoder is not None:
        if autoencoder_trainable:
            autoencoder.train()
        else:
            autoencoder.eval()
    diffusion_predict_next = str(diffusion_objective or "epsilon_mse").lower() in _DIFFUSION_PREDICT_NEXT_OBJECTIVES

    debug_param_name, debug_param = _find_named_param(model, nan_debug_param) if nan_debug else (None, None)
    if nan_debug and _rank0():
        if debug_param is None:
            print(f"[nan_debug] param_not_found={nan_debug_param}", flush=True)
        else:
            print(f"[nan_debug] tracking_param={debug_param_name}", flush=True)

    for step_idx, batch in enumerate(_iter_train_batches(train_dl, steps_per_epoch), start=1):
        last_step_idx = step_idx
        do_step = (step_idx % accum_steps == 0)
        if num_steps is not None and step_idx == num_steps:
            do_step = True

        if (step_idx - 1) % accum_steps == 0:
            optim.zero_grad(set_to_none=True)
        if nan_debug and step_idx <= nan_debug_steps:
            name, tensor = _first_nonfinite_named(model.named_parameters(), use_grad=False)
            if name is not None:
                rank_env = int(os.environ.get("RANK", "-1"))
                print(
                    f"[nan_debug] rank={rank_env} step={step_idx} pre_forward param={name} stats={_tensor_stats(tensor, name)}",
                    flush=True,
                )
                raise FloatingPointError("Non-finite parameter before forward")
        param_before = None
        param_snapshot = None
        optim_state_snapshot = None
        params = [p for group in optim.param_groups for p in group["params"]]
        if nan_debug and do_step and step_idx <= nan_debug_steps and debug_param is not None:
            param_before = debug_param.detach().clone()
        if nan_tolerate_steps > 0 and do_step and step_idx <= nan_tolerate_steps:
            param_snapshot = [p.detach().clone() for p in params]

        if warmup_phases and do_step:
            opt_step = opt_step_idx + 1
            if opt_step <= warmup_total:
                for limit, factor in warmup_phases:
                    if opt_step <= limit:
                        for pg in optim.param_groups:
                            base_lr = float(pg.get("base_lr", pg["lr"]))
                            pg["lr"] = base_lr * factor
                        break
            elif opt_step == warmup_total + 1:
                for pg in optim.param_groups:
                    base_lr = float(pg.get("base_lr", pg["lr"]))
                    pg["lr"] = base_lr
        elif lr_warmup_steps > 0 and do_step:
            if num_steps is not None:
                global_step = (epoch - 1) * num_steps + step_idx
            else:
                global_step = step_idx
            opt_step = opt_step_idx + 1
            if opt_step <= lr_warmup_steps:
                start_lr = lr_warmup_start_lr if lr_warmup_start_lr is not None else 0.0
                frac = float(opt_step) / float(max(lr_warmup_steps, 1))
                for pg in optim.param_groups:
                    base_lr = float(pg.get("base_lr", pg["lr"]))
                    pg["lr"] = start_lr + (base_lr - start_lr) * frac

        sync_ctx = nullcontext()
        if isinstance(model, DDP) and accum_steps > 1 and not do_step:
            sync_ctx = model.no_sync()
        with sync_ctx:
            x, y, cond, theta = _prepare_batch(batch, device, cond_cfg, use_chlast)
        weight = batch.get("weight", None)
        if weight is not None:
            weight = weight.to(device, non_blocking=True)
            if use_chlast and weight.dim() == 4:
                weight = weight.contiguous(memory_format=torch.channels_last)

        x, y, theta = _apply_latent_pipeline(
            x=x,
            y=y,
            theta=theta,
            autoencoder=autoencoder,
            autoencoder_trainable=autoencoder_trainable,
            latent_amp_enabled=latent_amp_enabled,
            latent_amp_dtype=latent_amp_dtype,
            latent_cfg=latent_cfg,
            model_family=model_family,
            use_chlast=use_chlast,
        )
        if theta is not None and (not theta_logged) and _rank0():
            theta_det = theta.detach().float()
            theta_mean = float(theta_det.mean().item())
            theta_std = float(theta_det.std(unbiased=False).item())
            theta_min = float(theta_det.min().item())
            theta_max = float(theta_det.max().item())
            print(
                f"[theta] epoch={epoch} step={step_idx} post_norm mean={theta_mean:.6g} "
                f"std={theta_std:.6g} min={theta_min:.6g} max={theta_max:.6g}",
                flush=True,
            )
            theta_logged = True

        if nan_debug and nan_debug_input_stats and step_idx <= nan_debug_steps:
            rank_env = int(os.environ.get("RANK", "-1"))
            stats = [_tensor_stats(x, "x"), _tensor_stats(y, "y")]
            if cond is not None:
                stats.append(_tensor_stats(cond, "cond"))
            if theta is not None:
                stats.append(_tensor_stats(theta, "theta"))
            if weight is not None:
                stats.append(_tensor_stats(weight, "weight"))
            print(f"[nan_debug] rank={rank_env} step={step_idx} input_stats={stats}", flush=True)

        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            if model_family == "diffusion":
                t = timestep_sampler_obj.sample(batch_size=y.shape[0])
                t = torch.tensor(t, device=device, dtype=torch.long) if not torch.is_tensor(t) else t.to(device=device, dtype=torch.long)
                sched_kind, x_noisy, eps, log_snr_t = _sample_diffusion_noisy_pair(
                    y=y,
                    x=x,
                    t=t,
                    noise_schedule_obj=noise_schedule_obj,
                    use_chlast=use_chlast,
                    device=device,
                )
                x_in = _build_diffusion_model_input(
                    x_noisy=x_noisy,
                    source=x,
                    noise_schedule_obj=noise_schedule_obj,
                    sched_kind=sched_kind,
                )
                region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                t_model = t.view(-1, 1) if t.dim() == 1 else t
                pred = _forward_diffusion(
                    model=model,
                    x_in=x_in,
                    t_model=t_model,
                    cond=cond,
                    theta=theta,
                    region_info=region_info,
                )
                loss = loss_fns["diffusion"](
                    pred,
                    eps,
                    target=y,
                    source=x,
                    noisy=x_noisy,
                    region_info=region_info,
                    t=t,
                    num_steps=getattr(noise_schedule_obj, "timesteps", None),
                    log_snr=log_snr_t,
                    schedule=noise_schedule_obj,
                )
                metric_target = y if diffusion_predict_next else eps
            elif model_family == "flow_matching":
                objective = str(flow_objective or "default").lower()
                if objective in _SFM_FLOW_OBJECTIVES:
                    flow_noise_field = torch.zeros_like(x)
                    flow_noise_scalar = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
                else:
                    flow_noise_field, flow_noise_scalar = _build_flow_noise(
                        ref_tensor=x,
                        noise_std=float(flow_stochastic_std),
                        mode=flow_stochastic_mode,
                        perturb_source=flow_stochastic_perturb_source,
                    )
                if objective in _SFM_FLOW_OBJECTIVES:
                    if x.shape != y.shape:
                        raise ValueError(
                            "sfm_latent_source_denoise_concat requires x/y to have identical latent shapes; "
                            f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                        )
                    sigma_z = _resolve_sfm_sigma_z(
                        source=x,
                        target=y,
                        base_sigma_z=float(sfm_sigma_z),
                        sigma_min=float(sfm_sigma_min),
                        sigma_max=float(sfm_sigma_max),
                        adaptive=bool(sfm_adaptive_sigma),
                        ema_beta=float(sfm_sigma_ema_beta),
                        state=sfm_state,
                        update_state=True,
                    )
                    sigma_floor = max(float(sfm_sigma_min), 1e-6)
                    sigma = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    sigma = sigma * max(sigma_z - sigma_floor, 0.0) + sigma_floor
                    t_match = 1.0 - (sigma / max(sigma_z, sigma_floor))
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    t_view = t_match.view(-1, 1, 1, 1)
                    sigma_view = sigma.view(-1, 1, 1, 1)
                    sigma_z_scalar = torch.full((y.shape[0], 1), sigma_z, device=y.device, dtype=y.dtype)
                    eps = torch.randn_like(y)
                    e = (x - y) / max(sigma_z, sigma_floor)
                    x_t = y + sigma_view * (e + eps)
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                    cond_t = torch.cat([cond_t, sigma_z_scalar], dim=1)
                    x_in = torch.cat([x_t, x], dim=1)
                    pred = _forward_flow_concat(
                        model=model,
                        x_in=x_in,
                        cond_t=cond_t,
                        theta=theta,
                    )
                    weight = (sigma_z_scalar / sigma.clamp_min(sigma_floor)).pow(2).view(-1, 1, 1, 1)
                    denoise = ((pred.float() - y.float()).pow(2) * weight.float()).mean()
                    enc_reg = (e.float().pow(2)).mean()
                    loss = denoise + float(sfm_encoder_reg_lambda) * enc_reg
                    metric_target = y
                    u_t = y
                elif objective in _DBFM_FLOW_OBJECTIVES:
                    if x.shape != y.shape:
                        raise ValueError(
                            "dbfm_source_anchored requires x/y to have identical latent shapes; "
                            f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                        )
                    z0 = x
                    t_match = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    if flow_noise_field is not None:
                        z0 = z0 + flow_noise_field
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = (1.0 - t_view) * z0 + t_view * y
                    u_t = y - z0
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    cond_t = t_match
                    cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                    pred = _forward_flow_dbfm(
                        model=model,
                        x_t=x_t,
                        t_match=t_match,
                        cond_t=cond_t,
                        theta=theta,
                    )
                    endpoint_pred = x_t + (1.0 - t_view) * pred
                    loss = loss_fns["flow"](
                        pred.float(),
                        u_t.float(),
                        target=y,
                        t=t_match,
                        endpoint_pred=endpoint_pred.float(),
                        target_state=y,
                    )
                    metric_target = u_t
                elif objective in _RECTIFIED_FLOW_OBJECTIVES:
                    source_anchored = objective.startswith("rectified_flow_source_anchored")
                    noise_source_concat = objective in _NOISE_SOURCE_CONCAT_FLOW_OBJECTIVES
                    if source_anchored:
                        if x.shape != y.shape:
                            raise ValueError(
                                "rectified_flow_source_anchored requires x/y to have identical latent shapes; "
                                f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                            )
                        z0 = x
                    else:
                        z0 = torch.randn_like(y)
                    t_match = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    if (source_anchored or noise_source_concat) and flow_stochastic_perturb_source and flow_noise_field is not None:
                        z0 = z0 + flow_noise_field
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = (1.0 - t_view) * z0 + t_view * y
                    u_t = y - z0
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                    cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                    if objective.endswith("_concat") or source_anchored or noise_source_concat:
                        source_ctx = x if noise_source_concat else z0
                        x_in = torch.cat([x_t, source_ctx], dim=1)
                        pred = _forward_flow_concat(
                            model=model,
                            x_in=x_in,
                            cond_t=cond_t,
                            theta=theta,
                        )
                    else:
                        pred = _forward_flow_nonconcat(
                            model=model,
                            x_t=x_t,
                            t_match=t_match,
                            cond_t=cond_t,
                            theta=theta,
                        )
                    endpoint_pred = x_t + (1.0 - t_view) * pred
                    loss = loss_fns["flow"](
                        pred.float(),
                        u_t.float(),
                        target=y,
                        t=t_match,
                        endpoint_pred=endpoint_pred.float(),
                        target_state=y,
                    )
                    metric_target = u_t
                else:
                    x0_noise = torch.randn_like(y)
                    t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x0_noise, x1=y)
                    t_match = t_match.to(device).view(-1, 1)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = x_t.to(device)
                    u_t = u_t.to(device)
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                    cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                    x_in = torch.cat([x_t, x], dim=1)
                    pred = _forward_flow_concat(
                        model=model,
                        x_in=x_in,
                        cond_t=cond_t,
                        theta=theta,
                    )
                    # Compute the loss in fp32 for stability even under autocast.
                    endpoint_pred = x_t + (1.0 - t_view) * pred
                    loss = loss_fns["flow"](
                        pred.float(),
                        u_t.float(),
                        target=y,
                        t=t_match,
                        endpoint_pred=endpoint_pred.float(),
                        target_state=y,
                    )
                    metric_target = u_t
            else:
                pred_out = _forward_surrogate(model, x, cond, theta)
                pred_main = pred_out.get("recon", pred_out.get("pred")) if isinstance(pred_out, dict) else pred_out
                dataset_weight = weight if (use_weight_loss and weight is not None) else None
                if num_steps is not None:
                    global_step = (epoch - 1) * num_steps + step_idx
                else:
                    global_step = step_idx
                metric_owner = model.module if isinstance(model, DDP) else model
                loss = loss_fns["surrogate"](pred_out, y, weight=dataset_weight, step=global_step, epoch=epoch)
                metric_target = y
                if hasattr(metric_owner, "compute_metrics"):
                    try:
                        task_metrics = metric_owner.compute_metrics(pred_main.detach(), metric_target.detach())  # type: ignore[arg-type]
                    except Exception:
                        task_metrics = {}
                    for name, val in task_metrics.items():
                        train_task_metric_sum[name] = train_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                        train_task_metric_count[name] = train_task_metric_count.get(name, 0) + 1
                pred = pred_main

        with torch.no_grad():
            obj_sum += float(loss.detach().cpu())
            obj_count += 1

        if not torch.isfinite(loss):
            # Collect lightweight stats to pinpoint the offending batch.
            def _stat_block(t: torch.Tensor, name: str):
                t_det = t.detach()
                finite_mask = torch.isfinite(t_det)
                finite_cnt = int(finite_mask.sum().item())
                total_cnt = t_det.numel()
                if finite_cnt > 0:
                    finite_min = float(t_det[finite_mask].min().item())
                    finite_max = float(t_det[finite_mask].max().item())
                else:
                    finite_min = float("nan")
                    finite_max = float("nan")
                return {
                    "name": name,
                    "finite": f"{finite_cnt}/{total_cnt}",
                    "min": finite_min,
                    "max": finite_max,
                    "has_nan": bool(torch.isnan(t_det).any().item()),
                    "has_inf": bool(torch.isinf(t_det).any().item()),
                }

            debug_blocks = []
            debug_blocks.append(_stat_block(loss.unsqueeze(0), "loss"))
            debug_blocks.append(_stat_block(pred, "pred"))
            debug_blocks.append(_stat_block(metric_target, "target"))
            debug_blocks.append(_stat_block(x, "x"))
            if model_family == "flow_matching":
                debug_blocks.append(_stat_block(t_match, "t_match"))
                debug_blocks.append(_stat_block(u_t, "u_t"))
                debug_blocks.append(_stat_block(x_t, "x_t"))
            if model_family == "diffusion":
                debug_blocks.append(_stat_block(t, "t"))
                debug_blocks.append(_stat_block(x_noisy, "x_noisy"))
                debug_blocks.append(_stat_block(eps, "eps"))

            rank_env = int(os.environ.get("RANK", "-1"))
            gid_info = [(str(g), int(k)) for g, k in zip(batch["gid"], batch["pair_index"])]
            print(
                f"[nan_guard] rank={rank_env} epoch={epoch} step={step_idx} gid/pair={gid_info} stats={debug_blocks}",
                flush=True,
            )
            if nan_tolerate_steps > 0 and step_idx <= nan_tolerate_steps:
                optim.zero_grad(set_to_none=True)
                continue
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

        loss_scaled = loss / float(accum_steps) if accum_steps > 1 else loss
        step_ok = True
        step_applied = False
        use_detect_anomaly = bool(detect_anomaly and step_idx <= detect_anomaly_steps)
        if scaler.is_enabled():
            if use_detect_anomaly:
                with torch.autograd.detect_anomaly():
                    scaler.scale(loss_scaled).backward()
            else:
                scaler.scale(loss_scaled).backward()
            if do_step:
                scaler.unscale_(optim)
                if nan_debug and step_idx <= nan_debug_steps:
                    name, tensor = _first_nonfinite_named(model.named_parameters(), use_grad=True)
                    if name is not None:
                        rank_env = int(os.environ.get("RANK", "-1"))
                        print(
                            f"[nan_debug] rank={rank_env} step={step_idx} grad param={name} stats={_tensor_stats(tensor, name)}",
                            flush=True,
                        )
                        raise FloatingPointError("Non-finite gradient detected")
                grad_norm = _grad_norm_or_clip(params, grad_clip)
                if nan_debug and step_idx <= nan_debug_steps and debug_param is not None:
                    rank_env = int(os.environ.get("RANK", "-1"))
                    grad_pre = _tensor_stats(debug_param.grad, f"{debug_param_name}.grad") if debug_param.grad is not None else {"name": f"{debug_param_name}.grad", "missing": True}
                    print(
                        f"[nan_debug] rank={rank_env} step={step_idx} grad_norm={float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm} grad_pre={grad_pre}",
                        flush=True,
                    )
                if torch.isfinite(grad_norm):
                    if nan_tolerate_steps > 0 and step_idx <= nan_tolerate_steps:
                        optim_state_snapshot = copy.deepcopy(optim.state_dict())
                    scaler.step(optim)
                    step_applied = True
                    opt_step_idx += 1
                else:
                    step_ok = False
                scaler.update()
        else:
            if use_detect_anomaly:
                with torch.autograd.detect_anomaly():
                    loss_scaled.backward()
            else:
                loss_scaled.backward()
            if do_step:
                if nan_debug and step_idx <= nan_debug_steps:
                    name, tensor = _first_nonfinite_named(model.named_parameters(), use_grad=True)
                    if name is not None:
                        rank_env = int(os.environ.get("RANK", "-1"))
                        print(
                            f"[nan_debug] rank={rank_env} step={step_idx} grad param={name} stats={_tensor_stats(tensor, name)}",
                            flush=True,
                        )
                        raise FloatingPointError("Non-finite gradient detected")
                grad_norm = _grad_norm_or_clip(params, grad_clip)
                if nan_debug and step_idx <= nan_debug_steps and debug_param is not None:
                    rank_env = int(os.environ.get("RANK", "-1"))
                    grad_pre = _tensor_stats(debug_param.grad, f"{debug_param_name}.grad") if debug_param.grad is not None else {"name": f"{debug_param_name}.grad", "missing": True}
                    print(
                        f"[nan_debug] rank={rank_env} step={step_idx} grad_norm={float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm} grad_pre={grad_pre}",
                        flush=True,
                    )
                if torch.isfinite(grad_norm):
                    if nan_tolerate_steps > 0 and step_idx <= nan_tolerate_steps:
                        optim_state_snapshot = copy.deepcopy(optim.state_dict())
                    optim.step()
                    step_applied = True
                    opt_step_idx += 1
                else:
                    step_ok = False

        if do_step and not step_ok:
            optim.zero_grad(set_to_none=True)

        if do_step and step_ok and nan_debug and step_idx <= nan_debug_steps:
            if debug_param is not None and param_before is not None:
                update = debug_param.detach() - param_before
                rank_env = int(os.environ.get("RANK", "-1"))
                update_stats = _tensor_stats(update, f"{debug_param_name}.update")
                if debug_param.grad is not None:
                    grad_stats = _tensor_stats(debug_param.grad, f"{debug_param_name}.grad")
                else:
                    grad_stats = {"name": f"{debug_param_name}.grad", "missing": True}
                if not torch.isfinite(debug_param).all():
                    print(
                        f"[nan_debug] rank={rank_env} step={step_idx} post_step param={debug_param_name} "
                        f"param={_tensor_stats(debug_param, debug_param_name)} grad={grad_stats} update={update_stats}",
                        flush=True,
                    )
        if do_step:
            name, tensor = _first_nonfinite_named(model.named_parameters(), use_grad=False)
            if name is not None:
                rank_env = int(os.environ.get("RANK", "-1"))
                print(
                    f"[nan_debug] rank={rank_env} step={step_idx} post_step param={name} stats={_tensor_stats(tensor, name)}",
                    flush=True,
                )
                if nan_tolerate_steps > 0 and step_idx <= nan_tolerate_steps and param_snapshot is not None:
                    for p, snap in zip(params, param_snapshot):
                        p.data.copy_(snap)
                    if optim_state_snapshot is not None:
                        optim.load_state_dict(optim_state_snapshot)
                    if step_applied:
                        opt_step_idx = max(opt_step_idx - 1, 0)
                    optim.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError("Non-finite parameter after optimizer step")

        if do_step and step_applied and not startup_logged and opt_step_idx == 1:
            rank_env = int(os.environ.get("RANK", "-1"))
            lr_now = float(optim.param_groups[0].get("lr", 0.0)) if optim.param_groups else 0.0
            print(
                f"[startup] rank={rank_env} epoch={epoch} step={step_idx} opt_step={opt_step_idx} lr={lr_now:.6g}",
                flush=True,
            )
            startup_logged = True

        with torch.no_grad():
            pred_metric = pred.float()
            target_metric = metric_target.float()
            mse_batch = F.mse_loss(pred_metric, target_metric, reduction="mean")
            if want_mae:
                mae_batch = F.l1_loss(pred_metric, target_metric, reduction="mean")

            elems = metric_target.numel()
            se_sum += float(mse_batch.detach().cpu()) * elems
            elem_count += elems
            if want_mae:
                mae_sum += float(mae_batch.detach().cpu()) * elems
            log_se_sum += float(mse_batch.detach().cpu()) * elems
            log_elem_count += elems
            if want_mae and log_mae_sum is not None:
                log_mae_sum += float(mae_batch.detach().cpu()) * elems

            ps_mse = (pred_metric.detach() - target_metric.detach())
            ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
            if want_vrmse and channel_stats is not None:
                _update_channel_stats(channel_stats, pred, metric_target)
            if want_spectral and spectral_train:
                spec_vals = spectral_rmse_bands(pred, metric_target, **spectral_cfg)
                for k, v in spec_vals.items():
                    if k not in spec_sum:
                        spec_sum[k] = torch.zeros((), dtype=torch.float64, device=metric_target.device)
                    spec_sum[k] += v.detach().to(dtype=torch.float64)
                spec_count += 1

        for g, k, m in zip(batch["gid"], batch["pair_index"], ps_mse):
            gid_stats_local[g][0] += float(m)
            gid_stats_local[g][1] += 1
            seen_pairs_local.add((g, int(k)))

        if log_interval > 0 and _rank0() and (step_idx % log_interval == 0):
            int_mse = log_se_sum / max(log_elem_count, 1)
            int_rmse = math.sqrt(max(int_mse, 0.0))
            msg = f"epoch={epoch} step={step_idx}"
            if num_steps:
                msg += f"/{num_steps}"
            msg += f" train_rmse={int_rmse:.6f}"
            if want_mae and log_mae_sum is not None:
                int_mae = log_mae_sum / max(log_elem_count, 1)
                msg += f" train_mae={int_mae:.6f}"
            msg += f" lr={optim.param_groups[0]['lr']:.6g}"
            print(msg, flush=True)
            log_se_sum, log_elem_count = 0.0, 0
            if want_mae:
                log_mae_sum = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    se_sum_g, elem_count_g = _allreduce_sum_count(device, se_sum, elem_count)
    train_mse = se_sum_g / max(elem_count_g, 1)
    train_rmse = math.sqrt(max(train_mse, 0.0))
    train_mae = None
    if want_mae:
        mae_sum_g, _ = _allreduce_sum_count(device, mae_sum, elem_count)
        train_mae = mae_sum_g / max(elem_count_g, 1)

    vrmse_stats = _compute_vrmse_from_stats(channel_stats, vrmse_eps) if want_vrmse else None

    gid_stats_all = _all_gather_object(dict(gid_stats_local))
    seen_sets_all = _all_gather_object(seen_pairs_local)
    gid_stats_merged = defaultdict(lambda: [0.0, 0])
    if _rank0():
        for d in gid_stats_all:
            for g, (s, c) in d.items():
                gid_stats_merged[g][0] += float(s)
                gid_stats_merged[g][1] += int(c)
        seen_all = set().union(*seen_sets_all)
    else:
        seen_all = set()

    coverage = None
    if _rank0() and total_pairs_by_gid is not None:
        coverage = {}
        cov_counts = Counter(g for g, _ in seen_all)
        for g, tot in total_pairs_by_gid.items():
            frac = cov_counts.get(g, 0) / max(tot, 1)
            coverage[g] = frac

    train_task_avg = {k: v / max(train_task_metric_count.get(k, 1), 1) for k, v in train_task_metric_sum.items()}
    if sfm_state is not None and str(flow_objective or "").lower() in _SFM_FLOW_OBJECTIVES:
        if "sigma_z_ema" in sfm_state:
            train_task_avg["sfm_sigma_z"] = float(sfm_state["sigma_z_ema"])
        if "sigma_z_batch" in sfm_state:
            train_task_avg["sfm_sigma_z_batch_rmse"] = float(sfm_state["sigma_z_batch"])
    out = {
        "mse": train_mse,
        "rmse": train_rmse,
        "objective": (obj_sum / max(obj_count, 1)),
        "mae": train_mae,
        "gid_stats": gid_stats_merged,
        "coverage": coverage,
        "task_avg": train_task_avg,
        "step_count": int(last_step_idx),
        "opt_step_count": int(opt_step_idx),
    }
    if want_spectral and spectral_train and spec_sum:
        count_t = _allreduce_sum_tensor(torch.tensor(float(spec_count), device=device, dtype=torch.float64))
        denom = max(float(count_t.item()), 1.0)
        for k, v in spec_sum.items():
            v_sum = _allreduce_sum_tensor(v)
            out[k] = float((v_sum / denom).item())
        if "spec_rmse_all" in out:
            out["spectral_rmse"] = out["spec_rmse_all"]
    if vrmse_stats is not None:
        out.update(vrmse_stats)
    return out


def _validate_epoch(
    model,
    val_dl,
    device,
    cond_cfg,
    autoencoder,
    latent_amp_enabled: bool,
    latent_amp_dtype,
    latent_cfg,
    use_chlast: bool,
    amp_enabled: bool,
    amp_dtype,
    model_family: str,
    loss_fns: Dict[str, Any],
    flow_matcher,
    flow_objective: str,
    diffusion_objective: str,
    noise_schedule_obj,
    timestep_sampler_obj,
    region_selector_obj,
    want_mae: bool,
    want_vrmse: bool,
    vrmse_eps: float,
    metric_channels: int,
    vrmse_ref_var: Optional[torch.Tensor] = None,
    want_spectral: bool = False,
    spectral_cfg: Optional[Dict[str, Any]] = None,
    want_endpoint_rmse: bool = False,
    flow_stochastic_std: float = 0.0,
    flow_stochastic_mode: str = "scalar",
    flow_stochastic_perturb_source: bool = True,
    flow_val_nfe: int = 20,
    flow_val_num_samples: int = 1,
    flow_val_deterministic: bool = False,
    flow_val_prob_metrics: bool = True,
    sfm_sigma_z: float = 0.05,
    sfm_sigma_min: float = 1e-3,
    sfm_sigma_max: float = 0.25,
    sfm_adaptive_sigma: bool = True,
    sfm_sigma_ema_beta: float = 0.02,
    sfm_encoder_reg_lambda: float = 0.0,
    sfm_state: Optional[Dict[str, float]] = None,
):
    if val_dl is None:
        return None
    model.eval()
    v_se_sum, v_elem_count = 0.0, 0
    v_mae_sum = 0.0 if want_mae else None
    v_obj_sum, v_obj_count = 0.0, 0
    v_gid_stats_local = defaultdict(lambda: [0.0, 0]) if model_family != "diffusion" else None
    val_task_metric_sum: Dict[str, float] = {}
    val_task_metric_count: Dict[str, int] = {}
    channel_stats: Optional[Dict[str, torch.Tensor]] = _init_channel_stats(metric_channels, device) if want_vrmse else None
    spec_sum: Dict[str, torch.Tensor] = {}
    spec_count = 0
    diffusion_predict_next = str(diffusion_objective or "epsilon_mse").lower() in _DIFFUSION_PREDICT_NEXT_OBJECTIVES
    endpoint_se_sum = 0.0
    endpoint_elem_count = 0
    endpoint_spec_sum: Dict[str, torch.Tensor] = {}
    endpoint_spec_count = 0
    endpoint_crps_term1_sum = 0.0
    endpoint_crps_term2_sum = 0.0
    endpoint_crps_elem_count = 0
    endpoint_spread_var_sum = 0.0
    endpoint_spread_elem_count = 0
    flow_val_nfe = max(1, int(flow_val_nfe))
    flow_val_num_samples = max(1, int(flow_val_num_samples))
    flow_eval_noise_std = 0.0 if bool(flow_val_deterministic) else float(flow_stochastic_std)
    flow_eval_num_samples = 1 if bool(flow_val_deterministic) else flow_val_num_samples
    flow_prob_metrics_enabled = bool(flow_val_prob_metrics and model_family == "flow_matching")

    with torch.inference_mode():
        if autoencoder is not None:
            autoencoder.eval()
        for batch in val_dl:
            x, y, cond, theta = _prepare_batch(batch, device, cond_cfg, use_chlast)
            x, y, theta = _apply_latent_pipeline(
                x=x,
                y=y,
                theta=theta,
                autoencoder=autoencoder,
                autoencoder_trainable=False,
                latent_amp_enabled=latent_amp_enabled,
                latent_amp_dtype=latent_amp_dtype,
                latent_cfg=latent_cfg,
                model_family=model_family,
                use_chlast=use_chlast,
            )
            if model_family == "diffusion":
                t = timestep_sampler_obj.sample(batch_size=y.shape[0])
                t = torch.tensor(t, device=device, dtype=torch.long) if not torch.is_tensor(t) else t.to(device=device, dtype=torch.long)
                sched_kind, x_noisy, eps, log_snr_t = _sample_diffusion_noisy_pair(
                    y=y,
                    x=x,
                    t=t,
                    noise_schedule_obj=noise_schedule_obj,
                    use_chlast=use_chlast,
                    device=device,
                )
                x_in = _build_diffusion_model_input(
                    x_noisy=x_noisy,
                    source=x,
                    noise_schedule_obj=noise_schedule_obj,
                    sched_kind=sched_kind,
                )
                region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    t_model = t.view(-1, 1) if t.dim() == 1 else t
                    pred = _forward_diffusion(
                        model=model,
                        x_in=x_in,
                        t_model=t_model,
                        cond=cond,
                        theta=theta,
                        region_info=region_info,
                    )
                    vobj = loss_fns["diffusion"](
                        pred,
                        eps,
                        target=y,
                        source=x,
                        noisy=x_noisy,
                        region_info=region_info,
                        t=t,
                        num_steps=getattr(noise_schedule_obj, "timesteps", None),
                        log_snr=log_snr_t,
                        schedule=noise_schedule_obj,
                    )
                    metric_target = y if diffusion_predict_next else eps
                    vloss = F.mse_loss(pred.float(), metric_target.float())
                    if want_mae:
                        vmae = F.l1_loss(pred.float(), metric_target.float())
                v_obj_sum += float(vobj.detach().cpu())
                v_obj_count += 1
                if want_vrmse and channel_stats is not None:
                    _update_channel_stats(channel_stats, pred, metric_target)
                if want_spectral and spectral_cfg is not None:
                    spec_vals = spectral_rmse_bands(pred, metric_target, **spectral_cfg)
                    for k, v in spec_vals.items():
                        if k not in spec_sum:
                            spec_sum[k] = torch.zeros((), dtype=torch.float64, device=metric_target.device)
                        spec_sum[k] += v.detach().to(dtype=torch.float64)
                    spec_count += 1
                elems = metric_target.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
                if want_endpoint_rmse and sched_kind == "unidb":
                    endpoint_pred = None
                    if diffusion_predict_next:
                        # For UniDB predict-next objectives, model output is directly x0/next latent.
                        endpoint_pred = pred
                    elif hasattr(noise_schedule_obj, "predict_x0_from_noisy"):
                        try:
                            endpoint_pred = noise_schedule_obj.predict_x0_from_noisy(
                                x_t=x_noisy,
                                noise_target=pred,
                                t=t,
                                mu=x,
                            )
                        except Exception:
                            endpoint_pred = None
                    if endpoint_pred is not None:
                        endpoint_mse = F.mse_loss(endpoint_pred.float(), y.float(), reduction="mean")
                        endpoint_elems = y.numel()
                        endpoint_se_sum += float(endpoint_mse.detach().cpu()) * endpoint_elems
                        endpoint_elem_count += endpoint_elems
                        if want_spectral and spectral_cfg is not None:
                            endpoint_spec_vals = spectral_rmse_bands(endpoint_pred, y, **spectral_cfg)
                            for k, v in endpoint_spec_vals.items():
                                if k not in endpoint_spec_sum:
                                    endpoint_spec_sum[k] = torch.zeros((), dtype=torch.float64, device=y.device)
                                endpoint_spec_sum[k] += v.detach().to(dtype=torch.float64)
                            endpoint_spec_count += 1
            elif model_family == "flow_matching":
                objective = str(flow_objective or "default").lower()
                if objective in _SFM_FLOW_OBJECTIVES:
                    flow_noise_field = torch.zeros_like(x)
                    flow_noise_scalar = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
                else:
                    flow_noise_field, flow_noise_scalar = _build_flow_noise(
                        ref_tensor=x,
                        noise_std=float(flow_stochastic_std),
                        mode=flow_stochastic_mode,
                        perturb_source=flow_stochastic_perturb_source,
                    )
                if objective in _SFM_FLOW_OBJECTIVES:
                    if x.shape != y.shape:
                        raise ValueError(
                            "sfm_latent_source_denoise_concat requires x/y to have identical latent shapes; "
                            f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                        )
                    sigma_z = _resolve_sfm_sigma_z(
                        source=x,
                        target=y,
                        base_sigma_z=float(sfm_sigma_z),
                        sigma_min=float(sfm_sigma_min),
                        sigma_max=float(sfm_sigma_max),
                        adaptive=bool(sfm_adaptive_sigma),
                        ema_beta=float(sfm_sigma_ema_beta),
                        state=sfm_state,
                        update_state=False,
                    )
                    sigma_floor = max(float(sfm_sigma_min), 1e-6)
                    sigma = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    sigma = sigma * max(sigma_z - sigma_floor, 0.0) + sigma_floor
                    t_match = 1.0 - (sigma / max(sigma_z, sigma_floor))
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    t_view = t_match.view(-1, 1, 1, 1)
                    sigma_view = sigma.view(-1, 1, 1, 1)
                    sigma_z_scalar = torch.full((y.shape[0], 1), sigma_z, device=y.device, dtype=y.dtype)
                    eps = torch.randn_like(y)
                    e = (x - y) / max(sigma_z, sigma_floor)
                    x_t = y + sigma_view * (e + eps)
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                        cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                        cond_t = torch.cat([cond_t, sigma_z_scalar], dim=1)
                        x_in = torch.cat([x_t, x], dim=1)
                        pred = _forward_flow_concat(
                            model=model,
                            x_in=x_in,
                            cond_t=cond_t,
                            theta=theta,
                        )
                        weight = (sigma_z_scalar / sigma.clamp_min(sigma_floor)).pow(2).view(-1, 1, 1, 1)
                        denoise = ((pred.float() - y.float()).pow(2) * weight.float()).mean()
                        enc_reg = (e.float().pow(2)).mean()
                        vloss = denoise + float(sfm_encoder_reg_lambda) * enc_reg
                        if want_mae:
                            vmae = F.l1_loss(pred.float(), y.float())
                    v_obj_sum += float(vloss.detach().cpu())
                    v_obj_count += 1
                    if want_vrmse and channel_stats is not None:
                        _update_channel_stats(channel_stats, pred, y)
                    if want_spectral and spectral_cfg is not None:
                        spec_vals = spectral_rmse_bands(pred, y, **spectral_cfg)
                        for k, v in spec_vals.items():
                            if k not in spec_sum:
                                spec_sum[k] = torch.zeros((), dtype=torch.float64, device=y.device)
                            spec_sum[k] += v.detach().to(dtype=torch.float64)
                        spec_count += 1
                    u_t = y
                elif objective in _DBFM_FLOW_OBJECTIVES:
                    if x.shape != y.shape:
                        raise ValueError(
                            "dbfm_source_anchored requires x/y to have identical latent shapes; "
                            f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                        )
                    z0 = x
                    t_match = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    if flow_stochastic_perturb_source:
                        z0 = z0 + flow_noise_field
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = (1.0 - t_view) * z0 + t_view * y
                    u_t = y - z0
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                        cond_t = t_match
                        cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                        pred = _forward_flow_dbfm(
                            model=model,
                            x_t=x_t,
                            t_match=t_match,
                            cond_t=cond_t,
                            theta=theta,
                        )
                        endpoint_pred = x_t + (1.0 - t_view) * pred
                        vloss = loss_fns["flow"](
                            pred.float(),
                            u_t.float(),
                            target=y,
                            t=t_match,
                            endpoint_pred=endpoint_pred.float(),
                            target_state=y,
                        )
                        if want_mae:
                            vmae = F.l1_loss(pred.float(), u_t.float())
                    v_obj_sum += float(vloss.detach().cpu())
                    v_obj_count += 1
                    if want_vrmse and channel_stats is not None:
                        _update_channel_stats(channel_stats, pred, u_t)
                    if want_spectral and spectral_cfg is not None:
                        spec_vals = spectral_rmse_bands(pred, u_t, **spectral_cfg)
                        for k, v in spec_vals.items():
                            if k not in spec_sum:
                                spec_sum[k] = torch.zeros((), dtype=torch.float64, device=u_t.device)
                            spec_sum[k] += v.detach().to(dtype=torch.float64)
                        spec_count += 1
                elif objective in _RECTIFIED_FLOW_OBJECTIVES:
                    source_anchored = objective.startswith("rectified_flow_source_anchored")
                    noise_source_concat = objective in _NOISE_SOURCE_CONCAT_FLOW_OBJECTIVES
                    if source_anchored:
                        if x.shape != y.shape:
                            raise ValueError(
                                "rectified_flow_source_anchored requires x/y to have identical latent shapes; "
                                f"got x={tuple(x.shape)} y={tuple(y.shape)}."
                            )
                        z0 = x
                    else:
                        z0 = torch.randn_like(y)
                    t_match = torch.rand(y.shape[0], 1, device=y.device, dtype=y.dtype)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    if (source_anchored or noise_source_concat) and flow_stochastic_perturb_source and flow_noise_field is not None:
                        z0 = z0 + flow_noise_field
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = (1.0 - t_view) * z0 + t_view * y
                    u_t = y - z0
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                        cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                        cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                        if objective.endswith("_concat") or source_anchored or noise_source_concat:
                            source_ctx = x if noise_source_concat else z0
                            x_in = torch.cat([x_t, source_ctx], dim=1)
                            pred = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
                        else:
                            pred = _forward_flow_nonconcat(
                                model=model,
                                x_t=x_t,
                                t_match=t_match,
                                cond_t=cond_t,
                                theta=theta,
                            )
                        endpoint_pred = x_t + (1.0 - t_view) * pred
                        vloss = loss_fns["flow"](
                            pred.float(),
                            u_t.float(),
                            target=y,
                            t=t_match,
                            endpoint_pred=endpoint_pred.float(),
                            target_state=y,
                        )
                        if want_mae:
                            vmae = F.l1_loss(pred.float(), u_t.float())
                    v_obj_sum += float(vloss.detach().cpu())
                    v_obj_count += 1
                    if want_vrmse and channel_stats is not None:
                        _update_channel_stats(channel_stats, pred, u_t)
                    if want_spectral and spectral_cfg is not None:
                        spec_vals = spectral_rmse_bands(pred, u_t, **spectral_cfg)
                        for k, v in spec_vals.items():
                            if k not in spec_sum:
                                spec_sum[k] = torch.zeros((), dtype=torch.float64, device=u_t.device)
                            spec_sum[k] += v.detach().to(dtype=torch.float64)
                        spec_count += 1
                else:
                    x0_noise = torch.randn_like(y)
                    t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x0_noise, x1=y)
                    t_match = t_match.to(device).view(-1, 1)
                    t_match = torch.clamp(t_match, 1e-4, 1.0 - 1e-4)
                    t_view = t_match.view(-1, 1, 1, 1)
                    x_t = x_t.to(device)
                    u_t = u_t.to(device)
                    if use_chlast:
                        x_t = x_t.contiguous(memory_format=torch.channels_last)
                    cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                    cond_t = torch.cat([cond_t, flow_noise_scalar], dim=1) if flow_noise_scalar is not None else cond_t
                    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                        x_in = torch.cat([x_t, x], dim=1)
                        pred = _forward_flow_concat(model=model, x_in=x_in, cond_t=cond_t, theta=theta)
                        endpoint_pred = x_t + (1.0 - t_view) * pred
                        vloss = loss_fns["flow"](
                            pred.float(),
                            u_t.float(),
                            target=y,
                            t=t_match,
                            endpoint_pred=endpoint_pred.float(),
                            target_state=y,
                        )
                        if want_mae:
                            vmae = F.l1_loss(pred.float(), u_t.float())
                    v_obj_sum += float(vloss.detach().cpu())
                    v_obj_count += 1
                    if want_vrmse and channel_stats is not None:
                        _update_channel_stats(channel_stats, pred, u_t)
                    if want_spectral and spectral_cfg is not None:
                        spec_vals = spectral_rmse_bands(pred, u_t, **spectral_cfg)
                        for k, v in spec_vals.items():
                            if k not in spec_sum:
                                spec_sum[k] = torch.zeros((), dtype=torch.float64, device=u_t.device)
                            spec_sum[k] += v.detach().to(dtype=torch.float64)
                        spec_count += 1
                elems = u_t.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
                if want_endpoint_rmse:
                    with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                        endpoint_samples = _predict_flow_rollout_samples(
                            model=model,
                            x=x,
                            cond=cond,
                            theta=theta,
                            flow_objective=objective,
                            nfe=flow_val_nfe,
                            flow_noise_std=flow_eval_noise_std,
                            flow_noise_mode=flow_stochastic_mode,
                            flow_noise_perturb_source=flow_stochastic_perturb_source,
                            num_samples=flow_eval_num_samples,
                        )
                    endpoint_pred = endpoint_samples.mean(dim=0)
                    endpoint_mse = F.mse_loss(endpoint_pred.float(), y.float(), reduction="mean")
                    endpoint_elems = y.numel()
                    endpoint_se_sum += float(endpoint_mse.detach().cpu()) * endpoint_elems
                    endpoint_elem_count += endpoint_elems
                    if flow_prob_metrics_enabled:
                        term1_sum, term2_sum, var_sum, prob_elems = _flow_ensemble_batch_sums(endpoint_samples, y)
                        endpoint_crps_term1_sum += term1_sum
                        endpoint_crps_term2_sum += term2_sum
                        endpoint_crps_elem_count += prob_elems
                        endpoint_spread_var_sum += var_sum
                        endpoint_spread_elem_count += prob_elems
                    if want_spectral and spectral_cfg is not None:
                        endpoint_spec_vals = spectral_rmse_bands(endpoint_pred, y, **spectral_cfg)
                        for k, v in endpoint_spec_vals.items():
                            if k not in endpoint_spec_sum:
                                endpoint_spec_sum[k] = torch.zeros((), dtype=torch.float64, device=y.device)
                            endpoint_spec_sum[k] += v.detach().to(dtype=torch.float64)
                        endpoint_spec_count += 1
                if want_endpoint_rmse:
                    ps_mse = (endpoint_pred.detach().float() - y.detach().float())
                else:
                    ps_mse = (pred.detach().float() - u_t.detach().float())
                ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                for g, m in zip(batch["gid"], ps_mse):
                    v_gid_stats_local[g][0] += float(m)
                    v_gid_stats_local[g][1] += 1
            else:
                with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                    pred_out = _forward_surrogate(model, x, cond, theta)
                    pred_main = pred_out.get("recon", pred_out.get("pred")) if isinstance(pred_out, dict) else pred_out
                    vloss = F.mse_loss(pred_main.float(), y.float())
                    if want_mae:
                        vmae = F.l1_loss(pred_main.float(), y.float())
                    metric_owner = model.module if isinstance(model, DDP) else model
                    if hasattr(metric_owner, "compute_metrics"):
                        try:
                            task_metrics = metric_owner.compute_metrics(pred_main.detach(), y.detach())  # type: ignore[arg-type]
                        except Exception:
                            task_metrics = {}
                        for name, val in task_metrics.items():
                            val_task_metric_sum[name] = val_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                            val_task_metric_count[name] = val_task_metric_count.get(name, 0) + 1
                v_obj_sum += float(vloss.detach().cpu())
                v_obj_count += 1
                if want_vrmse and channel_stats is not None:
                    _update_channel_stats(channel_stats, pred_main, y)
                if want_spectral and spectral_cfg is not None:
                    spec_vals = spectral_rmse_bands(pred_main, y, **spectral_cfg)
                    for k, v in spec_vals.items():
                        if k not in spec_sum:
                            spec_sum[k] = torch.zeros((), dtype=torch.float64, device=y.device)
                        spec_sum[k] += v.detach().to(dtype=torch.float64)
                    spec_count += 1
                elems = y.numel()
                v_se_sum += float(vloss.detach().cpu()) * elems
                if want_mae and v_mae_sum is not None:
                    v_mae_sum += float(vmae.detach().cpu()) * elems
                v_elem_count += elems
                ps_mse = (pred_main.detach().float() - y.detach().float())
                ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                for g, m in zip(batch["gid"], ps_mse):
                    v_gid_stats_local[g][0] += float(m)
                    v_gid_stats_local[g][1] += 1

    v_se_sum_g, v_elem_count_g = _allreduce_sum_count(device, v_se_sum, v_elem_count)
    val_mse = v_se_sum_g / max(v_elem_count_g, 1)
    val_rmse = math.sqrt(max(val_mse, 0.0))
    val_mae = None
    if want_mae and v_mae_sum is not None:
        v_mae_sum_g, _ = _allreduce_sum_count(device, v_mae_sum, v_elem_count)
        val_mae = v_mae_sum_g / max(v_elem_count_g, 1)

    vrmse_stats = (
        _compute_vrmse_from_stats(channel_stats, vrmse_eps, ref_var=vrmse_ref_var) if want_vrmse else None
    )

    val_gid_stats_merged = None
    if v_gid_stats_local is not None:
        v_gid_stats_all = _all_gather_object(dict(v_gid_stats_local))
        if _rank0():
            val_gid_stats_merged = defaultdict(lambda: [0.0, 0])
            for d in v_gid_stats_all:
                for g, (s, c) in d.items():
                    val_gid_stats_merged[g][0] += float(s)
                    val_gid_stats_merged[g][1] += int(c)

    val_task_avg = (
        {k: v / max(val_task_metric_count.get(k, 1), 1) for k, v in val_task_metric_sum.items()}
        if val_task_metric_sum
        else {}
    )
    if sfm_state is not None and str(flow_objective or "").lower() in _SFM_FLOW_OBJECTIVES:
        if "sigma_z_ema" in sfm_state:
            val_task_avg["sfm_sigma_z"] = float(sfm_state["sigma_z_ema"])
        if "sigma_z_batch" in sfm_state:
            val_task_avg["sfm_sigma_z_batch_rmse"] = float(sfm_state["sigma_z_batch"])
    out = {
        "mse": val_mse,
        "rmse": val_rmse,
        "objective": (v_obj_sum / max(v_obj_count, 1)),
        "mae": val_mae,
        "gid_stats": val_gid_stats_merged,
        "task_avg": val_task_avg,
    }
    if want_endpoint_rmse and endpoint_elem_count > 0:
        endpoint_se_sum_g, endpoint_elem_count_g = _allreduce_sum_count(
            device, endpoint_se_sum, endpoint_elem_count
        )
        endpoint_mse = endpoint_se_sum_g / max(endpoint_elem_count_g, 1)
        out["endpoint_mse"] = endpoint_mse
        out["endpoint_rmse"] = math.sqrt(max(endpoint_mse, 0.0))
        if model_family == "flow_matching":
            out["endpoint_eval"] = "flow_rollout"
            out["endpoint_eval_nfe"] = int(flow_val_nfe)
            out["endpoint_eval_num_samples"] = int(flow_eval_num_samples)
            out["endpoint_eval_noise_std"] = float(flow_eval_noise_std)
    if flow_prob_metrics_enabled and want_endpoint_rmse and endpoint_crps_elem_count > 0:
        term1_sum_g, crps_elem_count_g = _allreduce_sum_count(
            device, endpoint_crps_term1_sum, endpoint_crps_elem_count
        )
        term2_sum_g, _ = _allreduce_sum_count(
            device, endpoint_crps_term2_sum, endpoint_crps_elem_count
        )
        spread_var_sum_g, spread_elem_count_g = _allreduce_sum_count(
            device, endpoint_spread_var_sum, endpoint_spread_elem_count
        )
        endpoint_crps = (term1_sum_g - term2_sum_g) / max(crps_elem_count_g, 1)
        spread_var_mean = spread_var_sum_g / max(spread_elem_count_g, 1)
        endpoint_spread = math.sqrt(max(spread_var_mean, 0.0))
        endpoint_rmse_val = float(out.get("endpoint_rmse", float("nan")))
        endpoint_ssr = (
            (endpoint_spread / endpoint_rmse_val)
            if (math.isfinite(endpoint_rmse_val) and endpoint_rmse_val > 0.0)
            else float("nan")
        )
        out["endpoint_crps"] = float(endpoint_crps)
        out["endpoint_spread"] = float(endpoint_spread)
        out["endpoint_ssr"] = float(endpoint_ssr)
        out["endpoint_ssr_distance"] = (
            float(abs(endpoint_ssr - 1.0))
            if math.isfinite(endpoint_ssr)
            else float("nan")
        )
    if want_spectral and endpoint_spec_sum:
        endpoint_count_t = _allreduce_sum_tensor(
            torch.tensor(float(endpoint_spec_count), device=device, dtype=torch.float64)
        )
        endpoint_denom = max(float(endpoint_count_t.item()), 1.0)
        for k, v in endpoint_spec_sum.items():
            v_sum = _allreduce_sum_tensor(v)
            out[f"endpoint_{k}"] = float((v_sum / endpoint_denom).item())
        if "endpoint_spec_rmse_all" in out:
            out["endpoint_spectral_rmse"] = out["endpoint_spec_rmse_all"]
    if want_spectral and spec_sum:
        count_t = _allreduce_sum_tensor(torch.tensor(float(spec_count), device=device, dtype=torch.float64))
        denom = max(float(count_t.item()), 1.0)
        for k, v in spec_sum.items():
            v_sum = _allreduce_sum_tensor(v)
            out[k] = float((v_sum / denom).item())
        if "spec_rmse_all" in out:
            out["spectral_rmse"] = out["spec_rmse_all"]
    if vrmse_stats is not None:
        out.update(vrmse_stats)
    return out
