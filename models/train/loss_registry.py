"""Loss registry for rapid solidification trainers."""

from __future__ import annotations

from typing import Dict, Any, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from models.train.core.loss_functions import relative_mass_error

try:
    from models.train.core.wavelet_weight import (
        wavelet_importance_weighted_mse,
        wavelet_importance_per_channel,
    )
except Exception:
    wavelet_importance_weighted_mse = None
    wavelet_importance_per_channel = None
try:
    from models.train.core.wavelet_mask import wavelet_saliency, time_dependent_mask, wavelet_energy_map
except Exception:
    wavelet_saliency = None
    time_dependent_mask = None
    wavelet_energy_map = None
try:
    from models.train.core.wavelet_ae_loss import wavelet_highfreq_mse, wavelet_energy_masked_mse
except Exception:
    wavelet_highfreq_mse = None
    wavelet_energy_masked_mse = None


def build_surrogate_loss(loss_cfg: Dict[str, Any]):
    """Returns a callable(pred, target, weight=None) computing the base surrogate loss."""
    base_kind = str(loss_cfg.get("base", "mse")).lower()
    vmse_eps = float(loss_cfg.get("vmse_eps", 1e-2))
    weight_wavelet = float(loss_cfg.get("weight_wavelet_loss", 0.0))
    wavelet_weight_cfg = loss_cfg.get("wavelet_weight", {})
    wavelet_theta = float(wavelet_weight_cfg.get("theta", loss_cfg.get("theta", 0.8)))
    wavelet_alpha = float(wavelet_weight_cfg.get("alpha", 1.25))
    wavelet_beta = float(wavelet_weight_cfg.get("beta_w", 6.0))
    wavelet_norm = bool(wavelet_weight_cfg.get("normalize", False))
    wavelet_clip = wavelet_weight_cfg.get("clip_max", None)
    wavelet_J = int(wavelet_weight_cfg.get("J", 1))
    wavelet_wave = wavelet_weight_cfg.get("wave", "haar")
    wavelet_mode = wavelet_weight_cfg.get("mode", "zero")
    recon_weight = float(loss_cfg.get("recon_weight", 1.0))
    scale_weight = float(loss_cfg.get("scale_consistency_weight", 0.0))
    scale_recon_weight = float(loss_cfg.get("scale_recon_weight", 0.0))
    scales_cfg = loss_cfg.get("scale_consistency_scales", None)
    if scales_cfg is None:
        levels = int(loss_cfg.get("scale_consistency_levels", 0))
        if levels > 0:
            scales = [2 ** i for i in range(levels)]
        elif scale_weight > 0.0:
            scales = [1, 2, 4]
        else:
            scales = []
    else:
        scales = list(scales_cfg)
    scales = [int(s) for s in scales if int(s) > 0]
    recon_scales_cfg = loss_cfg.get("scale_recon_scales", None)
    if recon_scales_cfg is None:
        levels = int(loss_cfg.get("scale_recon_levels", 0))
        if levels > 0:
            recon_scales = [2 ** i for i in range(levels)]
        elif scale_recon_weight > 0.0:
            recon_scales = [2, 4]
        else:
            recon_scales = []
    else:
        recon_scales = list(recon_scales_cfg)
    recon_scales = [int(s) for s in recon_scales if int(s) > 0]

    wavelet_energy_weight = float(loss_cfg.get("wavelet_energy_weight", 0.0))
    wavelet_energy_J = int(loss_cfg.get("wavelet_energy_J", 1))
    wavelet_energy_wave = loss_cfg.get("wavelet_wave", "haar")
    wavelet_energy_mode = loss_cfg.get("wavelet_mode", "zero")
    ae_wavelet_cfg = loss_cfg.get("ae_wavelet", {})
    ae_wavelet_enabled = bool(ae_wavelet_cfg.get("enabled", False))
    ae_wavelet_J = int(ae_wavelet_cfg.get("J", wavelet_energy_J))
    ae_wavelet_wave = ae_wavelet_cfg.get("wave", wavelet_energy_wave)
    ae_wavelet_mode = ae_wavelet_cfg.get("mode", wavelet_energy_mode)
    ae_wavelet_weight = float(ae_wavelet_cfg.get("lambda_wav", 0.0))
    ae_mask_weight = float(ae_wavelet_cfg.get("mask_weight", 0.0))
    ae_mask_alpha = float(ae_wavelet_cfg.get("mask_alpha", 1.0))
    ae_mask_percentile = float(ae_wavelet_cfg.get("mask_percentile", 0.93))
    ae_mask_warmup_steps = int(ae_wavelet_cfg.get("mask_warmup_steps", 0))
    ae_mask_warmup_epochs = int(ae_wavelet_cfg.get("mask_warmup_epochs", 0))

    def _spectral_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
        target_f = torch.fft.rfft2(target.float(), norm="ortho")
        return F.mse_loss(torch.abs(pred_f), torch.abs(target_f))

    def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def _multi_scale_spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for s in scales:
            if s <= 1:
                p, t = pred, target
            else:
                if pred.shape[-2] < s or pred.shape[-1] < s:
                    continue
                p = _downsample(pred, s)
                t = _downsample(target, s)
            losses.append(_spectral_mse(p, t))
        if not losses:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()

    def _multi_scale_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for s in recon_scales:
            if s <= 1:
                p, t = pred, target
            else:
                if pred.shape[-2] < s or pred.shape[-1] < s:
                    continue
                p = _downsample(pred, s)
                t = _downsample(target, s)
            losses.append(F.mse_loss(p, t))
        if not losses:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()

    def _wavelet_energy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if wavelet_energy_map is None:
            raise ImportError("pytorch_wavelets required for wavelet energy loss.")
        pred_e = wavelet_energy_map(pred.float(), J=wavelet_energy_J, wave=wavelet_energy_wave, mode=wavelet_energy_mode)
        tgt_e = wavelet_energy_map(target.float(), J=wavelet_energy_J, wave=wavelet_energy_wave, mode=wavelet_energy_mode)
        return F.mse_loss(pred_e, tgt_e)

    def _warmup_factor(step: int | None, epoch: int | None) -> float:
        if ae_mask_warmup_steps > 0 and step is not None:
            return min(1.0, float(step) / float(ae_mask_warmup_steps))
        if ae_mask_warmup_epochs > 0 and epoch is not None:
            return min(1.0, float(epoch) / float(ae_mask_warmup_epochs))
        return 1.0

    structured_cfg = loss_cfg.get("structured_latent", {})
    structured_weight = float(structured_cfg.get("weight", 0.0))
    structured_kind = str(structured_cfg.get("base", base_kind)).lower()
    structured_vmse_eps = float(structured_cfg.get("vmse_eps", vmse_eps))

    def _base_loss(pred: torch.Tensor, target: torch.Tensor, weight_map, kind: str, eps: float) -> torch.Tensor:
        if kind == "vmse":
            pred_f = pred.float()
            target_f = target.float()
            err_map = (pred_f - target_f).pow(2)
            if weight_map is not None:
                err_map = err_map * weight_map.float()
            err_flat = err_map.flatten(2)
            var = target_f.flatten(2).var(dim=2)
            eps_safe = max(float(eps), 0.0)
            denom = (var + eps_safe).clamp_min(1e-12)
            return (err_flat.mean(dim=2) / denom).mean()
        if kind == "mse":
            base_map = (pred - target).pow(2)
            if weight_map is not None:
                base_map = base_map * weight_map
            return base_map.mean()
        raise ValueError(f"Unknown base loss '{kind}'")

    def _loss(pred, target, weight=None, **kwargs):
        structured_pred = None
        pred_main = pred
        if isinstance(pred, dict):
            structured_pred = pred.get("structured", None)
            pred_main = pred.get("recon", pred.get("pred", None))
            if pred_main is None:
                raise ValueError("Structured outputs require 'recon' or 'pred' key.")
        base_mse = _base_loss(pred_main, target, weight, base_kind, vmse_eps)
        wavelet_ctx = torch.autocast(device_type="cuda", enabled=False) if pred_main.is_cuda else nullcontext()
        if weight_wavelet > 0.0:
            if wavelet_importance_weighted_mse is None and wavelet_importance_per_channel is None:
                raise ImportError("pytorch_wavelets required for wavelet-weighted loss.")
            with wavelet_ctx:
                pred_f = pred_main.float()
                target_f = target.float()
                if wavelet_importance_weighted_mse is None:
                    with torch.no_grad():
                        a_w, _ = wavelet_importance_per_channel(
                            target_f,
                            J=wavelet_J,
                            wave=wavelet_wave,
                            mode=wavelet_mode,
                            theta=wavelet_theta,
                            alpha=wavelet_alpha,
                            beta_w=wavelet_beta,
                        )
                        if wavelet_norm:
                            denom = a_w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            a_w = a_w / denom
                        if wavelet_clip is not None:
                            a_w = a_w.clamp_max(float(wavelet_clip))
                    wavelet_mse = (a_w * (pred_f - target_f).pow(2)).mean()
                else:
                    wavelet_mse = wavelet_importance_weighted_mse(
                        pred_f,
                        target_f,
                        J=wavelet_J,
                        wave=wavelet_wave,
                        mode=wavelet_mode,
                        theta=wavelet_theta,
                        alpha=wavelet_alpha,
                        beta_w=wavelet_beta,
                        normalize=wavelet_norm,
                        clip_max=wavelet_clip,
                    )
            base = (1.0 - weight_wavelet) * base_mse + weight_wavelet * wavelet_mse
        else:
            base = base_mse

        if structured_pred is not None and structured_weight > 0.0:
            struct_loss = _base_loss(structured_pred, target, None, structured_kind, structured_vmse_eps)
            base = base + structured_weight * struct_loss
        base = recon_weight * base
        if scale_weight > 0.0 and scales:
            base = base + scale_weight * _multi_scale_spectral_loss(pred, target)
        if scale_recon_weight > 0.0 and recon_scales:
            base = base + scale_recon_weight * _multi_scale_recon_loss(pred, target)
        if wavelet_energy_weight > 0.0:
            with wavelet_ctx:
                base = base + wavelet_energy_weight * _wavelet_energy_loss(pred.float(), target.float())
        if ae_wavelet_enabled:
            with wavelet_ctx:
                pred_f = pred.float()
                target_f = target.float()
                if ae_wavelet_weight > 0.0:
                    if wavelet_highfreq_mse is None:
                        raise ImportError("pytorch_wavelets required for AE wavelet loss.")
                    base = base + ae_wavelet_weight * wavelet_highfreq_mse(
                        pred_f,
                        target_f,
                        J=ae_wavelet_J,
                        wave=ae_wavelet_wave,
                        mode=ae_wavelet_mode,
                    )
                if ae_mask_weight > 0.0:
                    if wavelet_energy_masked_mse is None:
                        raise ImportError("pytorch_wavelets required for AE wavelet mask loss.")
                    warmup = _warmup_factor(kwargs.get("step"), kwargs.get("epoch"))
                    base = base + (ae_mask_weight * warmup) * wavelet_energy_masked_mse(
                        pred_f,
                        target_f,
                        J=ae_wavelet_J,
                        wave=ae_wavelet_wave,
                        mode=ae_wavelet_mode,
                        mask_alpha=ae_mask_alpha,
                        mask_percentile=ae_mask_percentile,
                    )
        return base

    return _loss


def build_diffusion_loss(loss_cfg: Dict[str, Any]):
    weight_wavelet = float(loss_cfg.get("weight_wavelet_loss", 0.0))
    wavelet_weight_cfg = loss_cfg.get("wavelet_weight", {})
    wavelet_theta = float(wavelet_weight_cfg.get("theta", loss_cfg.get("theta", 0.8)))
    wavelet_alpha = float(wavelet_weight_cfg.get("alpha", 1.25))
    wavelet_beta = float(wavelet_weight_cfg.get("beta_w", 6.0))
    wavelet_norm = bool(wavelet_weight_cfg.get("normalize", False))
    wavelet_clip = wavelet_weight_cfg.get("clip_max", None)
    wavelet_J = int(wavelet_weight_cfg.get("J", 1))
    wavelet_wave = wavelet_weight_cfg.get("wave", "haar")
    wavelet_mode = wavelet_weight_cfg.get("mode", "zero")
    mask_cfg = loss_cfg.get("wavelet_mask", {})
    mask_enabled = bool(mask_cfg.get("enabled", False))
    p_min = float(mask_cfg.get("p_min", 0.3))
    J = int(mask_cfg.get("J", 1))
    wave = mask_cfg.get("wave", "haar")
    mode = mask_cfg.get("mode", "zero")
    diff_w_cfg = loss_cfg.get("diffusion_weighting", "none")
    if isinstance(diff_w_cfg, dict):
        weighting_kind = str(diff_w_cfg.get("kind", "none")).lower()
        min_snr_gamma = float(diff_w_cfg.get("min_snr_gamma", diff_w_cfg.get("gamma", 5.0)))
    else:
        weighting_kind = str(diff_w_cfg).lower()
        min_snr_gamma = float(loss_cfg.get("min_snr_gamma", 5.0))
    diffusion_objective = str(loss_cfg.get("diffusion_objective", "epsilon_mse")).lower()
    diffusion_matching_loss = str(loss_cfg.get("diffusion_matching_loss", "l1")).lower()
    predict_next_objectives = {"unidb_predict_next", "predict_next", "predict_x0", "x0_mse", "next_field_mse"}
    physics_aux_cfg = loss_cfg.get("physics_aux", {})
    physics_aux_enabled = bool(physics_aux_cfg.get("enabled", False))
    aux_pixel_weight = float(physics_aux_cfg.get("pixel_weight", 0.0))
    aux_cmean_weight = float(
        physics_aux_cfg.get(
            "concentration_mean_weight",
            physics_aux_cfg.get("cmean_weight", 0.0),
        )
    )
    aux_mass_weight = float(physics_aux_cfg.get("mass_rel_weight", 0.0))
    aux_sub_batch = int(physics_aux_cfg.get("sub_batch_size", 0))
    conc_channels_raw = physics_aux_cfg.get("concentration_channels", [1])
    mass_channels_raw = physics_aux_cfg.get("mass_channels", [])
    if isinstance(conc_channels_raw, int):
        concentration_channels = [int(conc_channels_raw)]
    elif isinstance(conc_channels_raw, (list, tuple)):
        concentration_channels = [int(c) for c in conc_channels_raw]
    else:
        concentration_channels = [1]
    if isinstance(mass_channels_raw, int):
        mass_channels = [int(mass_channels_raw)]
    elif isinstance(mass_channels_raw, (list, tuple)):
        mass_channels = [int(c) for c in mass_channels_raw]
    else:
        mass_channels = []
    mass_pixel_h = float(physics_aux_cfg.get("pixel_h", 1.0))
    mass_pixel_w = float(physics_aux_cfg.get("pixel_w", 1.0))

    def _safe_select_channels(x: torch.Tensor, channels: list[int]) -> Optional[torch.Tensor]:
        if x.dim() < 2 or not channels:
            return None
        valid = [c for c in channels if 0 <= int(c) < int(x.shape[1])]
        if not valid:
            return None
        idx = torch.as_tensor(valid, device=x.device, dtype=torch.long)
        return torch.index_select(x, dim=1, index=idx)

    def _physics_aux_loss(state_pred: Optional[torch.Tensor], target_state: torch.Tensor) -> Optional[torch.Tensor]:
        if (not physics_aux_enabled) or state_pred is None:
            return None
        if aux_sub_batch > 0 and state_pred.shape[0] > aux_sub_batch:
            state_pred = state_pred[:aux_sub_batch]
            target_state = target_state[:aux_sub_batch]
        aux_total = state_pred.new_zeros(())
        has_term = False

        if aux_pixel_weight > 0.0:
            aux_total = aux_total + aux_pixel_weight * F.mse_loss(state_pred.float(), target_state.float())
            has_term = True

        if aux_cmean_weight > 0.0:
            pred_c = _safe_select_channels(state_pred, concentration_channels)
            true_c = _safe_select_channels(target_state, concentration_channels)
            if pred_c is not None and true_c is not None:
                pred_mean = pred_c.float().mean(dim=(-2, -1))
                true_mean = true_c.float().mean(dim=(-2, -1))
                aux_total = aux_total + aux_cmean_weight * F.mse_loss(pred_mean, true_mean)
                has_term = True

        if aux_mass_weight > 0.0 and state_pred.dim() == 4 and target_state.dim() == 4:
            valid_mass = [c for c in mass_channels if 0 <= int(c) < int(state_pred.shape[1])]
            if valid_mass:
                rel_terms = []
                for c in valid_mass:
                    rel = relative_mass_error(
                        state_pred[:, c].float(),
                        target_state[:, c].float(),
                        pixel_size=(mass_pixel_h, mass_pixel_w),
                    ).mean()
                    rel_terms.append(rel)
                if rel_terms:
                    aux_total = aux_total + aux_mass_weight * torch.stack(rel_terms).mean()
                    has_term = True

        return aux_total if has_term else None

    def _matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if diffusion_matching_loss in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        if diffusion_matching_loss in {"l2", "mse"}:
            return F.mse_loss(pred, target)
        raise ValueError(
            f"Unknown diffusion_matching_loss '{diffusion_matching_loss}'. Use one of {{'l1','l2'}}."
        )

    def _wavelet_weighted_loss(pred, target, y_ref):
        if wavelet_importance_weighted_mse is None and wavelet_importance_per_channel is None:
            raise ImportError("pytorch_wavelets required for wavelet-weighted loss.")
        if wavelet_importance_weighted_mse is None:
            with torch.no_grad():
                a_w, _ = wavelet_importance_per_channel(
                    y_ref,
                    J=wavelet_J,
                    wave=wavelet_wave,
                    mode=wavelet_mode,
                    theta=wavelet_theta,
                    alpha=wavelet_alpha,
                    beta_w=wavelet_beta,
                )
                if wavelet_norm:
                    denom = a_w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                    a_w = a_w / denom
                if wavelet_clip is not None:
                    a_w = a_w.clamp_max(float(wavelet_clip))
            return (a_w * (pred - target).pow(2)).mean()
        return wavelet_importance_weighted_mse(
            pred,
            target,
            y_ref=y_ref,
            J=wavelet_J,
            wave=wavelet_wave,
            mode=wavelet_mode,
            theta=wavelet_theta,
            alpha=wavelet_alpha,
            beta_w=wavelet_beta,
            normalize=wavelet_norm,
            clip_max=wavelet_clip,
        )

    def _loss(pred_eps, true_eps, **kwargs):
        target = kwargs.get("target", true_eps)
        t = kwargs.get("t", None)
        num_steps = kwargs.get("num_steps", None)
        log_snr_t = kwargs.get("log_snr", None)
        if diffusion_objective in predict_next_objectives:
            base = _matching_loss(pred_eps, target)
            if weight_wavelet > 0.0:
                weighted = _wavelet_weighted_loss(pred_eps, target, target)
                total = (1.0 - weight_wavelet) * base + weight_wavelet * weighted
            else:
                total = base
            aux = _physics_aux_loss(pred_eps, target)
            if aux is not None:
                total = total + aux
            return total
        if diffusion_objective in {"unidb_reverse_step", "unidb", "reverse_step_matching"}:
            schedule = kwargs.get("schedule", None)
            source = kwargs.get("source", None)
            noisy = kwargs.get("noisy", None)
            if schedule is None or getattr(schedule, "kind", None) != "unidb":
                raise ValueError("unidb_reverse_step objective requires a UniDB schedule.")
            if source is None or noisy is None:
                raise ValueError("unidb_reverse_step objective requires 'source' and 'noisy' tensors.")
            if t is None:
                raise ValueError("unidb_reverse_step objective requires sampled timesteps 't'.")
            pi = None
            if hasattr(schedule, "compute_residual_modulator"):
                try:
                    pi = schedule.compute_residual_modulator(target, source)
                except Exception:
                    pi = None
            score = schedule.get_score_from_noise(pred_eps, t, pi=pi)
            xt_1_expectation = schedule.reverse_sde_step_mean(noisy, score, t, source)
            xt_1_optimum = schedule.reverse_optimum_step(noisy, target, t, source)
            total = _matching_loss(xt_1_expectation, xt_1_optimum)
            endpoint_pred = None
            if hasattr(schedule, "predict_x0_from_noisy"):
                try:
                    endpoint_pred = schedule.predict_x0_from_noisy(
                        x_t=noisy,
                        noise_target=pred_eps,
                        t=t,
                        mu=source,
                    )
                except Exception:
                    endpoint_pred = None
            aux = _physics_aux_loss(endpoint_pred, target)
            if aux is not None:
                total = total + aux
            return total

        loss_map = (pred_eps - true_eps).pow(2)

        if mask_enabled and wavelet_saliency is not None and time_dependent_mask is not None and t is not None and num_steps:
            # LWD-style gating: wavelet saliency + time-dependent mask
            with torch.no_grad():
                sal = wavelet_saliency(target, J=J, wave=wave, mode=mode)
                tau = t.float() / float(max(int(num_steps) - 1, 1))
                mask = time_dependent_mask(sal, tau, p_min=p_min)
            masked = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
            return masked

        if weighting_kind in {"none", ""}:
            base = loss_map.mean()
        elif weighting_kind in {"min_snr", "min_snr_gamma"}:
            if log_snr_t is None:
                base = loss_map.mean()
            else:
                snr = torch.exp(log_snr_t.float()).clamp_min(1e-8)
                w = torch.clamp_max(snr, min_snr_gamma) / snr
                while w.dim() < loss_map.dim():
                    w = w.unsqueeze(-1)
                base = (loss_map * w.to(device=loss_map.device, dtype=loss_map.dtype)).mean()
        else:
            raise ValueError(
                f"Unknown diffusion_weighting.kind '{weighting_kind}'. "
                "Use one of {'none','min_snr'}."
            )
        if weight_wavelet > 0.0:
            weighted = _wavelet_weighted_loss(pred_eps, true_eps, target)
            total = (1.0 - weight_wavelet) * base + weight_wavelet * weighted
        else:
            total = base

        endpoint_pred = None
        schedule = kwargs.get("schedule", None)
        source = kwargs.get("source", None)
        noisy = kwargs.get("noisy", None)
        if schedule is not None and hasattr(schedule, "predict_x0_from_noisy") and source is not None and noisy is not None and t is not None:
            try:
                endpoint_pred = schedule.predict_x0_from_noisy(
                    x_t=noisy,
                    noise_target=pred_eps,
                    t=t,
                    mu=source,
                )
            except Exception:
                endpoint_pred = None
        aux = _physics_aux_loss(endpoint_pred, target)
        if aux is not None:
            total = total + aux
        return total

    return _loss


def build_flow_loss(loss_cfg: Dict[str, Any]):
    mask_cfg = loss_cfg.get("wavelet_mask", {})
    mask_enabled = bool(mask_cfg.get("enabled", False))
    p_min = float(mask_cfg.get("p_min", 0.3))
    J = int(mask_cfg.get("J", 1))
    wave = mask_cfg.get("wave", "haar")
    mode = mask_cfg.get("mode", "zero")
    flow_matching_loss = str(loss_cfg.get("flow_matching_loss", "l2")).lower()
    physics_aux_cfg = loss_cfg.get("physics_aux", {})
    physics_aux_enabled = bool(physics_aux_cfg.get("enabled", False))
    aux_pixel_weight = float(physics_aux_cfg.get("pixel_weight", 0.0))
    aux_cmean_weight = float(
        physics_aux_cfg.get(
            "concentration_mean_weight",
            physics_aux_cfg.get("cmean_weight", 0.0),
        )
    )
    aux_mass_weight = float(physics_aux_cfg.get("mass_rel_weight", 0.0))
    aux_sub_batch = int(physics_aux_cfg.get("sub_batch_size", 0))
    conc_channels_raw = physics_aux_cfg.get("concentration_channels", [1])
    mass_channels_raw = physics_aux_cfg.get("mass_channels", [])
    if isinstance(conc_channels_raw, int):
        concentration_channels = [int(conc_channels_raw)]
    elif isinstance(conc_channels_raw, (list, tuple)):
        concentration_channels = [int(c) for c in conc_channels_raw]
    else:
        concentration_channels = [1]
    if isinstance(mass_channels_raw, int):
        mass_channels = [int(mass_channels_raw)]
    elif isinstance(mass_channels_raw, (list, tuple)):
        mass_channels = [int(c) for c in mass_channels_raw]
    else:
        mass_channels = []
    mass_pixel_h = float(physics_aux_cfg.get("pixel_h", 1.0))
    mass_pixel_w = float(physics_aux_cfg.get("pixel_w", 1.0))

    def _safe_select_channels(x: torch.Tensor, channels: list[int]) -> Optional[torch.Tensor]:
        if x.dim() < 2 or not channels:
            return None
        valid = [c for c in channels if 0 <= int(c) < int(x.shape[1])]
        if not valid:
            return None
        idx = torch.as_tensor(valid, device=x.device, dtype=torch.long)
        return torch.index_select(x, dim=1, index=idx)

    def _physics_aux_loss(state_pred: Optional[torch.Tensor], target_state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (not physics_aux_enabled) or state_pred is None or target_state is None:
            return None
        if aux_sub_batch > 0 and state_pred.shape[0] > aux_sub_batch:
            state_pred = state_pred[:aux_sub_batch]
            target_state = target_state[:aux_sub_batch]
        aux_total = state_pred.new_zeros(())
        has_term = False

        if aux_pixel_weight > 0.0:
            aux_total = aux_total + aux_pixel_weight * F.mse_loss(state_pred.float(), target_state.float())
            has_term = True

        if aux_cmean_weight > 0.0:
            pred_c = _safe_select_channels(state_pred, concentration_channels)
            true_c = _safe_select_channels(target_state, concentration_channels)
            if pred_c is not None and true_c is not None:
                pred_mean = pred_c.float().mean(dim=(-2, -1))
                true_mean = true_c.float().mean(dim=(-2, -1))
                aux_total = aux_total + aux_cmean_weight * F.mse_loss(pred_mean, true_mean)
                has_term = True

        if aux_mass_weight > 0.0 and state_pred.dim() == 4 and target_state.dim() == 4:
            valid_mass = [c for c in mass_channels if 0 <= int(c) < int(state_pred.shape[1])]
            if valid_mass:
                rel_terms = []
                for c in valid_mass:
                    rel = relative_mass_error(
                        state_pred[:, c].float(),
                        target_state[:, c].float(),
                        pixel_size=(mass_pixel_h, mass_pixel_w),
                    ).mean()
                    rel_terms.append(rel)
                if rel_terms:
                    aux_total = aux_total + aux_mass_weight * torch.stack(rel_terms).mean()
                    has_term = True

        return aux_total if has_term else None

    def _loss(pred_u, true_u, **kwargs):
        target = kwargs.get("target", None)
        tau = kwargs.get("t", None)
        if flow_matching_loss in {"l2", "mse"}:
            loss_map = (pred_u - true_u).pow(2)
        elif flow_matching_loss in {"l1", "mae"}:
            loss_map = (pred_u - true_u).abs()
        else:
            raise ValueError(
                f"Unknown flow_matching_loss '{flow_matching_loss}'. "
                "Use one of {'l1','l2'}."
            )

        if mask_enabled and wavelet_saliency is not None and time_dependent_mask is not None and target is not None and tau is not None:
            with torch.no_grad():
                sal = wavelet_saliency(target, J=J, wave=wave, mode=mode)
                tau_f = tau.view(-1).float()
                mask = time_dependent_mask(sal, tau_f, p_min=p_min)
            masked = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
            base = masked
        else:
            base = loss_map.mean()

        endpoint_pred = kwargs.get("endpoint_pred", None)
        target_state = kwargs.get("target_state", target)
        aux = _physics_aux_loss(endpoint_pred, target_state)
        if aux is not None:
            base = base + aux
        return base

    return _loss
