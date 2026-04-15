"""Loss registry for rapid solidification trainers."""

from __future__ import annotations

from typing import Dict, Any, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from models.train.core.loss_functions import relative_mass_error, composite_loss

try:
    from models.train.core.wavelet_weight import (
        wavelet_importance_weighted_mse,
        wavelet_importance_per_channel,
        wavelet_bandpass_importance_per_channel,
        wavelet_multiband_importance_per_channel,
    )
except Exception:
    wavelet_importance_weighted_mse = None
    wavelet_importance_per_channel = None
    wavelet_bandpass_importance_per_channel = None
    wavelet_multiband_importance_per_channel = None
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
    wavelet_thicken_radius = int(wavelet_weight_cfg.get("thicken_radius", 0))
    wavelet_thicken_mode = str(wavelet_weight_cfg.get("thicken_mode", "none")).strip().lower()
    wavelet_shared_map = bool(wavelet_weight_cfg.get("shared_map", False))
    wavelet_shared_channels_raw = wavelet_weight_cfg.get("shared_channels", [0, 1])
    if isinstance(wavelet_shared_channels_raw, int):
        wavelet_shared_channels = [int(wavelet_shared_channels_raw)]
    elif isinstance(wavelet_shared_channels_raw, (list, tuple)):
        wavelet_shared_channels = [int(c) for c in wavelet_shared_channels_raw]
    else:
        wavelet_shared_channels = []
    wavelet_shared_reduce = str(wavelet_weight_cfg.get("shared_reduce", "mean")).strip().lower()
    recon_weight = float(loss_cfg.get("recon_weight", 1.0))
    physics_aux_cfg = loss_cfg.get("physics_aux", {})
    physics_aux_enabled = bool(physics_aux_cfg.get("enabled", False))
    aux_cmean_weight = float(
        physics_aux_cfg.get(
            "concentration_mean_weight",
            physics_aux_cfg.get("cmean_weight", 0.0),
        )
    )
    conc_channels_raw = physics_aux_cfg.get("concentration_channels", [1])
    if isinstance(conc_channels_raw, int):
        concentration_channels = [int(conc_channels_raw)]
    elif isinstance(conc_channels_raw, (list, tuple)):
        concentration_channels = [int(c) for c in conc_channels_raw]
    else:
        concentration_channels = [1]
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

    wavelet_curriculum_cfg = loss_cfg.get("wavelet_curriculum", {})
    wavelet_curriculum_enabled = bool(wavelet_curriculum_cfg.get("enabled", False))
    wavelet_curriculum_stages = wavelet_curriculum_cfg.get("stages", [])
    if not isinstance(wavelet_curriculum_stages, (list, tuple)):
        wavelet_curriculum_stages = []

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

    def _resolve_wavelet_curriculum(epoch_val: int | None) -> tuple[float, float]:
        mix = float(weight_wavelet)
        gain = 1.0
        if (not wavelet_curriculum_enabled) or epoch_val is None:
            return float(max(0.0, min(1.0, mix))), float(max(0.0, gain))

        selected = None
        try:
            e = int(epoch_val)
        except Exception:
            return float(max(0.0, min(1.0, mix))), float(max(0.0, gain))

        for stage in wavelet_curriculum_stages:
            if not isinstance(stage, dict):
                continue
            end_epoch = stage.get("max_epoch", stage.get("end_epoch", stage.get("until_epoch", None)))
            if end_epoch is None:
                selected = stage
                break
            try:
                if e <= int(end_epoch):
                    selected = stage
                    break
            except Exception:
                continue

        if selected is None and wavelet_curriculum_stages:
            last = wavelet_curriculum_stages[-1]
            if isinstance(last, dict):
                selected = last

        if isinstance(selected, dict):
            if "mix" in selected:
                mix = float(selected.get("mix", mix))
            if "gain" in selected:
                gain = float(selected.get("gain", gain))
            elif "multiplier" in selected:
                gain = float(selected.get("multiplier", gain))

        mix = float(max(0.0, min(1.0, mix)))
        gain = float(max(0.0, gain))
        return mix, gain

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

    def _safe_select_channels(x: torch.Tensor, channels: list[int]) -> Optional[torch.Tensor]:
        if x.dim() < 2 or not channels:
            return None
        valid = [c for c in channels if 0 <= int(c) < int(x.shape[1])]
        if not valid:
            return None
        idx = torch.as_tensor(valid, device=x.device, dtype=torch.long)
        return x.index_select(1, idx)

    def _thicken_weight_map(a_w: torch.Tensor) -> torch.Tensor:
        if wavelet_thicken_radius <= 0 or wavelet_thicken_mode in {"none", "off", "false", "0"}:
            return a_w
        r = int(max(0, wavelet_thicken_radius))
        k = int(2 * r + 1)
        if k <= 1:
            return a_w
        if wavelet_thicken_mode in {"max", "dilate"}:
            return F.max_pool2d(a_w, kernel_size=k, stride=1, padding=r)
        if wavelet_thicken_mode in {"mean", "avg", "blur"}:
            return F.avg_pool2d(a_w, kernel_size=k, stride=1, padding=r)
        raise ValueError(
            "Unknown wavelet thicken mode '%s'. Use one of {'none','max','mean'}." % wavelet_thicken_mode
        )

    def _apply_shared_wavelet_map(a_w: torch.Tensor) -> torch.Tensor:
        if (not wavelet_shared_map) or (a_w.dim() != 4) or (a_w.shape[1] <= 1):
            return a_w
        c = int(a_w.shape[1])
        if wavelet_shared_channels:
            idx = [i for i in wavelet_shared_channels if 0 <= int(i) < c]
        else:
            idx = list(range(c))
        if not idx:
            idx = list(range(c))
        sel = a_w[:, idx, :, :]
        if wavelet_shared_reduce in {"max", "amax"}:
            shared = sel.amax(dim=1, keepdim=True)
        elif wavelet_shared_reduce in {"sum"}:
            shared = sel.sum(dim=1, keepdim=True)
        else:
            shared = sel.mean(dim=1, keepdim=True)
        return shared.expand(-1, c, -1, -1)

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
        epoch_val = kwargs.get("epoch", None)
        wave_mix, wave_gain = _resolve_wavelet_curriculum(epoch_val)
        if wave_mix > 0.0:
            with wavelet_ctx:
                pred_f = pred_main.float()
                target_f = target.float()
                if wavelet_importance_weighted_mse is None and wavelet_importance_per_channel is None:
                    if wavelet_energy_map is None:
                        raise ImportError("pytorch_wavelets required for wavelet-weighted loss.")
                    with torch.no_grad():
                        a_w = wavelet_energy_map(
                            target_f,
                            J=wavelet_J,
                            wave=wavelet_wave,
                            mode=wavelet_mode,
                        )
                        if wavelet_norm:
                            denom = a_w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            a_w = a_w / denom
                        if wavelet_clip is not None:
                            a_w = a_w.clamp_max(float(wavelet_clip))
                        a_w = _thicken_weight_map(a_w)
                        a_w = _apply_shared_wavelet_map(a_w)
                    wavelet_mse = (a_w * (pred_f - target_f).pow(2)).mean()
                elif wavelet_importance_weighted_mse is None:
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
                        a_w = _thicken_weight_map(a_w)
                        a_w = _apply_shared_wavelet_map(a_w)
                    wavelet_mse = (a_w * (pred_f - target_f).pow(2)).mean()
                else:
                    # Backward-compatible path: older wavelet helpers do not accept
                    # normalize/clip keyword arguments.
                    if wavelet_norm or (wavelet_clip is not None) or (wavelet_thicken_radius > 0 and wavelet_thicken_mode not in {"none", "off", "false", "0"}):
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
                            a_w = _thicken_weight_map(a_w)
                            a_w = _apply_shared_wavelet_map(a_w)
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
                        )
            base = (1.0 - wave_mix) * base_mse + wave_mix * (wave_gain * wavelet_mse)
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
        if physics_aux_enabled and aux_cmean_weight > 0.0:
            pred_c = _safe_select_channels(pred_main, concentration_channels)
            true_c = _safe_select_channels(target, concentration_channels)
            if pred_c is not None and true_c is not None:
                pred_mean = pred_c.float().mean(dim=(-2, -1))
                true_mean = true_c.float().mean(dim=(-2, -1))
                base = base + aux_cmean_weight * F.mse_loss(pred_mean, true_mean)
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
    wavelet_strategy = str(wavelet_weight_cfg.get("strategy", "importance")).lower()
    wavelet_level_weights_raw = wavelet_weight_cfg.get("level_weights", None)
    if isinstance(wavelet_level_weights_raw, (list, tuple)):
        wavelet_level_weights = [float(v) for v in wavelet_level_weights_raw]
    elif isinstance(wavelet_level_weights_raw, str):
        wavelet_level_weights = [float(v.strip()) for v in wavelet_level_weights_raw.split(",") if v.strip()]
    else:
        wavelet_level_weights = None
    wavelet_lowpass_weight = float(wavelet_weight_cfg.get("lowpass_weight", 0.2))
    wavelet_power = float(wavelet_weight_cfg.get("power", 1.5))
    wavelet_norm_quantile = float(wavelet_weight_cfg.get("norm_quantile", 0.99))
    wavelet_combine_norm = bool(wavelet_weight_cfg.get("combine_norm", True))
    wavelet_rescale_max = bool(wavelet_weight_cfg.get("rescale_max", False))
    wavelet_mask_quantile = float(wavelet_weight_cfg.get("mask_quantile", 0.95))
    wavelet_band_sigma = float(wavelet_weight_cfg.get("band_sigma", 16.0))
    wavelet_clip_min = wavelet_weight_cfg.get("clip_min", None)
    wavelet_thicken_radius = int(wavelet_weight_cfg.get("thicken_radius", 0))
    wavelet_thicken_mode = str(wavelet_weight_cfg.get("thicken_mode", "none")).strip().lower()
    wavelet_shared_map = bool(wavelet_weight_cfg.get("shared_map", False))
    wavelet_shared_channels_raw = wavelet_weight_cfg.get("shared_channels", [0, 1])
    if isinstance(wavelet_shared_channels_raw, int):
        wavelet_shared_channels = [int(wavelet_shared_channels_raw)]
    elif isinstance(wavelet_shared_channels_raw, (list, tuple)):
        wavelet_shared_channels = [int(c) for c in wavelet_shared_channels_raw]
    else:
        wavelet_shared_channels = []
    wavelet_shared_reduce = str(wavelet_weight_cfg.get("shared_reduce", "mean")).strip().lower()
    mask_cfg = loss_cfg.get("wavelet_mask", {})
    mask_enabled = bool(mask_cfg.get("enabled", False))
    p_min = float(mask_cfg.get("p_min", 0.3))
    p_max = float(mask_cfg.get("p_max", 0.95))
    mask_late_focus = bool(mask_cfg.get("late_focus", True))
    mask_smooth = float(mask_cfg.get("smooth", 0.0))
    mask_saliency_quantile = float(mask_cfg.get("saliency_quantile", 0.99))
    J = int(mask_cfg.get("J", 1))
    wave = mask_cfg.get("wave", "haar")
    mode = mask_cfg.get("mode", "zero")

    wavelet_curriculum_cfg = loss_cfg.get("wavelet_curriculum", {})
    wavelet_curriculum_enabled = bool(wavelet_curriculum_cfg.get("enabled", False))
    wavelet_curriculum_stages = wavelet_curriculum_cfg.get("stages", [])
    if not isinstance(wavelet_curriculum_stages, (list, tuple)):
        wavelet_curriculum_stages = []
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
    fdbm_drift_objectives = {"fdbm_drift_mse", "fdbm_drift", "bridge_fdbm_drift"}
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
    aux_curv_weight = float(physics_aux_cfg.get("curv_mean_rel_weight", 0.0))
    aux_perim_weight = float(physics_aux_cfg.get("perim_rel_weight", 0.0))
    aux_edge_weight = float(physics_aux_cfg.get("edge_strength_rel_weight", 0.0))
    aux_phi_channel = int(physics_aux_cfg.get("phi_channel", 0))
    aux_geom_cfg = physics_aux_cfg.get("geom", {}) if isinstance(physics_aux_cfg.get("geom", {}), dict) else {}
    aux_geom_level = float(aux_geom_cfg.get("level", 0.5))
    aux_geom_eps_band = float(aux_geom_cfg.get("eps_band", 0.02))
    aux_geom_eps_delta = float(aux_geom_cfg.get("eps_delta", 0.02))
    aux_geom_rel_eps = float(aux_geom_cfg.get("rel_eps", 1e-3))
    aux_geom_grad_eps = float(aux_geom_cfg.get("grad_eps", 1e-6))
    aux_geom_pixel_h = float(aux_geom_cfg.get("pixel_h", 1.0))
    aux_geom_pixel_w = float(aux_geom_cfg.get("pixel_w", 1.0))
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

    def _safe_select_channels(x: torch.Tensor, channels: list[int]) -> Optional[torch.Tensor]:
        if x.dim() < 2 or not channels:
            return None
        valid = [c for c in channels if 0 <= int(c) < int(x.shape[1])]
        if not valid:
            return None
        idx = torch.as_tensor(valid, device=x.device, dtype=torch.long)
        return torch.index_select(x, dim=1, index=idx)

    def _resolve_wavelet_curriculum(epoch_val: Optional[int]) -> tuple[float, float, float, float, float, bool]:
        mix = float(weight_wavelet)
        gain = 1.0
        pmin_cur = float(p_min)
        pmax_cur = float(p_max)
        sal_q_cur = float(mask_saliency_quantile)
        mask_on = bool(mask_enabled)
        if (not wavelet_curriculum_enabled) or epoch_val is None:
            return mix, gain, pmin_cur, pmax_cur, sal_q_cur, mask_on

        try:
            e = int(epoch_val)
        except Exception:
            return mix, gain, pmin_cur, pmax_cur, sal_q_cur, mask_on

        selected = None
        for st in wavelet_curriculum_stages:
            if not isinstance(st, dict):
                continue
            end_val = st.get("max_epoch", st.get("end_epoch", st.get("until_epoch", None)))
            if end_val is None:
                selected = st
                break
            try:
                if e <= int(end_val):
                    selected = st
                    break
            except Exception:
                continue

        if selected is None and wavelet_curriculum_stages:
            last = wavelet_curriculum_stages[-1]
            if isinstance(last, dict):
                selected = last

        if isinstance(selected, dict):
            if "mix" in selected:
                mix = float(selected.get("mix", mix))
            elif "weight_wavelet_loss" in selected:
                mix = float(selected.get("weight_wavelet_loss", mix))
            elif "mix_scale" in selected:
                mix = mix * float(selected.get("mix_scale", 1.0))

            if "gain" in selected:
                gain = float(selected.get("gain", gain))
            elif "multiplier" in selected:
                gain = float(selected.get("multiplier", gain))

            if "p_min" in selected:
                pmin_cur = float(selected.get("p_min", pmin_cur))
            if "p_max" in selected:
                pmax_cur = float(selected.get("p_max", pmax_cur))
            if "saliency_quantile" in selected:
                sal_q_cur = float(selected.get("saliency_quantile", sal_q_cur))
            if "mask_enabled" in selected:
                mask_on = bool(selected.get("mask_enabled"))

        mix = float(max(0.0, min(1.0, mix)))
        gain = float(max(0.0, gain))
        pmin_cur = float(max(0.0, min(0.999, pmin_cur)))
        pmax_cur = float(max(pmin_cur, min(0.999, pmax_cur)))
        sal_q_cur = float(max(0.5, min(0.999, sal_q_cur)))
        return mix, gain, pmin_cur, pmax_cur, sal_q_cur, mask_on

    def _physics_aux_loss(state_pred: Optional[torch.Tensor], target_state: torch.Tensor) -> Optional[torch.Tensor]:
        if (not physics_aux_enabled) or state_pred is None:
            return None
        if aux_sub_batch > 0 and state_pred.shape[0] > aux_sub_batch:
            state_pred = state_pred[:aux_sub_batch]
            target_state = target_state[:aux_sub_batch]
        aux_total = state_pred.new_zeros(())
        has_term = False

        if any(
            w > 0.0
            for w in (
                aux_pixel_weight,
                aux_mass_weight,
                aux_curv_weight,
                aux_perim_weight,
                aux_edge_weight,
            )
        ):
            mass_ch = mass_channels if aux_mass_weight > 0.0 else []
            phi_ch = aux_phi_channel if any(w > 0.0 for w in (aux_curv_weight, aux_perim_weight, aux_edge_weight)) else None
            comp_weights = {
                "mse": aux_pixel_weight,
                "mass_rel": aux_mass_weight,
                "curv_mean_rel": aux_curv_weight,
                "perim_rel": aux_perim_weight,
                "edge_strength_rel": aux_edge_weight,
            }
            geom_cfg = {
                "level": aux_geom_level,
                "eps_band": aux_geom_eps_band,
                "eps_delta": aux_geom_eps_delta,
                "rel_eps": aux_geom_rel_eps,
                "grad_eps": aux_geom_grad_eps,
                "pixel_h": aux_geom_pixel_h,
                "pixel_w": aux_geom_pixel_w,
            }
            comp, _ = composite_loss(
                state_pred.float(),
                target_state.float(),
                phi_channel=phi_ch,
                mass_channels=mass_ch,
                weights=comp_weights,
                geom_cfg=geom_cfg,
            )
            aux_total = aux_total + comp
            has_term = True

        if aux_cmean_weight > 0.0:
            pred_c = _safe_select_channels(state_pred, concentration_channels)
            true_c = _safe_select_channels(target_state, concentration_channels)
            if pred_c is not None and true_c is not None:
                pred_mean = pred_c.float().mean(dim=(-2, -1))
                true_mean = true_c.float().mean(dim=(-2, -1))
                aux_total = aux_total + aux_cmean_weight * F.mse_loss(pred_mean, true_mean)
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

    def _thicken_weight_map(a_w: torch.Tensor) -> torch.Tensor:
        if wavelet_thicken_radius <= 0 or wavelet_thicken_mode in {"none", "off", "false", "0"}:
            return a_w
        r = int(max(0, wavelet_thicken_radius))
        k = int(2 * r + 1)
        if k <= 1:
            return a_w
        if wavelet_thicken_mode in {"max", "dilate"}:
            return F.max_pool2d(a_w, kernel_size=k, stride=1, padding=r)
        if wavelet_thicken_mode in {"mean", "avg", "blur"}:
            return F.avg_pool2d(a_w, kernel_size=k, stride=1, padding=r)
        raise ValueError(
            f"Unknown wavelet thicken mode '{wavelet_thicken_mode}'. Use one of {{'none','max','mean'}}."
        )

    def _apply_shared_wavelet_map(a_w: torch.Tensor) -> torch.Tensor:
        if (not wavelet_shared_map) or (a_w.dim() != 4) or (a_w.shape[1] <= 1):
            return a_w
        c = int(a_w.shape[1])
        if wavelet_shared_channels:
            idx = [i for i in wavelet_shared_channels if 0 <= int(i) < c]
        else:
            idx = list(range(c))
        if not idx:
            idx = list(range(c))
        sel = a_w[:, idx, :, :]
        if wavelet_shared_reduce in {"max", "amax"}:
            shared = sel.amax(dim=1, keepdim=True)
        elif wavelet_shared_reduce in {"sum"}:
            shared = sel.sum(dim=1, keepdim=True)
        else:
            shared = sel.mean(dim=1, keepdim=True)
        return shared.expand(-1, c, -1, -1)

    def _wavelet_weighted_loss(pred, target, y_ref):
        strategy = wavelet_strategy

        if strategy in {"multiband", "multi", "midband"} and wavelet_multiband_importance_per_channel is not None:
            try:
                with torch.no_grad():
                    a_w, _ = wavelet_multiband_importance_per_channel(
                        y_ref,
                        J=wavelet_J,
                        wave=wavelet_wave,
                        mode=wavelet_mode,
                        level_weights=wavelet_level_weights,
                        lowpass_weight=wavelet_lowpass_weight,
                        beta_w=wavelet_beta,
                        power=wavelet_power,
                        norm_quantile=wavelet_norm_quantile,
                        normalize_mean=wavelet_norm,
                        rescale_max=wavelet_rescale_max,
                        clip_min=(float(wavelet_clip_min) if wavelet_clip_min is not None else None),
                        clip_max=(float(wavelet_clip) if wavelet_clip is not None else None),
                        combine_norm=wavelet_combine_norm,
                    )
                a_w = _thicken_weight_map(a_w)
                a_w = _apply_shared_wavelet_map(a_w)
                return (a_w * (pred - target).pow(2)).mean()
            except ImportError:
                # Fall back to default importance weighting when wavelet backend is unavailable.
                pass

        if strategy in {"bandpass", "band"} and wavelet_bandpass_importance_per_channel is not None:
            try:
                with torch.no_grad():
                    a_w, _ = wavelet_bandpass_importance_per_channel(
                        y_ref,
                        J=wavelet_J,
                        wave=wavelet_wave,
                        mode=wavelet_mode,
                        beta_w=wavelet_beta,
                        power=wavelet_power,
                        band_sigma=wavelet_band_sigma,
                        mask_quantile=wavelet_mask_quantile,
                        norm_quantile=wavelet_norm_quantile,
                        normalize_mean=wavelet_norm,
                        clip_min=(float(wavelet_clip_min) if wavelet_clip_min is not None else None),
                        clip_max=(float(wavelet_clip) if wavelet_clip is not None else None),
                    )
                a_w = _thicken_weight_map(a_w)
                a_w = _apply_shared_wavelet_map(a_w)
                return (a_w * (pred - target).pow(2)).mean()
            except ImportError:
                # Fall back to default importance weighting when wavelet backend is unavailable.
                pass

        if wavelet_importance_weighted_mse is None and wavelet_importance_per_channel is None:
            if wavelet_energy_map is None:
                raise ImportError("pytorch_wavelets required for wavelet-weighted loss.")
            with torch.no_grad():
                a_w = wavelet_energy_map(
                    y_ref,
                    J=wavelet_J,
                    wave=wavelet_wave,
                    mode=wavelet_mode,
                )
                if wavelet_norm:
                    denom = a_w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                    a_w = a_w / denom
                if wavelet_clip is not None:
                    a_w = a_w.clamp_max(float(wavelet_clip))
            a_w = _thicken_weight_map(a_w)
            a_w = _apply_shared_wavelet_map(a_w)
            return (a_w * (pred - target).pow(2)).mean()

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
            a_w = _thicken_weight_map(a_w)
            a_w = _apply_shared_wavelet_map(a_w)
            return (a_w * (pred - target).pow(2)).mean()

        # Backward-compatible path for helper signatures without normalize/clip.
        if wavelet_norm or (wavelet_clip is not None):
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
            a_w = _thicken_weight_map(a_w)
            a_w = _apply_shared_wavelet_map(a_w)
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
        )

    def _loss(pred_eps, true_eps, **kwargs):
        target = kwargs.get("target", true_eps)
        t = kwargs.get("t", None)
        num_steps = kwargs.get("num_steps", None)
        log_snr_t = kwargs.get("log_snr", None)
        epoch_val = kwargs.get("epoch", None)
        wave_mix, wave_gain, p_min_cur, p_max_cur, sal_q_cur, mask_on = _resolve_wavelet_curriculum(epoch_val)
        if diffusion_objective in fdbm_drift_objectives:
            schedule = kwargs.get("schedule", None)
            noisy = kwargs.get("noisy", None)
            if schedule is None or getattr(schedule, "kind", None) != "bridge":
                raise ValueError("fdbm_drift objective requires a bridge schedule (kind='bridge').")
            if noisy is None:
                raise ValueError("fdbm_drift objective requires 'noisy' tensor.")
            if t is None:
                raise ValueError("fdbm_drift objective requires sampled timesteps 't'.")
            t_long = t.view(-1).to(dtype=torch.long)
            if not hasattr(schedule, "a"):
                raise ValueError("fdbm_drift objective requires schedule coefficients 'a'.")
            a_t = schedule.a.to(device=noisy.device, dtype=noisy.dtype)[t_long].view(-1, 1, 1, 1)
            sigma = float(getattr(schedule, "sigma", 1.0))
            beta_t_diff = (sigma * sigma) * (1.0 - a_t)
            beta_t_diff = beta_t_diff.clamp_min(1e-6)
            # FDBM-style drift target: (x_T - x_t) / beta_t_diff.
            drift_true = (target - noisy) / beta_t_diff
            base = _matching_loss(pred_eps, drift_true)
            if wave_mix > 0.0:
                weighted = _wavelet_weighted_loss(pred_eps, drift_true, target)
                total = (1.0 - wave_mix) * base + wave_mix * (wave_gain * weighted)
            else:
                total = base
            aux_state_pred = kwargs.get("aux_state_pred", None)
            aux_target_state = kwargs.get("aux_target_state", target)
            aux = _physics_aux_loss(aux_state_pred, aux_target_state)
            if aux is not None:
                total = total + aux
            return total
        if diffusion_objective in predict_next_objectives:
            if diffusion_matching_loss in {"l1", "mae"}:
                loss_map = (pred_eps - target).abs()
            else:
                loss_map = (pred_eps - target).pow(2)

            if mask_on and wavelet_saliency is not None and time_dependent_mask is not None and t is not None and num_steps:
                with torch.no_grad():
                    sal = wavelet_saliency(
                        target,
                        J=J,
                        wave=wave,
                        mode=mode,
                        quantile=sal_q_cur,
                    )
                    tau = t.float() / float(max(int(num_steps) - 1, 1))
                    mask = time_dependent_mask(
                        sal,
                        tau,
                        p_min=p_min_cur,
                        p_max=p_max_cur,
                        late_focus=mask_late_focus,
                        smooth=mask_smooth,
                    )
                base = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                base = loss_map.mean()

            if wave_mix > 0.0:
                weighted = _wavelet_weighted_loss(pred_eps, target, target)
                total = (1.0 - wave_mix) * base + wave_mix * (wave_gain * weighted)
            else:
                total = base
            aux_state_pred = kwargs.get("aux_state_pred", pred_eps)
            aux_target_state = kwargs.get("aux_target_state", target)
            aux = _physics_aux_loss(aux_state_pred, aux_target_state)
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
            aux_state_pred = kwargs.get("aux_state_pred", endpoint_pred)
            aux_target_state = kwargs.get("aux_target_state", target)
            aux = _physics_aux_loss(aux_state_pred, aux_target_state)
            if aux is not None:
                total = total + aux
            return total

        loss_map = (pred_eps - true_eps).pow(2)

        if mask_on and wavelet_saliency is not None and time_dependent_mask is not None and t is not None and num_steps:
            # LWD-style gating: wavelet saliency + time-dependent mask
            with torch.no_grad():
                sal = wavelet_saliency(
                    target,
                    J=J,
                    wave=wave,
                    mode=mode,
                    quantile=sal_q_cur,
                )
                tau = t.float() / float(max(int(num_steps) - 1, 1))
                mask = time_dependent_mask(
                    sal,
                    tau,
                    p_min=p_min_cur,
                    p_max=p_max_cur,
                    late_focus=mask_late_focus,
                    smooth=mask_smooth,
                )
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
        if wave_mix > 0.0:
            weighted = _wavelet_weighted_loss(pred_eps, true_eps, target)
            total = (1.0 - wave_mix) * base + wave_mix * (wave_gain * weighted)
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
        aux_state_pred = kwargs.get("aux_state_pred", endpoint_pred)
        aux_target_state = kwargs.get("aux_target_state", target)
        aux = _physics_aux_loss(aux_state_pred, aux_target_state)
        if aux is not None:
            total = total + aux
        return total

    return _loss


def build_flow_loss(loss_cfg: Dict[str, Any]):
    mask_cfg = loss_cfg.get("wavelet_mask", {})
    mask_enabled = bool(mask_cfg.get("enabled", False))
    p_min = float(mask_cfg.get("p_min", 0.3))
    p_max = float(mask_cfg.get("p_max", 0.95))
    mask_late_focus = bool(mask_cfg.get("late_focus", True))
    mask_smooth = float(mask_cfg.get("smooth", 0.0))
    mask_saliency_quantile = float(mask_cfg.get("saliency_quantile", 0.99))
    J = int(mask_cfg.get("J", 1))
    wave = mask_cfg.get("wave", "haar")
    mode = mask_cfg.get("mode", "zero")
    wavelet_curriculum_cfg = loss_cfg.get("wavelet_curriculum", {})
    wavelet_curriculum_enabled = bool(wavelet_curriculum_cfg.get("enabled", False))
    wavelet_curriculum_stages = wavelet_curriculum_cfg.get("stages", [])
    if not isinstance(wavelet_curriculum_stages, (list, tuple)):
        wavelet_curriculum_stages = []
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

    def _flow_curriculum(epoch_val):
        enabled_cur = bool(mask_enabled)
        p_min_cur = float(p_min)
        p_max_cur = float(p_max)
        sal_q_cur = float(mask_saliency_quantile)
        if (not wavelet_curriculum_enabled) or epoch_val is None:
            return enabled_cur, p_min_cur, p_max_cur, sal_q_cur
        try:
            ep = int(epoch_val)
        except Exception:
            return enabled_cur, p_min_cur, p_max_cur, sal_q_cur
        for stage in wavelet_curriculum_stages:
            if not isinstance(stage, dict):
                continue
            max_epoch = stage.get("max_epoch", None)
            if max_epoch is not None:
                try:
                    if ep > int(max_epoch):
                        continue
                except Exception:
                    pass
            enabled_cur = bool(stage.get("mask_enabled", enabled_cur))
            p_min_cur = float(stage.get("p_min", p_min_cur))
            p_max_cur = float(stage.get("p_max", p_max_cur))
            sal_q_cur = float(stage.get("saliency_quantile", sal_q_cur))
            break
        return enabled_cur, p_min_cur, p_max_cur, sal_q_cur

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
        mask_enabled_cur, p_min_cur, p_max_cur, sal_q_cur = _flow_curriculum(kwargs.get("epoch", None))
        if flow_matching_loss in {"l2", "mse"}:
            loss_map = (pred_u - true_u).pow(2)
        elif flow_matching_loss in {"l1", "mae"}:
            loss_map = (pred_u - true_u).abs()
        else:
            raise ValueError(
                f"Unknown flow_matching_loss '{flow_matching_loss}'. "
                "Use one of {'l1','l2'}."
            )

        if mask_enabled_cur and wavelet_saliency is not None and time_dependent_mask is not None and target is not None and tau is not None:
            with torch.no_grad():
                sal = wavelet_saliency(
                    target,
                    J=J,
                    wave=wave,
                    mode=mode,
                    quantile=sal_q_cur,
                )
                tau_f = tau.view(-1).float()
                mask = time_dependent_mask(
                    sal,
                    tau_f,
                    p_min=p_min_cur,
                    p_max=p_max_cur,
                    late_focus=mask_late_focus,
                    smooth=mask_smooth,
                )
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
