#!/usr/bin/env python3
"""
Bridge preflight checker for active PF surrogate bridge-like training configs.

Usage:
  python scripts/bridge_preflight_check.py --config <yaml>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import h5py  # type: ignore
except Exception:
    h5py = None


UNIDB_ALLOWED = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
    "unidb_reverse_step",
    "unidb",
    "reverse_step_matching",
}

FDBM_BRIDGE_ALLOWED = {
    "fdbm_drift_mse",
    "fdbm_drift",
    "bridge_fdbm_drift",
}

PREDICT_NEXT_OBJECTIVES = {
    "unidb_predict_next",
    "predict_next",
    "predict_x0",
    "x0_mse",
    "next_field_mse",
}

FLOWMATCH_OBJECTIVES = {
    "dbfm_source_anchored",
    "dbfm_rectified_flow",
    "dbfm_flow",
    "dbfm",
    "rectified_flow_constant_displacement",
    "rectified_flow_constant_displacement_concat",
    "rectified_flow_source_anchored",
    "rectified_flow_source_anchored_concat",
    "rectified_flow_noise_source_concat",
    "rectified_flow_noise_cond_concat",
    "sfm_latent_source_denoise_concat",
    "sfm_latent_source_concat",
    "rectified_flow",
}


@dataclass
class Item:
    name: str
    status: str  # PASS / WARN / FAIL
    detail: str


def _get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _check(cfg: Dict[str, Any]) -> List[Item]:
    out: List[Item] = []

    model_family = str(_get(cfg, "train", "model_family", default="surrogate")).lower()

    diff = dict(cfg.get("diffusion", {}) or {})
    sched_name = str(diff.get("noise_schedule", "")).lower()
    sched_kwargs = dict(diff.get("schedule_kwargs", {}) or {})
    sampler_kwargs = dict(diff.get("sampler_kwargs", {}) or {})
    timesteps = int(sched_kwargs.get("timesteps", 0) or 0)

    loss = dict(cfg.get("loss", {}) or {})
    diff_obj = str(loss.get("diffusion_objective", "epsilon_mse")).lower()
    flow_obj = str(_get(cfg, "train", "objective", default="")).lower() or "rectified_flow"
    trainer = dict(cfg.get("trainer", {}) or {})
    trainer_metrics = dict(_get(cfg, "trainer", "metrics", default={}) or {})
    use_val = bool(trainer.get("use_val", True))
    sched_monitor_cfg = dict(trainer.get("scheduler_monitor", {}) or {})
    es_cfg = dict(trainer.get("early_stop", {}) or {})
    default_split = "val" if use_val else "train"
    sched_split = str(sched_monitor_cfg.get("split", default_split) or default_split).strip().lower()
    es_split = str(es_cfg.get("split", default_split) or default_split).strip().lower()

    model_params = dict(_get(cfg, "model", "params", default={}) or {})
    in_ch = int(model_params.get("in_channels", 0) or 0)
    out_ch = int(model_params.get("out_channels", 0) or 0)
    use_control = bool(model_params.get("use_control_branch", False))
    hint_channels = int(model_params.get("hint_channels", 0) or 0)

    cond = dict(cfg.get("conditioning", {}) or {})
    use_theta = bool(cond.get("use_theta", False))
    theta_channels = int(cond.get("theta_channels", 0) or 0)

    dl_args = dict(_get(cfg, "dataloader", "args", default={}) or {})
    add_thermal = bool(dl_args.get("add_thermal", False))
    thermal_precomp = bool(dl_args.get("thermal_require_precomputed", False))
    normalize_images = bool(dl_args.get("normalize_images", False))
    normalize_force = bool(dl_args.get("normalize_force", False))

    input_mode = str(sched_kwargs.get("input_mode", "")).lower()
    residual_mode = str(sched_kwargs.get("residual_mode", "none")).lower()
    residual_scale = float(sched_kwargs.get("residual_scale", 1.0) or 1.0)
    residual_power = float(sched_kwargs.get("residual_power", 1.0) or 1.0)
    residual_clip = sched_kwargs.get("residual_clip", None)
    residual_pi_floor = float(sched_kwargs.get("residual_pi_floor", 0.0) or 0.0)

    # 1) objective vs schedule
    if model_family == "diffusion":
        if sched_name.startswith("unidb"):
            if diff_obj in UNIDB_ALLOWED:
                out.append(Item("objective_schedule", "PASS", f"schedule={sched_name}, diffusion_objective={diff_obj}"))
            elif diff_obj in FDBM_BRIDGE_ALLOWED:
                out.append(Item("objective_schedule", "FAIL", f"FDBM/fractional objective '{diff_obj}' cannot be used with UniDB schedule '{sched_name}'"))
            else:
                out.append(Item("objective_schedule", "FAIL", f"UniDB schedule with incompatible objective '{diff_obj}'"))
        elif sched_name.startswith("bridge"):
            if diff_obj in UNIDB_ALLOWED:
                out.append(Item(
                    "objective_schedule",
                    "FAIL",
                    f"UniDB objective '{diff_obj}' with bridge schedule '{sched_name}'",
                ))
            else:
                out.append(Item("objective_schedule", "PASS", f"schedule={sched_name}, diffusion_objective={diff_obj}"))
        else:
            if diff_obj in UNIDB_ALLOWED or diff_obj in FDBM_BRIDGE_ALLOWED:
                out.append(Item(
                    "objective_schedule",
                    "FAIL",
                    f"bridge-specific objective '{diff_obj}' requires unidb_* or bridge_* schedule (got '{sched_name}')",
                ))
            else:
                out.append(Item("objective_schedule", "PASS", f"schedule={sched_name}, diffusion_objective={diff_obj}"))
    elif model_family == "flow_matching":
        if flow_obj in FLOWMATCH_OBJECTIVES:
            out.append(Item("objective_schedule", "PASS", f"model_family=flow_matching, train.objective={flow_obj}"))
        else:
            out.append(
                Item(
                    "objective_schedule",
                    "FAIL",
                    f"flow_matching requires train.objective in {sorted(FLOWMATCH_OBJECTIVES)}, got '{flow_obj}'",
                )
            )
    else:
        if diff_obj in UNIDB_ALLOWED:
            out.append(Item("objective_schedule", "WARN", f"non-bridge family '{model_family}' with diffusion objective '{diff_obj}'"))
        else:
            out.append(Item("objective_schedule", "WARN", f"non-bridge family '{model_family}', objective checks skipped"))

    # 2) timestep range
    if model_family == "diffusion":
        t_min = int(sampler_kwargs.get("t_min", 1) or 1)
        t_max = int(sampler_kwargs.get("t_max", timesteps if timesteps > 0 else 0) or 0)
        include_terminal = bool(sampler_kwargs.get("include_terminal_unidb", False))
        if timesteps <= 0:
            out.append(Item("timesteps", "FAIL", "schedule_kwargs.timesteps must be > 0"))
        else:
            ok_range = (t_min >= 1) and (t_max <= timesteps)
            if ok_range:
                out.append(Item("timesteps", "PASS", f"t_min={t_min}, t_max={t_max}, timesteps={timesteps}, include_terminal={include_terminal}"))
            else:
                out.append(Item("timesteps", "FAIL", f"invalid sampler range: t_min={t_min}, t_max={t_max}, timesteps={timesteps}"))
    else:
        out.append(Item("timesteps", "PASS", f"N/A for model_family={model_family}"))

    # 3) channel wiring
    if input_mode == "delta_source_concat" and out_ch > 0:
        expected = 2 * out_ch
        if in_ch == expected:
            out.append(Item("channel_wiring", "PASS", f"in_channels={in_ch}, expected={expected} (delta_source_concat)"))
        else:
            out.append(Item("channel_wiring", "FAIL", f"in_channels={in_ch}, expected={expected} for delta_source_concat"))
    else:
        out.append(Item("channel_wiring", "WARN", f"input_mode={input_mode}; checker rule only covers delta_source_concat"))

    # 4) thermal path
    if use_theta:
        if add_thermal:
            if thermal_precomp:
                out.append(Item("thermal_path", "PASS", "use_theta=true with add_thermal=true and thermal_require_precomputed=true"))
            else:
                out.append(Item("thermal_path", "PASS", "use_theta=true with add_thermal=true"))
        else:
            out.append(Item("thermal_path", "FAIL", "use_theta=true requires add_thermal=true"))
    else:
        out.append(Item("thermal_path", "WARN", "conditioning.use_theta=false"))

    if use_control:
        if hint_channels == theta_channels:
            out.append(Item("control_hint_channels", "PASS", f"hint_channels={hint_channels}, theta_channels={theta_channels}"))
        else:
            out.append(Item("control_hint_channels", "WARN", f"hint_channels={hint_channels}, theta_channels={theta_channels}"))

    # 5) residual parameters
    if residual_mode != "none":
        bad = []
        if residual_scale <= 0:
            bad.append(f"residual_scale={residual_scale} <= 0")
        if residual_power <= 0:
            bad.append(f"residual_power={residual_power} <= 0")
        if residual_clip is not None and float(residual_clip) <= 0:
            bad.append(f"residual_clip={residual_clip} <= 0")
        if residual_pi_floor < 0:
            bad.append(f"residual_pi_floor={residual_pi_floor} < 0")
        if bad:
            out.append(Item("residual_params", "FAIL", "; ".join(bad)))
        else:
            out.append(Item("residual_params", "PASS", f"mode={residual_mode}, scale={residual_scale}, power={residual_power}"))
    else:
        out.append(Item("residual_params", "WARN", "residual_mode=none"))

    # 5b) endpoint mode vs schedule family (avoid UniDB/FDBM mixups)
    endpoint_mode_raw = str(diff.get("val_endpoint_mode", "source_rollout_dbim")).strip().lower()
    if endpoint_mode_raw in {"rollout_dbim", "source_rollout_dbim"}:
        endpoint_mode = "source_rollout_dbim"
    elif endpoint_mode_raw in {"source_rollout_unidb_sde", "rollout_unidb_sde", "unidb_sde", "unidb_exact", "unidb_reverse_sde"}:
        endpoint_mode = "source_rollout_unidb_sde"
    elif endpoint_mode_raw in {"teacher_forced", "legacy", "single_step"}:
        endpoint_mode = "teacher_forced"
    else:
        endpoint_mode = "source_rollout_dbim"

    if endpoint_mode == "source_rollout_unidb_sde" and not sched_name.startswith("unidb"):
        out.append(Item(
            "endpoint_mode_schedule",
            "FAIL",
            f"val_endpoint_mode={endpoint_mode} requires unidb_* schedule, got '{sched_name}'",
        ))
    elif endpoint_mode == "source_rollout_dbim" and sched_name.startswith("unidb"):
        out.append(Item(
            "endpoint_mode_schedule",
            "FAIL",
            f"val_endpoint_mode={endpoint_mode} is bridge/DBIM mode and cannot be used with UniDB schedule '{sched_name}'",
        ))
    else:
        out.append(Item("endpoint_mode_schedule", "PASS", f"val_endpoint_mode={endpoint_mode}, schedule={sched_name}"))

    # 6) metrics visibility
    want_endpoint = bool(trainer_metrics.get("endpoint_rmse", False))
    want_spec = bool(trainer_metrics.get("spectral_rmse", False))
    if want_endpoint and want_spec:
        out.append(Item("metrics_endpoint", "PASS", "endpoint_rmse + spectral_rmse enabled"))
    else:
        out.append(Item("metrics_endpoint", "WARN", f"endpoint_rmse={want_endpoint}, spectral_rmse={want_spec}"))

    # 6a) monitor split must match validation availability
    if (not use_val) and (sched_split == "val" or es_split == "val"):
        out.append(
            Item(
                "monitor_split_vs_use_val",
                "WARN",
                f"trainer.use_val=false but monitor split targets val (scheduler={sched_split}, early_stop={es_split}); runtime will fall back to train metrics",
            )
        )
    else:
        out.append(
            Item(
                "monitor_split_vs_use_val",
                "PASS",
                f"trainer.use_val={use_val}, scheduler_split={sched_split}, early_stop_split={es_split}",
            )
        )

    # 6b) diffusion stochastic/probabilistic validation settings
    if model_family == "diffusion":
        diff_val_num_samples = int(diff.get("val_num_samples", 1) or 1)
        diff_val_deterministic = bool(diff.get("val_deterministic", False))
        diff_val_prob_metrics = bool(diff.get("val_probabilistic_metrics", False))
        diff_val_monitor = str(diff.get("val_monitor_metric_stochastic", "endpoint_crps")).lower()
        if diff_val_prob_metrics and diff_val_deterministic:
            out.append(
                Item(
                    "diffusion_prob_metrics",
                    "WARN",
                    "val_probabilistic_metrics=true but val_deterministic=true; probabilistic metrics are disabled",
                )
            )
        elif diff_val_prob_metrics and diff_val_num_samples <= 1:
            out.append(
                Item(
                    "diffusion_prob_metrics",
                    "WARN",
                    "val_probabilistic_metrics=true but val_num_samples<=1; ensemble metrics need >1 sample",
                )
            )
        elif diff_val_prob_metrics:
            out.append(
                Item(
                    "diffusion_prob_metrics",
                    "PASS",
                    f"enabled num_samples={diff_val_num_samples}, monitor_metric={diff_val_monitor}",
                )
            )
        else:
            out.append(
                Item(
                    "diffusion_prob_metrics",
                    "WARN",
                    "disabled (set diffusion.val_probabilistic_metrics=true for stochastic evaluation)",
                )
            )

    # 7) validation dataset path check (when validation enabled)
    h5_train = _get(cfg, "paths", "h5", "train", default=None)
    h5_val = _get(cfg, "paths", "h5", "val", default=None)
    if use_val:
        if not h5_val:
            out.append(Item("val_dataset", "FAIL", "trainer.use_val=true but paths.h5.val is missing"))
        elif not Path(str(h5_val)).exists():
            out.append(Item("val_dataset", "FAIL", f"trainer.use_val=true but val H5 does not exist: {h5_val}"))
        else:
            out.append(Item("val_dataset", "PASS", f"val H5 present: {h5_val}"))
    else:
        out.append(Item("val_dataset", "WARN", "trainer.use_val=false; val metrics/checkpointing will not be computed"))

    # 8) wavelet dependency presence (preflight environment)
    weight_wavelet = float(loss.get("weight_wavelet_loss", 0.0) or 0.0)
    wavelet_mask_enabled = bool(_get(loss, "wavelet_mask", "enabled", default=False))
    ae_wavelet_enabled = bool(_get(loss, "ae_wavelet", "enabled", default=False))
    wavelet_needed = (weight_wavelet > 0.0) or wavelet_mask_enabled or ae_wavelet_enabled
    if wavelet_needed:
        has_wavelets = importlib.util.find_spec("pytorch_wavelets") is not None
        if has_wavelets:
            out.append(Item("wavelet_dependency", "PASS", "pytorch_wavelets importable in preflight env"))
        else:
            out.append(
                Item(
                    "wavelet_dependency",
                    "WARN",
                    "wavelet losses/masks enabled but pytorch_wavelets is not importable in preflight env; verify runtime container/venv has it",
                )
            )
    else:
        out.append(Item("wavelet_dependency", "PASS", "wavelet losses/masks disabled"))

    # 9) normalization check against H5 attrs (if available)
    if h5_train and h5py is not None and Path(str(h5_train)).exists():
        try:
            data_key = str(dl_args.get("data_key", "")).strip()
            key_label = data_key or "root"
            with h5py.File(str(h5_train), "r") as hf:
                attrs = hf.attrs
                if data_key:
                    schema = str(
                        attrs.get(
                            f"{data_key}_normalization_schema",
                            attrs.get(
                                f"{data_key}_norm_type",
                                attrs.get("normalization_schema", attrs.get("norm_type_images", "")),
                            ),
                        )
                    ).lower()
                else:
                    schema = str(attrs.get("normalization_schema", attrs.get("norm_type_images", ""))).lower()
            if schema == "zscore" and (not normalize_images):
                out.append(Item("normalization", "PASS", f"schema({key_label})=zscore with normalize_images=false (expected pre-normalized workflow)"))
            elif schema == "zscore" and normalize_images and normalize_force:
                out.append(Item("normalization", "WARN", f"schema({key_label})=zscore + normalize_images/force may double-normalize"))
            elif schema == "":
                out.append(Item("normalization", "WARN", f"normalization schema missing for data_key={key_label}"))
            else:
                out.append(Item("normalization", "PASS", f"schema({key_label})={schema}, normalize_images={normalize_images}"))
        except Exception as e:
            out.append(Item("normalization", "WARN", f"could not inspect H5 attrs: {e}"))
    else:
        out.append(Item("normalization", "WARN", "train H5 missing or h5py unavailable; skipped schema check"))

    # 8) objective inference note
    if model_family == "flow_matching":
        out.append(Item("inference_contract", "PASS", "flow_matching: eval/rollout integrates learned velocity field"))
    elif diff_obj in PREDICT_NEXT_OBJECTIVES:
        out.append(Item("inference_contract", "PASS", "predict-next objective: eval/rollout must use direct endpoint prediction"))
    else:
        out.append(Item("inference_contract", "PASS", "epsilon/reverse-step objective: eval/rollout should invert from noise prediction"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight checks for bridge configs.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="Return non-zero on WARN in addition to FAIL.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    items = _check(cfg)

    fail_n = sum(1 for x in items if x.status == "FAIL")
    warn_n = sum(1 for x in items if x.status == "WARN")
    pass_n = sum(1 for x in items if x.status == "PASS")

    print(f"[bridge_preflight] config={args.config}")
    for x in items:
        print(f"[{x.status:<4}] {x.name}: {x.detail}")
    print(f"[summary] PASS={pass_n} WARN={warn_n} FAIL={fail_n}")

    if args.json_out is not None:
        payload = {
            "config": str(args.config),
            "pass": pass_n,
            "warn": warn_n,
            "fail": fail_n,
            "items": [x.__dict__ for x in items],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    if fail_n > 0:
        return 2
    if args.strict and warn_n > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
