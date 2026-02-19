"""
Config loading and validation helpers.
"""

from __future__ import annotations

from typing import Any, Dict
import yaml


def _validate_config(cfg: Dict[str, Any]):
    required = ["paths", "dataloader", "loader", "model", "trainer"]
    for k in required:
        if k not in cfg:
            raise KeyError(f"Missing required config key: {k}")
    if "train" not in cfg["paths"].get("h5", {}):
        raise KeyError("paths.h5.train is required.")
    if "file" not in cfg["dataloader"] or "class" not in cfg["dataloader"]:
        raise KeyError("dataloader.file and dataloader.class are required.")
    if "batch_size" not in cfg["loader"]:
        raise KeyError("loader.batch_size is required.")


def _load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("trainer", {})
    cfg.setdefault("train", {})
    cfg.setdefault("task", {})
    cfg.setdefault("model", {})
    cfg.setdefault("diffusion", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("flow_matching", {})
    cfg.setdefault("adaptive", {})
    cfg.setdefault("mlflow", {})

    cfg["trainer"].setdefault("seed", 17)
    cfg["trainer"].setdefault("deterministic", False)
    cfg["trainer"].setdefault("log_interval", 0)
    cfg["trainer"].setdefault("epochs", 1)
    cfg["trainer"].setdefault("flow_monitor_force_endpoint", True)
    cfg["train"].setdefault("model_family", "surrogate")
    cfg["task"].setdefault("name", "surrogate")
    cfg["diffusion"].setdefault("noise_schedule", "linear")
    cfg["diffusion"].setdefault("timestep_sampler", "uniform")
    cfg["flow_matching"].setdefault("sigma", 0.0)
    cfg["flow_matching"].setdefault("val_rollout_nfe", 20)
    cfg["flow_matching"].setdefault("val_num_samples", 1)
    cfg["flow_matching"].setdefault("val_deterministic", False)
    cfg["flow_matching"].setdefault("val_probabilistic_metrics", True)
    cfg["flow_matching"].setdefault("val_monitor_metric_stochastic", "endpoint_crps")
    cfg["flow_matching"].setdefault("sfm_sigma_z", 0.05)
    cfg["flow_matching"].setdefault("sfm_sigma_min", 1e-3)
    cfg["flow_matching"].setdefault("sfm_sigma_max", 0.25)
    cfg["flow_matching"].setdefault("sfm_adaptive_sigma", True)
    cfg["flow_matching"].setdefault("sfm_sigma_ema_beta", 0.02)
    cfg["flow_matching"].setdefault("sfm_encoder_reg_lambda", 0.0)
    cfg["adaptive"].setdefault("region_selector", "none")
    cfg["adaptive"].setdefault("enable_adaptive_resolution", False)
    cfg["loss"].setdefault("weight_wavelet_loss", 0.0)
    cfg["loader"].setdefault("pin_memory", True)
    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    cond_cfg.setdefault("enabled", False)
    cond_cfg.setdefault("use_theta", False)
    cond_cfg.setdefault("theta_channels", 1)
    if cond_cfg.get("enabled", False):
        raise ValueError(
            "Scalar conditioning has been removed from the active training path. "
            "Set conditioning.enabled=false and use conditioning.use_theta with add_thermal."
        )
    cond_cfg["cond_dim"] = 0
    cond_cfg["source"] = "none"
    cfg["conditioning"] = cond_cfg
    return cfg
