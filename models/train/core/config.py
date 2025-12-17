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
    cfg["train"].setdefault("model_family", "surrogate")
    cfg["task"].setdefault("name", "surrogate")
    cfg["diffusion"].setdefault("noise_schedule", "linear")
    cfg["diffusion"].setdefault("timestep_sampler", "uniform")
    cfg["flow_matching"].setdefault("sigma", 0.0)
    cfg["adaptive"].setdefault("region_selector", "none")
    cfg["adaptive"].setdefault("enable_adaptive_resolution", False)
    cfg["loss"].setdefault("weight_wavelet_loss", 0.0)
    cfg["loader"].setdefault("pin_memory", True)
    cfg["conditioning"] = cfg.get("conditioning", {"enabled": True, "cond_dim": 2, "source": "field"})
    return cfg
