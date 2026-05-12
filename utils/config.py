from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/scratch/project_2008261/pf_surrogate_modelling")
DEFAULT_CONFIG = PACKAGE_ROOT / "utils" / "deterministic_afno_bridge_config.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "diffusion_bridge_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_predictnext_nomass_afno8_gputest_overfit_det_nonfrac_q1" / "UNetFiLMAttn"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoint.best.pth"
DEFAULT_VAL_H5 = PROJECT_ROOT / "data" / "latent_best_psgd_e279_dev" / "val_latent_experimental_midtrain.h5"


def load_yaml(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def wavelet_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trainer.use_wavelet_weights": bool(cfg.get("trainer", {}).get("use_wavelet_weights", False)),
        "loss.weight_wavelet_loss": float(cfg.get("loss", {}).get("weight_wavelet_loss", 0.0)),
    }
