"""Lightweight registries for rapid-solidification models/backbones."""

from __future__ import annotations

from typing import Any, Callable, Dict

from models.backbones.ads_convnext import ADSAutoregressiveSurrogate
from models.backbones.uafno_cond import UAFNO_PreSkip_Full
from models.backbones.unet_conv_att_cond import UNet_SSA_PreSkip_Full
from models.backbones.unet_film_bottleneck import UNetFiLMAttn
from models.backbones.fno_field import FNO_Field2D
from models.backbones.uvit_film import UVitFiLMVelocity

BackboneBuilder = Callable[..., Any]

_BACKBONE_REGISTRY: Dict[str, BackboneBuilder] = {}


def _register_default_backbones() -> None:
    register_backbone("unet", UNet_SSA_PreSkip_Full)
    register_backbone("unet_ssa", UNet_SSA_PreSkip_Full)
    register_backbone("unet_film_attn", UNetFiLMAttn)
    register_backbone("unet_bottleneck_attn", UNetFiLMAttn)
    register_backbone("uafno", UAFNO_PreSkip_Full)
    register_backbone("fno", FNO_Field2D)
    register_backbone("ads_convnext", ADSAutoregressiveSurrogate)
    register_backbone("uvit", UVitFiLMVelocity)


def register_backbone(name: str, builder: BackboneBuilder) -> None:
    """Registers a backbone under a canonical string key."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Backbone name must be non-empty.")
    _BACKBONE_REGISTRY[key] = builder


def available_backbones() -> Dict[str, BackboneBuilder]:
    return dict(_BACKBONE_REGISTRY)


def build_backbone(name: str, cfg: Dict[str, Any]) -> Any:
    key = str(name).strip().lower()
    if key not in _BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Registered: {sorted(_BACKBONE_REGISTRY)}")
    builder = _BACKBONE_REGISTRY[key]
    return builder(**cfg)


def build_model(model_family: str, backbone: str, model_cfg: Dict[str, Any]) -> Any:
    """
    Builds a model given the descriptor fields. Diffusion-specific wiring will
    be added once the diffusion trainer lands; currently both families reuse
    the backbone registry.
    """
    family = str(model_family or "surrogate").strip().lower()
    params = model_cfg.get("params", model_cfg)

    if family in {"surrogate", "diffusion", "flow_matching"}:
        return build_backbone(backbone, params)
    raise ValueError(f"Unknown model_family '{model_family}'")


_register_default_backbones()
