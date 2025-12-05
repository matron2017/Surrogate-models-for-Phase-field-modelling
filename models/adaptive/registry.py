"\"\"\"Region/adaptive components registry.\"\"\""

from __future__ import annotations

from typing import Dict, Optional


class IdentityRegionSelector:
    """Default selector that performs no masking but keeps the interface uniform."""

    def __call__(self, batch, t):
        return None


class QuadtreeRegionSelector:
    """Placeholder for future QDM-style implementations."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch, t):
        raise NotImplementedError("QuadtreeRegionSelector not yet implemented.")


class EdgeMaskRegionSelector:
    """Placeholder for edge-aware masks."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch, t):
        raise NotImplementedError("EdgeMaskRegionSelector not yet implemented.")


_REGISTRY: Dict[str, type] = {
    "none": IdentityRegionSelector,
    "quadtree": QuadtreeRegionSelector,
    "edge_mask": EdgeMaskRegionSelector,
}


def build_region_selector(name: str, **kwargs):
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown region_selector '{name}'. Registered: {sorted(_REGISTRY)}")
    cls = _REGISTRY[key]
    return cls(**kwargs)
