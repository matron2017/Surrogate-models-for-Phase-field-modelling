#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    label: str
    cmap: str
    symmetric: bool = False
    clamp_nonnegative: bool = False


DEFAULT_FIELD_SPECS = (
    FieldSpec(label="Phase", cmap="coolwarm", symmetric=True),
    FieldSpec(label="Concentration", cmap="viridis", clamp_nonnegative=True),
)


def default_field_specs(channel_count: int) -> List[FieldSpec]:
    specs = list(DEFAULT_FIELD_SPECS[: max(0, int(channel_count))])
    while len(specs) < int(channel_count):
        specs.append(FieldSpec(label=f"Channel {len(specs)}", cmap="viridis"))
    return specs


def _stable_limits(gt: np.ndarray, pred: np.ndarray, spec: FieldSpec) -> Tuple[float, float]:
    stacked = np.stack([np.asarray(gt), np.asarray(pred)], axis=0)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return (-1.0, 1.0) if spec.symmetric else (0.0, 1.0)

    if spec.symmetric:
        vmax = float(np.max(np.abs(finite)))
        vmax = max(vmax, 1e-8)
        return -vmax, vmax

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if spec.clamp_nonnegative and vmin >= 0.0:
        vmin = 0.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        delta = max(abs(vmax), 1.0) * 1e-6
        vmin -= delta
        vmax += delta
    return vmin, vmax


def render_gt_pred_rows(
    out_path: Path,
    gt_arrays: Sequence[np.ndarray],
    pred_arrays: Sequence[np.ndarray],
    title: str,
    *,
    field_specs: Optional[Sequence[FieldSpec]] = None,
    left_title: str = "Ground truth",
    right_title: str = "Autoregressive prediction",
    footer_lines: Iterable[str] | None = None,
    origin: str = "upper",
    dpi: int = 160,
) -> None:
    if len(gt_arrays) != len(pred_arrays):
        raise ValueError("gt_arrays and pred_arrays must have the same length")
    if not gt_arrays:
        raise ValueError("at least one field is required")

    specs = list(field_specs) if field_specs is not None else default_field_specs(len(gt_arrays))
    if len(specs) != len(gt_arrays):
        raise ValueError("field_specs length must match number of arrays")

    fig_height = max(5.5, 4.8 * len(gt_arrays))
    fig, axes = plt.subplots(len(gt_arrays), 2, figsize=(16, fig_height), squeeze=False, facecolor="white")

    for row, (gt, pred, spec) in enumerate(zip(gt_arrays, pred_arrays, specs)):
        vmin, vmax = _stable_limits(gt, pred, spec)
        im_left = axes[row, 0].imshow(gt, cmap=spec.cmap, vmin=vmin, vmax=vmax, origin=origin)
        axes[row, 1].imshow(pred, cmap=spec.cmap, vmin=vmin, vmax=vmax, origin=origin)

        axes[row, 0].set_title(f"{spec.label} | {left_title}")
        axes[row, 1].set_title(f"{spec.label} | {right_title}")
        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        cbar = fig.colorbar(im_left, ax=axes[row, :], fraction=0.046, pad=0.02, shrink=0.92)
        cbar.ax.set_ylabel(spec.label)

    fig.suptitle(title, y=0.98)
    footer_text = "\n".join([line for line in (footer_lines or []) if line])
    if footer_text:
        fig.text(0.5, 0.02, footer_text, ha="center", va="bottom", fontsize=10)
        rect = (0.0, 0.06, 1.0, 0.95)
    else:
        rect = (0.0, 0.03, 1.0, 0.95)
    fig.tight_layout(rect=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)