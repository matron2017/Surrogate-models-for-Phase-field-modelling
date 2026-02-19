#!/usr/bin/env python3
"""
Plot GT bridge denoising stages for selected dataset samples.

For each requested sample, this script saves one PNG with:
  - INPUT (source x)
  - 3 intermediate denoising bridge states
  - OUTPUT (target y)

Exactly 2 channels are shown per sample. Channels are auto-selected as:
  - high-change channel (largest mean |y-x|)
  - low-change channel (smallest mean |y-x|)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.pf_dataloader import PFPairDataset


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _build_dataset(cfg: Dict[str, Any], split: str) -> PFPairDataset:
    dcfg = dict(cfg.get("dataloader", {}) or {})
    args = dict(dcfg.get("args", {}) or {})
    split_args = dict(dcfg.get(f"{split}_args", {}) or {})
    ds_args = {**args, **split_args}
    h5_map = dict((cfg.get("paths", {}) or {}).get("h5", {}) or {})
    if split not in h5_map:
        raise KeyError(f"Missing paths.h5.{split} in config.")
    ds_args["h5_path"] = str(h5_map[split])
    return PFPairDataset(**ds_args)


def _split_theta(x: torch.Tensor, cond_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bool(cond_cfg.get("use_theta", False)):
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if x.shape[0] <= theta_channels:
            raise ValueError(f"Input channels {x.shape[0]} <= theta_channels {theta_channels}.")
        return x[:-theta_channels, ...], x[-theta_channels:, ...]
    return x, None


def _parse_indices(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("No sample indices provided.")
    return out


def _parse_fracs(s: str) -> List[float]:
    vals: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        vals.append(min(0.999, max(0.001, v)))
    if len(vals) != 3:
        raise ValueError("Expected exactly 3 fractions, e.g. '0.85,0.55,0.25'.")
    return sorted(vals, reverse=True)


def _k_from_frac(schedule, frac: float) -> int:
    T = int(schedule.timesteps)
    k = int(round(float(frac) * (T - 1)))
    return max(1, min(T - 1, k))


def _uni_coeffs(schedule, t_idx: int, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.tensor([int(t_idx)], dtype=torch.long, device=ref.device)
    m_t = schedule._m(t, ref)
    n_t = schedule._n(t, ref)
    s_t = schedule.f_sigma(t, ref)
    return m_t, n_t, s_t


def _q_sample_unidb(
    schedule,
    x_source: torch.Tensor,
    y_target: torch.Tensor,
    t_idx: int,
    eps: torch.Tensor,
) -> torch.Tensor:
    m_t, n_t, s_t = _uni_coeffs(schedule, t_idx, y_target)
    return n_t * x_source + m_t * y_target + s_t * eps


def _stats_text(arr: np.ndarray) -> str:
    return f"mean={float(np.mean(arr)):.4f}\nmin={float(np.min(arr)):.4f}\nmax={float(np.max(arr)):.4f}"


def _pick_two_very_different_channels(x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    # Per-channel average absolute change; choose one max-change and one min-change channel.
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dim() != 3:
        raise ValueError(f"Expected channel-first [C,H,W], got shape {tuple(x.shape)}")
    score = torch.mean((y - x).abs().reshape(x.shape[0], -1), dim=1)
    hi = int(torch.argmax(score).item())
    lo = int(torch.argmin(score).item())
    if lo == hi and x.shape[0] > 1:
        # fallback in pathological tie case
        order = torch.argsort(score)
        lo = int(order[0].item())
        hi = int(order[-1].item())
    return hi, lo


def _save_sample_panel(
    out_path: Path,
    *,
    arrays_by_channel: Dict[int, Sequence[np.ndarray]],
    stage_titles: Sequence[str],
    sample_idx: int,
    gid: str,
    pair_index: int,
    k_values: Sequence[int],
) -> None:
    channels = list(arrays_by_channel.keys())
    nrows = len(channels)
    ncols = len(stage_titles)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 3.2 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for ridx, ch in enumerate(channels):
        arrays = list(arrays_by_channel[ch])
        vmin = float(min(np.nanmin(a) for a in arrays))
        vmax = float(max(np.nanmax(a) for a in arrays))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

        for cidx, (arr, ttl) in enumerate(zip(arrays, stage_titles)):
            ax = axes[ridx, cidx]
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if ridx == 0:
                ax.set_title(ttl, fontsize=10, fontweight="bold")
            if cidx == 0:
                ax.set_ylabel(f"channel {ch}", fontsize=10, fontweight="bold")
            ax.text(
                0.02,
                0.02,
                _stats_text(arr),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.72),
            )
            if cidx == ncols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        (
            f"GT Bridge Denoising Stages | sample={sample_idx} gid={gid} pair={pair_index}\n"
            f"Stages: INPUT -> k={k_values[0]} -> k={k_values[1]} -> k={k_values[2]} -> OUTPUT"
        ),
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot 5-stage GT bridge denoising for selected samples.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m_rdbmres_controlhint.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-indices", type=str, default="289,313")
    ap.add_argument("--fracs", type=str, default="0.85,0.55,0.25")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "visuals" / "bridge_denoising_steps_samples_289_313",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.bridge_config.expanduser().resolve())
    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"Expected UniDB schedule, got kind={getattr(schedule, 'kind', 'unknown')}")

    ds = _build_dataset(cfg, split=str(args.split))
    req_indices = _parse_indices(args.sample_indices)
    fracs = _parse_fracs(args.fracs)
    k_values = [_k_from_frac(schedule, f) for f in fracs]
    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_titles = [
        "INPUT\n(source x)",
        f"DENOISE 1\n(k={k_values[0]})",
        f"DENOISE 2\n(k={k_values[1]})",
        f"DENOISE 3\n(k={k_values[2]})",
        "OUTPUT\n(target y)",
    ]

    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bridge_config": str(args.bridge_config.expanduser().resolve()),
        "split": str(args.split),
        "requested_sample_indices": req_indices,
        "k_values": k_values,
        "samples": [],
    }

    for idx_req in req_indices:
        if idx_req < 0 or idx_req >= len(ds):
            raise IndexError(f"Sample index {idx_req} out of bounds for split={args.split} len={len(ds)}")
        sample = ds[idx_req]
        x = sample["input"].float()
        y = sample["target"].float()
        gid = str(sample.get("gid", "unknown"))
        pair_index = int(sample.get("pair_index", -1))
        x_state, _theta = _split_theta(x, cond_cfg)
        if x_state.shape != y.shape:
            raise ValueError(
                f"Shape mismatch at sample {idx_req}: x_state={tuple(x_state.shape)} y={tuple(y.shape)}"
            )

        ch_hi, ch_lo = _pick_two_very_different_channels(x_state, y)
        channels = [ch_hi, ch_lo]

        g = torch.Generator(device=x_state.device)
        g.manual_seed(int(args.seed) + int(idx_req))
        eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

        stage_tensors: List[torch.Tensor] = [x_state]
        for k in k_values:
            stage_tensors.append(_q_sample_unidb(schedule, x_state, y, t_idx=int(k), eps=eps))
        stage_tensors.append(y)

        arrays_by_channel: Dict[int, Sequence[np.ndarray]] = {}
        for ch in channels:
            arrays_by_channel[int(ch)] = [st[ch].cpu().numpy() for st in stage_tensors]

        out_png = out_dir / f"sample_{idx_req:04d}_gt_denoising_5stage_2ch.png"
        _save_sample_panel(
            out_png,
            arrays_by_channel=arrays_by_channel,
            stage_titles=stage_titles,
            sample_idx=int(idx_req),
            gid=gid,
            pair_index=pair_index,
            k_values=k_values,
        )

        summary["samples"].append(
            {
                "sample_index": int(idx_req),
                "gid": gid,
                "pair_index": pair_index,
                "channels": channels,
                "out_png": str(out_png),
            }
        )
        print(f"[saved] {out_png}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

