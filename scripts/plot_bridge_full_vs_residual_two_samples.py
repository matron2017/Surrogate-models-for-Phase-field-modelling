#!/usr/bin/env python3
"""
Create bridge visuals for selected samples showing:
1) full-state bridge noising x_k
2) equivalent residual-space noising r_k = x_k - x

No model predictions are used.
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
    ds_args["h5_path"] = str((cfg.get("paths", {}) or {}).get("h5", {})[split])
    return PFPairDataset(**ds_args)


def _split_theta(x: torch.Tensor, cond_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if bool(cond_cfg.get("use_theta", False)):
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if x.shape[0] <= theta_channels:
            raise ValueError(f"Input channels {x.shape[0]} <= theta_channels {theta_channels}.")
        return x[:-theta_channels, ...], x[-theta_channels:, ...]
    return x, None


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


def _plot_sample(
    out_path: Path,
    x: torch.Tensor,
    y: torch.Tensor,
    x_k: torch.Tensor,
    r_gt: torch.Tensor,
    r_k: torch.Tensor,
    eps_k: torch.Tensor,
    sample_idx: int,
    k: int,
    gid: str,
    src_step: Optional[int],
    dst_step: Optional[int],
    channels: Sequence[int],
) -> None:
    nrows = len(channels)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.2 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    total_channels = int(y.shape[0])
    h, w = int(y.shape[-2]), int(y.shape[-1])
    step_text = (
        f"Euler {src_step} -> {dst_step}"
        if src_step is not None and dst_step is not None
        else "Euler steps unavailable"
    )
    for row, ch in enumerate(channels):
        arrs = [
            x[ch].cpu().numpy(),
            y[ch].cpu().numpy(),
            x_k[ch].cpu().numpy(),
            r_gt[ch].cpu().numpy(),
            r_k[ch].cpu().numpy(),
            eps_k[ch].cpu().numpy(),
        ]
        titles = [
            f"Source latent x ({step_text})",
            f"Target latent y ({step_text})",
            f"Bridge state x_k (k={k})",
            "True change (y - x)",
            "Noisy change (x_k - x)",
            "Noise term sigma(k) * eps",
        ]
        val_min = float(min(np.nanmin(a) for a in arrs[:3]))
        val_max = float(max(np.nanmax(a) for a in arrs[:3]))
        res_abs = float(max(np.nanmax(np.abs(a)) for a in arrs[3:]))
        res_abs = max(res_abs, 1e-8)
        for col, (arr, title) in enumerate(zip(arrs, titles)):
            ax = axes[row, col]
            if col <= 2:
                im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=val_min, vmax=val_max, interpolation="nearest")
            else:
                im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-res_abs, vmax=res_abs, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)
        axes[row, 0].set_ylabel(f"Channel {ch}/{total_channels - 1}", fontsize=11)
    fig.suptitle(
        f"Latent phase-field bridge view | sample={sample_idx} ({gid}) | {step_text}\n"
        f"Latent map resolution: {h}x{w}.",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Simple reading: source latent x is moved toward target latent y through bridge timestep k. "
        "This is latent-space representation of phase-field data.\n"
        "Full: x_k = n(k)*x + m(k)*y + sigma(k)*eps. Residual form: x_k - x = m(k)*(y-x) + sigma(k)*eps.",
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.75),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_channels(s: str) -> List[int]:
    vals: List[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("Expected at least one channel index in --channels.")
    return vals


def _resolve_pair_steps(ds: PFPairDataset, gid: str, pair_index: int) -> Tuple[Optional[int], Optional[int]]:
    if not bool(getattr(ds, "use_pairs_idx", True)):
        return int(pair_index), int(pair_index)
    h5 = ds._get_h5()
    g = h5[gid]
    if "pairs_idx" not in g:
        return None, None
    pair = np.asarray(g["pairs_idx"][int(pair_index)])
    if pair.size < 2:
        return None, None
    return int(pair[0]), int(pair[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot GT full-vs-residual bridge visuals for two sample indices.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-indices", type=str, default="220,240")
    ap.add_argument("--channels", type=str, default="28", help="Comma-separated latent channels, e.g. '28'.")
    ap.add_argument("--frac", type=float, default=0.60, help="Bridge fraction used for this comparison view.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "visuals" / "bridge_rdbm_ablations_run1",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.bridge_config.expanduser().resolve())
    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"Expected UniDB schedule, got kind={getattr(schedule, 'kind', 'unknown')}.")
    ds = _build_dataset(cfg, split=str(args.split))
    cond_cfg = dict(cfg.get("conditioning", {}) or {})

    k = _k_from_frac(schedule, float(args.frac))
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_idxs = [int(s.strip()) for s in str(args.sample_indices).split(",") if s.strip()]
    channels = _parse_channels(args.channels)
    used: List[Dict[str, Any]] = []

    for idx_in in raw_idxs:
        idx = int(max(0, min(len(ds) - 1, idx_in)))
        sample = ds[idx]
        x = sample["input"].float()
        y = sample["target"].float()
        x_state, _ = _split_theta(x, cond_cfg)
        if x_state.shape != y.shape:
            raise ValueError(f"Shape mismatch at idx={idx}: {x_state.shape} vs {y.shape}")

        g = torch.Generator(device=x_state.device)
        g.manual_seed(int(args.seed) + int(idx))
        eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

        m_k, n_k, s_k = _uni_coeffs(schedule, k, y)
        x_k = n_k * x_state + m_k * y + s_k * eps
        r_gt = y - x_state
        r_k = x_k - x_state
        eps_k = s_k * eps

        gid = str(sample.get("gid", "n/a"))
        pair_index = int(sample.get("pair_index", idx))
        src_step, dst_step = _resolve_pair_steps(ds, gid, pair_index)
        out_path = out_dir / f"sample_{idx_in:04d}_GT_full_vs_residual_k{k}_ch{'-'.join(str(c) for c in channels)}.png"
        _plot_sample(
            out_path,
            x_state,
            y,
            x_k,
            r_gt,
            r_k,
            eps_k,
            sample_idx=idx_in,
            k=k,
            gid=gid,
            src_step=src_step,
            dst_step=dst_step,
            channels=channels,
        )
        used.append(
            {
                "requested_index": idx_in,
                "used_index": idx,
                "gid": gid,
                "pair_index": pair_index,
                "source_euler_step": src_step,
                "target_euler_step": dst_step,
                "channels": channels,
                "out_path": str(out_path),
            }
        )

    notes = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "k_used": int(k),
        "channels": channels,
        "schedule_kind": getattr(schedule, "kind", "unknown"),
        "formula_full": "x_k = n(k)*x + m(k)*y + sigma(k)*eps",
        "formula_residual": "r_k = x_k - x = m(k)*(y-x) + sigma(k)*eps",
        "answer_to_question": "In bridge papers like RDBM, process is source-conditioned full-state bridge; equivalently it denoises residuals around source.",
        "samples": used,
    }
    (out_dir / "residual_vs_full_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
