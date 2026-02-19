#!/usr/bin/env python3
"""
Create single-channel 20-step bridge summaries:
- one figure per sample
- shows x -> 3 intermediate denoising states -> y
- shows residual maps below for meaningful adjacent pairs
- compares plain bridge noise vs residual-modulated bridge noise
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


def _uni_coeffs(schedule, t_idx: int, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.tensor([int(t_idx)], dtype=torch.long, device=ref.device)
    m_t = schedule._m(t, ref)
    n_t = schedule._n(t, ref)
    s_t = schedule.f_sigma(t, ref)
    return m_t, n_t, s_t


def _make_pi(
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    mode: str,
    normalize: bool,
    scale: float,
    power: float,
    clip_max: Optional[float],
    eps: float,
) -> torch.Tensor:
    residual = y - x
    m = str(mode).strip().lower()
    if m == "abs":
        pi = residual.abs()
    elif m == "signed":
        pi = residual
    else:
        raise ValueError(f"Unsupported residual mode '{mode}'.")
    if float(power) != 1.0:
        if m == "abs":
            pi = pi.clamp_min(0.0).pow(float(power))
        else:
            pi = torch.sign(pi) * torch.pow(pi.abs().clamp_min(float(eps)), float(power))
    if bool(normalize):
        den = pi.abs().mean().clamp_min(float(eps))
        pi = pi / den
    if clip_max is not None:
        c = float(clip_max)
        if m == "abs":
            pi = pi.clamp(min=0.0, max=c)
        else:
            pi = pi.clamp(min=-c, max=c)
    if float(scale) != 1.0:
        pi = float(scale) * pi
    return pi


def _descending_k_indices(T: int, n_steps: int) -> List[int]:
    vals = torch.linspace(float(T - 1), 0.0, steps=max(2, int(n_steps))).round().to(torch.long).tolist()
    out: List[int] = []
    for v in vals:
        iv = int(max(0, min(T - 1, int(v))))
        if not out or out[-1] != iv:
            out.append(iv)
    if out[-1] != 0:
        out.append(0)
    return out


def _select_three_k(k_list: Sequence[int]) -> List[int]:
    # Pick meaningful early/mid/late points from the 20-step schedule.
    n = len(k_list)
    if n < 5:
        raise ValueError("Need at least 5 steps in k_list to select 3 intermediates.")
    idxs = [max(1, int(round((n - 1) * q))) for q in (0.25, 0.5, 0.75)]
    # Ensure unique and sorted in denoising order.
    uniq = []
    seen = set()
    for i in idxs:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    while len(uniq) < 3:
        cand = min(n - 2, (uniq[-1] + 1) if uniq else 1)
        if cand not in seen:
            uniq.append(cand)
            seen.add(cand)
        else:
            break
    return [int(k_list[i]) for i in uniq[:3]]


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


def _plot_single_channel(
    out_path: Path,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    plain_states: Sequence[torch.Tensor],
    residual_states: Sequence[torch.Tensor],
    k_selected: Sequence[int],
    channel: int,
    gid: str,
    sample_idx: int,
    src_step: Optional[int],
    dst_step: Optional[int],
    total_channels: int,
    dpi: int,
) -> None:
    # 5 columns: x, k1, k2, k3, y
    # 4 rows: plain states, plain residual deltas, residual-mod states, residual-mod residual deltas
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), constrained_layout=True)

    state_plain = [x[channel], plain_states[0][channel], plain_states[1][channel], plain_states[2][channel], y[channel]]
    state_resid = [x[channel], residual_states[0][channel], residual_states[1][channel], residual_states[2][channel], y[channel]]
    state_plain_np = [a.detach().cpu().numpy() for a in state_plain]
    state_resid_np = [a.detach().cpu().numpy() for a in state_resid]

    state_all = state_plain_np + state_resid_np
    vmin = float(min(np.nanmin(a) for a in state_all))
    vmax = float(max(np.nanmax(a) for a in state_all))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    # Residual maps for meaningful adjacent pairs.
    # Cols: total(y-x), k1-x, k2-k1, k3-k2, y-k3
    res_plain_np = [
        state_plain_np[4] - state_plain_np[0],
        state_plain_np[1] - state_plain_np[0],
        state_plain_np[2] - state_plain_np[1],
        state_plain_np[3] - state_plain_np[2],
        state_plain_np[4] - state_plain_np[3],
    ]
    res_resid_np = [
        state_resid_np[4] - state_resid_np[0],
        state_resid_np[1] - state_resid_np[0],
        state_resid_np[2] - state_resid_np[1],
        state_resid_np[3] - state_resid_np[2],
        state_resid_np[4] - state_resid_np[3],
    ]
    res_abs = float(max(np.nanmax(np.abs(a)) for a in (res_plain_np + res_resid_np)))
    if not np.isfinite(res_abs) or res_abs < 1e-12:
        res_abs = 1e-12

    k1, k2, k3 = [int(k) for k in k_selected]
    col_titles = [
        "Input x",
        f"Intermediate k={k1}",
        f"Intermediate k={k2}",
        f"Intermediate k={k3}",
        "Output y",
    ]
    res_titles = [
        "Total (y - x)",
        f"k={k1} - x",
        f"k={k2} - k={k1}",
        f"k={k3} - k={k2}",
        f"y - k={k3}",
    ]

    for j in range(5):
        im0 = axes[0, j].imshow(state_plain_np[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0, j].set_title(col_titles[j], fontsize=10)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        if j == 4:
            cb = fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)

        im1 = axes[1, j].imshow(res_plain_np[j], origin="lower", cmap="RdBu_r", vmin=-res_abs, vmax=res_abs, interpolation="nearest")
        axes[1, j].set_title(res_titles[j], fontsize=10)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        if j == 4:
            cb = fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)

        im2 = axes[2, j].imshow(state_resid_np[j], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[2, j].set_title(col_titles[j], fontsize=10)
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])
        if j == 4:
            cb = fig.colorbar(im2, ax=axes[2, j], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)

        im3 = axes[3, j].imshow(res_resid_np[j], origin="lower", cmap="RdBu_r", vmin=-res_abs, vmax=res_abs, interpolation="nearest")
        axes[3, j].set_title(res_titles[j], fontsize=10)
        axes[3, j].set_xticks([])
        axes[3, j].set_yticks([])
        if j == 4:
            cb = fig.colorbar(im3, ax=axes[3, j], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("Plain bridge\nstates", fontsize=11)
    axes[1, 0].set_ylabel("Plain bridge\nresiduals", fontsize=11)
    axes[2, 0].set_ylabel("Residual-mod\nstates", fontsize=11)
    axes[3, 0].set_ylabel("Residual-mod\nresiduals", fontsize=11)

    step_text = (
        f"Euler {src_step} -> {dst_step}"
        if src_step is not None and dst_step is not None
        else "Euler steps unavailable"
    )
    h, w = int(y.shape[-2]), int(y.shape[-1])
    fig.suptitle(
        f"20-step bridge summary | sample={sample_idx} ({gid}) | channel {channel}/{total_channels - 1}\n"
        f"{step_text} | latent resolution {h}x{w} | shown intermediates from 20-step schedule: k={k1}, {k2}, {k3}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Rows 1-2: plain bridge (global noise). Rows 3-4: residual-modulated bridge. "
        "Residual rows show pairwise changes between meaningful adjacent states.",
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.75),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-channel 20-step plain vs residual bridge summaries.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-indices", type=str, default="220,240")
    ap.add_argument("--channel", type=int, default=28)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--residual-mode", type=str, default="abs", choices=["abs", "signed"])
    ap.add_argument("--residual-normalize", dest="residual_normalize", action="store_true", default=True)
    ap.add_argument("--no-residual-normalize", dest="residual_normalize", action="store_false")
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--residual-power", type=float, default=1.0)
    ap.add_argument("--residual-clip", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=280)
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
        raise ValueError(f"Expected UniDB schedule, got kind={getattr(schedule, 'kind', 'unknown')}")
    ds = _build_dataset(cfg, split=str(args.split))
    cond_cfg = dict(cfg.get("conditioning", {}) or {})

    k_list = _descending_k_indices(int(schedule.timesteps), int(args.n_steps))
    k_selected = _select_three_k(k_list)
    req_indices = [int(s.strip()) for s in str(args.sample_indices).split(",") if s.strip()]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "requested_indices": req_indices,
        "channel": int(args.channel),
        "k_list_20step": k_list,
        "k_selected": k_selected,
        "n_steps_actual": len(k_list),
        "residual_mode": str(args.residual_mode),
        "residual_normalize": bool(args.residual_normalize),
        "residual_scale": float(args.residual_scale),
        "residual_power": float(args.residual_power),
        "residual_clip": None if args.residual_clip is None else float(args.residual_clip),
        "samples": [],
    }

    for idx_req in req_indices:
        idx = int(max(0, min(len(ds) - 1, idx_req)))
        sample = ds[idx]
        x = sample["input"].float()
        y = sample["target"].float()
        x_state, _ = _split_theta(x, cond_cfg)
        if x_state.shape != y.shape:
            raise ValueError(f"Shape mismatch at idx={idx}: {x_state.shape} vs {y.shape}")
        if int(args.channel) < 0 or int(args.channel) >= int(y.shape[0]):
            raise ValueError(f"Channel {args.channel} out of range [0, {int(y.shape[0]) - 1}]")

        pi = _make_pi(
            y=y,
            x=x_state,
            mode=str(args.residual_mode),
            normalize=bool(args.residual_normalize),
            scale=float(args.residual_scale),
            power=float(args.residual_power),
            clip_max=args.residual_clip,
            eps=1e-6,
        )

        g = torch.Generator(device=x_state.device)
        g.manual_seed(int(args.seed) + int(idx))
        eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

        plain_states: List[torch.Tensor] = []
        residual_states: List[torch.Tensor] = []
        for k in k_selected:
            m_k, n_k, s_k = _uni_coeffs(schedule, k, y)
            mean_k = n_k * x_state + m_k * y
            plain_states.append(mean_k + s_k * eps)
            residual_states.append(mean_k + s_k * (pi * eps))

        gid = str(sample.get("gid", "n/a"))
        pair_index = int(sample.get("pair_index", idx))
        src_step, dst_step = _resolve_pair_steps(ds, gid, pair_index)
        out_path = out_dir / f"sample_{idx_req:04d}_20step_5stage_plain_vs_residual_ch{int(args.channel)}.png"
        _plot_single_channel(
            out_path,
            x=x_state,
            y=y,
            plain_states=plain_states,
            residual_states=residual_states,
            k_selected=k_selected,
            channel=int(args.channel),
            gid=gid,
            sample_idx=int(idx_req),
            src_step=src_step,
            dst_step=dst_step,
            total_channels=int(y.shape[0]),
            dpi=int(args.dpi),
        )

        manifest["samples"].append(
            {
                "requested_index": int(idx_req),
                "used_index": int(idx),
                "gid": gid,
                "pair_index": pair_index,
                "source_euler_step": src_step,
                "target_euler_step": dst_step,
                "out_path": str(out_path),
                "pi_stats": {
                    "min": float(pi.min().item()),
                    "max": float(pi.max().item()),
                    "mean": float(pi.mean().item()),
                    "abs_mean": float(pi.abs().mean().item()),
                },
            }
        )

    (out_dir / "plain_vs_residual_20step_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
