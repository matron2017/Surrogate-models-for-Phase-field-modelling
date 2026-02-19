#!/usr/bin/env python3
"""
Plot 5-stage GT bridge denoising for one chosen channel, comparing:
  - plain bridge noising
  - residual-modulated bridge noising

Stages shown:
  INPUT x -> k1 -> k2 -> k3 -> OUTPUT y

Also writes a delta-focused panel per sample to make "what changed where"
explicit for both styles.
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
    vals: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("No sample indices provided.")
    return vals


def _parse_fracs(s: str) -> List[float]:
    vals: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(min(0.999, max(0.001, float(tok))))
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


def _stats(arr: np.ndarray) -> str:
    return f"mean={float(np.mean(arr)):.4f}\nmin={float(np.min(arr)):.4f}\nmax={float(np.max(arr)):.4f}"


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


def _save_panel(
    out_path: Path,
    *,
    plain_arrays: Sequence[np.ndarray],
    resid_arrays: Sequence[np.ndarray],
    stage_titles: Sequence[str],
    sample_idx: int,
    gid: str,
    pair_index: int,
    channel: int,
    total_channels: int,
    src_step: Optional[int],
    dst_step: Optional[int],
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(3.25 * 5, 3.2 * 2), constrained_layout=True)
    all_arrays = list(plain_arrays) + list(resid_arrays)
    vmin = float(min(np.nanmin(a) for a in all_arrays))
    vmax = float(max(np.nanmax(a) for a in all_arrays))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    for ridx, row_name in enumerate(("Plain bridge", "Residual-mod bridge")):
        row_arrays = plain_arrays if ridx == 0 else resid_arrays
        for cidx, (arr, ttl) in enumerate(zip(row_arrays, stage_titles)):
            ax = axes[ridx, cidx]
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            if ridx == 0:
                ax.set_title(ttl, fontsize=10, fontweight="bold")
            if cidx == 0:
                ax.set_ylabel(f"{row_name}\nchannel {channel}/{total_channels - 1}", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.02,
                _stats(arr),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.72),
            )
            if cidx == 4:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=8)

    step_txt = (
        f"Euler {src_step} -> {dst_step}"
        if src_step is not None and dst_step is not None
        else "Euler steps unavailable"
    )
    fig.suptitle(
        (
            f"GT 5-stage denoising | sample={sample_idx} gid={gid} pair={pair_index} | {step_txt}\n"
            "Same channel, two training styles: plain vs residual-modulated bridge"
        ),
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_delta_panel(
    out_path: Path,
    *,
    plain_arrays: Sequence[np.ndarray],
    resid_arrays: Sequence[np.ndarray],
    stage_titles: Sequence[str],
    sample_idx: int,
    gid: str,
    pair_index: int,
    channel: int,
    total_channels: int,
    src_step: Optional[int],
    dst_step: Optional[int],
) -> None:
    # Deltas by columns:
    # 0 total (y-x), 1 k1-x, 2 k2-k1, 3 k3-k2, 4 y-k3
    plain_d = [
        plain_arrays[4] - plain_arrays[0],
        plain_arrays[1] - plain_arrays[0],
        plain_arrays[2] - plain_arrays[1],
        plain_arrays[3] - plain_arrays[2],
        plain_arrays[4] - plain_arrays[3],
    ]
    resid_d = [
        resid_arrays[4] - resid_arrays[0],
        resid_arrays[1] - resid_arrays[0],
        resid_arrays[2] - resid_arrays[1],
        resid_arrays[3] - resid_arrays[2],
        resid_arrays[4] - resid_arrays[3],
    ]
    method_d = [r - p for r, p in zip(resid_d, plain_d)]

    fig, axes = plt.subplots(3, 5, figsize=(3.25 * 5, 3.2 * 3), constrained_layout=True)
    all_d = plain_d + resid_d + method_d
    vmax = float(max(np.nanmax(np.abs(a)) for a in all_d))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = 1e-12

    k1 = stage_titles[1].split("k=")[-1].split(")")[0] if "k=" in stage_titles[1] else "k1"
    k2 = stage_titles[2].split("k=")[-1].split(")")[0] if "k=" in stage_titles[2] else "k2"
    k3 = stage_titles[3].split("k=")[-1].split(")")[0] if "k=" in stage_titles[3] else "k3"
    col_titles = [
        "Total change\ny - x",
        f"Step 1 change\nk={k1} - x",
        f"Step 2 change\nk={k2} - k={k1}",
        f"Step 3 change\nk={k3} - k={k2}",
        f"Final change\ny - k={k3}",
    ]
    rows = [
        ("Plain bridge deltas", plain_d),
        ("Residual-mod deltas", resid_d),
        ("Residual - Plain\n(delta diff)", method_d),
    ]

    for ridx, (row_name, row_arrays) in enumerate(rows):
        for cidx, (arr, ttl) in enumerate(zip(row_arrays, col_titles)):
            ax = axes[ridx, cidx]
            im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            if ridx == 0:
                ax.set_title(ttl, fontsize=10, fontweight="bold")
            if cidx == 0:
                ax.set_ylabel(f"{row_name}\nchannel {channel}/{total_channels - 1}", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.02,
                _stats(arr),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.72),
            )
            if cidx == 4:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=8)

    step_txt = (
        f"Euler {src_step} -> {dst_step}"
        if src_step is not None and dst_step is not None
        else "Euler steps unavailable"
    )
    fig.suptitle(
        (
            f"GT denoising delta comparison | sample={sample_idx} gid={gid} pair={pair_index} | {step_txt}\n"
            "Red/blue shows signed local change. Third row isolates what residual modulation changed."
        ),
        fontsize=12,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot 5-stage GT denoising: plain vs residual-modulated, one channel.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m_rdbmres_controlhint.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-indices", type=str, default="313")
    ap.add_argument("--channel", type=int, default=1)
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
    sched_kwargs = dict(diff_cfg.get("schedule_kwargs", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **sched_kwargs)
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
        "channel": int(args.channel),
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
            raise ValueError(f"Shape mismatch: x_state={tuple(x_state.shape)} y={tuple(y.shape)}")
        if int(args.channel) < 0 or int(args.channel) >= int(y.shape[0]):
            raise ValueError(f"Channel {args.channel} out of range [0, {int(y.shape[0]) - 1}]")

        g = torch.Generator(device=x_state.device)
        g.manual_seed(int(args.seed) + int(idx_req))
        eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

        # plain and residual-modulated versions
        pi = _make_pi(
            y=y,
            x=x_state,
            mode=str(sched_kwargs.get("residual_mode", "abs")),
            normalize=bool(sched_kwargs.get("residual_normalize", True)),
            scale=float(sched_kwargs.get("residual_scale", 1.0)),
            power=float(sched_kwargs.get("residual_power", 1.0)),
            clip_max=sched_kwargs.get("residual_clip", None),
            eps=float(sched_kwargs.get("residual_eps", 1e-6)),
        )

        plain_tensors: List[torch.Tensor] = [x_state]
        resid_tensors: List[torch.Tensor] = [x_state]
        for k in k_values:
            m_k, n_k, s_k = _uni_coeffs(schedule, int(k), y)
            mean_k = n_k * x_state + m_k * y
            plain_tensors.append(mean_k + s_k * eps)
            resid_tensors.append(mean_k + s_k * (pi * eps))
        plain_tensors.append(y)
        resid_tensors.append(y)

        ch = int(args.channel)
        plain_arrays = [t[ch].cpu().numpy() for t in plain_tensors]
        resid_arrays = [t[ch].cpu().numpy() for t in resid_tensors]

        src_step, dst_step = _resolve_pair_steps(ds, gid, pair_index)
        out_png = out_dir / f"sample_{idx_req:04d}_gt_denoising_5stage_plain_vs_residual_ch{ch}.png"
        _save_panel(
            out_png,
            plain_arrays=plain_arrays,
            resid_arrays=resid_arrays,
            stage_titles=stage_titles,
            sample_idx=int(idx_req),
            gid=gid,
            pair_index=pair_index,
            channel=ch,
            total_channels=int(y.shape[0]),
            src_step=src_step,
            dst_step=dst_step,
        )
        print(f"[saved] {out_png}")

        out_png_deltas = out_dir / f"sample_{idx_req:04d}_gt_denoising_5stage_plain_vs_residual_ch{ch}_deltas.png"
        _save_delta_panel(
            out_png_deltas,
            plain_arrays=plain_arrays,
            resid_arrays=resid_arrays,
            stage_titles=stage_titles,
            sample_idx=int(idx_req),
            gid=gid,
            pair_index=pair_index,
            channel=ch,
            total_channels=int(y.shape[0]),
            src_step=src_step,
            dst_step=dst_step,
        )
        print(f"[saved] {out_png_deltas}")

        summary["samples"].append(
            {
                "sample_index": int(idx_req),
                "gid": gid,
                "pair_index": pair_index,
                "channel": ch,
                "source_euler_step": src_step,
                "target_euler_step": dst_step,
                "out_png": str(out_png),
                "out_png_deltas": str(out_png_deltas),
            }
        )

    (out_dir / "summary_plain_vs_residual_1ch.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote summary: {out_dir / 'summary_plain_vs_residual_1ch.json'}")


if __name__ == "__main__":
    main()
