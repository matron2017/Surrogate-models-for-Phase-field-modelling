#!/usr/bin/env python3
"""
Plot GT bridge intermediates for selected samples with:
1) no intermediate noise (mean path),
2) intermediate noise (plain bridge),
3) intermediate noise (soft residual-modulated bridge).

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


def _parse_fracs(s: str) -> List[float]:
    vals: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(min(0.999, max(0.001, float(tok))))
    if len(vals) != 2:
        raise ValueError("Expected exactly 2 fractions, e.g. '0.75,0.35'.")
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
        raise ValueError(f"Unsupported residual mode '{mode}' in this plotting script.")
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


def _plot_sample(
    out_path: Path,
    x: torch.Tensor,
    y: torch.Tensor,
    pi: torch.Tensor,
    items: Sequence[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    gid: str,
    sample_idx: int,
) -> None:
    # items: (k, mean_state, noisy_plain, noisy_residual)
    channels = [0, 1]
    ncols = 2 + len(items) * 3
    fig, axes = plt.subplots(2, ncols, figsize=(2.5 * ncols, 6.8), constrained_layout=True)

    for row, ch in enumerate(channels):
        # Build data list for this row
        arrs: List[np.ndarray] = [x[ch].cpu().numpy(), y[ch].cpu().numpy()]
        titles: List[str] = [f"ch{ch}: INPUT x", f"ch{ch}: OUTPUT y"]
        for k, mean_k, noisy_plain, noisy_res in items:
            arrs.extend(
                [
                    mean_k[ch].cpu().numpy(),
                    noisy_plain[ch].cpu().numpy(),
                    noisy_res[ch].cpu().numpy(),
                ]
            )
            titles.extend(
                [
                    f"ch{ch}: k={k} mean (no noise)",
                    f"ch{ch}: k={k} plain noisy",
                    f"ch{ch}: k={k} residual noisy",
                ]
            )
        # Shared scale for state maps
        state_vals = arrs
        vmin = float(min(np.nanmin(a) for a in state_vals))
        vmax = float(max(np.nanmax(a) for a in state_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

        for col, (arr, title) in enumerate(zip(arrs, titles)):
            ax = axes[row, col]
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=7)

    # Pi map inset-style row footer
    pi_min = float(pi.min().item())
    pi_max = float(pi.max().item())
    pi_mean = float(pi.mean().item())
    fig.suptitle(
        f"GT bridge intermediates: with/without noise | sample idx={sample_idx}, gid={gid}\n"
        "Columns per k: mean(no-noise) | plain noisy | residual-modulated noisy",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        f"soft residual constraint stats: pi min={pi_min:.4g}, max={pi_max:.4g}, mean={pi_mean:.4g}.  "
        "Residual-modulated uses x_k = mean_k + sigma_k*(pi*eps).",
        ha="left",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.75),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare bridge intermediate states with/without noise and residual modulation.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-indices", type=str, default="296,255")
    ap.add_argument("--fracs", type=str, default="0.75,0.35")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--residual-mode", type=str, default="abs", choices=["abs", "signed"])
    ap.add_argument("--residual-normalize", dest="residual_normalize", action="store_true", default=True)
    ap.add_argument("--no-residual-normalize", dest="residual_normalize", action="store_false")
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--residual-power", type=float, default=1.0)
    ap.add_argument("--residual-clip", type=float, default=None)
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
        raise ValueError(f"Expected UniDB schedule, got {getattr(schedule, 'kind', 'unknown')}")

    ds = _build_dataset(cfg, split=str(args.split))
    cond_cfg = dict(cfg.get("conditioning", {}) or {})

    fracs = _parse_fracs(args.fracs)
    k_high = _k_from_frac(schedule, fracs[0])
    k_mid = _k_from_frac(schedule, fracs[1])
    k_pre = 1
    k_list = [k_high, k_mid, k_pre]

    req_indices = [int(s.strip()) for s in str(args.sample_indices).split(",") if s.strip()]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "requested_indices": req_indices,
        "k_list": k_list,
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
            raise ValueError(f"shape mismatch at idx={idx}: {x_state.shape} vs {y.shape}")

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

        items: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for k in k_list:
            m_k, n_k, s_k = _uni_coeffs(schedule, k, y)
            mean_k = n_k * x_state + m_k * y
            noisy_plain = mean_k + s_k * eps
            noisy_res = mean_k + s_k * (pi * eps)
            items.append((k, mean_k, noisy_plain, noisy_res))

        out_path = out_dir / f"sample_{idx_req:04d}_with_without_intermediate_noise.png"
        _plot_sample(out_path, x_state, y, pi, items, gid=str(sample.get("gid", "n/a")), sample_idx=idx_req)

        manifest["samples"].append(
            {
                "requested_index": int(idx_req),
                "used_index": int(idx),
                "gid": str(sample.get("gid", "n/a")),
                "out_path": str(out_path),
                "pi_stats": {
                    "min": float(pi.min().item()),
                    "max": float(pi.max().item()),
                    "mean": float(pi.mean().item()),
                    "abs_mean": float(pi.abs().mean().item()),
                },
            }
        )

    (out_dir / "with_without_intermediate_noise_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
