#!/usr/bin/env python3
"""
Plot only ground-truth bridge intermediates (no model prediction paths).

Outputs per channel:
  - ch{c}_GT_bridge_states_only.png
  - ch{c}_GT_residual_and_noise_terms.png
Plus:
  - schedule_coefficients.png
  - README_EXPLAINED.md
  - summary.json
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
        v = float(tok)
        vals.append(min(0.999, max(0.001, v)))
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


def _q_sample_unidb(
    schedule,
    x_source: torch.Tensor,
    y_target: torch.Tensor,
    t_idx: int,
    eps: torch.Tensor,
) -> torch.Tensor:
    m_t, n_t, s_t = _uni_coeffs(schedule, t_idx, y_target)
    return n_t * x_source + m_t * y_target + s_t * eps


def _save_schedule_plot(schedule, out_path: Path, k_list: Sequence[int]) -> None:
    T = int(schedule.timesteps)
    k = torch.arange(1, T, dtype=torch.long)
    ref = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    m = schedule._m(k, ref).view(-1).cpu().numpy()
    n = schedule._n(k, ref).view(-1).cpu().numpy()
    sigma = schedule.f_sigma(k, ref).view(-1).cpu().numpy()
    tau = k.cpu().numpy() / float(T - 1)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    ax.plot(tau, n, lw=2, label="source weight n(t)")
    ax.plot(tau, m, lw=2, label="target weight m(t)")
    ax.plot(tau, sigma, lw=2, label="noise scale sigma(t)")
    for kk in k_list:
        tv = kk / float(T - 1)
        ax.axvline(tv, color="k", ls="--", lw=1.2, alpha=0.5)
        ax.text(tv, 0.98, f"k={kk}", va="top", ha="left", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_title("Ground-truth Bridge Schedule (UniDB)")
    ax.set_xlabel("timestep fraction (k/(T-1))")
    ax.set_ylabel("coefficient value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_panel(
    arrays: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: Path,
    suptitle: str,
    cmap: str = "RdBu_r",
    overlay_text: Optional[str] = None,
) -> None:
    if len(arrays) != len(titles):
        raise ValueError("arrays/titles mismatch")
    vmin = float(min(np.nanmin(a) for a in arrays))
    vmax = float(max(np.nanmax(a) for a in arrays))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    if overlay_text:
        fig.text(
            0.01,
            0.01,
            overlay_text,
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.72),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_readme(out_dir: Path, *, k_high: int, k_mid: int, k_pre: int) -> None:
    lines: List[str] = []
    lines.append("# GT-Only Bridge Visuals (No Predictions)")
    lines.append("")
    lines.append("This folder contains only **ground-truth bridge process** visualizations.")
    lines.append("No model-predicted trajectories are shown.")
    lines.append("")
    lines.append("## Stage Definitions")
    lines.append("- `INPUT`: source/past latent state `x`.")
    lines.append(f"- `INTERMEDIATE-1`: true noisy bridge state at `k={k_high}`.")
    lines.append(f"- `INTERMEDIATE-2`: true noisy bridge state at `k={k_mid}`.")
    lines.append(f"- `STATE-BEFORE-OUTPUT`: true noisy bridge state at `k={k_pre}`.")
    lines.append("- `OUTPUT`: target/next latent state `y` (k=0 stage).")
    lines.append("")
    lines.append("## Files")
    lines.append("- `schedule_coefficients.png`: source/target/noise coefficients over bridge time.")
    lines.append("- `ch{c}_GT_bridge_states_only.png`: INPUT -> INTERMEDIATES -> STATE-BEFORE-OUTPUT -> OUTPUT.")
    lines.append("- `ch{c}_GT_residual_and_noise_terms.png`: residual `y-x` and noise terms `sigma(k)*eps` at selected k.")
    lines.append("")
    lines.append("## Practical Meaning")
    lines.append("- These images answer: what does the **true bridge process** look like before any model prediction quality is involved?")
    lines.append("- If your model is weak now, these GT-only plots are still valid process references.")
    (out_dir / "README_EXPLAINED.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot GT-only bridge intermediates and residual/noise terms.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--channels", type=str, default="0,1")
    ap.add_argument("--fracs", type=str, default="0.75,0.35")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "visuals" / "bridge_gt_only",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.bridge_config.expanduser().resolve())
    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(f"Expected UniDB schedule; got kind={getattr(schedule, 'kind', 'unknown')}")

    ds = _build_dataset(cfg, split=str(args.split))
    if len(ds) <= 0:
        raise RuntimeError("Dataset is empty.")
    idx = int(max(0, min(len(ds) - 1, int(args.sample_index))))
    sample = ds[idx]

    x = sample["input"].float()
    y = sample["target"].float()
    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    x_state, _ = _split_theta(x, cond_cfg)
    if x_state.shape != y.shape:
        raise ValueError(f"Shape mismatch: x_state={x_state.shape}, y={y.shape}")

    fracs = _parse_fracs(args.fracs)
    k_high = _k_from_frac(schedule, fracs[0])
    k_mid = _k_from_frac(schedule, fracs[1])
    k_pre = 1

    g = torch.Generator(device=x_state.device)
    g.manual_seed(int(args.seed))
    eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

    x_high = _q_sample_unidb(schedule, x_state, y, t_idx=k_high, eps=eps)
    x_mid = _q_sample_unidb(schedule, x_state, y, t_idx=k_mid, eps=eps)
    x_pre = _q_sample_unidb(schedule, x_state, y, t_idx=k_pre, eps=eps)

    _m_h, _n_h, s_h = _uni_coeffs(schedule, k_high, x_high)
    _m_m, _n_m, s_m = _uni_coeffs(schedule, k_mid, x_mid)
    _m_p, _n_p, s_p = _uni_coeffs(schedule, k_pre, x_pre)

    residual = y - x_state
    noise_h = s_h * eps
    noise_m = s_m * eps
    noise_p = s_p * eps

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_schedule_plot(schedule, out_dir / "schedule_coefficients.png", [k_high, k_mid, k_pre])

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    for ch in channels:
        if ch < 0 or ch >= int(y.shape[0]):
            continue
        _save_panel(
            arrays=[
                x_state[ch].cpu().numpy(),
                x_high[ch].cpu().numpy(),
                x_mid[ch].cpu().numpy(),
                x_pre[ch].cpu().numpy(),
                y[ch].cpu().numpy(),
            ],
            titles=[
                "1) INPUT source/past x",
                f"2) GT noisy state (k={k_high})",
                f"3) GT noisy state (k={k_mid})",
                f"4) GT state-before-output (k={k_pre})",
                "5) OUTPUT target y",
            ],
            out_path=out_dir / f"ch{ch}_GT_bridge_states_only.png",
            suptitle=f"GT-only bridge states | channel={ch}",
            overlay_text=(
                "No predictions here. These are true bridge forward noisy states from source x to target y."
            ),
        )
        _save_panel(
            arrays=[
                residual[ch].cpu().numpy(),
                noise_h[ch].cpu().numpy(),
                noise_m[ch].cpu().numpy(),
                noise_p[ch].cpu().numpy(),
                eps[ch].cpu().numpy(),
            ],
            titles=[
                "1) residual (y - x)",
                f"2) noise term sigma(k={k_high})*eps",
                f"3) noise term sigma(k={k_mid})*eps",
                f"4) noise term sigma(k={k_pre})*eps",
                "5) base Gaussian eps",
            ],
            out_path=out_dir / f"ch{ch}_GT_residual_and_noise_terms.png",
            suptitle=f"GT residual/noise decomposition | channel={ch}",
            overlay_text=(
                "Residual shows true change. Noise terms show how schedule scales Gaussian noise at each bridge stage."
            ),
        )

    summary = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bridge_config": str(args.bridge_config),
        "split": str(args.split),
        "sample_index_used": int(idx),
        "gid": str(sample.get("gid", "n/a")),
        "pair_index": int(sample.get("pair_index", -1)),
        "channels_plotted": channels,
        "x_state_shape": list(map(int, x_state.shape)),
        "y_shape": list(map(int, y.shape)),
        "schedule_kind": getattr(schedule, "kind", "unknown"),
        "schedule_timesteps": int(getattr(schedule, "timesteps", -1)),
        "stage_indices": {"k_high": int(k_high), "k_mid": int(k_mid), "k_pre_output": int(k_pre), "k_output": 0},
        "notes": [
            "GT-only visuals. No model prediction branch included.",
            "Bridge noisy states computed by x_k = n(k)*x + m(k)*y + sigma(k)*eps.",
            "Residual/noise panel separates true change y-x from scheduled noise components.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_readme(out_dir, k_high=k_high, k_mid=k_mid, k_pre=k_pre)
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
