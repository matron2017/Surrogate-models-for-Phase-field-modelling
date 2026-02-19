#!/usr/bin/env python3
"""
Visualize flow-matching vs diffusion-bridge process intuition on one sample.

Outputs:
  - schedule_coefficients.png
  - flow_intermediates_ch{c}.png
  - bridge_intermediates_ch{c}.png
  - summary.json
"""

from __future__ import annotations

import argparse
import json
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
import sys

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
            raise ValueError(
                f"Input channels {x.shape[0]} <= theta_channels {theta_channels}; cannot split theta."
            )
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
    if len(vals) != 3:
        raise ValueError("Expected exactly 3 comma-separated fractions, e.g. '0.15,0.50,0.85'.")
    return vals


def _flow_states(x: torch.Tensor, y: torch.Tensor, fracs: Sequence[float]) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = [("past x", x)]
    for t in fracs:
        x_t = (1.0 - float(t)) * x + float(t) * y
        out.append((f"t={t:.2f}", x_t))
    out.append(("target y", y))
    return out


def _bridge_states_unidb(
    schedule,
    x: torch.Tensor,
    y: torch.Tensor,
    fracs: Sequence[float],
    seed: int,
) -> Tuple[List[Tuple[str, torch.Tensor]], List[Dict[str, float]]]:
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError("This visualization expects a UniDB schedule for bridge states.")
    x_b = x.unsqueeze(0).float()
    y_b = y.unsqueeze(0).float()

    g = torch.Generator(device=x_b.device)
    g.manual_seed(int(seed))
    eps = torch.randn(y_b.shape, generator=g, device=y_b.device, dtype=y_b.dtype)

    states: List[Tuple[str, torch.Tensor]] = [("past x", x)]
    meta: List[Dict[str, float]] = []
    T = int(schedule.timesteps)
    for frac in fracs:
        t_idx = int(round(float(frac) * (T - 1)))
        t_idx = max(1, min(T - 1, t_idx))
        t = torch.tensor([t_idx], dtype=torch.long, device=x_b.device)
        x_t, _ = schedule.sample_noisy_state(x0=y_b, mu=x_b, t=t, noise=eps)
        m_t = float(schedule._m(t, y_b).view(-1)[0].item())
        n_t = float(schedule._n(t, y_b).view(-1)[0].item())
        s_t = float(schedule.f_sigma(t, y_b).view(-1)[0].item())
        states.append((f"k={t_idx}", x_t[0]))
        meta.append({"frac": float(frac), "k": float(t_idx), "m": m_t, "n": n_t, "sigma": s_t})
    states.append(("target y", y))
    return states, meta


def _save_panel(states: Sequence[Tuple[str, torch.Tensor]], ch: int, out_path: Path, title: str) -> None:
    arrays = [s[1][ch].detach().cpu().numpy() for s in states]
    vmin = float(min(np.min(a) for a in arrays))
    vmax = float(max(np.max(a) for a in arrays))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite values encountered in visualization arrays.")
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, st), arr in zip(axes, states, arrays):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(f"{title} | channel={ch}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_schedule_plot(schedule, flow_fracs: Sequence[float], bridge_fracs: Sequence[float], out_path: Path) -> Dict[str, Any]:
    T = int(schedule.timesteps)
    k = torch.arange(1, T, dtype=torch.long)
    ref = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    m = schedule._m(k, ref).view(-1).cpu().numpy()
    n = schedule._n(k, ref).view(-1).cpu().numpy()
    sigma = schedule.f_sigma(k, ref).view(-1).cpu().numpy()
    tau = k.cpu().numpy() / float(T - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Flow schedule
    t = np.linspace(0.0, 1.0, 300)
    axes[0].plot(t, 1.0 - t, label="source weight (1-t)", lw=2)
    axes[0].plot(t, t, label="target weight (t)", lw=2)
    axes[0].plot(t, np.zeros_like(t), label="noise scale (0)", lw=2)
    for tv in flow_fracs:
        axes[0].axvline(float(tv), color="k", ls="--", lw=1, alpha=0.35)
    axes[0].set_title("Flow Matching Path Coefficients")
    axes[0].set_xlabel("t in [0,1]")
    axes[0].set_ylabel("coefficient value")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    # Bridge schedule
    axes[1].plot(tau, m, label="m(t): target weight", lw=2)
    axes[1].plot(tau, n, label="n(t): source weight", lw=2)
    axes[1].plot(tau, sigma, label="sigma(t): noise scale", lw=2)
    bridge_marks = []
    for tv in bridge_fracs:
        kk = int(round(float(tv) * (T - 1)))
        kk = max(1, min(T - 1, kk))
        tau_k = kk / float(T - 1)
        axes[1].axvline(tau_k, color="k", ls="--", lw=1, alpha=0.35)
        bridge_marks.append({"frac": float(tv), "k": int(kk), "tau": float(tau_k)})
    axes[1].set_title("UniDB Bridge Coefficients")
    axes[1].set_xlabel("timestep fraction (k/(T-1))")
    axes[1].set_ylabel("coefficient value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"bridge_marks": bridge_marks}


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize flow-vs-bridge schedules and intermediate states.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--channels", type=str, default="0,1")
    ap.add_argument("--flow-fracs", type=str, default="0.15,0.50,0.85")
    ap.add_argument("--bridge-fracs", type=str, default="0.10,0.50,0.90")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "visuals" / "flow_bridge_intuition")
    args = ap.parse_args()

    cfg = _load_cfg(args.bridge_config.expanduser().resolve())
    ds = _build_dataset(cfg, split=str(args.split))
    if len(ds) <= 0:
        raise RuntimeError("Dataset is empty.")
    idx = int(max(0, min(len(ds) - 1, int(args.sample_index))))
    sample = ds[idx]

    x = sample["input"].float()
    y = sample["target"].float()
    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    x_state, theta = _split_theta(x, cond_cfg)  # theta unused in process-only visualization
    _ = theta

    flow_fracs = _parse_fracs(args.flow_fracs)
    bridge_fracs = _parse_fracs(args.bridge_fracs)

    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(
            f"Expected UniDB schedule for this script, got kind={getattr(schedule, 'kind', 'unknown')}."
        )

    flow_states = _flow_states(x_state, y, flow_fracs)
    bridge_states, bridge_meta = _bridge_states_unidb(schedule, x_state, y, bridge_fracs, seed=int(args.seed))

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_meta = _save_schedule_plot(
        schedule=schedule,
        flow_fracs=flow_fracs,
        bridge_fracs=bridge_fracs,
        out_path=out_dir / "schedule_coefficients.png",
    )

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    for ch in channels:
        if ch < 0 or ch >= int(y.shape[0]):
            continue
        _save_panel(
            states=flow_states,
            ch=ch,
            out_path=out_dir / f"flow_intermediates_ch{ch}.png",
            title="Flow Matching: deterministic interpolation path",
        )
        _save_panel(
            states=bridge_states,
            ch=ch,
            out_path=out_dir / f"bridge_intermediates_ch{ch}.png",
            title="Diffusion Bridge: noisy UniDB forward states",
        )

    summary = {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "bridge_config": str(args.bridge_config),
        "split": str(args.split),
        "sample_index_used": idx,
        "gid": str(sample.get("gid", "n/a")),
        "pair_index": int(sample.get("pair_index", -1)),
        "x_state_shape": list(map(int, x_state.shape)),
        "y_shape": list(map(int, y.shape)),
        "schedule_kind": getattr(schedule, "kind", "unknown"),
        "schedule_timesteps": int(getattr(schedule, "timesteps", -1)),
        "flow_fracs": [float(v) for v in flow_fracs],
        "bridge_fracs": [float(v) for v in bridge_fracs],
        "bridge_intermediates_meta": bridge_meta,
        "schedule_marks": schedule_meta,
        "channels_plotted": channels,
        "notes": [
            "Flow panel stages are deterministic x_t=(1-t)x+t*y.",
            "Bridge panel stages are noisy states sampled from UniDB x_t=f_mean(y,x,t)+f_sigma(t)*eps.",
            "Bridge uses a fixed eps across selected stages to emphasize schedule effect."
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
