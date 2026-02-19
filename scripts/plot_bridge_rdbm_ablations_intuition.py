#!/usr/bin/env python3
"""
Plot RDBM-inspired bridge ablations on one PF sample with two intermediate stages.

Outputs (per channel):
  - schedule_coefficients.png
  - residual_modulation_ch{c}.png
  - dual_objective_toggle_ch{c}.png
  - ddim_vs_dbim_ch{c}.png
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
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
        raise ValueError("Expected exactly 2 fractions, e.g. '0.70,0.30'.")
    vals = sorted(vals, reverse=True)
    return vals


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    den = (x_max - x_min).clamp_min(1e-8)
    return (x - x_min) / den


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
    sigma_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m_t, n_t, s_t = _uni_coeffs(schedule, t_idx, y_target)
    if sigma_scale is not None:
        s_t = s_t * sigma_scale
    return n_t * x_source + m_t * y_target + s_t * eps


def _rdbm_step_pred_noise(
    x_t: torch.Tensor,
    x_source: torch.Tensor,
    t_now: int,
    t_next: int,
    noise_pred: torch.Tensor,
    schedule,
) -> torch.Tensor:
    m_now, _n_now, s_now = _uni_coeffs(schedule, t_now, x_t)
    m_next, _n_next, s_next = _uni_coeffs(schedule, t_next, x_t)
    if t_now == int(schedule.timesteps) - 1:
        return x_source
    return x_source + (m_next / m_now.clamp_min(1e-8)) * (x_t - x_source) - (
        ((m_next / m_now.clamp_min(1e-8)) * s_now) - s_next
    ) * noise_pred


def _rdbm_step_pred_xstart(
    x_t: torch.Tensor,
    x_source: torch.Tensor,
    t_now: int,
    t_next: int,
    xstart_pred: torch.Tensor,
    schedule,
) -> torch.Tensor:
    m_now, _n_now, s_now = _uni_coeffs(schedule, t_now, x_t)
    m_next, _n_next, s_next = _uni_coeffs(schedule, t_next, x_t)
    if t_now == int(schedule.timesteps) - 1:
        return x_source + m_next * (xstart_pred - x_source)
    return x_source + (s_next / s_now.clamp_min(1e-8)) * (x_t - x_source) + (
        m_next - (m_now * s_next / s_now.clamp_min(1e-8))
    ) * (xstart_pred - x_source)


def _dbim_like_step(
    x_t: torch.Tensor,
    x_source: torch.Tensor,
    xstart_pred: torch.Tensor,
    t_now: int,
    t_next: int,
    schedule,
    eta: float,
    g: torch.Generator,
) -> torch.Tensor:
    m_now, n_now, s_now = _uni_coeffs(schedule, t_now, x_t)
    m_next, n_next, s_next = _uni_coeffs(schedule, t_next, x_t)

    coeff_xs = s_next / s_now.clamp_min(1e-8)
    coeff_y = m_next - coeff_xs * m_now
    coeff_src = n_next - coeff_xs * n_now
    x_det = coeff_xs * x_t + coeff_y * xstart_pred + coeff_src * x_source
    if eta <= 0.0 or t_next <= 0:
        return x_det

    base_var = (s_next**2 - (coeff_xs * s_now) ** 2).clamp_min(0.0)
    sigma_add = float(eta) * torch.sqrt(base_var)
    z = torch.randn(x_t.shape, generator=g, device=x_t.device, dtype=x_t.dtype)
    return x_det + sigma_add * z


def _save_coeff_plot(schedule, k_high: int, k_mid: int, out_path: Path) -> Dict[str, Any]:
    T = int(schedule.timesteps)
    k = torch.arange(1, T, dtype=torch.long)
    ref = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    m = schedule._m(k, ref).view(-1).cpu().numpy()
    n = schedule._n(k, ref).view(-1).cpu().numpy()
    sigma = schedule.f_sigma(k, ref).view(-1).cpu().numpy()
    tau = k.cpu().numpy() / float(T - 1)

    fig, ax = plt.subplots(figsize=(8.8, 4.0), constrained_layout=True)
    ax.plot(tau, m, lw=2, label="target weight m(t)")
    ax.plot(tau, n, lw=2, label="source weight n(t)")
    ax.plot(tau, sigma, lw=2, label="noise scale sigma(t)")
    for kk, color in [(k_high, "k"), (k_mid, "gray")]:
        tv = kk / float(T - 1)
        ax.axvline(tv, color=color, ls="--", lw=1.2, alpha=0.6)
        ax.text(tv, 0.98, f"k={kk}", va="top", ha="left", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_title("UniDB bridge coefficients (selected intermediate stages)")
    ax.set_xlabel("timestep fraction (k/(T-1))")
    ax.set_ylabel("coefficient value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"k_high": int(k_high), "k_mid": int(k_mid), "timesteps": int(T)}


def _save_panel(
    arrays: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: Path,
    suptitle: str,
    overlay_text: Optional[str] = None,
    cmap: str = "RdBu_r",
) -> None:
    if len(arrays) != len(titles):
        raise ValueError("arrays/titles length mismatch")
    vmin = float(min(np.nanmin(a) for a in arrays))
    vmax = float(max(np.nanmax(a) for a in arrays))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), constrained_layout=True)
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
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="black", ec="none", alpha=0.72),
            color="white",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_explained_readme(
    out_dir: Path,
    *,
    channels: Sequence[int],
    k_high: int,
    k_mid: int,
    k_pre: int,
    alpha: float,
    err_scale: float,
    dbim_eta: float,
) -> None:
    lines: List[str] = []
    lines.append("# Bridge Ablation Plots: What Each Image Means")
    lines.append("")
    lines.append("These plots visualize `past -> next` transition intuition in latent space.")
    lines.append("")
    lines.append("## Terms")
    lines.append(f"- `k={k_high}`: earlier/noisier bridge stage (farther from final target).")
    lines.append(f"- `k={k_mid}`: middle bridge stage.")
    lines.append(f"- `k={k_pre}`: **state before output** (last bridge state before final output stage).")
    lines.append("- `k=0`: output stage.")
    lines.append("- `past x`: previous timestep latent (the condition/source).")
    lines.append("- `target y`: next timestep latent (ground-truth).")
    lines.append("")
    lines.append("## Files")
    lines.append("- `schedule_coefficients.png`: source weight, target weight, and noise strength across bridge time.")
    lines.append("- `residual_modulation_ch{c}_base.png`: baseline noising path with explicit stages: INPUT -> INTERMEDIATE -> PRE-OUTPUT -> OUTPUT.")
    lines.append("- `residual_modulation_ch{c}.png`: residual-aware noising path (same stages, but noise is modulated by |target-source|).")
    lines.append("- `dual_objective_toggle_ch{c}.png`: reverse path with explicit PRE-OUTPUT state for `pred_noise` and `pred_x_start`.")
    lines.append("- `ddim_vs_dbim_ch{c}.png`: deterministic (`DDIM-like`) vs stochastic (`DBIM-like`) reverse paths, with PRE-OUTPUT shown.")
    lines.append("- `ch{c}_A_*.png` ... `ch{c}_D_*.png`: same figures as above, copied with human-readable names.")
    lines.append("")
    lines.append("## Column Meaning In The Main Panels")
    lines.append("- `INPUT`: what the solver starts from (source `x` and/or noisy state at `k_high`).")
    lines.append("- `INTERMEDIATE`: state after some reverse updates (for example at `k_mid`).")
    lines.append("- `STATE BEFORE OUTPUT`: state at `k=1` (last stage before final output).")
    lines.append("- `OUTPUT`: final predicted `k=0` state, compared against target `y`.")
    lines.append("")
    lines.append("## Practical Reading Guide")
    lines.append("- If `residual_modulation` highlights changing fronts better than `base`, residual-aware scaling is helping where physics changes quickly.")
    lines.append("- In `dual_objective_toggle`, if one branch lands closer to `target y`, that objective is more stable/useful for this case.")
    lines.append("- In `ddim_vs_dbim`, deterministic is cleaner/reproducible; stochastic explores alternate plausible outcomes.")
    lines.append("")
    lines.append("## How Prediction Actually Runs (Important)")
    lines.append("- **Model input (bridge backbone):** the model is conditioned on `past/source` and a noisy bridge state at a chosen bridge timestep.")
    lines.append("- **Training:** not recursive through all 100 timesteps. One random bridge timestep is sampled per batch item, and loss is computed at that timestep.")
    lines.append("- **Inference for one `past -> next` prediction:** recursive reverse updates are applied over selected bridge timesteps.")
    lines.append("- If you choose full schedule, that can be ~100 inner bridge updates (here schedule has `T=100`).")
    lines.append("- If you choose fast sampling (common), you use fewer updates (for example 20), not full 100.")
    lines.append("- **These plots:** show selected stages (`k_high`, `k_mid`, `k=1`, `k=0`) for intuition, not all 100 states.")
    lines.append("- **Autoregressive forecasting across many physical frames is a separate outer loop:** after one `past -> next` prediction, the predicted `next` can become the new `past` for the next physical step.")
    lines.append("")
    lines.append("## Parameters Used In This Run")
    lines.append(f"- residual modulation alpha: `{alpha}`")
    lines.append(f"- synthetic objective error scale: `{err_scale}`")
    lines.append(f"- DBIM stochasticity eta: `{dbim_eta}`")
    lines.append(f"- channels plotted: `{list(channels)}`")
    lines.append("")
    (out_dir / "README_EXPLAINED.md").write_text("\n".join(lines) + "\n")


def _copy_alias(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot RDBM-inspired bridge ablation intuition on PF sample.")
    ap.add_argument(
        "--bridge-config",
        type=Path,
        default=ROOT / "configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--channels", type=str, default="0,1")
    ap.add_argument("--fracs", type=str, default="0.75,0.35", help="Two timestep fractions, high->mid.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-mod-alpha", type=float, default=1.0, help="Residual-aware sigma scale: 1 + alpha*norm(|y-x|)")
    ap.add_argument("--objective-error-scale", type=float, default=0.08, help="Synthetic model error scale for objective toggle demo.")
    ap.add_argument("--dbim-eta", type=float, default=0.7, help="Stochasticity for DBIM-like path in DDIM-vs-DBIM panel.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "visuals" / "bridge_rdbm_ablations",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.bridge_config.expanduser().resolve())
    diff_cfg = dict(cfg.get("diffusion", {}) or {})
    schedule = get_noise_schedule(diff_cfg["noise_schedule"], **diff_cfg.get("schedule_kwargs", {}))
    if getattr(schedule, "kind", "") != "unidb":
        raise ValueError(
            f"This script expects UniDB schedule from bridge config; got kind={getattr(schedule, 'kind', 'unknown')}"
        )

    ds = _build_dataset(cfg, split=str(args.split))
    if len(ds) <= 0:
        raise RuntimeError("Dataset is empty.")
    idx = int(max(0, min(len(ds) - 1, int(args.sample_index))))
    sample = ds[idx]

    x = sample["input"].float()
    y = sample["target"].float()
    cond_cfg = dict(cfg.get("conditioning", {}) or {})
    x_state, _theta = _split_theta(x, cond_cfg)
    if x_state.shape != y.shape:
        raise ValueError(f"State/target shape mismatch: {x_state.shape} vs {y.shape}")

    fracs = _parse_fracs(args.fracs)
    k_high = _k_from_frac(schedule, fracs[0])
    k_mid = _k_from_frac(schedule, fracs[1])
    k_pre = 1
    k_low = 0

    g = torch.Generator(device=x_state.device)
    g.manual_seed(int(args.seed))
    eps = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

    x_high = _q_sample_unidb(schedule, x_state, y, t_idx=k_high, eps=eps)
    x_mid = _q_sample_unidb(schedule, x_state, y, t_idx=k_mid, eps=eps)
    x_pre = _q_sample_unidb(schedule, x_state, y, t_idx=k_pre, eps=eps)

    residual_mag = torch.abs(y - x_state)
    residual_scale = 1.0 + float(args.noise_mod_alpha) * _normalize01(residual_mag)
    x_high_mod = _q_sample_unidb(schedule, x_state, y, t_idx=k_high, eps=eps, sigma_scale=residual_scale)
    x_mid_mod = _q_sample_unidb(schedule, x_state, y, t_idx=k_mid, eps=eps, sigma_scale=residual_scale)
    x_pre_mod = _q_sample_unidb(schedule, x_state, y, t_idx=k_pre, eps=eps, sigma_scale=residual_scale)

    m_high, _n_high, s_high = _uni_coeffs(schedule, k_high, x_high)
    true_noise_high = (x_high - x_state - m_high * (y - x_state)) / s_high.clamp_min(1e-8)

    err_scale = float(max(0.0, args.objective_error_scale))
    noise_pred = true_noise_high + err_scale * torch.randn(
        true_noise_high.shape, generator=g, device=true_noise_high.device, dtype=true_noise_high.dtype
    )
    xstart_pred = y + err_scale * torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)

    pn_mid = _rdbm_step_pred_noise(x_high, x_state, k_high, k_mid, noise_pred, schedule)
    pn_pre = _rdbm_step_pred_noise(pn_mid, x_state, k_mid, k_pre, noise_pred, schedule)
    pn_low = _rdbm_step_pred_noise(pn_pre, x_state, k_pre, k_low, noise_pred, schedule)

    px_mid = _rdbm_step_pred_xstart(x_high, x_state, k_high, k_mid, xstart_pred, schedule)
    px_pre = _rdbm_step_pred_xstart(px_mid, x_state, k_mid, k_pre, xstart_pred, schedule)
    px_low = _rdbm_step_pred_xstart(px_pre, x_state, k_pre, k_low, xstart_pred, schedule)

    dd_mid = _rdbm_step_pred_xstart(x_high, x_state, k_high, k_mid, y, schedule)
    dd_pre = _rdbm_step_pred_xstart(dd_mid, x_state, k_mid, k_pre, y, schedule)
    dd_low = _rdbm_step_pred_xstart(dd_pre, x_state, k_pre, k_low, y, schedule)

    db_mid = _dbim_like_step(x_high, x_state, y, k_high, k_mid, schedule, eta=float(args.dbim_eta), g=g)
    db_pre = _dbim_like_step(db_mid, x_state, y, k_mid, k_pre, schedule, eta=float(args.dbim_eta), g=g)
    db_low = _dbim_like_step(db_pre, x_state, y, k_pre, k_low, schedule, eta=float(args.dbim_eta), g=g)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    coeff_meta = _save_coeff_plot(schedule, k_high=k_high, k_mid=k_mid, out_path=out_dir / "schedule_coefficients.png")

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    for ch in channels:
        if ch < 0 or ch >= int(y.shape[0]):
            continue
        base_path = out_dir / f"residual_modulation_ch{ch}_base.png"
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
                f"2) INTERMEDIATE noisy state x_k (k={k_high})",
                f"3) INTERMEDIATE noisy state x_k (k={k_mid})",
                f"4) STATE BEFORE OUTPUT x_k (k={k_pre})",
                "5) OUTPUT target y (ground truth)",
            ],
            out_path=base_path,
            suptitle=f"Base bridge noisy states | channel={ch}",
            overlay_text=(
                "Pipeline: INPUT source x -> noisy bridge states (intermediate) -> "
                "STATE BEFORE OUTPUT (k=1) -> OUTPUT stage (k=0 target)."
            ),
        )
        mod_path = out_dir / f"residual_modulation_ch{ch}.png"
        _save_panel(
            arrays=[
                residual_scale[ch].cpu().numpy(),
                x_high_mod[ch].cpu().numpy(),
                x_mid_mod[ch].cpu().numpy(),
                x_pre_mod[ch].cpu().numpy(),
                y[ch].cpu().numpy(),
            ],
            titles=[
                "1) |target-source| noise modulation map",
                f"2) INTERMEDIATE mod-noisy state (k={k_high})",
                f"3) INTERMEDIATE mod-noisy state (k={k_mid})",
                f"4) STATE BEFORE OUTPUT mod-noisy (k={k_pre})",
                "5) OUTPUT target y (ground truth)",
            ],
            out_path=mod_path,
            suptitle=f"Residual-aware noise modulation | channel={ch}",
            overlay_text=(
                "Same pipeline, but noise strength is modulated by |target-source| "
                "so changing regions receive stronger perturbation."
            ),
        )
        obj_path = out_dir / f"dual_objective_toggle_ch{ch}.png"
        _save_panel(
            arrays=[
                x_state[ch].cpu().numpy(),
                x_high[ch].cpu().numpy(),
                pn_mid[ch].cpu().numpy(),
                pn_pre[ch].cpu().numpy(),
                pn_low[ch].cpu().numpy(),
                px_mid[ch].cpu().numpy(),
                px_pre[ch].cpu().numpy(),
                px_low[ch].cpu().numpy(),
                y[ch].cpu().numpy(),
            ],
            titles=[
                "1) INPUT source/past x",
                f"2) INPUT noisy state x_k (k={k_high})",
                f"3) pred_noise intermediate (k={k_mid})",
                f"4) pred_noise state-before-output (k={k_pre})",
                "5) pred_noise OUTPUT (k=0)",
                f"6) pred_x_start intermediate (k={k_mid})",
                f"7) pred_x_start state-before-output (k={k_pre})",
                "8) pred_x_start OUTPUT (k=0)",
                "9) target y",
            ],
            out_path=obj_path,
            suptitle=f"Dual objective toggle (synthetic error) | channel={ch}",
            overlay_text=(
                "Compare reverse objectives on same input: pred_noise path vs pred_x_start path. "
                "Both show intermediate, pre-output (k=1), and final output (k=0)."
            ),
        )
        ddim_dbim_path = out_dir / f"ddim_vs_dbim_ch{ch}.png"
        _save_panel(
            arrays=[
                x_state[ch].cpu().numpy(),
                x_high[ch].cpu().numpy(),
                dd_mid[ch].cpu().numpy(),
                dd_pre[ch].cpu().numpy(),
                dd_low[ch].cpu().numpy(),
                db_mid[ch].cpu().numpy(),
                db_pre[ch].cpu().numpy(),
                db_low[ch].cpu().numpy(),
                y[ch].cpu().numpy(),
            ],
            titles=[
                "1) INPUT source/past x",
                f"2) INPUT noisy state x_k (k={k_high})",
                f"3) DDIM deterministic intermediate (k={k_mid})",
                f"4) DDIM state-before-output (k={k_pre})",
                "5) DDIM OUTPUT (k=0)",
                f"6) DBIM stochastic intermediate (k={k_mid})",
                f"7) DBIM state-before-output (k={k_pre})",
                "8) DBIM OUTPUT (k=0)",
                "9) target y",
            ],
            out_path=ddim_dbim_path,
            suptitle=f"DDIM-like deterministic vs DBIM-like stochastic | channel={ch}",
            overlay_text=(
                "DDIM: deterministic reverse path. DBIM: stochastic reverse path (eta>0). "
                "Both start from same noisy input and end at output stage."
            ),
        )
        _copy_alias(base_path, out_dir / f"ch{ch}_A_base_bridge_noisy_states.png")
        _copy_alias(mod_path, out_dir / f"ch{ch}_B_residual_aware_noise_modulation.png")
        _copy_alias(obj_path, out_dir / f"ch{ch}_C_pred_noise_vs_pred_xstart_with_preoutput.png")
        _copy_alias(ddim_dbim_path, out_dir / f"ch{ch}_D_ddim_vs_dbim_with_preoutput.png")

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
        "fracs": [float(v) for v in fracs],
        "coeff_meta": coeff_meta,
        "stage_indices": {"k_high": int(k_high), "k_mid": int(k_mid), "k_pre_output": int(k_pre), "k_output": int(k_low)},
        "noise_mod_alpha": float(args.noise_mod_alpha),
        "objective_error_scale": float(args.objective_error_scale),
        "dbim_eta": float(args.dbim_eta),
        "notes": [
            "Residual-aware modulation uses sigma_eff = sigma * (1 + alpha * norm(|y-x|)).",
            "Dual objective plot compares RDBM-style pred_noise and pred_x_start updates with the same synthetic model error.",
            "DDIM-like branch is deterministic (eta=0); DBIM-like branch injects stochasticity via eta.",
            "These plots are process intuition diagnostics, not a measured model benchmark.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_explained_readme(
        out_dir=out_dir,
        channels=channels,
        k_high=k_high,
        k_mid=k_mid,
        k_pre=k_pre,
        alpha=float(args.noise_mod_alpha),
        err_scale=float(args.objective_error_scale),
        dbim_eta=float(args.dbim_eta),
    )
    print(str(out_dir), flush=True)


if __name__ == "__main__":
    main()
