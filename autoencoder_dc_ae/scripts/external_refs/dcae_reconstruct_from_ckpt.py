#!/usr/bin/env python3
"""Reconstruct one PDE sample using a fine-tuned DC-AE checkpoint.

This script is intended for quick post-train smoke checks where training used a tiny
subset (e.g., 3 frames). It reproduces the same per-channel normalisation logic
as train_dcae_finetune.py and saves physical-value plots with colorbars.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def _project_root() -> Path:
    default_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("PROJECT_ROOT", str(default_root))).expanduser().resolve()


def _default_repo_root() -> Path:
    return Path(
        os.environ.get(
            "DC_GEN_REPO_ROOT",
            str(_project_root() / "external_refs" / "DC-Gen"),
        )
    ).expanduser().resolve()


def _default_h5_path() -> Path:
    data_root = Path(os.environ.get("DATA_ROOT", str(_project_root() / "data"))).expanduser()
    return (data_root / "train.h5").resolve()


def _load_dcgen_repo(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_index(h5_path: Path, t_start: int, t_step: int, max_frames: int) -> list[tuple[str, int]]:
    idx: list[tuple[str, int]] = []
    with h5py.File(h5_path, "r") as f:
        for sim in sorted(f.keys()):
            n_t = f[sim]["images"].shape[0]
            for t in range(t_start, n_t, t_step):
                idx.append((sim, t))
                if len(idx) >= max_frames:
                    return idx
    return idx


def _read_x3(h5_path: Path, sim: str, t: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        g = f[sim]
        x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
        th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
    return np.concatenate([x2, th], axis=0)


def _compute_norm_stats(h5_path: Path, train_index: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(3, np.inf, dtype=np.float32)
    maxs = np.full(3, -np.inf, dtype=np.float32)
    for sim, t in train_index:
        x = _read_x3(h5_path, sim, t)
        for c in range(3):
            mins[c] = min(mins[c], float(x[c].min()))
            maxs[c] = max(maxs[c], float(x[c].max()))
    scale = maxs - mins
    scale[scale == 0] = 1.0
    return mins, scale


def _save_physical_png(path: Path, arr2d: np.ndarray, cmap: str, vmin: float, vmax: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=130)
    im = ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _channel_style(name: str, x: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if name == "phi":
        m = max(abs(lo), abs(hi), 1e-6)
        return ("coolwarm", -m, m)
    return ("viridis", lo, hi if hi > lo else lo + 1e-6)


def main() -> int:
    default_repo_root = _default_repo_root()
    default_h5_path = _default_h5_path()
    ap = argparse.ArgumentParser(description="Reconstruct one sample with fine-tuned DC-AE checkpoint")
    ap.add_argument("--repo-root", default=str(default_repo_root))
    ap.add_argument("--h5", default=str(default_h5_path))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-key", default="dc-ae-f32c32-in-1.0")
    ap.add_argument("--model-source", default=None,
                    help="HuggingFace repo ID or local snapshot path for from_pretrained")
    ap.add_argument("--sim", default="sim_0001")
    ap.add_argument("--t-index", type=int, default=100)
    ap.add_argument("--train-t-start", type=int, default=0)
    ap.add_argument("--train-t-step", type=int, default=100)
    ap.add_argument("--train-max-frames", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root)
    h5_path = Path(args.h5)
    model_source = args.model_source or os.environ.get("MODEL_SOURCE") or f"mit-han-lab/{args.model_key}"
    _load_dcgen_repo(repo_root)
    from dc_gen.ae_model_zoo import DCAE_HF

    train_index = _build_index(h5_path, args.train_t_start, args.train_t_step, args.train_max_frames)
    mins, scale = _compute_norm_stats(h5_path, train_index)

    x_phys = _read_x3(h5_path, args.sim, args.t_index)
    x_norm = ((x_phys - mins[:, None, None]) / scale[:, None, None] * 2.0 - 1.0).astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DCAE_HF.from_pretrained(model_source).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    x = torch.from_numpy(x_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)

    y_norm = y.squeeze(0).float().cpu().numpy()
    y_phys = ((y_norm + 1.0) / 2.0 * scale[:, None, None] + mins[:, None, None]).astype(np.float32)

    diff = y_phys - x_phys
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))

    ch_names = ["phi", "c", "theta"]
    for c, name in enumerate(ch_names):
        cmap, vmin, vmax = _channel_style(name, x_phys[c], y_phys[c])
        _save_physical_png(out_dir / f"input_{name}_physical.png", x_phys[c], cmap, vmin, vmax, f"Input {name}")
        _save_physical_png(out_dir / f"recon_{name}_physical.png", y_phys[c], cmap, vmin, vmax, f"Recon {name}")
        _save_physical_png(
            out_dir / f"absdiff_{name}_physical.png",
            np.abs(diff[c]),
            "magma",
            0.0,
            float(np.abs(diff[c]).max()) + 1e-12,
            f"|Recon-Input| {name}",
        )

    np.save(out_dir / "input_3x512x512.npy", x_phys)
    np.save(out_dir / "recon_3x512x512.npy", y_phys)
    np.save(out_dir / "diff_3x512x512.npy", diff)

    summary = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "model_key": args.model_key,
        "model_source": model_source,
        "sim": args.sim,
        "t_index": int(args.t_index),
        "train_index_used_for_norm": train_index,
        "latent_shape": list(z.shape),
        "mse_physical": mse,
        "mae_physical": mae,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
