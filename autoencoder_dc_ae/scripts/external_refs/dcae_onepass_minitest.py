#!/usr/bin/env python3
"""Ultra-quick one-pass minitest:
- take exactly 3 train samples
- do one optimizer pass over those 3 samples
- reconstruct one of the 3 and save plots
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
import torch.nn.functional as F


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


def _read_x3(h5_path: Path, sim: str, t: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        g = f[sim]
        x2 = np.asarray(g["images"][t, :2], dtype=np.float32)
        th = np.asarray(g["thermal_field"][t, :1], dtype=np.float32)
    return np.concatenate([x2, th], axis=0)


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


def _style(name: str, x: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if name == "phi":
        m = max(abs(lo), abs(hi), 1e-6)
        return ("coolwarm", -m, m)
    return ("viridis", lo, hi if hi > lo else lo + 1e-6)


def main() -> int:
    default_repo_root = _default_repo_root()
    default_h5_path = _default_h5_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(default_repo_root))
    ap.add_argument("--h5", default=str(default_h5_path))
    ap.add_argument("--model-key", default="dc-ae-f32c32-in-1.0")
    ap.add_argument("--model-source", default=None,
                    help="HuggingFace repo ID or local snapshot path for from_pretrained")
    ap.add_argument("--sim", default="sim_0001")
    ap.add_argument("--t-list", default="0,100,200")
    ap.add_argument("--recon-t", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_source = args.model_source or os.environ.get("MODEL_SOURCE") or f"mit-han-lab/{args.model_key}"

    _load_dcgen_repo(Path(args.repo_root))
    from dc_gen.ae_model_zoo import DCAE_HF

    t_list = [int(x.strip()) for x in args.t_list.split(",") if x.strip()]
    if len(t_list) != 3:
        raise ValueError("t-list must contain exactly 3 entries")

    xs_phys = np.stack([_read_x3(Path(args.h5), args.sim, t) for t in t_list], axis=0)  # (3,3,H,W)
    mins = xs_phys.reshape(3, 3, -1).min(axis=(0, 2))
    maxs = xs_phys.reshape(3, 3, -1).max(axis=(0, 2))
    scale = maxs - mins
    scale[scale == 0] = 1.0

    xs_norm = ((xs_phys - mins[None, :, None, None]) / scale[None, :, None, None] * 2.0 - 1.0).astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DCAE_HF.from_pretrained(model_source).to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    x = torch.from_numpy(xs_norm).to(device)

    # one pass over the 3-sample mini-batch
    use_amp = torch.cuda.is_available()
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        y0 = model.decode(model.encode(x))
        loss0 = F.l1_loss(y0, x)
    optim.zero_grad(set_to_none=True)
    loss0.backward()
    optim.step()

    with torch.no_grad():
        y1 = model.decode(model.encode(x))
        loss1 = F.l1_loss(y1, x)

    # reconstruct selected sample after the one-pass update
    idx = t_list.index(args.recon_t)
    in_norm = x[idx:idx + 1]
    with torch.no_grad():
        z = model.encode(in_norm)
        out_norm = model.decode(z)

    in_norm_np = in_norm.squeeze(0).float().cpu().numpy()
    out_norm_np = out_norm.squeeze(0).float().cpu().numpy()
    in_phys = ((in_norm_np + 1.0) / 2.0 * scale[:, None, None] + mins[:, None, None]).astype(np.float32)
    out_phys = ((out_norm_np + 1.0) / 2.0 * scale[:, None, None] + mins[:, None, None]).astype(np.float32)
    diff = out_phys - in_phys

    ch_names = ["phi", "c", "theta"]
    for c, name in enumerate(ch_names):
        cmap, vmin, vmax = _style(name, in_phys[c], out_phys[c])
        _save_physical_png(out_dir / f"input_{name}_physical.png", in_phys[c], cmap, vmin, vmax, f"Input {name}")
        _save_physical_png(out_dir / f"recon_{name}_physical.png", out_phys[c], cmap, vmin, vmax, f"Recon {name}")
        _save_physical_png(out_dir / f"absdiff_{name}_physical.png", np.abs(diff[c]), "magma", 0.0, float(np.abs(diff[c]).max()) + 1e-12, f"|Diff| {name}")

    ckpt_path = out_dir / "checkpoint.onepass.pth"
    torch.save({"model": model.state_dict(), "loss_before": float(loss0.detach()), "loss_after": float(loss1)}, ckpt_path)

    summary = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "sim": args.sim,
        "t_list": t_list,
        "recon_t": args.recon_t,
        "model_source": model_source,
        "loss_before_onepass": float(loss0),
        "loss_after_onepass": float(loss1),
        "latent_shape": list(z.shape),
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "command.txt").write_text("python " + " ".join(sys.argv) + "\n")
    np.save(out_dir / "input_3x512x512.npy", in_phys)
    np.save(out_dir / "recon_3x512x512.npy", out_phys)
    np.save(out_dir / "diff_3x512x512.npy", diff)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
