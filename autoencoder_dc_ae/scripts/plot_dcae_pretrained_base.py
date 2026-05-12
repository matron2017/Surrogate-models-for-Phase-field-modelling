#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, datetime as dt
from pathlib import Path
import h5py, numpy as np, torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
DC_GEN_REPO_ROOT = Path(os.environ.get('DC_GEN_REPO_ROOT', PROJECT_ROOT / 'external_refs' / 'DC-Gen')).resolve()
for p in [PROJECT_ROOT, DC_GEN_REPO_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.train_dcae_finetune import PDEFieldDataset
from dc_gen.ae_model_zoo import DCAE_HF

CH_NAMES = ['phi', 'c', 'theta']


def read_x3(h5_path: Path, sim: str | None, t: int) -> tuple[str, np.ndarray]:
    with h5py.File(h5_path, 'r') as f:
        gid = sim or sorted(f.keys())[0]
        g = f[gid]
        x2 = np.asarray(g['images'][t, :2], dtype=np.float32)
        th = np.asarray(g['thermal_field'][t, :1], dtype=np.float32)
    return gid, np.concatenate([x2, th], axis=0)


def style(name: str, a: np.ndarray, b: np.ndarray):
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    if name == 'phi':
        m = max(abs(lo), abs(hi), 1e-6)
        return 'coolwarm', -m, m
    return 'viridis', lo, hi if hi > lo else lo + 1e-6


def save_img(path: Path, arr: np.ndarray, title: str, cmap: str, vmin: float, vmax: float):
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def save_panel(path: Path, x: np.ndarray, y: np.ndarray, diff: np.ndarray, title: str):
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), dpi=150, facecolor='white')
    for r, name in enumerate(CH_NAMES):
        cmap, vmin, vmax = style(name, x[r], y[r])
        limd = float(max(np.nanmax(np.abs(diff[r])), 1e-12))
        panels = [(x[r], f'Input {name}', cmap, vmin, vmax), (y[r], f'Pretrained recon {name}', cmap, vmin, vmax), (np.abs(diff[r]), f'|diff| {name}', 'magma', 0.0, limd)]
        for c, (arr, ttl, cm, lo, hi) in enumerate(panels):
            im = axes[r, c].imshow(arr, cmap=cm, vmin=lo, vmax=hi, origin='upper')
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].set_title(ttl)
            fig.colorbar(im, ax=axes[r, c], fraction=0.047, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(path); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-h5', type=Path, default=PROJECT_ROOT / 'data' / 'train.h5')
    ap.add_argument('--sample-h5', type=Path, default=PROJECT_ROOT / 'data' / 'val.h5')
    ap.add_argument('--sim', default=None)
    ap.add_argument('--t-index', type=int, default=0)
    ap.add_argument('--stats-max-frames', type=int, default=200)
    ap.add_argument('--model-key', default='dc-ae-f32c32-in-1.0')
    ap.add_argument('--model-source', default=None)
    ap.add_argument('--out-dir', type=Path, default=PROJECT_ROOT / 'plots' / 'pretrained_base')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_ds = PDEFieldDataset(str(args.train_h5), max_frames=args.stats_max_frames, augment=False)
    mins = stats_ds.norm_min.astype(np.float32)
    scale = stats_ds.norm_scale.astype(np.float32)
    gid, x_phys = read_x3(args.sample_h5, args.sim, args.t_index)
    x_norm = ((x_phys - mins[:, None, None]) / scale[:, None, None] * 2.0 - 1.0).astype(np.float32)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_source = args.model_source or os.environ.get('MODEL_SOURCE') or f'mit-han-lab/{args.model_key}'
    model = DCAE_HF.from_pretrained(model_source).to(device).eval()
    x = torch.from_numpy(x_norm).unsqueeze(0).to(device)
    with torch.inference_mode():
        z = model.encode(x)
        recon_norm = model.decode(z)
    y_norm = recon_norm.squeeze(0).float().cpu().numpy()
    y_phys = ((y_norm + 1.0) / 2.0 * scale[:, None, None] + mins[:, None, None]).astype(np.float32)
    diff = y_phys - x_phys

    for i, name in enumerate(CH_NAMES):
        cmap, vmin, vmax = style(name, x_phys[i], y_phys[i])
        save_img(args.out_dir / f'input_{name}.png', x_phys[i], f'Input {name}', cmap, vmin, vmax)
        save_img(args.out_dir / f'pretrained_recon_{name}.png', y_phys[i], f'Pretrained recon {name}', cmap, vmin, vmax)
        save_img(args.out_dir / f'absdiff_{name}.png', np.abs(diff[i]), f'|pretrained recon-input| {name}', 'magma', 0.0, float(np.abs(diff[i]).max()) + 1e-12)
    save_panel(args.out_dir / 'pretrained_base_panel.png', x_phys, y_phys, diff, f'Base pretrained DCAE only | {gid} t={args.t_index}')
    np.save(args.out_dir / 'input_3x512x512.npy', x_phys)
    np.save(args.out_dir / 'pretrained_recon_3x512x512.npy', y_phys)
    np.save(args.out_dir / 'diff_3x512x512.npy', diff)

    summary = {
        'timestamp_utc': dt.datetime.now(dt.UTC).isoformat(),
        'uses_finetuned_checkpoint': False,
        'model_source': model_source,
        'sample_h5': str(args.sample_h5),
        'train_h5_for_norm_stats': str(args.train_h5),
        'sim': gid,
        't_index': int(args.t_index),
        'input_shape': list(x.shape),
        'latent_shape': list(z.shape),
        'recon_shape': list(recon_norm.shape),
        'mse_physical': float(np.mean(diff * diff)),
        'mae_physical': float(np.mean(np.abs(diff))),
        'channel_mae': {name: float(np.mean(np.abs(diff[i]))) for i, name in enumerate(CH_NAMES)},
        'out_dir': str(args.out_dir),
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
