#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import math
import re

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

ROOT = Path('/scratch/project_462001338/pf_surrogate_modelling')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _prepare_batch
from models.train.core.loops import _predict_bridge_rollout_unidb_sde


def _sanitize(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(s))


def _tensor_stats(t: torch.Tensor) -> dict:
    td = t.detach().float()
    finite = torch.isfinite(td)
    finite_n = int(finite.sum().item())
    total_n = int(td.numel())
    out = {
        'shape': list(td.shape),
        'finite': f'{finite_n}/{total_n}',
        'has_nan': bool(torch.isnan(td).any().item()),
        'has_inf': bool(torch.isinf(td).any().item()),
    }
    if finite_n > 0:
        vals = td[finite]
        out.update(
            {
                'min': float(vals.min().item()),
                'max': float(vals.max().item()),
                'mean': float(vals.mean().item()),
                'std': float(vals.std(unbiased=False).item()),
            }
        )
    else:
        out.update({'min': math.nan, 'max': math.nan, 'mean': math.nan, 'std': math.nan})
    return out


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def _read_norm_attrs(h5_path: str, data_key: str, thermal_key: str) -> dict:
    with h5py.File(h5_path, 'r') as f:
        attrs = f.attrs

        def _get(*keys: str):
            for k in keys:
                if k in attrs:
                    v = attrs[k]
                    if isinstance(v, bytes):
                        v = v.decode('utf-8', errors='ignore')
                    if isinstance(v, np.ndarray) and v.shape == ():
                        v = v.item()
                    return str(v)
            return ''

        return {
            'normalization_schema': _get(f'{data_key}_normalization_schema', f'{data_key}_norm_type', 'normalization_schema', 'norm_type_images'),
            'normalization_norm': _get(f'{data_key}_norm', f'{data_key}_normalization'),
            'thermal_schema': _get(f'{thermal_key}_normalization_schema', f'{thermal_key}_norm_type', 'normalization_schema', 'norm_type_images'),
            'thermal_norm': _get(f'{thermal_key}_norm', f'{thermal_key}_normalization'),
            'has_channel_mean_std': bool('channel_mean' in attrs and 'channel_std' in attrs),
            'has_thermal_mean_std': bool(f'{thermal_key}_channel_mean' in attrs and f'{thermal_key}_channel_std' in attrs),
        }


def _make_panel(out_path: Path, x: np.ndarray, y: np.ndarray, pred: np.ndarray, theta: np.ndarray, title: str):
    # 2 rows (phi, c), 6 cols: source, target, pred, abs err, target-source, theta
    fig, ax = plt.subplots(2, 6, figsize=(18, 6))
    labels = ['source', 'target', 'prediction', 'abs_error', 'target-source', 'theta']
    for r, name in [(0, 'phi'), (1, 'c')]:
        arrs = [
            x[r],
            y[r],
            pred[r],
            np.abs(pred[r] - y[r]),
            y[r] - x[r],
            theta,
        ]
        cmaps = ['viridis', 'viridis', 'viridis', 'magma', 'coolwarm', 'plasma']
        for c in range(6):
            ax[r, c].imshow(arrs[c], cmap=cmaps[c])
            ax[r, c].set_title(f'{labels[c]} ({name})')
            ax[r, c].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--nfe', type=int, default=25)
    ap.add_argument('--max-items', type=int, default=8)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out-json', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    use_cuda = args.device.startswith('cuda') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    model = registry_build_model('diffusion', cfg['model']['backbone'], cfg['model'])
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device).eval()

    diff_cfg = dict(cfg.get('diffusion', {}) or {})
    schedule = get_noise_schedule(diff_cfg['noise_schedule'], **diff_cfg.get('schedule_kwargs', {}))
    noise_mode = str(diff_cfg.get('val_rollout_noise_mode', 'standard'))

    dcfg = cfg['dataloader']
    base_args = dict(dcfg.get('args', {}) or {})
    split_args = dict(dcfg.get(f'{args.split}_args', {}) or {})
    split_args['augment'] = False
    split_args['max_items'] = int(args.max_items)

    h5_path = cfg['paths']['h5'][args.split]
    ds = PFPairDataset(h5_path=h5_path, **{**base_args, **split_args})

    n = min(len(ds), int(args.max_items))
    fixed_indices = list(range(n))
    subset = Subset(ds, fixed_indices)
    dl = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    cond_cfg = dict(cfg.get('conditioning', {}) or {})
    theta_key = str(base_args.get('thermal_norm_data_key', 'thermal_field') or 'thermal_field')
    norm_attrs = _read_norm_attrs(h5_path=h5_path, data_key=str(base_args.get('data_key', 'images') or 'images'), thermal_key=theta_key)

    # Pairing audit via pairs_idx.
    pair_meta = {}
    with h5py.File(h5_path, 'r') as f:
        for ds_idx in fixed_indices:
            gid, k = ds.items[int(ds_idx)]
            pair = f[gid]['pairs_idx'][int(k)] if 'pairs_idx' in f[gid] else [int(k), int(k)]
            pair_meta[(gid, int(k))] = {'src_i': int(pair[0]), 'tgt_j': int(pair[1])}

    cached = []
    raw_batch_audit = None
    model_batch_audit = None
    with torch.inference_mode():
        for local_idx, batch in enumerate(dl):
            ds_idx = fixed_indices[local_idx]
            gid_ref, k_ref = ds.items[int(ds_idx)]

            gid_b = str(batch['gid'][0]) if isinstance(batch.get('gid', None), list) else str(batch['gid'])
            pair_b = int(batch['pair_index'][0].item()) if torch.is_tensor(batch['pair_index']) else int(batch['pair_index'])
            if gid_b != str(gid_ref) or pair_b != int(k_ref):
                raise RuntimeError(f'pairing mismatch dataset_idx={ds_idx}: batch(gid={gid_b},pair={pair_b}) expected(gid={gid_ref},pair={k_ref})')

            x_raw = batch['input']
            y_raw = batch['target']
            if not torch.isfinite(x_raw).all() or not torch.isfinite(y_raw).all():
                raise RuntimeError(f'non-finite raw tensors at dataset_idx={ds_idx}')

            x, y, _, theta = _prepare_batch(batch, device=device, cond_cfg=cond_cfg, use_chlast=False)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (theta is not None and not torch.isfinite(theta).all()):
                raise RuntimeError(f'non-finite model input tensors at dataset_idx={ds_idx}')

            if raw_batch_audit is None:
                raw_batch_audit = {
                    'dataset_index': int(ds_idx),
                    'gid': gid_b,
                    'pair_index': pair_b,
                    'source_stats': _tensor_stats(x_raw),
                    'target_stats': _tensor_stats(y_raw),
                    'theta_stats': _tensor_stats(x_raw[:, -int(cond_cfg.get('theta_channels', 1)):, ...]) if bool(cond_cfg.get('use_theta', False)) else None,
                }
            if model_batch_audit is None:
                model_batch_audit = {
                    'dataset_index': int(ds_idx),
                    'gid': gid_b,
                    'pair_index': pair_b,
                    'source_stats': _tensor_stats(x),
                    'target_stats': _tensor_stats(y),
                    'theta_stats': _tensor_stats(theta) if theta is not None else None,
                }

            cached.append(
                {
                    'dataset_index': int(ds_idx),
                    'gid': gid_b,
                    'pair_index': pair_b,
                    'pair_src_i': int(pair_meta[(gid_b, pair_b)]['src_i']),
                    'pair_tgt_j': int(pair_meta[(gid_b, pair_b)]['tgt_j']),
                    'x': x,
                    'y': y,
                    'theta': theta,
                }
            )

        if len(cached) > 1:
            theta_shuf = [cached[(i + 1) % len(cached)]['theta'] for i in range(len(cached))]
        else:
            theta_shuf = [cached[0]['theta']] if cached else []

        rows = []
        for i, row in enumerate(cached):
            x = row['x']
            y = row['y']
            theta = row['theta']
            theta_zero = torch.zeros_like(theta) if theta is not None else None
            theta_s = theta_shuf[i] if theta is not None else None

            pred_normal = _predict_bridge_rollout_unidb_sde(
                model=model,
                schedule=schedule,
                x=x,
                cond=None,
                theta=theta,
                nfe=int(args.nfe),
                eta=0.0,
                predict_next=False,
                noise_mode=noise_mode,
            )
            pred_shuf = _predict_bridge_rollout_unidb_sde(
                model=model,
                schedule=schedule,
                x=x,
                cond=None,
                theta=theta_s,
                nfe=int(args.nfe),
                eta=0.0,
                predict_next=False,
                noise_mode=noise_mode,
            )
            pred_zero = _predict_bridge_rollout_unidb_sde(
                model=model,
                schedule=schedule,
                x=x,
                cond=None,
                theta=theta_zero,
                nfe=int(args.nfe),
                eta=0.0,
                predict_next=False,
                noise_mode=noise_mode,
            )

            rmse_copy = _rmse(x, y)
            rmse_normal = _rmse(pred_normal, y)
            rmse_shuf = _rmse(pred_shuf, y)
            rmse_zero = _rmse(pred_zero, y)
            phi_rmse = _rmse(pred_normal[:, 0:1], y[:, 0:1])
            c_rmse = _rmse(pred_normal[:, 1:2], y[:, 1:2]) if y.shape[1] > 1 else float('nan')

            x_np = x[0].detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            p_np = pred_normal[0].detach().cpu().numpy()
            th_np = theta[0, 0].detach().cpu().numpy() if theta is not None else np.zeros_like(x_np[0])
            panel_path = args.out_dir / f"sample_{i:03d}_idx_{row['dataset_index']:05d}_{_sanitize(row['gid'])}_pair_{row['pair_index']:05d}.png"
            title = (
                f"idx={row['dataset_index']} gid={row['gid']} pair={row['pair_index']} "
                f"rmse_pred={rmse_normal:.6f} rmse_copy={rmse_copy:.6f} gap={rmse_copy-rmse_normal:.6f}"
            )
            _make_panel(panel_path, x=x_np, y=y_np, pred=p_np, theta=th_np, title=title)

            rows.append(
                {
                    'dataset_index': int(row['dataset_index']),
                    'gid': row['gid'],
                    'pair_index': int(row['pair_index']),
                    'pair_src_i': int(row['pair_src_i']),
                    'pair_tgt_j': int(row['pair_tgt_j']),
                    'rmse_copy': rmse_copy,
                    'rmse_normal': rmse_normal,
                    'rmse_theta_shuffled': rmse_shuf,
                    'rmse_theta_zero': rmse_zero,
                    'copy_gap': rmse_copy - rmse_normal,
                    'phi_rmse': phi_rmse,
                    'c_rmse': c_rmse,
                    'panel': str(panel_path),
                }
            )

    def _mean(key: str):
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        'n': len(rows),
        'val_endpoint_rmse': _mean('rmse_normal'),
        'copy_baseline_rmse': _mean('rmse_copy'),
        'copy_gap': _mean('copy_gap'),
        'phi_rmse': _mean('phi_rmse'),
        'c_rmse': _mean('c_rmse'),
        'theta_shuffled_rmse': _mean('rmse_theta_shuffled'),
        'theta_zero_rmse': _mean('rmse_theta_zero'),
    }
    if aggregate['val_endpoint_rmse'] is not None and aggregate['copy_baseline_rmse'] is not None:
        aggregate['beats_copy'] = bool(aggregate['val_endpoint_rmse'] < aggregate['copy_baseline_rmse'])

    payload = {
        'ckpt': str(args.ckpt),
        'split': args.split,
        'nfe': int(args.nfe),
        'device': str(device),
        'seed': int(args.seed),
        'normalization_runtime': {
            'dataset_args': {
                'normalize_images': bool(base_args.get('normalize_images', False)),
                'normalize_force': bool(base_args.get('normalize_force', False)),
                'normalize_source': str(base_args.get('normalize_source', 'file')),
                'normalize_thermal': bool(base_args.get('normalize_thermal', False)),
                'thermal_norm_source': str(base_args.get('thermal_norm_source', 'file')),
                'thermal_norm_data_key': theta_key,
            },
            'h5_attrs': norm_attrs,
        },
        'data_audit': {
            'pairing_assertions': 'passed',
            'raw_batch': raw_batch_audit,
            'model_input_batch': model_batch_audit,
        },
        'aggregate': aggregate,
        'rows': rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['aggregate'], indent=2))
    print(f'json={args.out_json}')


if __name__ == '__main__':
    main()
