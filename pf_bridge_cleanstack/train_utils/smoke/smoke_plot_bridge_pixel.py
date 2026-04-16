#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import sys

ROOT = Path('/scratch/project_462001338/pf_surrogate_modelling')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.scheduler_registry import get_noise_schedule
from models.train.core.pf_dataloader import PFPairDataset
from models.train.core.utils import _prepare_batch
from models.train.core.diffusion_forward import _build_diffusion_model_input

DIFF_PRED = {'unidb_predict_next', 'predict_next', 'predict_x0', 'x0_mse', 'next_field_mse'}


def bridge_coeffs(sch, t_long: torch.Tensor, ref: torch.Tensor):
    kind = str(getattr(sch, 'kind', '')).lower()
    if kind == 'unidb':
        b_t = sch._m(t_long, ref)
        a_t = sch._n(t_long, ref)
        c_t = sch.f_sigma(t_long, ref)
        return a_t, b_t, c_t
    if hasattr(sch, 'a') and hasattr(sch, 'b') and hasattr(sch, 'c'):
        a_t = sch.a.to(ref.device, dtype=ref.dtype)[t_long].view(-1, 1, 1, 1)
        b_t = sch.b.to(ref.device, dtype=ref.dtype)[t_long].view(-1, 1, 1, 1)
        c_t = sch.c.to(ref.device, dtype=ref.dtype)[t_long].view(-1, 1, 1, 1)
        return a_t, b_t, c_t
    raise ValueError(f'unsupported bridge schedule kind={kind}')


def find_dataset_index_by_pair(ds, pair_index: int):
    p = int(pair_index)
    for i in range(len(ds)):
        s = ds[i]
        if int(s.get('pair_index', -1)) == p:
            return i, str(s.get('gid', ''))
    raise ValueError(f'pair_index={p} not found in dataset (len={len(ds)})')


def rollout(model, sch, x, theta, nfe, predict_next=True):
    b = x.shape[0]
    T = int(sch.timesteps)
    ts = torch.linspace(T - 1, 1, steps=max(1, int(nfe))).round().long().tolist()
    uniq = []
    for t in ts:
        ti = max(1, min(T - 1, int(t)))
        if not uniq or uniq[-1] != ti:
            uniq.append(ti)
    if uniq[-1] != 1:
        uniq.append(1)

    xc = x
    cond = torch.zeros((b, 1), device=x.device, dtype=x.dtype)
    sched_kind = str(getattr(sch, 'kind', '')).lower()
    for i, s_idx in enumerate(uniq):
        s = torch.full((b,), int(s_idx), device=x.device, dtype=torch.long)
        sm = s.view(-1, 1).to(dtype=x.dtype)
        xin = _build_diffusion_model_input(x_noisy=xc, source=x, noise_schedule_obj=sch, sched_kind=sched_kind)
        try:
            pred = model(xin, cond, sm, hint=theta)
        except TypeError:
            pred = model(xin, cond, sm, theta=theta)

        a, beta, c = bridge_coeffs(sch, s, xc)
        yhat = pred if predict_next else (xc - a * x - c * pred) / beta.clamp_min(1e-6)
        if i == 0:
            xc = a * x + beta * yhat

        tnext = int(uniq[i + 1]) if i + 1 < len(uniq) else 0
        t = torch.full((b,), tnext, device=x.device, dtype=torch.long)
        at, bt, ct = bridge_coeffs(sch, t, xc)
        cx = ct / c.clamp_min(1e-8)
        cy = bt - cx * beta
        cs = at - cx * a
        xc = cx * xc + cy * yhat + cs * x

    return xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--pair-index', type=int, default=None)
    ap.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--nfe', type=int, default=20)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    a = ap.parse_args()

    d = torch.device(a.device if (a.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['config']
    m = registry_build_model('diffusion', cfg['model']['backbone'], cfg['model'])
    m.load_state_dict(ck['model'], strict=True)
    m = m.to(d).eval()
    sch = get_noise_schedule(cfg['diffusion']['noise_schedule'], **cfg['diffusion'].get('schedule_kwargs', {}))

    dcfg = cfg['dataloader']
    base = dict(dcfg.get('args', {}) or {})
    sp = dict(dcfg.get(f'{a.split}_args', {}) or {})
    if a.pair_index is not None:
        sp.pop('max_items', None)
    h5 = cfg['paths']['h5'][a.split]
    ds = PFPairDataset(h5_path=h5, **{**base, **sp})

    chosen_idx = int(a.index)
    chosen_pair = None
    chosen_gid = ''
    if a.pair_index is not None:
        chosen_idx, chosen_gid = find_dataset_index_by_pair(ds, int(a.pair_index))
        chosen_pair = int(a.pair_index)

    dl = DataLoader(Subset(ds, [int(chosen_idx)]), batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    if chosen_pair is None:
        chosen_pair = int(batch.get('pair_index', torch.tensor([-1]))[0].item()) if isinstance(batch.get('pair_index', None), torch.Tensor) else int(batch.get('pair_index', -1))
        chosen_gid = str(batch.get('gid', [''])[0]) if isinstance(batch.get('gid', None), list) else str(batch.get('gid', ''))

    x, y, cond, theta = _prepare_batch(batch, d, cond_cfg=cfg.get('conditioning', {}), use_chlast=False)

    predict_next = str(cfg.get('loss', {}).get('diffusion_objective', '')).lower() in DIFF_PRED
    with torch.inference_mode():
        pred = rollout(m, sch, x, theta, a.nfe, predict_next=predict_next)

    xin = x[0].detach().cpu().numpy()
    ygt = y[0].detach().cpu().numpy()
    ypr = pred[0].detach().cpu().numpy()
    rmse_pred = float(np.sqrt(np.mean((ypr - ygt) ** 2)))
    rmse_copy = float(np.sqrt(np.mean((xin - ygt) ** 2)))
    ratio = rmse_pred / max(rmse_copy, 1e-12)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for c, name in [(0, 'phase'), (1, 'conc')]:
        ax[c, 0].imshow(xin[c], cmap='viridis'); ax[c, 0].set_title(f'in {name}')
        ax[c, 1].imshow(ygt[c], cmap='viridis'); ax[c, 1].set_title('gt')
        ax[c, 2].imshow(ypr[c], cmap='viridis'); ax[c, 2].set_title('pred')
        ax[c, 3].imshow(ypr[c] - ygt[c], cmap='coolwarm'); ax[c, 3].set_title('pred-gt')
        for j in range(4):
            ax[c, j].axis('off')
    fig.suptitle(f"kind={getattr(sch, 'kind', '')}; pair={chosen_pair}; gid={chosen_gid}; RMSE pred={rmse_pred:.4e} copy={rmse_copy:.4e} ratio={ratio:.3f}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=180)

    print({'rmse_pred': rmse_pred, 'rmse_copy': rmse_copy, 'ratio_pred_over_copy': ratio, 'plot': str(a.out), 'pair_index': int(chosen_pair), 'gid': chosen_gid, 'dataset_index': int(chosen_idx)})


if __name__ == '__main__':
    main()
