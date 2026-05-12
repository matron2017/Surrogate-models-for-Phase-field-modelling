#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--old', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cur = read_csv(args.current)
    old = read_csv(args.old)
    if cur is None and old is None:
        raise SystemExit('no metrics found')
    metrics = ['train_total', 'train_l1', 'train_grad', 'train_spec', 'val_l1']
    summary = {}
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        if old is not None and metric in old.columns:
            ax.plot(old['epoch'], old[metric], label='old project_2008488', lw=1.8)
            summary[f'old_{metric}_last'] = float(old[metric].iloc[-1])
            summary[f'old_{metric}_min'] = float(old[metric].min())
        if cur is not None and metric in cur.columns:
            ax.plot(cur['epoch'], cur[metric], label='current project_2008261', lw=2.2, marker='o', ms=3)
            summary[f'current_{metric}_last'] = float(cur[metric].iloc[-1])
            summary[f'current_{metric}_min'] = float(cur[metric].min())
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f'{metric}.png')
        plt.close(fig)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
