#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if len(args.labels) != len(args.metrics):
        raise ValueError("labels and metrics must have same length")

    data = []
    for label, path in zip(args.labels, args.metrics):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        blob = json.loads(p.read_text())
        data.append((label, blob))

    labels = [d[0] for d in data]
    overall_rmse = [d[1]["overall_rmse"] for d in data]
    mape_mean = [d[1]["mape_image_summary_percent"]["mean"] for d in data]
    per_ch_rmse = np.array([d[1]["per_channel_rmse"] for d in data], dtype=float)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall bar plot
    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - width / 2, overall_rmse, width, label="overall RMSE (normalized)")
    ax1.set_ylabel("overall RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Overall RMSE vs mean MAPE% (full val set)")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, mape_mean, width, color="orange", label="mean MAPE% (per-image)")
    ax2.set_ylabel("mean MAPE%")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "bar_overall_rmse_vs_mape.png", dpi=200)
    plt.close(fig)

    # Per-channel RMSE bar plot
    ch = per_ch_rmse.shape[1]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / ch
    for c in range(ch):
        ax.bar(x - 0.4 + (c + 0.5) * width, per_ch_rmse[:, c], width, label=f"channel {c} RMSE")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Per-channel RMSE (full val set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_per_channel_rmse.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
