#!/usr/bin/env python3
"""Generate a tiny HDF5 placeholder dataset with PF-like tensors and scalars."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def synth_field(num_frames: int) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    frames = []
    for idx in range(num_frames):
        phase = 2 * np.pi * (idx / max(1, num_frames - 1))
        ch0 = np.sin(np.pi * xx + phase) * np.cos(np.pi * yy)
        ch1 = np.cos(np.pi * xx - phase) * np.sin(np.pi * yy)
        noise = 0.05 * np.random.default_rng(idx).standard_normal(size=ch0.shape, dtype=np.float32)
        field = np.stack([ch0, ch1 + noise], axis=0).astype(np.float32)
        frames.append(field)
    return np.stack(frames, axis=0)


def synth_scalars(num_frames: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    thermal = 1.0e6 + 2.0e5 * rng.random(num_frames, dtype=np.float32)
    time_norm = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    return np.stack([thermal, time_norm], axis=1)


def write_placeholder(path: Path, num_frames: int) -> None:
    fields = synth_field(num_frames)
    scalars = synth_scalars(num_frames)
    if (path.exists()):
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("images", data=fields, compression="gzip")
        h5.create_dataset("scalars", data=scalars, compression="gzip")
        h5.attrs["description"] = "Placeholder PF dataset: 2x1024x1024 frames with scalars"
    print(f"Wrote placeholder dataset to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create placeholder PF tensors")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/diffusion_prototype/data/placeholder_smoke.h5"),
                        help="Output HDF5 path")
    parser.add_argument("--frames", type=int, default=4, help="Number of sequential frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_placeholder(args.output, max(2, args.frames))


if __name__ == "__main__":
    main()
