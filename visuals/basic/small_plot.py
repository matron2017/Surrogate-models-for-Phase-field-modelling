#!/usr/bin/env python3
# Legacy reference: scripts/visualise_consecutive.py
# Consecutive-frame RAW PNGs with viridis and colourbars, same style as single-image plots.
# No second-person pronouns in code or comments.

from __future__ import annotations
import os, json
from typing import Dict, List, Tuple

import h5py
import numpy as np

# Import helpers alongside this script.
from solid_data_visual import (
    _auto_denormalise_images,
    _resize_hw,
    _robust_min_max,
    _save_png,
    _engi_time,
)

# ---------------- configuration ----------------
H5_PATH = "/scratch/project_2008261/rapid_solidification/data/rapid_solidification/simulation_train.h5"
OUT_DIR = "/scratch/project_2008261/rapid_solidification/results/visuals_basic/selected_consecutive"
SIZE_PX = 512
ROBUST_LOW, ROBUST_HIGH = 1.0, 99.0
UNITS: Dict[str, str] = {}  # e.g., {"Field_0": "K", "Field_1": "-"}

# Predefined runs:
# 1) G2.3: steps 270000→300000, stride 1000
# 2) 1.4G: steps 270000→300000, stride 1000
RUNS: List[Tuple[str, int, int, int]] = [
    ("G2.3", 270000, 300000, 1000),
    ("1.4G", 270000, 300000, 1000),
]

# ---------------- helpers ----------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _resolve_group_by_original_name(h5_path: str, sim_hint: str) -> str:
    base = os.path.dirname(h5_path)
    meta_path = os.path.join(base, "sim_meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        for gid, info in meta.items():
            name = str(info.get("original_name", ""))
            if sim_hint in name:
                return gid
    with h5py.File(h5_path, "r") as h5:
        keys = sorted([k for k in h5.keys() if isinstance(h5[k], h5py.Group)])
    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"Could not resolve group for hint={sim_hint!r}.")

def _visualise_range(
    h5_path: str,
    sim_hint: str,
    start_step: int,
    end_step: int,
    every: int,
    out_root: str,
    size_px: int,
    robust_low: float,
    robust_high: float,
    units: Dict[str, str],
) -> None:
    gid = _resolve_group_by_original_name(h5_path, sim_hint)
    out_dir = os.path.join(out_root, gid)
    _ensure_dir(out_dir)

    with h5py.File(h5_path, "r") as h5:
        assert gid in h5, f"group {gid!r} not found"
        grp = h5[gid]
        imgs_z = grp["images"][...]              # (T,C,H,W), z-scored
        times  = grp["times"][...].astype(np.int64)
        attrs  = dict(h5.attrs); attrs.update({k: grp.attrs[k] for k in grp.attrs.keys()})
        eff_dt = float(attrs.get("effective_dt", np.nan))

    imgs_raw, _ = _auto_denormalise_images(imgs_z, attrs)
    T, C, H, W = imgs_raw.shape

    wanted_steps = list(range(start_step, end_step + 1, max(1, int(every))))
    step_to_idx = {int(s): int(i) for i, s in enumerate(times.tolist())}
    idxs: List[int] = []
    steps_found: List[int] = []
    for s in wanted_steps:
        if s in step_to_idx:
            idxs.append(step_to_idx[s])
            steps_found.append(s)
    if not idxs:
        raise RuntimeError(f"No requested steps found in group {gid!r} for hint {sim_hint!r}.")

    sel = imgs_raw[idxs]  # (K,C,H,W)
    vmins: List[float] = []; vmaxs: List[float] = []
    for ch in range(C):
        vmin, vmax = _robust_min_max(sel[:, ch], robust_low, robust_high)
        vmins.append(vmin); vmaxs.append(vmax)

    for s, k in zip(steps_found, idxs):
        for ch in range(C):
            arr = _resize_hw(imgs_raw[k, ch], size_px)
            title = f"Field_{ch} — step={s}"
            if np.isfinite(eff_dt) and eff_dt > 0.0:
                title += f" | {_engi_time(float(s) * eff_dt)}"
            unit = units.get(f"Field_{ch}", "")
            out_path = os.path.join(out_dir, f"frame_step{s}_f{ch}.png")
            _save_png(out_path, arr, vmins[ch], vmaxs[ch], title, cmap="viridis", unit_label=unit)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "h5_path": h5_path,
                "group": gid,
                "sim_hint": sim_hint,
                "steps": steps_found,
                "every": every,
                "size_px": size_px,
                "robust_low": robust_low,
                "robust_high": robust_high,
                "vmins": vmins,
                "vmaxs": vmaxs,
                "effective_dt": (float(eff_dt) if np.isfinite(eff_dt) else None),
            },
            f,
            indent=2,
        )
    print(f"[{sim_hint}] saved {len(steps_found)*C} PNGs to {out_dir}")

# ---------------- main ----------------
def main() -> None:
    for sim_hint, start_step, end_step, every in RUNS:
        _visualise_range(
            h5_path=H5_PATH,
            sim_hint=sim_hint,
            start_step=start_step,
            end_step=end_step,
            every=every,
            out_root=OUT_DIR,
            size_px=SIZE_PX,
            robust_low=ROBUST_LOW,
            robust_high=ROBUST_HIGH,
            units=UNITS,
        )

if __name__ == "__main__":
    main()
