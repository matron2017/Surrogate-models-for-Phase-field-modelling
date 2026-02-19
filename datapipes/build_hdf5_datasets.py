#!/usr/bin/env python3
"""
build_hdf5_datasets.py — YAML-driven.

Imports the spatial extractor and performs:
- Per-channel z-score based on TRAINING frames only
- Thermal gradient G z-score based on TRAINING simulations only
- Absolute time z-score based on TRAINING frames only
Applies those statistics to train/val/test consistently.
Writes split-wise HDF5 and a JSON normalisation manifest.

Respects debug and runtime options in the YAML.
"""

from __future__ import annotations
import os, sys, json, time, logging
from typing import Dict, List, Tuple, Any
import numpy as np
import h5py
import yaml

from vtu_spatial_processing import load_full, extract_thermal_gradient

logger = logging.getLogger("build_hdf5")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ── YAML config ───────────────────────────────────────────────────────────────

def load_yaml_config() -> Dict[str, Any]:
    default_cfg = "/scratch/project_2008261/pf_surrogate_modelling/configs/data/phase_field_data.yaml"
    cfg_path = os.environ.get("PF_DATA_CONFIG", default_cfg)
    logger.info(f"[config] PF_DATA_CONFIG={cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# ── Normalisation helpers ─────────────────────────────────────────────────────

def compute_mean_std_channelwise(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # images: (N, C, H, W)
    means = images.mean(axis=(0, 2, 3))
    vars_ = images.var(axis=(0, 2, 3))
    stds  = np.sqrt(vars_.astype(np.float64)).astype(np.float32, copy=False)
    return means.astype(np.float32), stds

def apply_zscore_channelwise(images: np.ndarray,
                             ch_mean: np.ndarray,
                             ch_std:  np.ndarray,
                             eps: float) -> np.ndarray:
    out = images.astype(np.float32, copy=True)
    for c in range(out.shape[1]):
        denom = float(ch_std[c]) if np.isfinite(ch_std[c]) and ch_std[c] > 0 else eps
        out[:, c] = (out[:, c] - float(ch_mean[c])) / denom
    return out

def _resolve_bounds(prm: Dict[str, Any], H: int, W: int) -> Tuple[float, float, float, float, float, float]:
    dx_units = float(prm.get('dx', 1.0))
    if not np.isfinite(dx_units) or dx_units <= 0:
        dx_units = 1.0
    dx_m = float(prm.get('physical_grid_spacing', dx_units * float(prm.get('W0', 1.0))))
    if not np.isfinite(dx_m) or dx_m <= 0:
        dx_m = dx_units

    x_min = float(prm.get('x_min', np.nan))
    x_max = float(prm.get('x_max', np.nan))
    y_min = float(prm.get('y_min', np.nan))
    y_max = float(prm.get('y_max', np.nan))
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min):
        x_min, x_max = 0.0, dx_units * W
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
        y_min, y_max = 0.0, dx_units * H

    # Bounds are stored in dx-units; convert to meters for thermal field.
    x_min_m, x_max_m = x_min * dx_m, x_max * dx_m
    y_min_m, y_max_m = y_min * dx_m, y_max * dx_m
    return x_min_m, x_max_m, y_min_m, y_max_m, dx_m, dx_m

def _build_thermal_stack(
    ts: np.ndarray,
    prm: Dict[str, Any],
    eff_dt: float,
    H: int,
    W: int,
    axis: str,
    use_x0: bool,
    T0: float,
    t0: float,
    G_raw: float,
) -> np.ndarray:
    x_min, x_max, y_min, y_max, dx, dy = _resolve_bounds(prm, H, W)
    x_centers = x_min + (np.arange(W, dtype=np.float64) + 0.5) * ((x_max - x_min) / W)
    y_centers = y_min + (np.arange(H, dtype=np.float64) + 0.5) * ((y_max - y_min) / H)

    V = float(prm.get('pulling_speed', 0.0))
    if not np.isfinite(V):
        V = 0.0

    if use_x0:
        x0_dx = float(prm.get('x0_dx', 0.0))
        if not np.isfinite(x0_dx):
            x0_dx = 0.0
        x0 = x0_dx * dx
    else:
        x0 = 0.0

    t_abs = ts.astype(np.float64) * float(eff_dt)
    t_shift = t_abs - float(t0)

    if axis == "y":
        base = y_centers - x0
        line = T0 + G_raw * (base[None, :] - V * t_shift[:, None])
        grid = np.tile(line[:, :, None], (1, 1, W))
    else:
        base = x_centers - x0
        line = T0 + G_raw * (base[None, :] - V * t_shift[:, None])
        grid = np.tile(line[:, None, :], (1, H, 1))

    return grid[:, None, :, :].astype(np.float32, copy=False)

def _resolve_T0(cfg_T0: Any, prm: Dict[str, Any]) -> float:
    if isinstance(cfg_T0, str):
        if cfg_T0.strip().lower() in ("tl", "liquidus", "auto"):
            tl = prm.get('Tl', None)
            if tl is not None and np.isfinite(tl):
                return float(tl)
    try:
        return float(cfg_T0)
    except Exception:
        return 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_yaml_config()

    BASE = str(cfg['base_data_dir'])
    OUT  = str(cfg['output_dir'])
    os.makedirs(OUT, exist_ok=True)

    FIELDS = list(cfg['fields'])
    assert len(FIELDS) == 2, f"Expected exactly 2 fields; got {len(FIELDS)}"
    GRID_SETTINGS = {'min_res': int(cfg['grid']['min_res']), 'max_res': int(cfg['grid']['max_res'])}
    TAU0 = float(cfg.get('tau0', 1.0))
    PAIR_STRIDES = [int(s) for s in (cfg.get('pairing', {}) or {}).get('strides', [1]) if int(s) > 0] or [1]

    splits: Dict[str, List[str]] = {
        'train': list(cfg['splits']['train']),
        'val':   list(cfg['splits']['val']),
        'test':  list(cfg['splits']['test']),
    }

    rt = cfg.get('runtime', {}) or {}
    log_level   = str(rt.get('log_level', 'INFO')).upper()
    logger.setLevel(getattr(logging, log_level))
    log_first_n = int(rt.get('log_first_n', 3))
    log_every   = int(rt.get('log_every', 25))
    precheck_vtu = bool(rt.get('precheck_vtu', True))
    dry_run      = bool(rt.get('dry_run', False))
    max_runtime  = float(rt.get('max_runtime_minutes', 300.0)) * 60.0 if rt.get('max_runtime_minutes', None) is not None else None

    dbg = cfg.get('debug', {}) or {}
    debug_enable  = bool(dbg.get('enable', False))
    debug_sim     = str(dbg.get('sim', ''))
    debug_first_n = dbg.get('first_n_frames', None)
    debug_split   = str(dbg.get('split', 'train'))
    if debug_enable:
        assert debug_sim, "debug.sim must be provided when debug.enable=true"
        splits = {'train': [], 'val': [], 'test': []}
        splits[debug_split] = [debug_sim]
        logger.info(f"[debug] limiting to sim={debug_sim}, first_n_frames={debug_first_n}, split={debug_split}")

    norm_cfg     = cfg.get('normalization', {}) or {}
    img_norm_cfg = norm_cfg.get('images', {}) or {}
    tg_norm_cfg   = norm_cfg.get('thermal_gradient', {}) or {}
    time_norm_cfg = norm_cfg.get('absolute_time', {}) or {}

    use_img_norm = bool(img_norm_cfg.get('apply', True))
    use_tg_norm  = bool(tg_norm_cfg.get('apply', True))
    use_time_norm = bool(time_norm_cfg.get('apply', True))

    eps_img = float(img_norm_cfg.get('eps', 1e-6))
    eps_tg  = float(tg_norm_cfg.get('eps', 1e-12))
    eps_time = float(time_norm_cfg.get('eps', 1e-12))

    thermal_cfg = cfg.get('thermal_field', {}) or {}
    thermal_enable = bool(thermal_cfg.get('enable', False))
    thermal_axis = str(thermal_cfg.get('axis', 'x')).lower()
    thermal_use_x0 = bool(thermal_cfg.get('use_x0', True))
    thermal_T0 = thermal_cfg.get('T0', 0.0)
    thermal_t0 = float(thermal_cfg.get('t0', 0.0))
    thermal_norm = bool(thermal_cfg.get('normalize', False))
    eps_thermal_field = float(thermal_cfg.get('eps', 1e-6))

    # GPU detection is only needed inside spatial module; keep a cap to 1 here and let spatial use device 0.
    try:
        import cupy as cp
        ngpu_detected = cp.cuda.runtime.getDeviceCount()
    except Exception:
        ngpu_detected = 1
    max_gpus = int(cfg.get('accelerators', {}).get('max_gpus', ngpu_detected))
    NGPU = max(1, min(ngpu_detected, max_gpus))
    logger.info(f"Using {NGPU} GPU(s)")

    start_time = time.monotonic()

    # ── Build or load training-only z-score stats ─────────────────────────────
    norm_path = os.path.join(OUT, 'normalization_config.json')
    ch_mean = ch_std = None
    tg_mean = 0.0; tg_std = 1.0
    time_mean = 0.0; time_std = 1.0
    have_cache = False

    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        if norm.get('method_images') == 'zscore' and len(norm.get('channel_mean', [])) == len(FIELDS):
            ch_mean = np.array(norm['channel_mean'], dtype=np.float32)
            ch_std  = np.array(norm['channel_std'],  dtype=np.float32)
            tg_mean = float(norm.get('thermal_mean', 0.0))
            tg_std  = float(norm.get('thermal_std',  1.0))
            time_mean = float(norm.get('time_mean', 0.0))
            time_std  = float(norm.get('time_std',  1.0))
            eps_thermal_field = float(norm.get('eps_thermal_field', eps_thermal_field))
            thermal_norm = str(norm.get('thermal_field_norm', 'none')) == 'zscore'
            thermal_axis = str(norm.get('thermal_field_axis', thermal_axis))
            thermal_T0 = float(norm.get('thermal_field_T0', thermal_T0))
            thermal_t0 = float(norm.get('thermal_field_t0', thermal_t0))
            eps_img  = float(norm.get('eps_images',  eps_img))
            eps_tg   = float(norm.get('eps_thermal', eps_tg))
            eps_time = float(norm.get('eps_time', eps_time))
            have_cache = True

    if not have_cache:
        imgs_accum: List[np.ndarray] = []
        tg_vals: List[float] = []
        time_vals: List[np.ndarray] = []
        thermal_sum = 0.0
        thermal_sumsq = 0.0
        thermal_count = 0

        for sim in splits.get('train', []):
            seq, ts, pidx, ptime, pstride, prm, fails, support_counts, support_mean_x = load_full(
                sim=sim, base=BASE, ngpu=NGPU,
                log_first_n=log_first_n, log_every=log_every, use_precheck=precheck_vtu,
                field_list=FIELDS, grid_settings=GRID_SETTINGS, pair_strides=PAIR_STRIDES,
                frame_limit=(debug_first_n if (debug_enable and sim == debug_sim and debug_first_n) else None),
            )
            if seq.size:
                imgs_accum.append(seq)  # (T,C,H,W)
            sim_dir = os.path.join(BASE, sim)
            tg_vals.append(extract_thermal_gradient(sim_dir, sim))
            eff_dt = float(prm.get('dt', 1.0)) * float(cfg.get('tau0', 1.0))
            time_vals.append(ts.astype(np.float64) * eff_dt)
            if thermal_enable and thermal_norm and seq.size:
                G_raw = extract_thermal_gradient(sim_dir, sim)
                T0_sim = _resolve_T0(thermal_T0, prm)
                thermal = _build_thermal_stack(
                    ts=ts, prm=prm, eff_dt=eff_dt,
                    H=seq.shape[-2], W=seq.shape[-1],
                    axis=thermal_axis, use_x0=thermal_use_x0,
                    T0=T0_sim, t0=thermal_t0, G_raw=G_raw,
                )
                thermal_sum += float(thermal.sum(dtype=np.float64))
                thermal_sumsq += float(np.square(thermal, dtype=np.float64).sum())
                thermal_count += int(thermal.size)

        if not imgs_accum:
            raise RuntimeError("No training images available to compute z-score stats.")
        all_imgs = np.concatenate(imgs_accum, axis=0)  # training-only frames
        ch_mean, ch_std = compute_mean_std_channelwise(all_imgs)

        tg_arr = np.asarray(tg_vals, dtype=np.float64) if tg_vals else np.array([0.0], dtype=np.float64)
        tg_mean = float(np.mean(tg_arr)); tg_std = float(np.std(tg_arr)) if tg_arr.size > 1 else 1.0

        time_all = np.concatenate(time_vals, axis=0) if time_vals else np.array([0.0], dtype=np.float64)
        time_mean = float(np.mean(time_all))
        time_std  = float(np.std(time_all)) if time_all.size > 1 else 1.0

        thermal_mean = 0.0
        thermal_std = 1.0
        if thermal_enable and thermal_norm and thermal_count > 0:
            thermal_mean = thermal_sum / thermal_count
            var = max(thermal_sumsq / thermal_count - thermal_mean ** 2, 0.0)
            thermal_std = float(np.sqrt(var)) if var > 0 else 1.0

        norm = {
            'method_images'  : 'zscore',
            'channel_mean'   : ch_mean.tolist(),
            'channel_std'    : ch_std.tolist(),
            'thermal_mean'   : tg_mean,
            'thermal_std'    : tg_std,
            'method_time'    : 'zscore',
            'time_mean'      : time_mean,
            'time_std'       : time_std,
            'eps_images'     : eps_img,
            'eps_thermal'    : eps_tg,
            'eps_time'       : eps_time,
            'thermal_field_mean': thermal_mean,
            'thermal_field_std': thermal_std,
            'eps_thermal_field': eps_thermal_field,
            'thermal_field_norm': 'zscore' if thermal_norm else 'none',
            'thermal_field_axis': thermal_axis,
            'thermal_field_T0': thermal_T0,
            'thermal_field_t0': thermal_t0,
            'fields'         : list(FIELDS),
            'pair_strides'   : list(PAIR_STRIDES),
            'computed_from'  : 'training_only',
        }
        with open(norm_path, 'w') as f:
            json.dump(norm, f, indent=2)
        logger.info(f"[norm] training-only zscore: ch_mean={ch_mean} ch_std={ch_std} "
                    f"tg_mean={tg_mean:.6g} tg_std={tg_std:.6g} time_mean={time_mean:.6g} time_std={time_std:.6g}")

    # ── Output maps ────────────────────────────────────────────────────────────
    map_path      = os.path.join(OUT, 'sim_map.json')
    meta_path     = os.path.join(OUT, 'sim_meta.json')
    manifest_path = os.path.join(OUT, 'sim_manifest.json')
    sim_map      = json.load(open(map_path))       if os.path.exists(map_path)       else {}
    sim_meta     = json.load(open(meta_path))      if os.path.exists(meta_path)      else {}
    sim_manifest = json.load(open(manifest_path))  if os.path.exists(manifest_path)  else {}
    existing = [int(gid.split('_')[1]) for gid in sim_map.values()] or []
    sid = max(existing) + 1 if existing else 1

    # ── Process splits with fixed statistics ───────────────────────────────────
    for split, sim_list in splits.items():
        if max_runtime and (time.monotonic() - start_time) > max_runtime:
            logger.info("Max runtime reached; stopping.")
            break

        fout = os.path.join(OUT, f'simulation_{split}.h5')
        mode = 'a' if not dry_run else 'w'
        with h5py.File(fout, mode) as h5f:
            h5f.attrs['normalization_schema'] = 'zscore'
            h5f.attrs['fields'] = np.array(FIELDS, dtype=h5py.string_dtype(encoding='utf-8'))
            h5f.attrs['channel_mean'] = ch_mean
            h5f.attrs['channel_std']  = ch_std
            h5f.attrs['thermal_mean'] = float(tg_mean)
            h5f.attrs['thermal_std']  = float(tg_std)
            h5f.attrs['time_mean']    = float(time_mean)
            h5f.attrs['time_std']     = float(time_std)
            h5f.attrs['thermal_field_mean'] = float(norm.get('thermal_field_mean', 0.0))
            h5f.attrs['thermal_field_std']  = float(norm.get('thermal_field_std', 1.0))
            h5f.attrs['zscore_eps_thermal_field'] = float(norm.get('eps_thermal_field', eps_thermal_field))
            h5f.attrs['thermal_field_norm'] = str(norm.get('thermal_field_norm', 'none'))
            h5f.attrs['thermal_field_axis'] = str(norm.get('thermal_field_axis', thermal_axis))
            h5f.attrs['thermal_field_T0'] = str(norm.get('thermal_field_T0', thermal_T0))
            h5f.attrs['thermal_field_t0'] = float(norm.get('thermal_field_t0', thermal_t0))
            h5f.attrs['thermal_field_enabled'] = bool(thermal_enable)
            h5f.attrs['tau0']         = float(TAU0)
            h5f.attrs['zscore_eps_images']   = float(eps_img)
            h5f.attrs['zscore_eps_thermal']  = float(eps_tg)
            h5f.attrs['zscore_eps_time']     = float(eps_time)
            h5f.attrs['pair_strides'] = np.array(PAIR_STRIDES, dtype=np.int32)

            for sim in sim_list:
                logger.info(f"[sim] begin {sim}")
                seq, ts, pidx, ptime, pstride, prm, fails, support_counts, support_mean_x = load_full(
                    sim=sim, base=BASE, ngpu=NGPU,
                    log_first_n=log_first_n, log_every=log_every, use_precheck=precheck_vtu,
                    field_list=FIELDS, grid_settings=GRID_SETTINGS, pair_strides=PAIR_STRIDES,
                    frame_limit=(debug_first_n if (debug_enable and sim == debug_sim and debug_first_n) else None),
                )
                if dry_run:
                    logger.info(f"[sim] dry-run {sim}: frames={len(ts)} pairs={len(pidx)}")
                    continue

                # Images: apply training z-score
                seq_norm = apply_zscore_channelwise(seq, ch_mean, ch_std, eps_img) if use_img_norm else seq.astype(np.float32, copy=True)

                support_counts = support_counts.astype(np.int32, copy=False)
                support_mean_x = support_mean_x.astype(np.float32, copy=False)
                valid_mask = (support_counts > 0).astype(np.uint8)

                # Thermal gradient: z-score using training stats
                sim_dir = os.path.join(BASE, sim)
                G_raw = extract_thermal_gradient(sim_dir, sim)
                G_norm_val = (G_raw - tg_mean) / (tg_std if tg_std > 0 else eps_tg) if use_tg_norm else G_raw

                prm['tau0'] = float(prm.get('tau0', TAU0))
                eff_dt = float(prm.get('dt', 1.0)) * TAU0
                time_phys = ts.astype(np.float64) * eff_dt
                if use_time_norm:
                    denom = time_std if time_std > 0 else eps_time
                    time_norm = ((time_phys - time_mean) / denom).astype(np.float32)
                else:
                    time_norm = time_phys.astype(np.float32)

                # Create or resume group
                if sim in sim_map:
                    gid = sim_map[sim]
                    grp = h5f[gid]
                else:
                    gid = f'sim_{sid:04d}'; sid += 1
                    sim_map[sim] = gid
                    grp = h5f.create_group(gid)

                # Datasets
                if 'images' in grp:
                    old = grp['images'].shape[0]
                    grp['images'].resize((old + seq_norm.shape[0],) + seq_norm.shape[1:]); grp['images'][old:] = seq_norm
                    grp['times'].resize((old + ts.size,)); grp['times'][old:] = ts
                    grp['pairs_idx'].resize((grp['pairs_idx'].shape[0] + pidx.shape[0], 2)); grp['pairs_idx'][-pidx.shape[0]:] = pidx
                    grp['pairs_time'].resize((grp['pairs_time'].shape[0] + pidx.shape[0], 2)); grp['pairs_time'][-pidx.shape[0]:] = ptime
                    grp['pairs_stride'].resize((grp['pairs_stride'].shape[0] + pstride.shape[0],)); grp['pairs_stride'][-pstride.shape[0]:] = pstride
                    if 'time_phys' in grp:
                        grp['time_phys'].resize((grp['time_phys'].shape[0] + time_phys.shape[0],))
                        grp['time_phys'][-time_phys.shape[0]:] = time_phys
                    else:
                        grp.create_dataset('time_phys', data=time_phys, maxshape=(None,), compression='gzip', chunks=(min(4096, len(time_phys)),))
                    if 'time_phys_norm' in grp:
                        grp['time_phys_norm'].resize((grp['time_phys_norm'].shape[0] + time_norm.shape[0],))
                        grp['time_phys_norm'][-time_norm.shape[0]:] = time_norm
                    else:
                        grp.create_dataset('time_phys_norm', data=time_norm, maxshape=(None,), compression='gzip', chunks=(min(4096, len(time_norm)),))
                    if 'bin_counts' in grp:
                        grp['bin_counts'].resize((old + support_counts.shape[0],) + support_counts.shape[1:])
                        grp['bin_counts'][old:] = support_counts
                    else:
                        ds = grp.create_dataset(
                            'bin_counts',
                            shape=(old + support_counts.shape[0],) + support_counts.shape[1:],
                            maxshape=(None,) + support_counts.shape[1:],
                            dtype=np.int32,
                            compression='gzip',
                            chunks=(1,) + support_counts.shape[1:],
                        )
                        ds[:old] = 0
                        ds[old:] = support_counts
                    if 'valid_mask' in grp:
                        grp['valid_mask'].resize((old + valid_mask.shape[0],) + valid_mask.shape[1:])
                        grp['valid_mask'][old:] = valid_mask
                    else:
                        ds = grp.create_dataset(
                            'valid_mask',
                            shape=(old + valid_mask.shape[0],) + valid_mask.shape[1:],
                            maxshape=(None,) + valid_mask.shape[1:],
                            dtype=np.uint8,
                            compression='gzip',
                            chunks=(1,) + valid_mask.shape[1:],
                        )
                        ds[:old] = 0
                        ds[old:] = valid_mask
                    if 'x_centroid' in grp:
                        grp['x_centroid'].resize((old + support_mean_x.shape[0],) + support_mean_x.shape[1:])
                        grp['x_centroid'][old:] = support_mean_x
                    else:
                        ds = grp.create_dataset(
                            'x_centroid',
                            shape=(old + support_mean_x.shape[0],) + support_mean_x.shape[1:],
                            maxshape=(None,) + support_mean_x.shape[1:],
                            dtype=np.float32,
                            compression='gzip',
                            chunks=(1,) + support_mean_x.shape[1:],
                        )
                        ds[:old] = np.nan
                        ds[old:] = support_mean_x
                else:
                    grp.create_dataset('images',     data=seq_norm, maxshape=(None,)+seq_norm.shape[1:], compression='gzip', chunks=(1,)+seq_norm.shape[1:])
                    grp.create_dataset('times',      data=ts,       maxshape=(None,),                     compression='gzip', chunks=(min(1024, len(ts)),))
                    grp.create_dataset('pairs_idx',  data=pidx,     maxshape=(None,2),                    compression='gzip', chunks=(min(1024, len(pidx)),2))
                    grp.create_dataset('pairs_time', data=ptime,    maxshape=(None,2),                    compression='gzip', chunks=(min(1024, len(ptime)),2))
                    grp.create_dataset('pairs_stride',   data=pstride,     maxshape=(None,), compression='gzip', chunks=(min(4096, len(pstride)),))
                    grp.create_dataset('time_phys',      data=time_phys,     maxshape=(None,), compression='gzip', chunks=(min(4096, len(time_phys)),))
                    grp.create_dataset('time_phys_norm', data=time_norm,     maxshape=(None,), compression='gzip', chunks=(min(4096, len(time_norm)),))
                    grp.create_dataset('bin_counts', data=support_counts, maxshape=(None,)+support_counts.shape[1:],
                                       compression='gzip', chunks=(1,)+support_counts.shape[1:])
                    grp.create_dataset('valid_mask', data=valid_mask, maxshape=(None,)+valid_mask.shape[1:],
                                       compression='gzip', chunks=(1,)+valid_mask.shape[1:])
                    grp.create_dataset('x_centroid', data=support_mean_x, maxshape=(None,)+support_mean_x.shape[1:],
                                       compression='gzip', chunks=(1,)+support_mean_x.shape[1:])

                if thermal_enable:
                    T0_sim = _resolve_T0(thermal_T0, prm)
                    thermal = _build_thermal_stack(
                        ts=ts, prm=prm, eff_dt=eff_dt,
                        H=seq_norm.shape[-2], W=seq_norm.shape[-1],
                        axis=thermal_axis, use_x0=thermal_use_x0,
                        T0=T0_sim, t0=thermal_t0, G_raw=G_raw,
                    )
                    if thermal_norm:
                        denom = float(norm.get('thermal_field_std', 1.0))
                        denom = denom if denom > 0 else eps_thermal_field
                        thermal = (thermal - float(norm.get('thermal_field_mean', 0.0))) / denom
                    if 'thermal_field' in grp:
                        old = grp['thermal_field'].shape[0]
                        grp['thermal_field'].resize((old + thermal.shape[0],) + thermal.shape[1:])
                        grp['thermal_field'][old:] = thermal
                    else:
                        grp.create_dataset('thermal_field', data=thermal, maxshape=(None,)+thermal.shape[1:],
                                           compression='gzip', chunks=(1,)+thermal.shape[1:])
                    grp.attrs['thermal_field_T0'] = float(T0_sim)
                    grp.attrs['thermal_field_t0'] = float(thermal_t0)
                    grp.attrs['thermal_field_axis'] = str(thermal_axis)

                # Attributes
                grp.attrs['norm_type_images']   = 'zscore' if use_img_norm else 'none'
                grp.attrs['zscore_eps_images']  = float(eps_img)
                grp.attrs['channel_mean']       = ch_mean
                grp.attrs['channel_std']        = ch_std

                grp.attrs['thermal_gradient_raw']  = float(G_raw)
                grp.attrs['norm_type_thermal']     = 'zscore' if use_tg_norm else 'none'
                grp.attrs['zscore_eps_thermal']    = float(eps_tg)
                grp.attrs['thermal_mean']          = float(tg_mean)
                grp.attrs['thermal_std']           = float(tg_std)

                grp.attrs['norm_type_time']        = 'zscore' if use_time_norm else 'none'
                grp.attrs['zscore_eps_time']       = float(eps_time)
                grp.attrs['time_mean']             = float(time_mean)
                grp.attrs['time_std']              = float(time_std)

                grp.attrs['effective_dt']          = float(eff_dt)
                grp.attrs['dx']                    = float(prm.get('dx', np.nan))
                grp.attrs['W0']                    = float(prm.get('W0', np.nan))
                grp.attrs['physical_grid_spacing'] = float(prm.get('physical_grid_spacing', np.nan))
                grp.attrs['x_min']                 = float(prm.get('x_min', np.nan))
                grp.attrs['x_max']                 = float(prm.get('x_max', np.nan))
                grp.attrs['y_min']                 = float(prm.get('y_min', np.nan))
                grp.attrs['y_max']                 = float(prm.get('y_max', np.nan))
                grp.attrs['pulling_speed']         = float(prm.get('pulling_speed', np.nan))
                grp.attrs['tau0']                  = float(prm.get('tau0', TAU0))
                grp.attrs['x0_dx']                 = float(prm.get('x0_dx', np.nan))
                grp.attrs['thermal_field_norm']    = str(norm.get('thermal_field_norm', 'none'))
                grp.attrs['x_centroid_units']      = 'dx'

                if 'thermal_gradient_series' in grp:
                    old = grp['thermal_gradient_series'].shape[0]
                    grp['thermal_gradient_series'].resize((old + len(ts),)); grp['thermal_gradient_series'][old:] = G_raw
                    grp['thermal_gradient_series_norm'].resize((old + len(ts),)); grp['thermal_gradient_series_norm'][old:] = G_norm_val
                else:
                    grp.create_dataset('thermal_gradient_series',
                                       data=np.full(len(ts), G_raw, dtype=np.float32),
                                       maxshape=(None,), compression='gzip', chunks=(min(4096, len(ts)),))
                    grp.create_dataset('thermal_gradient_series_norm',
                                       data=np.full(len(ts), G_norm_val, dtype=np.float32),
                                       maxshape=(None,), compression='gzip', chunks=(min(4096, len(ts)),))

                # Meta
                meta_entry = {
                    'original_name'  : sim,
                    'group_id'       : gid,
                    'num_timesteps'  : int(len(ts)),
                    'num_pairs'      : int(len(pidx)),
                    'pair_strides'   : sorted(list(set(PAIR_STRIDES))),
                    'physical_params': {k: (float(v) if isinstance(v, (int, float)) else v) for k,v in prm.items()},
                }
                sim_meta[gid] = meta_entry
                sim_manifest[sim] = meta_entry
                h5f.flush()
                logger.info(f"[sim] done {sim}: frames={len(ts)} pairs={len(pidx)}")

    # Persist maps
    with open(map_path, 'w')      as f: json.dump(sim_map,      f, indent=2)
    with open(meta_path,'w')      as f: json.dump(sim_meta,     f, indent=2)
    with open(manifest_path, 'w') as f: json.dump(sim_manifest, f, indent=2)

if __name__ == "__main__":
    main()
