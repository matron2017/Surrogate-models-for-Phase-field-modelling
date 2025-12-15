#!/usr/bin/env python3
"""
spatial_processing.py

Pure spatial stage:
- Reads VTU frames
- Extracts requested point-data fields
- Projects nodes → fixed grids (GPU, NaN-fill by nearest)
- Builds multi-stride index pairs
- Parses minimal physical parameters
- Returns arrays; no normalisation and no cross-split statistics

The logic mirrors the original spatial path.
"""

from __future__ import annotations
import os, re, glob, logging, signal
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import multiprocessing as mp

# Fresh CUDA contexts in child processes
mp.set_start_method('spawn', force=True)

import cupy as cp
import cupyx.scipy.ndimage as cnd
import vtk
from vtk.util import numpy_support

# Silence VTK warnings
vtk.vtkObject.GlobalWarningDisplayOff()

logger = logging.getLogger("spatial_processing")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

shutdown = False
def _handle_signal(signum, frame):
    global shutdown
    shutdown = True
    logger.info(f"Received signal {signum!r}, will shut down after current sim.")
signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_log_frame(i: int, total: int, first_n: int, every: int) -> bool:
    if i < first_n:
        return True
    if every > 0 and (i % every == 0 or i == total - 1):
        return True
    return False

def mesh_summary(mesh: vtk.vtkDataSet) -> Dict[str, Any]:
    pd = mesh.GetPointData()
    n_arrays = pd.GetNumberOfArrays() if pd is not None else 0
    arrays = []
    for j in range(n_arrays):
        arr = pd.GetArray(j)
        if arr is None:
            continue
        name = arr.GetName() or f"Array_{j}"
        arrays.append({"name": name, "components": arr.GetNumberOfComponents(), "tuples": arr.GetNumberOfTuples()})
    b = mesh.GetBounds() if hasattr(mesh, "GetBounds") else None
    return {
        "n_points": mesh.GetNumberOfPoints(),
        "n_cells":  mesh.GetNumberOfCells(),
        "bounds":   tuple(float(x) for x in b) if b is not None else None,
        "point_arrays": arrays,
    }

def log_frame_read(phase: str, sim: str, idx: int, total: int, fpath: str,
                   t_idx: Optional[int], size_bytes: Optional[int],
                   precheck: Optional[bool], level=logging.INFO) -> None:
    base = os.path.basename(fpath)
    sz = f"{(size_bytes or 0)/1024:.1f} KiB" if size_bytes is not None else "n/a"
    logger.log(level, f"[{phase}] sim={sim} frame={idx+1}/{total} file={base} size={sz} t={t_idx} precheck={precheck}")

def extract_time_increment(fname: str) -> Optional[int]:
    m = re.search(r'output-r(\d+)\.vtu', os.path.basename(fname))
    return int(m.group(1)) if m else None

def fill_nan_nearest(grid_np: np.ndarray) -> np.ndarray:
    """Nearest-neighbour NaN fill. GPU first; CPU fallback."""
    try:
        grid_cp = cp.asarray(grid_np)
        mask = cp.isnan(grid_cp)
        if bool(mask.all()):
            return np.zeros_like(grid_np)
        if not bool(mask.any()):
            return grid_np
        inds = cnd.distance_transform_edt(mask, return_distances=False, return_indices=True)
        if isinstance(inds, tuple):
            inds = inds[1]
        filled = grid_cp[tuple(inds)]
        return cp.asnumpy(filled)
    except Exception:
        try:
            from scipy import ndimage
            mask_cpu = np.isnan(grid_np)
            if not mask_cpu.any():
                return grid_np
            if mask_cpu.all():
                return np.zeros_like(grid_np)
            inds_cpu = ndimage.distance_transform_edt(mask_cpu, return_distances=False, return_indices=True)
            return grid_np[tuple(inds_cpu)]
        except Exception:
            return np.nan_to_num(grid_np, copy=False)

def vtk_array_to_1d_scalar(arr: vtk.vtkDataArray, want_name: str, sim: str, comp: int = 0) -> np.ndarray:
    """Convert VTK point-data array to 1-D float32 vector."""
    if arr is None:
        return np.array([], dtype=np.float32)
    np_arr = numpy_support.vtk_to_numpy(arr)
    if np_arr.ndim == 2 and np_arr.shape[1] > 1:
        if comp >= np_arr.shape[1]:
            comp = 0
        np_arr = np_arr[:, comp]
    np_arr = np.asarray(np_arr).astype(np.float32, copy=False)
    if np_arr.ndim != 1:
        np_arr = np_arr.reshape(-1)
    return np_arr

def convert_nodes_to_grid(coords: np.ndarray,
                          values: np.ndarray,
                          settings: Dict[str, int],
                          bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    x_range, y_range = xmax - xmin, ymax - ymin
    max_range = max(x_range, y_range)
    min_r, max_r = settings['min_res'], settings['max_res']
    if max_range == 0:
        nx = ny = min_r
    else:
        nx = int(max(min_r, max_r * (x_range / max_range)))
        ny = int(max(min_r, max_r * (y_range / max_range)))
    cp_coords = cp.asarray(coords)
    cp_vals   = cp.asarray(values if values.dtype != np.float64 else values.astype(np.float32, copy=False))
    xi = cp.floor((cp_coords[:, 0] - xmin) / (x_range or 1) * nx).astype(cp.int32)
    yi = cp.floor((cp_coords[:, 1] - ymin) / (y_range or 1) * ny).astype(cp.int32)
    xi, yi = cp.clip(xi, 0, nx - 1), cp.clip(yi, 0, ny - 1)
    idx = yi * nx + xi
    sums   = cp.bincount(idx, weights=cp_vals, minlength=nx * ny)
    counts = cp.bincount(idx, minlength=nx * ny)
    grid   = cp.where(counts > 0, sums / counts, cp.nan).reshape(ny, nx)
    return cp.asnumpy(grid)

# ── Thermal gradient parsing ──────────────────────────────────────────────────
_name_G = re.compile(r'\bG(?P<base>\d+(?:\.\d+)?)e(?P<exp>[+-]?\d+)\b', re.I)
_log_G  = re.compile(r'\bthermal[_\s-]*gradient\s*=\s*([0-9.eE+-]+)', re.I)
_dat_G  = re.compile(r'\bthermal[_\s-]*gradient\s*[:=]\s*([0-9.eE+-]+)', re.I)

def _parse_G_from_name(sim_name: str) -> Optional[float]:
    m = _name_G.search(sim_name)
    if not m: return None
    base = float(m.group('base')); exp = int(m.group('exp'))
    return base * (10.0 ** exp)

def _latest_runlog(sim_dir: str) -> Optional[str]:
    cands = glob.glob(os.path.join(sim_dir, 'log.run_*'))
    if not cands:
        return None
    cands.sort(key=lambda p: (os.path.getsize(p), os.path.getmtime(p)))
    return cands[-1]

def _parse_G_from_files(sim_dir: str) -> Optional[float]:
    dat = os.path.join(sim_dir, 'Simul_Params.dat')
    try:
        if os.path.isfile(dat):
            txt = open(dat, 'r', errors='ignore').read()
            m = _dat_G.search(txt)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    logp = _latest_runlog(sim_dir)
    if logp:
        try:
            txt = open(logp, 'r', errors='ignore').read()
            m = _log_G.search(txt)
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return None

def extract_thermal_gradient(sim_dir: str, sim_name: str) -> float:
    v = _parse_G_from_files(sim_dir)
    if v is not None:
        return float(v)
    v = _parse_G_from_name(sim_name)
    return float(v) if v is not None else 0.0

def parse_simul_params(sim_dir: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    dat_file = os.path.join(sim_dir, 'Simul_Params.dat')
    if not os.path.isfile(dat_file):
        params['thermal_gradient'] = extract_thermal_gradient(sim_dir, os.path.basename(sim_dir))
