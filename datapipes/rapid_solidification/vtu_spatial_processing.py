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
        return params
    lines = open(dat_file).read().splitlines()
    try:
        params['W0'] = float(lines[2].split()[0])
        a, b = lines[6].split()[:2]
        params['dx'], params['dt'] = float(a), float(b)
    except Exception:
        pass
    params['thermal_gradient'] = extract_thermal_gradient(sim_dir, os.path.basename(sim_dir))
    return params

def vtu_is_plausible(path: str, min_bytes: int = 50_000) -> bool:
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, 'rb') as fh:
            try:
                fh.seek(-4096, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read()
        return b'</VTKFile>' in tail
    except Exception:
        return False

# ── GPU worker ────────────────────────────────────────────────────────────────

def _worker_frames(gpu_id, indices, files, times, domain,
                   field_list, grid_settings,
                   result_buf, idx_buf, fail_buf,
                   log_first_n: int, log_every: int, use_precheck: bool):
    import cupy as _cp
    _cp.cuda.Device(gpu_id).use()
    rdr = vtk.vtkXMLUnstructuredGridReader()
    total = len(indices)
    for local_k, i in enumerate(indices):
        if shutdown:
            break
        f = files[i]
        size_b = os.path.getsize(f) if os.path.exists(f) else None
        pre_ok = (vtu_is_plausible(f) if use_precheck else None)
        if should_log_frame(local_k, total, log_first_n, log_every):
            log_frame_read("worker", os.path.basename(os.path.dirname(os.path.dirname(f))),
                           local_k, total, f, times[i], size_b, pre_ok, level=logging.DEBUG)
        if use_precheck and not pre_ok:
            fail_buf.append(f); continue
        try:
            rdr.SetFileName(f); rdr.Update()
            mesh = rdr.GetOutput()
            if mesh is None or mesh.GetPoints is None:
                raise RuntimeError("Empty mesh or no points")
            pts    = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
            coords = pts[:, :2]
            ch_imgs = []
            for fld in field_list:
                arr = mesh.GetPointData().GetArray(fld)
                vals = vtk_array_to_1d_scalar(arr, fld, sim=os.path.basename(os.path.dirname(f))) if arr is not None \
                       else np.full(coords.shape[0], np.nan, dtype=np.float32)
                if vals.size == 0:
                    vals = np.full(coords.shape[0], np.nan, dtype=np.float32)
                g = convert_nodes_to_grid(coords, vals, grid_settings, domain)
                ch_imgs.append(fill_nan_nearest(g))
            result_buf.append((i, np.stack(ch_imgs, 0)))
            idx_buf.append((i, times[i]))
        except Exception:
            logger.exception(f"[worker] exception on gpu={gpu_id} idx={i} file={f}")
            fail_buf.append(f)

# ── Core loader ───────────────────────────────────────────────────────────────

def load_full(sim: str, base: str, ngpu: int,
              log_first_n: int, log_every: int, use_precheck: bool,
              field_list: List[str], grid_settings: Dict[str,int],
              pair_strides: List[int],
              frame_limit: Optional[int] = None):
    """
    Returns:
      seq:        (T, C, H, W) float32, no normalisation
      ts:         (T,) int
      pidx:       (P, 2) int64
      ptime:      (P, 2) int64
      pstride:    (P,) int32
      prm:        dict with dt, dx, W0, thermal_gradient, effective_dt, physical_grid_spacing
      failures:   list[str]
    """
    path      = os.path.join(base, sim)
    raw_files = glob.glob(os.path.join(path, 'output-r*.vtu'))
    files     = sorted(raw_files, key=extract_time_increment)
    if not files:
        raise RuntimeError(f"No VTU in {sim}")

    if frame_limit is not None and frame_limit > 0:
        files = files[:frame_limit]

    reader = vtk.vtkXMLUnstructuredGridReader()
    domain = None
    probe_file = None
    for f in files:
        if use_precheck and not vtu_is_plausible(f):
            continue
        reader.SetFileName(f); reader.Update()
        mesh = reader.GetOutput()
        if mesh and mesh.GetPoints():
            pts = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
            domain = (float(pts[:,0].min()), float(pts[:,0].max()),
                      float(pts[:,1].min()), float(pts[:,1].max()))
            probe_file = f
            break
    if domain is None:
        raise RuntimeError(f"Empty geometry {sim}")

    try:
        reader.SetFileName(probe_file or files[0]); reader.Update()
        _mesh0 = reader.GetOutput()
        if _mesh0 and _mesh0.GetPointData():
            names = []
            for j in range(_mesh0.GetPointData().GetNumberOfArrays()):
                arr = _mesh0.GetPointData().GetArray(j)
                if arr is not None:
                    names.append(arr.GetName() or f"Array_{j}")
            missing = [f for f in field_list if f not in names]
            if missing:
                logger.warning(f"[fields] sim={sim} requested={field_list} missing={missing} (NaN→filled)")
    except Exception:
        logger.exception(f"[fields] sim={sim} failed to probe point arrays")

    times_all = [extract_time_increment(f) for f in files]
    tmp = [(f, t) for f, t in zip(files, times_all) if t is not None]
    if not tmp:
        raise RuntimeError(f"No valid time indices found in {sim}")
    files, times = zip(*tmp)
    files, times = list(files), list(times)
    n_total = len(files)

    def _process_frame(i: int, f: str) -> Optional[np.ndarray]:
        size_b = os.path.getsize(f) if os.path.exists(f) else None
        pre_ok = (vtu_is_plausible(f) if use_precheck else None)
        if should_log_frame(i, n_total, log_first_n, log_every):
            log_frame_read("seq", sim, i, n_total, f, times[i], size_b, pre_ok)
        if use_precheck and not pre_ok:
            return None
        try:
            reader.SetFileName(f); reader.Update()
            mesh = reader.GetOutput()
            if mesh is None or mesh.GetPoints() is None:
                return None
            pts    = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
            coords = pts[:, :2]
            ch_imgs = []
            for fld in field_list:
                arr  = mesh.GetPointData().GetArray(fld)
                vals = vtk_array_to_1d_scalar(arr, fld, sim=sim) if arr is not None \
                       else np.full(coords.shape[0], np.nan, dtype=np.float32)
                if vals.size == 0:
                    vals = np.full(coords.shape[0], np.nan, dtype=np.float32)
                g = convert_nodes_to_grid(coords, vals, grid_settings, domain)
                ch_imgs.append(fill_nan_nearest(g))
            return np.stack(ch_imgs, 0)
        except Exception:
            logger.exception(f"[seq] exception while parsing frame i={i} sim={sim} file={f}")
            return None

    failures: List[str] = []
    # Single-GPU path
    seq_list, ts_list = [], []
    for i, f in enumerate(files):
        if shutdown:
            break
        img = _process_frame(i, f)
        if img is not None:
            seq_list.append(img); ts_list.append(times[i])
        else:
            failures.append(f)
    if not seq_list:
        raise RuntimeError(f"All frames failed in {sim}")
    seq = np.stack(seq_list, 0).astype(np.float32, copy=False)
    ts  = np.array(ts_list, dtype=int)

    # Build pairs
    strides_clean = sorted(set(int(s) for s in (pair_strides or [1]) if int(s) > 0)) or [1]
    max_stride = max(strides_clean)
    if len(ts) - max_stride <= 0:
        raise RuntimeError(f"Insufficient frames after filtering in {sim}")

    pairs_idx_list, stride_tag_list = [], []
    for s in strides_clean:
        n = len(ts) - s
        if n <= 0:
            continue
        i0 = np.arange(n, dtype=np.int64)
        pairs_idx_list.append(np.stack([i0, i0 + s], 1))
        stride_tag_list.append(np.full(n, s, dtype=np.int32))
    pairs_idx   = np.concatenate(pairs_idx_list, axis=0)
    pair_stride = np.concatenate(stride_tag_list, axis=0)
    pairs_time  = ts[pairs_idx]

    prm        = parse_simul_params(path)
    prm['effective_dt']          = prm.get('dt', 1.0) * 1.0  # tau0 applied in caller if needed
    prm['physical_grid_spacing'] = prm.get('dx', 1.0) * prm.get('W0', 1.0)

    return seq, ts, pairs_idx, pairs_time, pair_stride, prm, failures
