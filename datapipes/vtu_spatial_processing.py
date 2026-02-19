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

def _grid_shape(bounds: Tuple[float, float, float, float], settings: Dict[str, int]) -> Tuple[int, int]:
    xmin, xmax, ymin, ymax = bounds
    x_range, y_range = xmax - xmin, ymax - ymin
    max_range = max(x_range, y_range)
    min_r, max_r = settings['min_res'], settings['max_res']
    if max_range == 0:
        nx = ny = min_r
    else:
        nx = int(max(min_r, max_r * (x_range / max_range)))
        ny = int(max(min_r, max_r * (y_range / max_range)))
    return nx, ny

def compute_support_bins(coords: np.ndarray,
                         settings: Dict[str, int],
                         bounds: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    x_range, y_range = xmax - xmin, ymax - ymin
    nx, ny = _grid_shape(bounds, settings)
    try:
        cp_coords = cp.asarray(coords)
        xi = cp.floor((cp_coords[:, 0] - xmin) / (x_range or 1) * nx).astype(cp.int32)
        yi = cp.floor((cp_coords[:, 1] - ymin) / (y_range or 1) * ny).astype(cp.int32)
        xi, yi = cp.clip(xi, 0, nx - 1), cp.clip(yi, 0, ny - 1)
        idx = yi * nx + xi
        counts = cp.bincount(idx, minlength=nx * ny)
        sums_x = cp.bincount(idx, weights=cp_coords[:, 0], minlength=nx * ny)
        mean_x = cp.where(counts > 0, sums_x / counts, cp.nan).reshape(ny, nx)
        return cp.asnumpy(counts).reshape(ny, nx), cp.asnumpy(mean_x).astype(np.float32, copy=False)
    except Exception:
        coords_np = np.asarray(coords)
        xi = np.floor((coords_np[:, 0] - xmin) / (x_range or 1) * nx).astype(np.int64)
        yi = np.floor((coords_np[:, 1] - ymin) / (y_range or 1) * ny).astype(np.int64)
        xi = np.clip(xi, 0, nx - 1)
        yi = np.clip(yi, 0, ny - 1)
        idx = yi * nx + xi
        counts = np.bincount(idx, minlength=nx * ny)
        sums_x = np.bincount(idx, weights=coords_np[:, 0], minlength=nx * ny)
        mean_x = np.where(counts > 0, sums_x / counts, np.nan).reshape(ny, nx)
        return counts.reshape(ny, nx), mean_x.astype(np.float32, copy=False)

def convert_nodes_to_grid(coords: np.ndarray,
                          values: np.ndarray,
                          settings: Dict[str, int],
                          bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    x_range, y_range = xmax - xmin, ymax - ymin
    nx, ny = _grid_shape(bounds, settings)
    try:
        cp_coords = cp.asarray(coords)
        cp_vals = cp.asarray(values if values.dtype != np.float64 else values.astype(np.float32, copy=False))
        xi = cp.floor((cp_coords[:, 0] - xmin) / (x_range or 1) * nx).astype(cp.int32)
        yi = cp.floor((cp_coords[:, 1] - ymin) / (y_range or 1) * ny).astype(cp.int32)
        xi, yi = cp.clip(xi, 0, nx - 1), cp.clip(yi, 0, ny - 1)
        idx = yi * nx + xi
        sums = cp.bincount(idx, weights=cp_vals, minlength=nx * ny)
        counts = cp.bincount(idx, minlength=nx * ny)
        grid = cp.where(counts > 0, sums / counts, cp.nan).reshape(ny, nx)
        return cp.asnumpy(grid)
    except Exception:
        coords_np = np.asarray(coords)
        vals_np = np.asarray(values if values.dtype != np.float64 else values.astype(np.float32, copy=False))
        xi = np.floor((coords_np[:, 0] - xmin) / (x_range or 1) * nx).astype(np.int64)
        yi = np.floor((coords_np[:, 1] - ymin) / (y_range or 1) * ny).astype(np.int64)
        xi = np.clip(xi, 0, nx - 1)
        yi = np.clip(yi, 0, ny - 1)
        idx = yi * nx + xi
        sums = np.bincount(idx, weights=vals_np, minlength=nx * ny)
        counts = np.bincount(idx, minlength=nx * ny)
        grid = np.where(counts > 0, sums / counts, np.nan).reshape(ny, nx)
        return grid.astype(np.float32, copy=False)

# ── Thermal gradient parsing ──────────────────────────────────────────────────
_name_G = re.compile(r'\bG(?P<base>\d+(?:\.\d+)?)e(?P<exp>[+-]?\d+)\b', re.I)
_log_G    = re.compile(r'\bthermal[_\s-]*gradient\s*=\s*([0-9.eE+-]+)', re.I)
_dat_G    = re.compile(r'\bthermal[_\s-]*gradient\s*[:=]\s*([0-9.eE+-]+)', re.I)
_log_V    = re.compile(r'\bpulling[_\s-]*speed\s*[:=]\s*([0-9.eE+-]+)', re.I)
_dat_V    = re.compile(r'\bpulling[_\s-]*speed\s*[:=]\s*([0-9.eE+-]+)', re.I)
_log_tau0 = re.compile(r'\btau0\s*[:=]\s*([0-9.eE+-]+)', re.I)
_dat_tau0 = re.compile(r'\btau0\s*[:=]\s*([0-9.eE+-]+)', re.I)
_log_dt   = re.compile(r'\bdt\s*=\s*([0-9.eE+-]+)\s*tau0\b', re.I)
_log_x0dx = re.compile(r'\b(?:x0[_\s-]*dx|x_directional_origo)\s*[:=]\s*([0-9.eE+-]+)\s*(?:dx)?', re.I)
_dat_x0dx = re.compile(r'\b(?:x0[_\s-]*dx|x_directional_origo)\s*[:=]\s*([0-9.eE+-]+)\s*(?:dx)?', re.I)

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

def _parse_scalar_from_files(sim_dir: str, dat_re: re.Pattern, log_re: re.Pattern) -> Optional[float]:
    dat = os.path.join(sim_dir, 'Simul_Params.dat')
    try:
        if os.path.isfile(dat):
            txt = open(dat, 'r', errors='ignore').read()
            m = dat_re.search(txt)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    logp = _latest_runlog(sim_dir)
    if logp:
        try:
            txt = open(logp, 'r', errors='ignore').read()
            m = log_re.search(txt)
            if m:
                return float(m.group(1))
        except Exception:
            pass
    return None

def _parse_dt_from_log(sim_dir: str) -> Optional[float]:
    logp = _latest_runlog(sim_dir)
    if logp:
        try:
            txt = open(logp, 'r', errors='ignore').read()
            m = _log_dt.search(txt)
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
        v = _parse_scalar_from_files(sim_dir, _dat_V, _log_V)
        if v is not None:
            params['pulling_speed'] = float(v)
        tau0 = _parse_scalar_from_files(sim_dir, _dat_tau0, _log_tau0)
        if tau0 is not None:
            params['tau0'] = float(tau0)
        x0dx = _parse_scalar_from_files(sim_dir, _dat_x0dx, _log_x0dx)
        if x0dx is not None:
            params['x0_dx'] = float(x0dx)
        return params
    lines = open(dat_file).read().splitlines()
    try:
        if len(lines) > 1:
            c_o, m0, ke = lines[0].split()[:3]
            params['c_o'] = float(c_o)
            params['m0'] = float(m0)
            params['ke'] = float(ke)
            params['Tm'] = float(lines[1].split()[0])
        params['W0'] = float(lines[2].split()[0])
        a, b = lines[6].split()[:2]
        params['dx'], params['dt'] = float(a), float(b)
    except Exception:
        pass
    params['thermal_gradient'] = extract_thermal_gradient(sim_dir, os.path.basename(sim_dir))
    v = _parse_scalar_from_files(sim_dir, _dat_V, _log_V)
    if v is not None:
        params['pulling_speed'] = float(v)
    tau0 = _parse_scalar_from_files(sim_dir, _dat_tau0, _log_tau0)
    if tau0 is not None:
        params['tau0'] = float(tau0)
    dt_log = _parse_dt_from_log(sim_dir)
    if dt_log is not None:
        params['dt'] = float(dt_log)
    x0dx = _parse_scalar_from_files(sim_dir, _dat_x0dx, _log_x0dx)
    if x0dx is not None:
        params['x0_dx'] = float(x0dx)
    if 'Tm' in params and 'm0' in params and 'c_o' in params:
        params['Tl'] = params['Tm'] - params['m0'] * params['c_o']
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
            try:
                support_counts, support_mean_x = compute_support_bins(coords, grid_settings, domain)
            except Exception:
                logger.exception(f"[worker] support binning failed gpu={gpu_id} idx={i} file={f}")
                nx, ny = _grid_shape(domain, grid_settings)
                support_counts = np.zeros((ny, nx), dtype=np.int32)
                support_mean_x = np.full((ny, nx), np.nan, dtype=np.float32)
            ch_imgs = []
            for fld in field_list:
                arr = mesh.GetPointData().GetArray(fld)
                vals = vtk_array_to_1d_scalar(arr, fld, sim=os.path.basename(os.path.dirname(f))) if arr is not None \
                       else np.full(coords.shape[0], np.nan, dtype=np.float32)
                if vals.size == 0:
                    vals = np.full(coords.shape[0], np.nan, dtype=np.float32)
                g = convert_nodes_to_grid(coords, vals, grid_settings, domain)
                ch_imgs.append(fill_nan_nearest(g))
            result_buf.append((i, np.stack(ch_imgs, 0), support_counts, support_mean_x))
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
      support_counts: (T, H, W) int32, per-bin point counts
      support_mean_x: (T, H, W) float32, per-bin mean x (dx units)
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

    def _process_frame(i: int, f: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            try:
                support_counts, support_mean_x = compute_support_bins(coords, grid_settings, domain)
            except Exception:
                logger.exception(f"[seq] support binning failed sim={sim} file={f}")
                nx, ny = _grid_shape(domain, grid_settings)
                support_counts = np.zeros((ny, nx), dtype=np.int32)
                support_mean_x = np.full((ny, nx), np.nan, dtype=np.float32)
            ch_imgs = []
            for fld in field_list:
                arr  = mesh.GetPointData().GetArray(fld)
                vals = vtk_array_to_1d_scalar(arr, fld, sim=sim) if arr is not None \
                       else np.full(coords.shape[0], np.nan, dtype=np.float32)
                if vals.size == 0:
                    vals = np.full(coords.shape[0], np.nan, dtype=np.float32)
                g = convert_nodes_to_grid(coords, vals, grid_settings, domain)
                ch_imgs.append(fill_nan_nearest(g))
            return np.stack(ch_imgs, 0), support_counts, support_mean_x
        except Exception:
            logger.exception(f"[seq] exception while parsing frame i={i} sim={sim} file={f}")
            return None

    failures: List[str] = []
    seq: np.ndarray
    ts: np.ndarray

    use_multi_gpu = ngpu > 1 and len(files) > 1
    if use_multi_gpu:
        manager = mp.Manager()
        result_buf = manager.list()
        idx_buf = manager.list()
        fail_buf = manager.list()

        splits: List[List[int]] = [[] for _ in range(ngpu)]
        for idx in range(len(files)):
            splits[idx % ngpu].append(idx)

        workers = []
        for gpu_id, chunk in enumerate(splits):
            if not chunk:
                continue
            p = mp.Process(
                target=_worker_frames,
                args=(gpu_id, chunk, files, times, domain,
                      field_list, grid_settings,
                      result_buf, idx_buf, fail_buf,
                      log_first_n, log_every, use_precheck),
                daemon=True,
            )
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        failures.extend(list(fail_buf))
        if not idx_buf:
            raise RuntimeError(f"All frames failed in {sim}")

        # Order results by original frame index
        result_map = {idx: (arr, cnt, mx) for idx, arr, cnt, mx in result_buf}
        ordered_indices = sorted(result_map.keys())
        seq = np.stack([result_map[i][0] for i in ordered_indices], axis=0).astype(np.float32, copy=False)
        support_counts = np.stack([result_map[i][1] for i in ordered_indices], axis=0).astype(np.int32, copy=False)
        support_mean_x = np.stack([result_map[i][2] for i in ordered_indices], axis=0).astype(np.float32, copy=False)
        ts = np.array([times[i] for i in ordered_indices], dtype=int)

    else:
        seq_list, ts_list = [], []
        support_counts_list, support_mean_x_list = [], []
        for i, f in enumerate(files):
            if shutdown:
                break
            img = _process_frame(i, f)
            if img is not None:
                seq_list.append(img[0])
                support_counts_list.append(img[1])
                support_mean_x_list.append(img[2])
                ts_list.append(times[i])
            else:
                failures.append(f)
        if not seq_list:
            raise RuntimeError(f"All frames failed in {sim}")
        seq = np.stack(seq_list, 0).astype(np.float32, copy=False)
        support_counts = np.stack(support_counts_list, 0).astype(np.int32, copy=False)
        support_mean_x = np.stack(support_mean_x_list, 0).astype(np.float32, copy=False)
        ts = np.array(ts_list, dtype=int)

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
    prm['x_min'], prm['x_max'], prm['y_min'], prm['y_max'] = domain
    prm['effective_dt']          = prm.get('dt', 1.0) * 1.0  # tau0 applied in caller if needed
    prm['physical_grid_spacing'] = prm.get('dx', 1.0) * prm.get('W0', 1.0)

    return seq, ts, pairs_idx, pairs_time, pair_stride, prm, failures, support_counts, support_mean_x
