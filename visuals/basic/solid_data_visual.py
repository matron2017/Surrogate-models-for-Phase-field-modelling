"""
solid_data_visual.py — RAW-value visuals for a single phase-field simulation.

- Loads z-scored HDF5 and auto-denormalises to RAW using file/group attrs.
- Fixed viridis colormap for both PNGs and videos (when enabled).
- Phase field channel scaled to [-1, 1] (normalisation only, no palette change).
- Concentration channel uses robust RAW percentiles for vmin/vmax.
- Overlays Euler step and physical time (engineering units or forced unit).
- Saves only every Nth PNG frame (configurable).
- Video export implemented but disabled by default via YAML.

Code comments avoid second-person phrasing.
"""

from __future__ import annotations
import os, sys, json, glob, warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import yaml
import h5py

# Headless-safe Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors  # use modern colormap access via matplotlib.colormaps

# Optional backends
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

try:
    import imageio as iio
    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        pass
    IMGIO_OK = True
except Exception:
    IMGIO_OK = False


# ────────────────────────── utilities ──────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _auto_denormalise_images(imgs: np.ndarray, attrs: Dict[str, object]) -> Tuple[np.ndarray, bool]:
    schema = str(attrs.get("normalization_schema", "")).lower()
    mean = attrs.get("channel_mean")
    std  = attrs.get("channel_std")
    if schema == "zscore" and mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        std  = np.array(std,  dtype=np.float32).reshape(1, -1, 1, 1)
        std  = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32, copy=False)
        return imgs.astype(np.float32) * std + mean, True
    return imgs.astype(np.float32), False

def _robust_min_max(data: np.ndarray, low: float, high: float) -> Tuple[float, float]:
    flat = data.reshape(-1)
    vmin = float(np.nanpercentile(flat, low))
    vmax = float(np.nanpercentile(flat, high))
    if not np.isfinite(vmin): vmin = float(np.nanmin(flat))
    if not np.isfinite(vmax): vmax = float(np.nanmax(flat))
    if vmax <= vmin: vmax = vmin + 1e-6
    return vmin, vmax

def _engi_time(seconds: float) -> str:
    if not np.isfinite(seconds): return "t=NA"
    s = abs(seconds)
    if s >= 1:    return f"t={seconds:,.3f}s"
    if s >= 1e-3: return f"t={seconds/1e-3:.3f}ms"
    if s >= 1e-6: return f"t={seconds/1e-6:.3f}µs"
    if s >= 1e-9: return f"t={seconds/1e-9:.3f}ns"
    if s >= 1e-12:return f"t={seconds/1e-12:.3f}ps"
    return f"t={seconds/1e-15:.3f}fs"

def _format_G(g: Optional[float], sigfigs: int = 3) -> str:
    if g is None or not np.isfinite(g): return "G=NA"
    return "G=" + f"{g:.{sigfigs}e}"

def _save_png(path: str, arr: np.ndarray, vmin: float, vmax: float,
              title: Optional[str], cmap: str, unit_label: str,
              overlay_label: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im = ax.imshow(arr, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    if unit_label: cb.set_label(unit_label)
    if overlay_label:
        ax.text(0.02, 0.02, overlay_label,
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, color="w",
                bbox=dict(facecolor="k", alpha=0.40, pad=2.0, lw=0.0))
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def _viridis_rgb_from_norm(a01: np.ndarray) -> np.ndarray:
    vir = matplotlib.colormaps.get_cmap("viridis")
    return (vir(np.clip(a01, 0, 1))[:, :, :3] * 255.0 + 0.5).astype(np.uint8)

def _make_colorbar_strip(height_px: int, vmin: float, vmax: float, unit: str,
                         width_px: int = 90, n_ticks: int = 5) -> np.ndarray:
    dpi = 100
    fig_h = height_px / dpi
    fig_w = width_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.35, 0.05, 0.25, 0.90])  # x, y, w, h
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=matplotlib.colormaps.get_cmap("viridis"),
        norm=norm,
        orientation="vertical",
    )
    if unit: cb.set_label(unit)
    if n_ticks > 0:
        ticks = np.linspace(vmin, vmax, n_ticks)
        cb.set_ticks(ticks)
    # Render to RGBA buffer for compatibility across Matplotlib builds
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (h, w, 4)
    buf = rgba[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    if buf.shape[0] != height_px:
        if CV2_OK:
            buf = cv2.resize(buf, (buf.shape[1], height_px), interpolation=cv2.INTER_AREA)
        else:
            idx = (np.linspace(0, buf.shape[0]-1, height_px)).astype(int)
            buf = buf[idx]
    return buf

def _overlay_text(rgb: np.ndarray, label: str) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape
    if CV2_OK:
        cv2.rectangle(out, (10, h-42), (min(10 + 520, w-10), h-10), (0, 0, 0), thickness=-1)
        cv2.putText(out, label, (18, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _hstack_with_colorbar(frame_rgb: np.ndarray, cb_rgb: np.ndarray) -> np.ndarray:
    h = frame_rgb.shape[0]
    if cb_rgb.shape[0] != h:
        if CV2_OK:
            cb_rgb = cv2.resize(cb_rgb, (cb_rgb.shape[1], h), interpolation=cv2.INTER_AREA)
        else:
            idx = (np.linspace(0, cb_rgb.shape[0]-1, h)).astype(int)
            cb_rgb = cb_rgb[idx]
    return np.concatenate([frame_rgb, cb_rgb], axis=1)

def _write_video_imageio_rgb(path: str, rgb_frames: List[np.ndarray], fps: int) -> None:
    if not rgb_frames: raise ValueError("No frames for video")
    with iio.get_writer(path, format="ffmpeg", mode="I",
                        fps=int(fps), codec="libx264", pixelformat="yuv420p") as w:
        for frame in rgb_frames:
            w.append_data(frame)

def _write_video_cv2_rgb(path: str, rgb_frames: List[np.ndarray], fps: int) -> None:
    if not CV2_OK: raise RuntimeError("OpenCV not available")
    if not rgb_frames: raise ValueError("No frames for video")
    h, w = rgb_frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h), True)
    for fr in rgb_frames:
        vw.write(fr)
    vw.release()


# ────────────────────────── core pipeline ──────────────────────────

def _select_single_simulation(h5_paths: List[str], group_cfg: str) -> Tuple[str, str]:
    # File selection: first existing file.
    for p in h5_paths:
        if os.path.exists(p):
            chosen_file = p
            break
    else:
        raise FileNotFoundError("No HDF5 files found among provided paths.")
    # Group selection: explicit id or first group containing 'images'.
    with h5py.File(chosen_file, "r") as h5:
        if group_cfg not in ("*", "all", "auto_first"):
            if group_cfg in h5: return chosen_file, group_cfg
            raise KeyError(f"Group {group_cfg!r} not found in {chosen_file}")
        for k, v in h5.items():
            if isinstance(v, h5py.Group) and "images" in v:
                return chosen_file, k
    raise RuntimeError(f"No valid group found in {chosen_file}")

def _process_one(cfg: dict, h5_path: str, group_id: str) -> None:
    out_root   = str(cfg["out_dir"])
    every      = int(cfg.get("every", 1))              # processing stride
    png_cfg    = cfg.get("png", {}) or {}
    png_save   = bool(png_cfg.get("save", True))
    png_stride = int(png_cfg.get("every_n", 30))       # save only every Nth processed frame
    size       = int(cfg.get("size", 512))
    cmap_name  = str(cfg.get("cmap", "viridis"))
    units_map  = cfg.get("units", {}) or {}
    use_phys   = bool(cfg.get("use_physical_time", True))
    time_cfg   = cfg.get("time", {}) or {}
    time_unit_override = str(time_cfg.get("unit", "auto"))  # "auto"|"s"|"ms"|"µs"|"ns"|"ps"|"fs"
    time_axis_label    = str(time_cfg.get("axis_label", "time (s)"))
    rob        = cfg.get("robust_percentiles", {}) or {"low": 1.0, "high": 99.0}
    v_cfg      = cfg.get("video", {}) or {}
    video_enable = bool(v_cfg.get("enable", False))    # disabled by default
    fps_base   = int(v_cfg.get("fps_base", 12))
    cb_width   = int(v_cfg.get("colorbar_width_px", 90))
    cb_ticks   = int(v_cfg.get("colorbar_ticks", 5))

    phase_cfg  = cfg.get("phase_field", {}) or {}
    phase_ch   = int(phase_cfg.get("channel", -1))
    phase_min  = float(phase_cfg.get("vmin", -1.0))
    phase_max  = float(phase_cfg.get("vmax",  1.0))

    # Load data
    with h5py.File(h5_path, "r") as h5:
        assert group_id in h5, f"Group {group_id!r} not found in {h5_path}"
        grp = h5[group_id]
        imgs_z = grp[str(cfg.get("io", {}).get("read_dataset", "images"))][...]  # (T,C,H,W)
        times  = grp[str(cfg.get("io", {}).get("times_dataset", "times"))][...]  # (T,)
        attrs = dict(h5.attrs)
        attrs.update({k: grp.attrs[k] for k in grp.attrs.keys()})

        G_raw  = float(grp.attrs.get("thermal_gradient_raw", np.nan)) if "thermal_gradient_raw" in grp.attrs else np.nan

    # Output directory: single simulation, no G in path
    out_dir = os.path.join(out_root, group_id)
    _ensure_dir(out_dir)
    frame_dir = os.path.join(out_dir, "frames"); _ensure_dir(frame_dir)

    # Denormalise to RAW
    imgs_raw, _ = _auto_denormalise_images(imgs_z, attrs)
    T, C, H, W = imgs_raw.shape

    # Frame selection for processing
    stride = max(1, int(every))
    sel_idx = np.arange(0, T, stride, dtype=int)

    # Time axes
    eff_dt = float(attrs.get("effective_dt", np.nan))
    unit_scale = {"s":1.0,"ms":1e-3,"µs":1e-6,"ns":1e-9,"ps":1e-12,"fs":1e-15}
    def _label_time(t_seconds: float) -> str:
        if time_unit_override != "auto" and time_unit_override in unit_scale:
            return f"t={t_seconds/unit_scale[time_unit_override]:.3f}{time_unit_override}"
        return _engi_time(t_seconds)

    # Colour scales per channel
    low = float(rob.get("low", 1.0)); high = float(rob.get("high", 99.0))
    vmins: List[float]; vmaxs: List[float]
    vmins, vmaxs = [], []
    for c in range(C):
        vmin, vmax = _robust_min_max(imgs_raw[:, c], low, high)
        vmins.append(vmin); vmaxs.append(vmax)
    if 0 <= phase_ch < C:
        vmins[phase_ch] = phase_min
        vmaxs[phase_ch] = phase_max

    # Constant per-channel colourbar images for video
    cb_per_ch: List[np.ndarray] = []
    for ch in range(C):
        unit = str(units_map.get(f"Field_{ch}", ""))
        cb = _make_colorbar_strip(height_px=size, vmin=vmins[ch], vmax=vmaxs[ch],
                                  unit=unit, width_px=cb_width, n_ticks=cb_ticks)
        cb_per_ch.append(cb)

    # Prepare per-channel video frames (kept in memory if enabled later)
    rgb_frames_per_ch: List[List[np.ndarray]] = [[] for _ in range(C)]

    # Iterate frames
    for j, k in enumerate(sel_idx):
        frame = imgs_raw[k]  # (C,H,W)
        # Euler step
        step = int(times[k]) if times.ndim == 1 else int(k)
        # Physical time label
        if use_phys and np.isfinite(eff_dt) and eff_dt > 0:
            t_label = _label_time(float(times[k]) * eff_dt)
        else:
            t_label = "t=NA"

        # Per channel rendering
        for ch in range(C):
            arr = frame[ch].astype(np.float32, copy=False)
            # Normalise for video mapping
            norm01 = (np.clip((arr - vmins[ch]) / (vmaxs[ch] - vmins[ch] + 1e-12), 0, 1)).astype(np.float32)
            rgb = _viridis_rgb_from_norm(norm01)
            label = f"step={step}   {t_label}"
            rgb = _overlay_text(rgb, label)
            rgb_with_cb = _hstack_with_colorbar(rgb, cb_per_ch[ch])
            if video_enable:
                rgb_frames_per_ch[ch].append(rgb_with_cb)

            # Save only every Nth processed frame as PNG
            if png_save and (j % max(1, png_stride) == 0):
                unit  = str(units_map.get(f"Field_{ch}", ""))
                title = f"Field_{ch} — step={step} | {t_label}"
                _save_png(
                    os.path.join(frame_dir, f"frame_{k:05d}_f{ch}.png"),
                    arr, vmins[ch], vmaxs[ch], title, cmap_name, unit,
                    overlay_label=label
                )

    # Video export (currently off by default)
    if video_enable:
        fps = max(2, int(fps_base / max(1, stride)))
        for ch in range(C):
            out_mp4 = os.path.join(out_dir, f"evolution_field{ch}.mp4")
            try:
                if not IMGIO_OK:
                    raise RuntimeError("imageio/ffmpeg not available")
                _write_video_imageio_rgb(out_mp4, rgb_frames_per_ch[ch], fps)
            except Exception as e:
                if CV2_OK:
                    warnings.warn(f"imageio unavailable ({e}); falling back to OpenCV.")
                    _write_video_cv2_rgb(out_mp4, rgb_frames_per_ch[ch], fps)
                else:
                    warnings.warn(f"Video export skipped for field {ch}: {e}")

    # Minimal manifest
    meta = {
        "h5_path": h5_path,
        "group": group_id,
        "T": int(T),
        "processed_stride": int(stride),
        "png_stride": int(png_stride),
        "size": int(size),
        "vmins": [float(x) for x in vmins],
        "vmaxs": [float(x) for x in vmaxs],
        "cmap": "viridis",
        "phase_field_channel": (int(phase_ch) if 0 <= phase_ch < C else None),
        "thermal_gradient_raw": (float(G_raw) if np.isfinite(G_raw) else None),
        "time_axis": (time_axis_label),
        "effective_dt": (float(eff_dt) if np.isfinite(eff_dt) else None),
        "video_enabled": bool(video_enable),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ────────────────────────── entry point ──────────────────────────

def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/project_2008261/rapid_solidification/configs/visuals/rapid_solid_visuals.yaml"
    cfg = _load_yaml(yaml_path)

    # Inputs: prefer h5_path, else first existing from h5_paths.
    if "h5_path" in cfg and cfg["h5_path"]:
        paths = [cfg["h5_path"]]
    else:
        paths_raw = cfg.get("h5_paths", [])
        paths = []
        for p in (paths_raw if isinstance(paths_raw, list) else [paths_raw]):
            if isinstance(p, str) and ("*" in p or "?" in p or "[" in p):
                paths.extend(sorted(glob.glob(p)))
            else:
                paths.append(p)

    group_sel = str(cfg.get("group", "auto_first"))

    # Single simulation selection
    h5_file, group_id = _select_single_simulation(paths, group_sel)

    # Process
    _process_one(cfg, h5_file, group_id)


if __name__ == "__main__":
    main()
