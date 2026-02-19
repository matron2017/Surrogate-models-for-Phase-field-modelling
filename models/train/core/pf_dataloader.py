from __future__ import annotations

import h5py
import torch
import threading
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Sequence, List, Tuple, Dict, Optional

class PFPairDataset(Dataset):
    """
    Paired-frame dataset with thermal-field support for PDE-to-PDE prediction.
    Input:  (C_in, H, W)
    Target: (C_out, H, W)
    Returns per item:
      {
        "input":  (C_in, H, W) float32,
        "target": (C_out, H, W) float32,
        "gid":    str,
        "pair_index": int,
        "weight": (C_out, H, W) float32  # optional, wavelet-based weights
      }
    """
    def __init__(
        self,
        h5_path: str,
        input_channels: Sequence[int] | None,    # indices into data_key channel dimension; None means all
        target_channels: Sequence[int] | None,   # indices into data_key
        limit_per_group: Optional[int] = None,
        max_items: Optional[int | float] = None,
        weight_h5: Optional[str] = None,  # path to side-car file with wavelet_weights
        weight_key: str = "wavelet_weights",
        identity_pairs: bool = False,     # when True, use i=j for autoencoder-style tasks
        use_pairs_idx: bool = True,       # when False, iterate all frames directly (k is frame index)
        data_key: str = "images",         # dataset name to read frames from
        return_cond: bool = False,        # deprecated compatibility flag; must remain false
        add_thermal: bool = False,        # append thermal field channel(s) to inputs/targets
        thermal_axis: str = "x",          # "x" or "y"
        thermal_use_x0: bool = True,
        thermal_T0: float = 0.0,
        thermal_on_target: bool = False,
        thermal_require_precomputed: bool = False,
        thermal_debug: bool = False,
        thermal_debug_prob: float = 0.01,
        augment: bool = False,
        augment_flip: bool = True,
        augment_flip_prob: float = 0.5,
        augment_roll: bool = False,
        augment_roll_prob: float = 0.5,
        augment_roll_max: Optional[int] = None,
        augment_swap: bool = False,
        augment_swap_prob: float = 0.5,
        augment_rotate: bool = False,     # random 90/180/270 rotations
        augment_rotate_prob: float = 0.5,
        normalize_images: bool = False,
        normalize_force: bool = False,
        normalize_source: str = "file",
    ):
        super().__init__()
        self.h5_path = h5_path
        if input_channels is None or input_channels == "all":
            self.input_channels = None
        else:
            self.input_channels = list(input_channels)
        if target_channels is None or target_channels == "all":
            self.target_channels = None
        else:
            self.target_channels = list(target_channels)
        self.weight_h5_path  = weight_h5
        self.weight_key      = weight_key
        self.identity_pairs  = bool(identity_pairs)
        self.use_pairs_idx   = bool(use_pairs_idx)
        self.data_key = str(data_key)
        self.max_items = max_items
        self.return_cond = bool(return_cond)
        if self.return_cond:
            raise ValueError(
                "PFPairDataset return_cond=True is no longer supported. "
                "Use thermal-field conditioning via add_thermal + conditioning.use_theta."
            )
        self.add_thermal = bool(add_thermal)
        self.thermal_axis = str(thermal_axis).lower()
        self.thermal_use_x0 = bool(thermal_use_x0)
        self.thermal_T0 = float(thermal_T0)
        self.thermal_on_target = bool(thermal_on_target)
        self.thermal_require_precomputed = bool(thermal_require_precomputed)
        self.thermal_debug = bool(thermal_debug)
        self.thermal_debug_prob = float(thermal_debug_prob)
        self.augment = bool(augment)
        self.augment_flip = bool(augment_flip)
        self.augment_flip_prob = float(augment_flip_prob)
        self.augment_roll = bool(augment_roll)
        self.augment_roll_prob = float(augment_roll_prob)
        self.augment_roll_max = augment_roll_max if augment_roll_max is None else int(augment_roll_max)
        self.augment_swap = bool(augment_swap)
        self.augment_swap_prob = float(augment_swap_prob)
        self.augment_rotate = bool(augment_rotate)
        self.augment_rotate_prob = float(augment_rotate_prob)
        self.normalize_images = bool(normalize_images)
        self.normalize_force = bool(normalize_force)
        self.normalize_source = str(normalize_source).lower()
        self._norm_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, float, bool]] = {}

        self._local   = threading.local()   # per-thread/per-process main HDF5 handle
        self._local_w = threading.local() if self.weight_h5_path is not None else None

        # Build a flat (gid, k) index, capture per-group shapes for diagnostics, and cache time stats.
        self.items: List[Tuple[str, int]] = []
        self.shapes: Dict[str, Tuple[int, int]] = {}
        self.time_stats: Dict[str, Tuple[float, float, float]] = {}  # gid -> (t_mean, t_std>0, eff_dt)
        self.thermal_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._thermal_warned: Dict[str, bool] = {}
        with h5py.File(self.h5_path, "r") as h5:
            file_norm = None
            if self.normalize_images:
                if "channel_mean" in h5.attrs and "channel_std" in h5.attrs:
                    mean = torch.tensor(h5.attrs["channel_mean"], dtype=torch.float32)
                    std = torch.tensor(h5.attrs["channel_std"], dtype=torch.float32)
                    eps = float(h5.attrs.get("zscore_eps_images", 1e-6))
                    norm_schema = str(h5.attrs.get("normalization_schema", "")).lower()
                    already = norm_schema == "zscore"
                    file_norm = (mean, std, eps, already)
            for gid in sorted(h5.keys()):
                g = h5[gid]
                if self.use_pairs_idx:
                    if "pairs_idx" not in g:
                        raise KeyError(f"Expected dataset 'pairs_idx' in group {gid} (use_pairs_idx=True).")
                    P = int(g["pairs_idx"].shape[0])
                else:
                    P = int(g[self.data_key].shape[0])
                use = P if limit_per_group is None else min(P, limit_per_group)
                if self.data_key not in g:
                    raise KeyError(f"Expected dataset '{self.data_key}' in group {gid}")
                H, W = g[self.data_key].shape[-2], g[self.data_key].shape[-1]
                self.shapes[gid] = (H, W)
                self.items.extend((gid, k) for k in range(use))

                eff_dt = float(g.attrs.get("effective_dt", g.file.attrs.get("effective_dt", 1.0)))
                eps_time = float(g.attrs.get("zscore_eps_time", g.file.attrs.get("zscore_eps_time", 1e-12)))
                if "time_phys" in g:
                    t_arr = g["time_phys"][:].astype("float64")
                    t_mean = float(t_arr.mean())
                    t_std = float(t_arr.std())
                elif "times" in g:
                    t_arr = g["times"][:].astype("float64") * eff_dt
                    t_mean = float(t_arr.mean())
                    t_std = float(t_arr.std())
                else:
                    t_mean = float(g.attrs.get("time_mean", g.file.attrs.get("time_mean", 0.0)))
                    t_std = float(g.attrs.get("time_std", g.file.attrs.get("time_std", 1.0)))

                denom = t_std if isinstance(t_std, (int, float)) and t_std > 0 else eps_time
                denom = denom if denom and denom > 0 else 1.0
                self.time_stats[gid] = (t_mean, float(denom), eff_dt)

                if self.add_thermal:
                    has_precomputed = ("thermal_field" in g) or ("thermal_images" in g)
                    if self.thermal_require_precomputed and not has_precomputed:
                        raise KeyError(
                            f"Expected precomputed thermal map dataset ('thermal_field' or 'thermal_images') "
                            f"in group {gid}, but none was found."
                        )
                    x_min = float(g.attrs.get("x_min", g.file.attrs.get("x_min", np.nan)))
                    x_max = float(g.attrs.get("x_max", g.file.attrs.get("x_max", np.nan)))
                    y_min = float(g.attrs.get("y_min", g.file.attrs.get("y_min", np.nan)))
                    y_max = float(g.attrs.get("y_max", g.file.attrs.get("y_max", np.nan)))
                    dx = float(g.attrs.get("physical_grid_spacing", g.file.attrs.get("physical_grid_spacing", np.nan)))
                    if not np.isfinite(dx) or dx <= 0:
                        dx_raw = float(g.attrs.get("dx", g.file.attrs.get("dx", 1.0)))
                        W0 = float(g.attrs.get("W0", g.file.attrs.get("W0", 1.0)))
                        dx = dx_raw * W0 if (np.isfinite(dx_raw) and np.isfinite(W0)) else 1.0
                    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min):
                        x_min, x_max = 0.0, dx * W
                    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min):
                        y_min, y_max = 0.0, dx * H
                    # x_min/x_max are stored in dx-units; convert to meters.
                    x_min_m, x_max_m = x_min * dx, x_max * dx
                    y_min_m, y_max_m = y_min * dx, y_max * dx
                    x_centers = x_min_m + (np.arange(W, dtype=np.float64) + 0.5) * ((x_max_m - x_min_m) / W)
                    y_centers = y_min_m + (np.arange(H, dtype=np.float64) + 0.5) * ((y_max_m - y_min_m) / H)
                    self.thermal_cache[gid] = {
                        "x_centers": x_centers,
                        "y_centers": y_centers,
                        "dx": float((x_max_m - x_min_m) / W),
                        "dy": float((y_max_m - y_min_m) / H),
                    }
                if file_norm is not None and self.normalize_source == "file":
                    self._norm_cache[gid] = file_norm

        if self.max_items is not None:
            max_items = self.max_items
            try:
                max_items = float(max_items)
            except (TypeError, ValueError):
                max_items = self.max_items
            if isinstance(max_items, float) and 0.0 < max_items < 1.0:
                max_items = max(1, int(len(self.items) * max_items))
            else:
                max_items = int(max_items)
            if max_items > 0:
                self.items = self.items[: max_items]

        # Optional consistency check between main and side-car file
        if self.weight_h5_path is not None:
            with h5py.File(self.weight_h5_path, "r") as hw:
                missing = [gid for gid, _ in self.items if gid not in hw]
                if len(missing) > 0:
                    raise RuntimeError(
                        f"Groups missing from weight file {self.weight_h5_path}: {sorted(set(missing))[:5]} ..."
                    )

    def _get_h5(self) -> h5py.File:
        h5 = getattr(self._local, "h5", None)
        if h5 is None or not hasattr(h5, "filename"):
            self._local.h5 = h5py.File(self.h5_path, "r")
        return self._local.h5

    def _get_h5_w(self) -> Optional[h5py.File]:
        if self.weight_h5_path is None:
            return None
        if self._local_w is None:
            self._local_w = threading.local()
        hw = getattr(self._local_w, "h5", None)
        if hw is None or not hasattr(hw, "filename"):
            self._local_w.h5 = h5py.File(self.weight_h5_path, "r")
        return self._local_w.h5

    def __len__(self) -> int:
        return len(self.items)

    def _get_time_abs(self, g: h5py.Group, idx: int, eff_dt: float) -> float:
        if "time_phys" in g:
            return float(g["time_phys"][idx])
        if "times" in g:
            return float(g["times"][idx] * eff_dt)
        return float(idx * eff_dt)

    def _get_thermal_params(self, g: h5py.Group, gid: str, idx: int) -> Tuple[float, float, float]:
        G_raw = float(g.attrs.get("thermal_gradient_raw", g.attrs.get("thermal_gradient", np.nan)))
        if not np.isfinite(G_raw) and "thermal_gradient_series" in g:
            G_raw = float(g["thermal_gradient_series"][idx])
        if not np.isfinite(G_raw):
            G_raw = 0.0

        V = float(g.attrs.get("pulling_speed", g.file.attrs.get("pulling_speed", np.nan)))
        if not np.isfinite(V):
            V = 0.0

        x0_dx = float(g.attrs.get("x0_dx", g.file.attrs.get("x0_dx", np.nan)))
        if not np.isfinite(x0_dx):
            x0_dx = 0.0

        if self.thermal_use_x0:
            dx = float(g.attrs.get("physical_grid_spacing", g.file.attrs.get("physical_grid_spacing", np.nan)))
            if not np.isfinite(dx) or dx <= 0:
                dx_raw = float(g.attrs.get("dx", g.file.attrs.get("dx", 1.0)))
                W0 = float(g.attrs.get("W0", g.file.attrs.get("W0", 1.0)))
                dx = dx_raw * W0 if (np.isfinite(dx_raw) and np.isfinite(W0)) else 1.0
            x0 = x0_dx * dx
        else:
            x0 = 0.0

        if gid not in self._thermal_warned and (not np.isfinite(G_raw) or not np.isfinite(V)):
            self._thermal_warned[gid] = True
            print(f"[thermal] missing attrs for {gid} (G_raw={G_raw}, V={V}); using defaults", flush=True)

        return G_raw, V, x0

    def _get_norm_stats(self, gid: str, g: h5py.Group) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, bool]]:
        if not self.normalize_images:
            return None
        cached = self._norm_cache.get(gid)
        if cached is not None:
            return cached
        mean = std = None
        eps = 1e-6
        already = False
        if self.normalize_source == "group":
            if "channel_mean" in g.attrs and "channel_std" in g.attrs:
                mean = torch.tensor(g.attrs["channel_mean"], dtype=torch.float32)
                std = torch.tensor(g.attrs["channel_std"], dtype=torch.float32)
                eps = float(g.attrs.get("zscore_eps_images", 1e-6))
                already = str(g.attrs.get("norm_type_images", "")).lower() == "zscore"
        elif self.normalize_source == "file":
            if "channel_mean" in g.file.attrs and "channel_std" in g.file.attrs:
                mean = torch.tensor(g.file.attrs["channel_mean"], dtype=torch.float32)
                std = torch.tensor(g.file.attrs["channel_std"], dtype=torch.float32)
                eps = float(g.file.attrs.get("zscore_eps_images", 1e-6))
                already = str(g.file.attrs.get("normalization_schema", "")).lower() == "zscore"
        if mean is None or std is None:
            return None
        stats = (mean, std, eps, already)
        self._norm_cache[gid] = stats
        return stats

    def _apply_normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float) -> torch.Tensor:
        denom = std.clone()
        denom = torch.where(denom > 0, denom, torch.full_like(denom, eps))
        return (x - mean.view(-1, 1, 1)) / denom.view(-1, 1, 1)

    def _select_norm_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        channels: Optional[Sequence[int]],
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if channels is None:
            if mean.numel() != tensor.shape[0]:
                raise ValueError("Normalization stats channel count does not match tensor.")
            return mean, std
        if max(channels) >= mean.numel():
            raise ValueError("Normalization stats do not cover requested channels.")
        idx = torch.as_tensor(channels, dtype=torch.long)
        return mean.index_select(0, idx), std.index_select(0, idx)

    def _apply_augmentations(
        self,
        xi: torch.Tensor,
        yj: torch.Tensor,
        wj: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.augment:
            return xi, yj, wj

        if self.augment_rotate and np.random.rand() < self.augment_rotate_prob:
            # Random 90/180/270 rotation.
            k = int(np.random.randint(1, 4))
            xi = torch.rot90(xi, k=k, dims=(-2, -1))
            yj = torch.rot90(yj, k=k, dims=(-2, -1))
            if wj is not None:
                wj = torch.rot90(wj, k=k, dims=(-2, -1))

        if self.augment_swap and np.random.rand() < self.augment_swap_prob:
            xi = xi.transpose(-2, -1)
            yj = yj.transpose(-2, -1)
            if wj is not None:
                wj = wj.transpose(-2, -1)

        if self.augment_flip:
            if np.random.rand() < self.augment_flip_prob:
                xi = xi.flip(-1)
                yj = yj.flip(-1)
                if wj is not None:
                    wj = wj.flip(-1)
            if np.random.rand() < self.augment_flip_prob:
                xi = xi.flip(-2)
                yj = yj.flip(-2)
                if wj is not None:
                    wj = wj.flip(-2)

        if self.augment_roll and np.random.rand() < self.augment_roll_prob:
            max_roll_h = self.augment_roll_max or xi.shape[-2]
            max_roll_w = self.augment_roll_max or xi.shape[-1]
            shift_y = int(np.random.randint(-max_roll_h, max_roll_h + 1))
            shift_x = int(np.random.randint(-max_roll_w, max_roll_w + 1))
            if shift_y != 0 or shift_x != 0:
                xi = torch.roll(xi, shifts=(shift_y, shift_x), dims=(-2, -1))
                yj = torch.roll(yj, shifts=(shift_y, shift_x), dims=(-2, -1))
                if wj is not None:
                    wj = torch.roll(wj, shifts=(shift_y, shift_x), dims=(-2, -1))

        return xi, yj, wj

    def _make_thermal_map(self, gid: str, g: h5py.Group, t_abs: float, idx: int, H: int, W: int) -> torch.Tensor:
        cache = self.thermal_cache.get(gid)
        if cache is None:
            x_centers = np.arange(W, dtype=np.float64) + 0.5
            y_centers = np.arange(H, dtype=np.float64) + 0.5
            dx = 1.0
            dy = 1.0
        else:
            x_centers = cache["x_centers"]
            y_centers = cache["y_centers"]
            dx = cache["dx"]
            dy = cache["dy"]

        G_raw, V, x0 = self._get_thermal_params(g, gid, idx)
        T0 = self.thermal_T0

        if self.thermal_axis == "y":
            coord = y_centers
            shift = V * t_abs
            T_line = T0 + G_raw * (coord - x0 - shift)
            T_map = np.tile(T_line[:, None], (1, W))
            slope = (T_line[1] - T_line[0]) / dy if H > 1 else 0.0
        else:
            coord = x_centers
            shift = V * t_abs
            T_line = T0 + G_raw * (coord - x0 - shift)
            T_map = np.tile(T_line[None, :], (H, 1))
            slope = (T_line[1] - T_line[0]) / dx if W > 1 else 0.0

        if self.thermal_debug and np.random.rand() < self.thermal_debug_prob:
            if not np.isfinite(T_map).all():
                raise RuntimeError(f"[thermal] non-finite values in {gid} at idx={idx}")
            if np.isfinite(G_raw) and (abs(G_raw) > 0 or abs(slope) > 0):
                if not np.isclose(slope, G_raw, rtol=1e-3, atol=1e-6):
                    raise RuntimeError(f"[thermal] slope mismatch in {gid}: slope={slope} G={G_raw}")

        return torch.from_numpy(T_map.astype(np.float32, copy=False)).unsqueeze(0)

    @staticmethod
    def _resize_map_if_needed(t_map: torch.Tensor, H: int, W: int, dtype: torch.dtype) -> torch.Tensor:
        if t_map.dim() == 2:
            t_map = t_map.unsqueeze(0)
        if t_map.dim() != 3:
            raise ValueError(f"Expected thermal map with 2D/3D shape, got {tuple(t_map.shape)}")
        if t_map.shape[-2:] == (H, W):
            return t_map.to(dtype=dtype)
        # Latent datasets can keep thermal_field at source resolution (e.g. 1024x1024).
        # Resize to the current sample resolution before channel concatenation.
        # For strong downsampling (e.g., 1024->64), area interpolation preserves
        # coarse averages better than bilinear and avoids aliasing artifacts.
        src_h, src_w = t_map.shape[-2], t_map.shape[-1]
        mode = "area" if (src_h > H or src_w > W) else "bilinear"
        if mode == "area":
            t_map = F.interpolate(
                t_map.unsqueeze(0).to(dtype=torch.float32),
                size=(H, W),
                mode=mode,
            ).squeeze(0)
        else:
            t_map = F.interpolate(
                t_map.unsqueeze(0).to(dtype=torch.float32),
                size=(H, W),
                mode=mode,
                align_corners=False,
            ).squeeze(0)
        return t_map.to(dtype=dtype)

    def __getitem__(self, idx: int):
        h5 = self._get_h5()
        gid, k = self.items[idx]
        g = h5[gid]

        # Pair indices
        if self.use_pairs_idx:
            if self.identity_pairs:
                j = int(g["pairs_idx"][k][1])
                i = j
            else:
                i, j = g["pairs_idx"][k]  # numpy scalars: int64
        else:
            i = int(k)
            j = i

        eff_dt = float(g.attrs.get("effective_dt", g.file.attrs.get("effective_dt", 1.0)))
        if self.use_pairs_idx and "pairs_time" in g:
            t_abs = float(g["pairs_time"][k][1] * eff_dt)
        else:
            t_abs = self._get_time_abs(g, j, eff_dt)

        # Frames
        x_all = g[self.data_key]  # (T, C, H, W), float32
        if self.input_channels is None:
            xi = torch.from_numpy(x_all[i]).contiguous()   # (C, H, W)
        else:
            xi = torch.from_numpy(x_all[i, self.input_channels]).contiguous()   # (C_in, H, W)
        if self.target_channels is None:
            yj = torch.from_numpy(x_all[j]).contiguous()  # (C, H, W)
        else:
            yj = torch.from_numpy(x_all[j, self.target_channels]).contiguous()  # (C_out, H, W)

        if self.add_thermal:
            H, W = xi.shape[-2], xi.shape[-1]
            if "thermal_field" in g or "thermal_images" in g:
                raw = g["thermal_field"][i] if "thermal_field" in g else g["thermal_images"][i]
                ti_map = torch.from_numpy(raw)
                if self.thermal_debug and np.random.rand() < self.thermal_debug_prob:
                    t_i = self._get_time_abs(g, i, eff_dt)
                    ref = self._make_thermal_map(gid, g, t_i, i, H, W)
                    ti_dbg = self._resize_map_if_needed(ti_map, H, W, dtype=ref.dtype)
                    # Some datasets store absolute thermal field with an unknown additive
                    # offset (T0), while _make_thermal_map can be relative. Compare centered
                    # maps to validate spatial structure/slope independently of global offset.
                    ti_dbg_center = ti_dbg - ti_dbg.mean(dim=(-2, -1), keepdim=True)
                    ref_center = ref - ref.mean(dim=(-2, -1), keepdim=True)
                    if not torch.allclose(ti_dbg_center, ref_center, rtol=1e-3, atol=1e-4):
                        raise RuntimeError(f"[thermal] precomputed mismatch in {gid} at idx={i}")
                ti_map = self._resize_map_if_needed(ti_map, H, W, dtype=xi.dtype)
            else:
                if self.thermal_require_precomputed:
                    raise RuntimeError(
                        f"Group {gid} is missing precomputed thermal maps at pair index {k}. "
                        "Set dataloader.args.thermal_require_precomputed=false to allow synthesized maps."
                    )
                t_i = self._get_time_abs(g, i, eff_dt)
                ti_map = self._make_thermal_map(gid, g, t_i, i, H, W).to(dtype=xi.dtype)
            xi = torch.cat([xi, ti_map], dim=0)
            if self.thermal_on_target:
                if "thermal_field" in g or "thermal_images" in g:
                    raw = g["thermal_field"][j] if "thermal_field" in g else g["thermal_images"][j]
                    tj_map = self._resize_map_if_needed(torch.from_numpy(raw), H, W, dtype=yj.dtype)
                else:
                    if self.thermal_require_precomputed:
                        raise RuntimeError(
                            f"Group {gid} target frame is missing precomputed thermal maps at pair index {k}. "
                            "Set dataloader.args.thermal_require_precomputed=false to allow synthesized maps."
                        )
                    tj_map = self._make_thermal_map(gid, g, t_abs, j, H, W).to(dtype=yj.dtype)
                yj = torch.cat([yj, tj_map], dim=0)
                if self.thermal_debug and np.random.rand() < self.thermal_debug_prob:
                    G_raw, V, _ = self._get_thermal_params(g, gid, j)
                    dt = t_abs - t_i
                    expected_shift = -G_raw * V * dt
                    actual_shift = float((tj_map[0, 0, 0] - ti_map[0, 0, 0]).item())
                    if not np.isclose(actual_shift, expected_shift, rtol=1e-3, atol=1e-6):
                        raise RuntimeError(
                            f"[thermal] time shift mismatch in {gid}: got={actual_shift} expected={expected_shift}"
                        )

        if self.normalize_images:
            stats = self._get_norm_stats(gid, g)
            if stats is None:
                raise RuntimeError("normalize_images=True but no channel_mean/channel_std found in HDF5 attrs.")
            mean, std, eps, already = stats
            if self.normalize_force or not already:
                mean_x, std_x = self._select_norm_stats(mean, std, self.input_channels, xi)
                mean_y, std_y = self._select_norm_stats(mean, std, self.target_channels, yj)
                xi = self._apply_normalize(xi, mean_x, std_x, eps)
                yj = self._apply_normalize(yj, mean_y, std_y, eps)

        # Optional wavelet-based weights for the target
        hw = self._get_h5_w()
        wj = None
        if hw is not None:
            gw = hw[gid]
            w_all = gw[self.weight_key]     # shape (T, C_out, H, W)
            wj = torch.from_numpy(w_all[j]).contiguous()  # (C_out, H, W)

        if self.augment:
            xi, yj, wj = self._apply_augmentations(xi, yj, wj)

        sample = {
            "input": xi,
            "target": yj,
            "gid": gid,
            "pair_index": int(k),
        }
        if wj is not None:
            sample["weight"] = wj
        return sample

    # Make pickling safe for DataLoader workers
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_local", None)
        state.pop("_local_w", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        self._local_w = threading.local() if getattr(self, "weight_h5_path", None) is not None else None

    def close(self):
        h5 = getattr(self._local, "h5", None)
        if h5 is not None:
            try:
                h5.close()
            finally:
                self._local.h5 = None

        if self._local_w is not None:
            hw = getattr(self._local_w, "h5", None)
            if hw is not None:
                try:
                    hw.close()
                finally:
                    self._local_w.h5 = None
