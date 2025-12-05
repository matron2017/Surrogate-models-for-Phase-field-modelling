import h5py
import torch
import threading
from torch.utils.data import Dataset
from typing import Sequence, List, Tuple, Dict, Optional

class PFPairDataset(Dataset):
    """
    Paired-frame dataset with scalar conditioning (no extra spatial channels).
    Input:  (C_in, H, W)
    Target: (C_out, H, W)
    Returns per item:
      {
        "input":  (C_in, H, W) float32,
        "target": (C_out, H, W) float32,
        "cond":   (2,) float32  # [Δt_norm, G_norm]
        "gid":    str,
        "pair_index": int,
        "weight": (C_out, H, W) float32  # optional, wavelet-based weights
      }
    """
    def __init__(
        self,
        h5_path: str,
        input_channels: Sequence[int],    # indices into 'images' channel dimension
        target_channels: Sequence[int],   # indices into 'images'
        limit_per_group: Optional[int] = None,
        weight_h5: Optional[str] = None,  # path to side-car file with wavelet_weights
        weight_key: str = "wavelet_weights",
    ):
        super().__init__()
        self.h5_path = h5_path
        self.input_channels  = list(input_channels)
        self.target_channels = list(target_channels)
        self.weight_h5_path  = weight_h5
        self.weight_key      = weight_key

        self._local   = threading.local()   # per-thread/per-process main HDF5 handle
        self._local_w = threading.local() if self.weight_h5_path is not None else None

        # Build a flat (gid, k) index and capture per-group shapes for diagnostics.
        self.items: List[Tuple[str, int]] = []
        self.shapes: Dict[str, Tuple[int, int]] = {}
        with h5py.File(self.h5_path, "r") as h5:
            for gid in sorted(h5.keys()):
                g = h5[gid]
                P = int(g["pairs_idx"].shape[0])
                use = P if limit_per_group is None else min(P, limit_per_group)
                H, W = g["images"].shape[-2], g["images"].shape[-1]
                self.shapes[gid] = (H, W)
                self.items.extend((gid, k) for k in range(use))

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

    def __getitem__(self, idx: int):
        h5 = self._get_h5()
        gid, k = self.items[idx]
        g = h5[gid]

        # Pair indices
        i, j = g["pairs_idx"][k]  # numpy scalars: int64

        # Scalars (float32)
        dt_n = float(g["pairs_dt_norm"][k])
        G_n  = float(g["thermal_gradient_series_norm"][i])

        # Frames
        x_all = g["images"]  # (T, C, H, W), float32
        xi = torch.from_numpy(x_all[i, self.input_channels]).contiguous()   # (C_in, H, W)
        yj = torch.from_numpy(x_all[j, self.target_channels]).contiguous()  # (C_out, H, W)

        # Conditioning vector only (no spatial maps)
        cond = torch.tensor([dt_n, G_n], dtype=xi.dtype)  # (2,)

        sample = {
            "input": xi,
            "target": yj,
            "cond": cond,
            "gid": gid,
            "pair_index": int(k),
        }

        # Optional wavelet-based weights for the target
        hw = self._get_h5_w()
        if hw is not None:
            gw = hw[gid]
            w_all = gw[self.weight_key]     # shape (T, C_out, H, W)
            wj = torch.from_numpy(w_all[j]).contiguous()  # (C_out, H, W)
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
