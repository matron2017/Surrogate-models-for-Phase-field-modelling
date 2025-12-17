"""Utility dataset reading HDF5 phase-field frames for VAE/diffusion training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    h5_path: str
    dataset_key: str = "images"
    pair_stride: int = 1
    pair_mode: bool = False


class PhaseFieldDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        dataset_key: str = "images",
        scalar_key: str | None = None,
        pair_mode: bool = False,
        pair_strides: Sequence[int] | int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.dataset_key = dataset_key
        self.scalar_key = scalar_key
        self.pair_mode = pair_mode
        if isinstance(pair_strides, int):
            pair_strides = [pair_strides]
        self.pair_strides: List[int] = [max(1, int(s)) for s in pair_strides]
        self.dtype = dtype
        self._handle: h5py.File | None = None

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        with h5py.File(self.h5_path, "r") as handle:
            if self.dataset_key not in handle:
                raise KeyError(f"Dataset '{self.dataset_key}' not found in {self.h5_path}")
            self._num_frames = int(handle[self.dataset_key].shape[0])
            if self.scalar_key:
                if self.scalar_key not in handle:
                    raise KeyError(f"Scalar dataset '{self.scalar_key}' missing from {self.h5_path}")
                scalar_len = int(handle[self.scalar_key].shape[0])
                if scalar_len != self._num_frames:
                    raise ValueError("Scalar dataset length must match frame count")

        self._indices: List[int | Tuple[int, int]] = []
        if self.pair_mode:
            for stride in self.pair_strides:
                for start in range(0, self._num_frames - stride):
                    self._indices.append((start, start + stride))
        else:
            self._indices = list(range(self._num_frames))

    def __len__(self) -> int:
        return len(self._indices)

    def _ensure_open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r", swmr=True)
        return self._handle

    def __getitem__(self, idx: int) -> dict:
        handle = self._ensure_open()
        ds = handle[self.dataset_key]
        item = self._indices[idx]
        scalars = None
        if self.scalar_key:
            idx_scalar = item[0] if self.pair_mode else item
            scalars = torch.from_numpy(handle[self.scalar_key][idx_scalar]).to(self.dtype)  # type: ignore[index]

        if self.pair_mode:
            src_idx, tgt_idx = item  # type: ignore[misc]
            current = torch.from_numpy(ds[src_idx]).to(self.dtype)
            nxt = torch.from_numpy(ds[tgt_idx]).to(self.dtype)
            sample = {"current": current, "next": nxt}
            if scalars is not None:
                sample["scalars"] = scalars
            return sample
        frame = torch.from_numpy(ds[item])  # type: ignore[arg-type]
        sample = {"frame": frame.to(self.dtype)}
        if scalars is not None:
            sample["scalars"] = scalars
        return sample

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


class ResidualPatchDataset(Dataset):
    """
    Samples residual patches (x_{t+Δ} - x_t) plus conditioning vectors from an HDF5 tensor.
    """

    def __init__(
        self,
        h5_path: str,
        dataset_key: str = "images",
        scalar_key: str = "scalars",
        patch_size: int = 64,
        pair_stride: int = 1,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.dataset_key = dataset_key
        self.scalar_key = scalar_key
        self.patch_size = int(patch_size)
        self.pair_stride = max(1, int(pair_stride))
        self.dtype = dtype
        self._handle: h5py.File | None = None
        self._rng = torch.Generator().manual_seed(seed)

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        with h5py.File(self.h5_path, "r") as handle:
            if dataset_key not in handle:
                raise KeyError(f"Dataset '{dataset_key}' missing in {self.h5_path}")
            if self.scalar_key not in handle:
                raise KeyError(f"Dataset '{self.scalar_key}' missing in {self.h5_path}")
            self._length = int(handle[dataset_key].shape[0])
            _, _, self.height, self.width = handle[dataset_key].shape

        if self.patch_size > self.height or self.patch_size > self.width:
            raise ValueError(f"patch_size={self.patch_size} exceeds tensor size {(self.height, self.width)}")

        self.indices: List[Tuple[int, int]] = [
            (start, start + self.pair_stride)
            for start in range(0, self._length - self.pair_stride)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def _ensure_open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r", swmr=True)
        return self._handle

    def __getitem__(self, idx: int) -> dict:
        handle = self._ensure_open()
        frames = handle[self.dataset_key]
        scalars = handle[self.scalar_key]
        src, tgt = self.indices[idx]

        xt = torch.from_numpy(frames[src]).to(self.dtype)
        xt_next = torch.from_numpy(frames[tgt]).to(self.dtype)
        residual = xt_next - xt

        max_i = self.height - self.patch_size
        max_j = self.width - self.patch_size
        i = int(torch.randint(0, max_i + 1, (1,), generator=self._rng))
        j = int(torch.randint(0, max_j + 1, (1,), generator=self._rng))

        xt_patch = xt[:, i : i + self.patch_size, j : j + self.patch_size]
        residual_patch = residual[:, i : i + self.patch_size, j : j + self.patch_size]
        cond_vec = torch.from_numpy(scalars[src]).to(self.dtype)

        return {
            "xt": xt_patch,
            "residual": residual_patch,
            "cond": cond_vec,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()
