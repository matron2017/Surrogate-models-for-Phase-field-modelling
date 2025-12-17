#!/usr/bin/env python3
"""
Precompute wavelet-based importance weights into a side-car HDF5 file.

- Input HDF5: original dataset (read-only).
- Output HDF5: same groups, each with dataset 'wavelet_weights' of shape (T, C_out, H, W).
"""

import argparse
from pathlib import Path
import json
import datetime

import h5py
import torch

import sys

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _import_wavelet():
    # prefer legacy path checked into this workspace
    legacy = ROOT / "solidification_modelling" / "scripts_legacy"
    if legacy.exists():
        sys.path.insert(0, str(legacy))
        from wavelet_weight import wavelet_importance_per_channel  # type: ignore  # noqa: E402
        return wavelet_importance_per_channel
    scripts_dir = ROOT / "models" / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
        from wavelet_weight import wavelet_importance_per_channel  # type: ignore  # noqa: E402
        return wavelet_importance_per_channel
    raise ModuleNotFoundError(f"wavelet_weight.py not found (checked {legacy} and {scripts_dir})")

wavelet_importance_per_channel = _import_wavelet()


def _parse_args():
    ap = argparse.ArgumentParser(description="Precompute wavelet weights side-car HDF5")
    ap.add_argument("--h5", type=str, required=True,
                    help="Path to original HDF5 (read-only)")
    ap.add_argument("--out", type=str, required=True,
                    help="Path to side-car HDF5 to create (will be overwritten)")
    ap.add_argument("--target-channels", type=int, nargs="+", required=True,
                    help="Indices in 'images' used as output channels")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Number of frames per wavelet batch")

    # Wavelet hyperparameters; defaults match wavelet_weight.py
    ap.add_argument("--J", type=int, default=1, help="Number of DWT levels")
    ap.add_argument("--wave", type=str, default="haar", help="Wavelet name")
    ap.add_argument("--mode", type=str, default="zero", help="DWT padding mode")
    ap.add_argument("--theta", type=float, default=0.8, help="Quantile threshold θ")
    ap.add_argument("--alpha", type=float, default=1.25, help="Base weight α")
    ap.add_argument("--beta-w", type=float, default=6.0, help="Max weight β_w")
    return ap.parse_args()


def _open_h5_read(path: Path) -> h5py.File:
    return h5py.File(str(path), "r")


def _open_h5_write(path: Path) -> h5py.File:
    if path.exists():
        path.unlink()
    return h5py.File(str(path), "w")


def _create_wavelet_dataset(group_out, T: int, C_out: int, H: int, W: int):
    """
    Create 'wavelet_weights' dataset with chunks along time dimension.
    """
    return group_out.create_dataset(
        "wavelet_weights",
        shape=(T, C_out, H, W),
        dtype="float32",
        chunks=(1, C_out, H, W),
        compression="gzip",
        compression_opts=4,
    )


def _attach_metadata(
    h5_out: h5py.File,
    h5_path: Path,
    target_channels,
    J: int,
    wave: str,
    mode: str,
    theta: float,
    alpha: float,
    beta_w: float,
):
    """
    Store parameters used to generate this side-car file as HDF5 root attributes.
    """
    h5_out.attrs["source_h5"] = str(h5_path)
    h5_out.attrs["created_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    h5_out.attrs["target_channels"] = json.dumps(list(map(int, target_channels)))

    params = {
        "J": int(J),
        "wave": str(wave),
        "mode": str(mode),
        "theta": float(theta),
        "alpha": float(alpha),
        "beta_w": float(beta_w),
    }
    h5_out.attrs["wavelet_params"] = json.dumps(params)
    h5_out.attrs["wavelet_weights_version"] = "v1"


def precompute_for_file(
    h5_path: Path,
    out_path: Path,
    target_channels,
    device_str: str,
    batch_size: int,
    J: int,
    wave: str,
    mode: str,
    theta: float,
    alpha: float,
    beta_w: float,
):
    device = torch.device(device_str)

    h5_in = _open_h5_read(h5_path)
    h5_out = _open_h5_write(out_path)

    # Attach metadata once at file creation
    _attach_metadata(
        h5_out=h5_out,
        h5_path=h5_path,
        target_channels=target_channels,
        J=J,
        wave=wave,
        mode=mode,
        theta=theta,
        alpha=alpha,
        beta_w=beta_w,
    )

    try:
        gids = sorted(h5_in.keys())
        print(f"Found {len(gids)} groups in {h5_path}")

        for gid in gids:
            g_in = h5_in[gid]
            images = g_in["images"]  # (T, C, H, W), float32
            T, C, H, W = images.shape
            C_out = len(target_channels)

            print(f"[{gid}] T={T} C={C} H={H} W={W} -> C_out={C_out}")

            g_out = h5_out.create_group(gid)
            dset_w = _create_wavelet_dataset(g_out, T, C_out, H, W)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                n = end - start

                arr = images[start:end, target_channels, :, :]
                x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    aw, _ = wavelet_importance_per_channel(
                        x,
                        J=J,
                        wave=wave,
                        mode=mode,
                        theta=theta,
                        alpha=alpha,
                        beta_w=beta_w,
                    )

                aw_np = aw.detach().cpu().numpy().astype("float32")
                if aw_np.shape != (n, C_out, H, W):
                    raise RuntimeError(
                        f"Unexpected aw shape {aw_np.shape}, expected {(n, C_out, H, W)}"
                    )

                dset_w[start:end, :, :, :] = aw_np

    finally:
        h5_in.close()
        h5_out.close()


def main():
    args = _parse_args()
    h5_path = Path(args.h5).resolve()
    out_path = Path(args.out).resolve()

    print(f"Input HDF5:   {h5_path}")
    print(f"Output HDF5:  {out_path}")
    print(f"Target chans: {args.target_channels}")
    print(f"Device:       {args.device}")

    precompute_for_file(
        h5_path=h5_path,
        out_path=out_path,
        target_channels=args.target_channels,
        device_str=args.device,
        batch_size=args.batch_size,
        J=args.J,
        wave=args.wave,
        mode=args.mode,
        theta=args.theta,
        alpha=args.alpha,
        beta_w=args.beta_w,
    )
    print("Done.")


if __name__ == "__main__":
    main()
