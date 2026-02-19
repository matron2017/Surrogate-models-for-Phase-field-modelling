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
    # Prefer local training implementation
    try:
        from models.train.core import wavelet_weight as ww  # type: ignore  # noqa: E402
        return ww
    except Exception:
        pass

    # Fall back to legacy paths if needed
    legacy = ROOT / "solidification_modelling" / "scripts_legacy"
    workspace_legacy = ROOT.parent / "solidification_modelling" / "scripts_legacy"
    if legacy.exists():
        sys.path.insert(0, str(legacy))
        import wavelet_weight as ww  # type: ignore  # noqa: E402
        return ww
    if workspace_legacy.exists():
        sys.path.insert(0, str(workspace_legacy))
        import wavelet_weight as ww  # type: ignore  # noqa: E402
        return ww
    scripts_dir = ROOT / "models" / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
        import wavelet_weight as ww  # type: ignore  # noqa: E402
        return ww
    raise ModuleNotFoundError(
        f"wavelet_weight.py not found (checked {legacy}, {workspace_legacy}, and {scripts_dir})"
    )


wavelet_weight = _import_wavelet()


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
    ap.add_argument(
        "--method",
        type=str,
        default="quantile",
        choices=("quantile", "bandpass", "multiband"),
        help="Weighting method: quantile (legacy), bandpass (interface + HF band), or multiband",
    )
    ap.add_argument("--bp-sigma-low", type=float, default=1.5, help="Bandpass blur σ1")
    ap.add_argument("--bp-sigma-high", type=float, default=10.0, help="Bandpass blur σ2")
    ap.add_argument("--bp-band-sigma", type=float, default=16.0, help="Interface band blur σ")
    ap.add_argument("--bp-mask-quantile", type=float, default=0.95, help="Gradient mask quantile")
    ap.add_argument("--bp-power", type=float, default=1.5, help="Importance power p")
    ap.add_argument("--bp-norm-quantile", type=float, default=0.99, help="E_bp normalization quantile")
    ap.add_argument("--bp-normalize-mean", action="store_true", help="Normalize mean weight to 1")
    ap.add_argument("--bp-clip-min", type=float, default=None, help="Optional min clamp")
    ap.add_argument("--bp-clip-max", type=float, default=500.0, help="Optional max clamp")
    ap.add_argument(
        "--mb-level-weights",
        type=str,
        default="1.0,0.7,0.4",
        help="Comma-separated detail band weights (per level)",
    )
    ap.add_argument("--mb-lowpass-weight", type=float, default=0.2, help="Low-pass band weight")
    ap.add_argument("--mb-norm-quantile", type=float, default=0.99, help="Band normalization quantile")
    ap.add_argument("--mb-power", type=float, default=1.5, help="Importance power p")
    ap.add_argument("--mb-normalize-mean", action="store_true", help="Normalize mean weight to 1")
    ap.add_argument("--mb-rescale-max", action="store_true", help="Rescale weights so max == beta_w")
    ap.add_argument("--mb-clip-min", type=float, default=None, help="Optional min clamp")
    ap.add_argument("--mb-clip-max", type=float, default=500.0, help="Optional max clamp")
    ap.add_argument("--mb-combine-norm", action="store_true", help="Normalize combined bands by sum of weights")
    ap.add_argument(
        "--top-quantile",
        type=float,
        default=None,
        help="Quantile for extra multiplier (defaults to --theta when --top-mult is set)",
    )
    ap.add_argument(
        "--top-mult",
        type=float,
        default=None,
        help="Extra multiplier for top-quantile wavelet energy weights (e.g. 500)",
    )
    ap.add_argument("--shard-rank", type=int, default=0, help="Shard index (0-based)")
    ap.add_argument("--shard-count", type=int, default=1, help="Total number of shards")
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
    top_quantile: float | None,
    top_mult: float | None,
    method: str,
    bandpass_params: dict | None,
    multiband_params: dict | None,
    shard_rank: int,
    shard_count: int,
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
        "method": str(method),
    }
    if top_quantile is not None:
        params["top_quantile"] = float(top_quantile)
    if top_mult is not None:
        params["top_mult"] = float(top_mult)
    if bandpass_params:
        params["bandpass"] = bandpass_params
    if multiband_params:
        params["multiband"] = multiband_params
    params["shard_rank"] = int(shard_rank)
    params["shard_count"] = int(shard_count)
    h5_out.attrs["wavelet_params"] = json.dumps(params)
    h5_out.attrs["wavelet_weights_version"] = "v2"


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
    top_quantile: float | None,
    top_mult: float | None,
    method: str,
    bandpass_params: dict | None,
    multiband_params: dict | None,
    shard_rank: int,
    shard_count: int,
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
        top_quantile=top_quantile,
        top_mult=top_mult,
        method=method,
        bandpass_params=bandpass_params,
        multiband_params=multiband_params,
        shard_rank=shard_rank,
        shard_count=shard_count,
    )

    try:
        gids = sorted(h5_in.keys())
        if shard_count > 1:
            gids = [g for idx, g in enumerate(gids) if idx % shard_count == shard_rank]
        print(f"Using shard {shard_rank + 1}/{shard_count}: {len(gids)} groups")
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
                    if method == "bandpass":
                        if not hasattr(wavelet_weight, "wavelet_bandpass_importance_per_channel"):
                            raise RuntimeError("Bandpass method requested but not available.")
                        aw, f_accum = wavelet_weight.wavelet_bandpass_importance_per_channel(
                            x,
                            J=J,
                            wave=wave,
                            mode=mode,
                            beta_w=beta_w,
                            power=float(bandpass_params["power"]),
                            sigma_low=float(bandpass_params["sigma_low"]),
                            sigma_high=float(bandpass_params["sigma_high"]),
                            band_sigma=float(bandpass_params["band_sigma"]),
                            mask_quantile=float(bandpass_params["mask_quantile"]),
                            norm_quantile=float(bandpass_params["norm_quantile"]),
                            normalize_mean=bool(bandpass_params["normalize_mean"]),
                            clip_min=bandpass_params["clip_min"],
                            clip_max=bandpass_params["clip_max"],
                        )
                    elif method == "multiband":
                        if not hasattr(wavelet_weight, "wavelet_multiband_importance_per_channel"):
                            raise RuntimeError("Multiband method requested but not available.")
                        aw, f_accum = wavelet_weight.wavelet_multiband_importance_per_channel(
                            x,
                            J=J,
                            wave=wave,
                            mode=mode,
                            level_weights=multiband_params["level_weights"],
                            lowpass_weight=float(multiband_params["lowpass_weight"]),
                            beta_w=beta_w,
                            power=float(multiband_params["power"]),
                            norm_quantile=float(multiband_params["norm_quantile"]),
                            normalize_mean=bool(multiband_params["normalize_mean"]),
                            rescale_max=bool(multiband_params["rescale_max"]),
                            clip_min=multiband_params["clip_min"],
                            clip_max=multiband_params["clip_max"],
                            combine_norm=bool(multiband_params["combine_norm"]),
                        )
                    else:
                        aw, f_accum = wavelet_weight.wavelet_importance_per_channel(
                            x,
                            J=J,
                            wave=wave,
                            mode=mode,
                            theta=theta,
                            alpha=alpha,
                            beta_w=beta_w,
                        )
                        if top_mult is not None and float(top_mult) != 1.0:
                            q_theta = float(theta if top_quantile is None else top_quantile)
                            f_flat = f_accum.view(n, C_out, -1)
                            q = torch.quantile(f_flat, q_theta, dim=-1, keepdim=True)
                            q_b = q.view(n, C_out, 1, 1)
                            aw = torch.where(f_accum > q_b, aw * float(top_mult), aw)

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
    shard_rank = int(args.shard_rank)
    shard_count = int(args.shard_count)
    if shard_count <= 0:
        raise ValueError("--shard-count must be > 0")
    if shard_rank < 0 or shard_rank >= shard_count:
        raise ValueError("--shard-rank must be within [0, shard-count)")
    if shard_count > 1:
        out_path = Path(str(out_path) + f".part{shard_rank:02d}")

    print(f"Input HDF5:   {h5_path}")
    print(f"Output HDF5:  {out_path}")
    print(f"Target chans: {args.target_channels}")
    print(f"Device:       {args.device}")

    bandpass_params = None
    if args.method == "bandpass":
        bandpass_params = {
            "sigma_low": float(args.bp_sigma_low),
            "sigma_high": float(args.bp_sigma_high),
            "band_sigma": float(args.bp_band_sigma),
            "mask_quantile": float(args.bp_mask_quantile),
            "power": float(args.bp_power),
            "norm_quantile": float(args.bp_norm_quantile),
            "normalize_mean": bool(args.bp_normalize_mean),
            "clip_min": args.bp_clip_min,
            "clip_max": args.bp_clip_max,
        }
    multiband_params = None
    if args.method == "multiband":
        level_weights = [float(x) for x in args.mb_level_weights.split(",") if x.strip()]
        multiband_params = {
            "level_weights": level_weights,
            "lowpass_weight": float(args.mb_lowpass_weight),
            "norm_quantile": float(args.mb_norm_quantile),
            "power": float(args.mb_power),
            "normalize_mean": bool(args.mb_normalize_mean),
            "rescale_max": bool(args.mb_rescale_max),
            "clip_min": args.mb_clip_min,
            "clip_max": args.mb_clip_max,
            "combine_norm": bool(args.mb_combine_norm),
        }

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
        top_quantile=args.top_quantile,
        top_mult=args.top_mult,
        method=args.method,
        bandpass_params=bandpass_params,
        multiband_params=multiband_params,
        shard_rank=shard_rank,
        shard_count=shard_count,
    )
    print("Done.")


if __name__ == "__main__":
    main()
