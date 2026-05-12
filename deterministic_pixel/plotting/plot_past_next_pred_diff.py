#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")

PF_ROOT = Path('/scratch/project_462001338/pf_surrogate_modelling')
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.plot_latent_flow_bridge_fields import (  # type: ignore
    _denorm_if_needed,
    _extract_norm_stats,
    _load_train_checkpoint,
    _predict_flow_source_anchored,
    _resolve_dataset_cfg,
    _split_theta,
)
from models.train.core.pf_dataloader import PFPairDataset  # type: ignore
from models.train.core.utils import _load_symbol  # type: ignore
from plot_pair_style import FieldSpec, render_gt_pred_rows


def _linspace_indices(n: int, k: int) -> List[int]:
    k = max(1, min(int(k), int(n)))
    if k == 1:
        return [0]
    return np.linspace(0, n - 1, k, dtype=int).tolist()


def _stats(a: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
    }


def _render_panel(out_png: Path, x: np.ndarray, y: np.ndarray, p: np.ndarray, title: str) -> Dict[str, Any]:
    cmax = min(2, int(x.shape[0]))
    all_stats: Dict[str, Any] = {}
    field_specs: List[FieldSpec] = []
    gt_arrays: List[np.ndarray] = []
    pred_arrays: List[np.ndarray] = []

    for ch in range(cmax):
        past = x[ch]
        nxt = y[ch]
        pred = p[ch]
        d_pn = nxt - past
        d_np = nxt - pred
        label = "Phase" if ch == 0 else "Concentration" if ch == 1 else f"Channel {ch}"
        field_specs.append(FieldSpec(label=label, cmap="coolwarm" if ch == 0 else "viridis", symmetric=(ch == 0), clamp_nonnegative=(ch == 1)))
        gt_arrays.append(nxt)
        pred_arrays.append(pred)
        all_stats[f"ch{ch}"] = {
            "past": _stats(past),
            "next": _stats(nxt),
            "pred": _stats(pred),
            "next-past": _stats(d_pn),
            "next-pred": _stats(d_np),
        }

    render_gt_pred_rows(
        out_png,
        gt_arrays=gt_arrays,
        pred_arrays=pred_arrays,
        field_specs=field_specs,
        title=title,
        left_title="Ground truth",
        right_title="Autoregressive prediction",
        origin="upper",
        dpi=220,
    )
    return all_stats


def _load_ae_model_and_cfg(ckpt_path: Path, cfg_path: Optional[Path], device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", None)
    if cfg is None:
        if cfg_path is None:
            raise RuntimeError("AE checkpoint has no embedded config and --config not provided")
        cfg = yaml.safe_load(cfg_path.read_text())
    mcfg = cfg["model"]
    model_file = str(mcfg["file"])
    if not Path(model_file).exists():
        old_root = "/scratch/project_2008261/pf_surrogate_modelling"
        if model_file.startswith(old_root):
            rel = model_file[len(old_root):].lstrip("/")
            cand = PF_ROOT / rel
            if cand.exists():
                model_file = str(cand)
    ModelClass = _load_symbol(model_file, mcfg["class"])
    model = ModelClass(**(mcfg.get("params", {}) or {}))
    state_dict = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, cfg


def _resolve_ae_dataset(cfg: Dict[str, Any], split: str) -> PFPairDataset:
    dcfg = cfg.get("dataloader", {}) or {}
    args = dict(dcfg.get("args", {}) or {})
    split_args = dict(dcfg.get(f"{split}_args", {}) or {})
    args.update(split_args)
    h5 = cfg.get("paths", {}).get("h5", {}).get(split)
    if isinstance(h5, dict):
        h5 = h5.get("h5_path", h5)
    args["h5_path"] = str(h5)
    return PFPairDataset(**args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["flow", "ae"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--indices", type=str, default="")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--flow-nfe", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.mode == "flow":
        model, cfg = _load_train_checkpoint(args.checkpoint, device)
        ds = _resolve_dataset_cfg(cfg, split=args.split, h5_override=None)
        mean, std, schema = _extract_norm_stats(Path(ds.h5_path), device=device)
        cond_cfg = dict(cfg.get("conditioning", {}) or {})
        fcfg = dict(cfg.get("flow_matching", {}) or {})
        objective = str(cfg.get("train", {}).get("objective", "rectified_flow_source_anchored_concat")).lower()
        noise_std = float(fcfg.get("noise_stochastic_std", 0.0))
        noise_mode = str(fcfg.get("noise_stochastic_mode", "scalar")).strip().lower()
        noise_perturb = bool(fcfg.get("noise_stochastic_perturb_source", False))

        idxs = [int(i) for i in args.indices.split(",") if i.strip()] if args.indices.strip() else _linspace_indices(len(ds), args.num_samples)
        for idx in idxs:
            s = ds[idx]
            gid = str(s.get("gid", ""))
            pair_index = int(s.get("pair_index", -1))
            x = s["input"].to(device).float()
            y = s["target"].unsqueeze(0).to(device).float()
            x_state, theta = _split_theta(x, cond_cfg)
            with torch.inference_mode():
                p = _predict_flow_source_anchored(
                    model=model,
                    x=x_state,
                    theta=theta,
                    nfe=max(10, int(args.flow_nfe)),
                    flow_objective=objective,
                    flow_noise_std=noise_std,
                    flow_noise_mode=noise_mode,
                    flow_noise_perturb_source=noise_perturb,
                )
                x_d = _denorm_if_needed(x_state, mean, std, schema)[0].detach().cpu().numpy()
                y_d = _denorm_if_needed(y, mean, std, schema)[0].detach().cpu().numpy()
                p_d = _denorm_if_needed(p, mean, std, schema)[0].detach().cpu().numpy()

            png = out_dir / f"flow_idx{idx:05d}_{gid}_pair{pair_index:04d}.png"
            stats = _render_panel(png, x_d, y_d, p_d, f"FLOW current | idx={idx} gid={gid} pair={pair_index}")
            outputs.append({"index": idx, "gid": gid, "pair_index": pair_index, "png": str(png), "stats": stats})

        manifest = {
            "mode": "flow",
            "checkpoint": str(args.checkpoint),
            "config_embedded": True,
            "split": args.split,
            "flow_nfe": int(max(10, int(args.flow_nfe))),
            "outputs": outputs,
        }

    else:
        model, cfg = _load_ae_model_and_cfg(args.checkpoint, args.config, device)
        ds = _resolve_ae_dataset(cfg, split=args.split)
        mean, std, schema = _extract_norm_stats(Path(ds.h5_path), device=device)

        idxs = [int(i) for i in args.indices.split(",") if i.strip()] if args.indices.strip() else _linspace_indices(len(ds), args.num_samples)
        for idx in idxs:
            s = ds[idx]
            gid = str(s.get("gid", ""))
            pair_index = int(s.get("pair_index", -1))
            x = s["input"].unsqueeze(0).to(device).float()
            y = s["target"].unsqueeze(0).to(device).float()
            cond = s.get("cond")
            cond = cond.unsqueeze(0).to(device).float() if cond is not None else None
            with torch.inference_mode():
                p = model(x, cond) if cond is not None else model(x)
                x_d = _denorm_if_needed(x, mean, std, schema)[0].detach().cpu().numpy()
                y_d = _denorm_if_needed(y, mean, std, schema)[0].detach().cpu().numpy()
                p_d = _denorm_if_needed(p, mean, std, schema)[0].detach().cpu().numpy()

            png = out_dir / f"ae_idx{idx:05d}_{gid}_pair{pair_index:04d}.png"
            stats = _render_panel(png, x_d, y_d, p_d, f"AE current | idx={idx} gid={gid} pair={pair_index}")
            outputs.append({"index": idx, "gid": gid, "pair_index": pair_index, "png": str(png), "stats": stats})

        manifest = {
            "mode": "ae",
            "checkpoint": str(args.checkpoint),
            "config_embedded": True,
            "split": args.split,
            "outputs": outputs,
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
