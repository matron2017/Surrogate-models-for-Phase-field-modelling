"""
Shared utilities for the modular trainer (config loading, seeding, batching, misc helpers).
"""

from __future__ import annotations

import datetime
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_symbol(py_path: str, symbol: str):
    """Load a symbol from a Python file path, adding likely package roots to sys.path if needed."""
    p = Path(py_path).resolve()

    def _guess_root(q: Path):
        for a in [q.parent, *q.parents]:
            if (a / "models").is_dir() or (a / "scripts").is_dir():
                return a
        return q.parent

    root = _guess_root(p)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        rel = p.relative_to(root).with_suffix("")
        mod_name = ".".join(rel.parts)
        mod = importlib.import_module(mod_name)
    except Exception:
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, symbol):
        raise AttributeError(f"Symbol '{symbol}' not found in {p}")
    return getattr(mod, symbol)


def _rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _seed_all(seed: int, deterministic: bool):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _seed_worker(worker_id: int):
    base_seed = torch.initial_seed() % 2**31
    import random
    import numpy as np

    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)


def _infer_backbone_name(model_cfg: Dict[str, Any]) -> str:
    file_path = str(model_cfg.get("file", "")).lower()
    class_name = str(model_cfg.get("class", "")).lower()
    for needle, label in [
        ("uafno", "uafno"),
        ("unet", "unet"),
        ("sinenet", "sinenet"),
        ("ssm", "ssm"),
    ]:
        if needle in file_path or needle in class_name:
            return label
    stem = Path(model_cfg.get("file", "")).stem
    return stem or (model_cfg.get("class") or "custom_model")


def _collate(batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("input", "target", "cond", "weight"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "gid" in batch[0]:
        out["gid"] = [b["gid"] for b in batch]
    if "pair_index" in batch[0]:
        out["pair_index"] = [b["pair_index"] for b in batch]
    return out


def _prepare_batch(batch: Dict[str, torch.Tensor], device, cond_cfg, use_chlast: bool):
    """
    Move a batch to device, split conditioning if requested, and enforce channels_last if enabled.
    Returns (x, y, cond_vec, theta_field).
    """
    x = batch["input"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)

    cond = None
    theta = None
    if cond_cfg.get("use_theta", False):
        theta_channels = int(cond_cfg.get("theta_channels", 1))
        if x.dim() != 4:
            raise AssertionError("Expected batched input for theta conditioning.")
        if x.size(1) <= theta_channels:
            raise ValueError(f"Input has {x.size(1)} channels but theta_channels={theta_channels}.")
        theta = x[:, -theta_channels:, ...]
        x = x[:, :-theta_channels, ...]
        if y.dim() == 4 and y.size(1) > x.size(1) and (y.size(1) - x.size(1) == theta_channels):
            y = y[:, :-theta_channels, ...]

        theta_norm = str(cond_cfg.get("theta_normalization", "none")).strip().lower()
        theta_eps = float(cond_cfg.get("theta_eps", 1e-6))
        if theta_norm in {"sample_zscore", "zscore_per_sample"}:
            mean = theta.mean(dim=(-2, -1), keepdim=True)
            std = theta.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(theta_eps)
            theta = (theta - mean) / std
        elif theta_norm in {"affine", "global_affine"}:
            theta_shift = float(cond_cfg.get("theta_shift", 0.0))
            theta_scale = float(cond_cfg.get("theta_scale", 1.0))
            if abs(theta_scale) < theta_eps:
                raise ValueError(f"conditioning.theta_scale must have |scale| >= theta_eps ({theta_eps}).")
            theta = (theta - theta_shift) / theta_scale
        elif theta_norm in {"none", ""}:
            pass
        else:
            raise ValueError(
                f"Unknown conditioning.theta_normalization='{theta_norm}'. "
                "Use one of {'none','sample_zscore','affine'}."
            )

    if cond_cfg.get("enabled", False):
        raise ValueError(
            "Scalar conditioning has been removed from the active training path. "
            "Set conditioning.enabled=false and use conditioning.use_theta with add_thermal."
        )

    if use_chlast:
        x = x.contiguous(memory_format=torch.channels_last)
        if theta is not None:
            theta = theta.contiguous(memory_format=torch.channels_last)

    return x, y, cond, theta


def _flatten_dict(prefix: str, obj: Any, out: Dict[str, Any]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_dict(key, v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def _extract_schedule_value(arr: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)
    out = arr.to(timesteps.device)[timesteps]
    view_shape = (timesteps.shape[0],) + (1,) * (len(x_shape) - 1)
    return out.view(view_shape)


def _q_sample(x0: torch.Tensor, t: torch.Tensor, schedule) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward process:
      - VP/DDPM: q(x_t | x0) = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps.
      - VE (variance-exploding): x_t = x0 + sigma_t * eps.
    Returns (x_t, eps).
    """
    if schedule is None:
        raise RuntimeError("Noise schedule is required for diffusion models.")
    if hasattr(schedule, "kind") and getattr(schedule, "kind") == "ve":
        sigmas = schedule.sigmas.to(x0.device)
        sigma_t = _extract_schedule_value(sigmas, t, x0.shape)
        eps = torch.randn_like(x0)
        x_t = x0 + sigma_t * eps
        return x_t, eps

    alpha_bar = schedule.alpha_bar.clamp(min=1e-12, max=1.0).to(x0.device)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_om_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    sqrt_ab_t = _extract_schedule_value(sqrt_alpha_bar, t, x0.shape)
    sqrt_om_ab_t = _extract_schedule_value(sqrt_om_alpha_bar, t, x0.shape)

    eps = torch.randn_like(x0)
    x_t = sqrt_ab_t * x0 + sqrt_om_ab_t * eps
    return x_t, eps


def _q_bridge_sample(
    x0: torch.Tensor,
    xT: torch.Tensor,
    t: torch.Tensor,
    schedule,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bridge forward process:
      x_t = a_t * xT + b_t * x0 + c_t * eps.
    Returns (x_t, eps).
    """
    if not hasattr(schedule, "kind") or getattr(schedule, "kind") != "bridge":
        raise RuntimeError("Bridge sampling requires a bridge schedule.")
    if x0.shape != xT.shape:
        raise ValueError(f"Bridge sampling expects x0/xT shapes to match (got {x0.shape} vs {xT.shape}).")

    eps = noise if noise is not None else torch.randn_like(x0)
    a_t = _extract_schedule_value(schedule.a, t, x0.shape)
    b_t = _extract_schedule_value(schedule.b, t, x0.shape)
    c_t = _extract_schedule_value(schedule.c, t, x0.shape)
    x_t = a_t * xT + b_t * x0 + c_t * eps
    return x_t, eps


def _q_unidb_sample(
    x0: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    schedule,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    UniDB forward process:
      x_t = f_mean(x0, t; mu) + f_sigma(t) * noise_target.
    noise_target is either:
      - eps (default UniDB), or
      - pi * eps (if residual modulation is enabled in schedule kwargs).
    Returns (x_t, noise_target).
    """
    if getattr(schedule, "kind", None) != "unidb":
        raise RuntimeError("UniDB sampling requires a UniDB schedule.")
    if x0.shape != mu.shape:
        raise ValueError(f"UniDB sampling expects x0/mu shapes to match (got {x0.shape} vs {mu.shape}).")
    return schedule.sample_noisy_state(x0=x0, mu=mu, t=t, noise=noise)


def _count_params(m: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def _fmt_params(n: int) -> str:
    return f"{n:,d} ({n/1e6:.3f} M)"


def _print_visible_gpus_once():
    if not torch.cuda.is_available():
        if _rank0():
            print("CUDA not available")
        return
    try:
        import pynvml as N

        N.nvmlInit()
        n = N.nvmlDeviceGetCount()
        if _rank0():
            print("Visible GPUs:")
            for i in range(n):
                h = N.nvmlDeviceGetHandleByIndex(i)
                name = N.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                print(f"  idx={i} name={name}")
        N.nvmlShutdown()
    except Exception:
        try:
            n = torch.cuda.device_count()
            if _rank0():
                print("Visible GPUs (torch):")
                for i in range(n):
                    print(f"  idx={i} name={torch.cuda.get_device_name(i)}")
        except Exception:
            if _rank0():
                print("Visible GPUs: unknown")


def _barrier_safe(local_rank: Optional[int] = None):
    if not _is_dist():
        return
    try:
        dist.barrier(device_ids=([local_rank] if local_rank is not None else None))
    except TypeError:
        dist.barrier()


def _allreduce_sum_count(device, se_sum: float, elem_cnt: int):
    if not _is_dist():
        return se_sum, elem_cnt
    t = torch.tensor([se_sum, float(elem_cnt)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())


def _allreduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _all_gather_object(obj):
    if not _is_dist():
        return [obj]
    obj_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_list, obj)
    return obj_list


def _now_utc_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_json_safe(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None
