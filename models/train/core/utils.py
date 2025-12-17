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


def _extract_cond_from_channels(x: torch.Tensor, cond_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B,C,H,W]; last cond_dim channels are replicated scalars per sample
    assert x.dim() == 4 and x.size(1) >= cond_dim
    cond_map = x[:, -cond_dim:, ...]  # [B,cond_dim,H,W]
    cond_vec = cond_map.mean(dim=(-2, -1))  # [B,cond_dim]
    x_trim = x[:, :-cond_dim, ...]  # [B,C-cond_dim,H,W]
    return x_trim, cond_vec


def _prepare_batch(batch: Dict[str, torch.Tensor], device, cond_cfg, use_chlast: bool):
    """
    Move a batch to device, split conditioning if requested, and enforce channels_last if enabled.
    Returns (x, y, cond_vec).
    """
    x = batch["input"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)

    cond = None
    if cond_cfg.get("enabled", True):
        source = str(cond_cfg.get("source", "field")).lower()
        cd = int(cond_cfg.get("cond_dim", 2))
        if source == "channels":
            if x.dim() == 3:
                raise AssertionError("Expected batched input for channels-based conditioning")
            x, cond = _extract_cond_from_channels(x, cd)
        else:
            if "cond" not in batch:
                raise KeyError("Conditioning enabled but 'cond' missing in batch.")
            cond_t = batch["cond"].to(device, non_blocking=True)
            assert cond_t.dim() == 2 and cond_t.size(1) == cd
            cond = cond_t

    if use_chlast:
        x = x.contiguous(memory_format=torch.channels_last)

    return x, y, cond


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
