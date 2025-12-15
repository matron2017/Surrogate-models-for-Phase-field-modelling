#!/usr/bin/env python3
# Minimal, explicit PyTorch trainer with optional DDP and AMP.
# Properties:
# - One-time GPU visibility print at start.
# - Conditioning via dense vector; supports source="field" or source="channels".
# - Model kwargs taken directly from YAML.
# - Epoch-only logging to CSV; final learning-curve plot at end.
# - Checkpoint policy: checkpoint.last.pth (each epoch) and checkpoint.best.pth (on improvement).
# - Scheduler and early stopping monitor default to 'val' if present else 'train'.
# - DDP + AMP preserved. No step logs. No GPU utilisation polling beyond the initial print.
# - Run-level JSON written at start and finalised at end.
# - Per-trajectory (gid) loss and coverage tracking written to CSV.
# - Comments avoid second-person phrasing.

import os, argparse, time, csv, math, signal, yaml, json, socket, platform, datetime
from typing import Dict, Any, Mapping, List, Tuple, Optional
import sys, importlib, importlib.util
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
try:
    from torchcfm import ConditionalFlowMatcher
except Exception:
    ConditionalFlowMatcher = None
try:
    import mlflow
except Exception:
    mlflow = None

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.backbones.registry import build_model as registry_build_model
from models.diffusion.configs import DiffusionConfig
from models.diffusion.scheduler_registry import get_noise_schedule
from models.diffusion.timestep_sampler import get_timestep_sampler
from models.train.adaptive.registry import build_region_selector
from models.train.loss_registry import build_surrogate_loss, build_diffusion_loss
from models.train.tasks.diffusion_task import DiffusionTask


# ------------------- utilities -------------------
def _load_symbol(py_path: str, symbol: str):
    p = Path(py_path).resolve()

    # Find a root that contains the top-level package folders (e.g. 'models', 'scripts')
    def _guess_root(q: Path):
        for a in [q.parent, *q.parents]:
            if (
                (a / "models").is_dir()
                or (a / "scripts").is_dir()
                or (a / "models").is_dir()
            ):
                return a
        return q.parent

    root = _guess_root(p)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Try importing as a proper package module to enable intra-package imports
    try:
        rel = p.relative_to(root).with_suffix("")
        mod_name = ".".join(rel.parts)  # e.g. models.backbones.uafno_cond
        mod = importlib.import_module(mod_name)
    except Exception:
        # Fallback: direct-by-path import
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
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

def _seed_worker(worker_id: int):
    base_seed = torch.initial_seed() % 2**31
    import random, numpy as np
    random.seed(base_seed + worker_id); np.random.seed(base_seed + worker_id)

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
    cond_map = x[:, -cond_dim:, ...]            # [B,cond_dim,H,W]
    cond_vec = cond_map.mean(dim=(-2, -1))      # [B,cond_dim]
    x_trim = x[:, :-cond_dim, ...]              # [B,C-cond_dim,H,W]
    return x_trim, cond_vec

def _prepare_batch(batch: Dict[str, torch.Tensor], device, cond_cfg, use_chlast: bool):
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
        else:  # "field"
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
    Diffusion forward process: q(x_t | x0) = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps.
    Returns x_t and the sampled eps.
    """
    if schedule is None:
        raise RuntimeError("Noise schedule is required for diffusion models.")
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
        if _rank0(): print("CUDA not available")
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
            if _rank0(): print("Visible GPUs: unknown")

def _barrier_safe(local_rank: Optional[int] = None):
    if not _is_dist(): return
    try:
        dist.barrier(device_ids=([local_rank] if local_rank is not None else None))
    except TypeError:
        dist.barrier()

def _allreduce_sum_count(device, se_sum: float, elem_cnt: int):
    if not _is_dist(): return se_sum, elem_cnt
    t = torch.tensor([se_sum, float(elem_cnt)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item())

def _all_gather_object(obj):
    if not _is_dist():
        return [obj]
    obj_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_list, obj)
    return obj_list

# --- run.json helpers ---
def _now_utc_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _load_json_safe(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

# ------------------- optim/sched -------------------
def _make_optimizer(params, cfg):
    name = cfg["optim"]["name"].lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"].get("weight_decay", 0.0))
    betas = tuple(cfg["optim"].get("betas", [0.9, 0.999]))
    if name == "adam":  return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw": return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name}")

def _make_scheduler(optim, cfg):
    s = cfg["sched"]["name"].lower()
    if s == "none": return None, None
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg["sched"].get("T_max", 100))), None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=int(cfg["sched"].get("step_size", 30)),
                                               gamma=float(cfg["sched"].get("gamma", 0.1))), None
    if s == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optim, gamma=float(cfg["sched"].get("gamma", 0.97))), None
    if s == "plateau":
        monitor = cfg["sched"].get("monitor", {"split": "train", "mode": "min"})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode=monitor.get("mode", "min"),
            factor=float(cfg["sched"].get("factor", 0.5)),
            patience=int(cfg["sched"].get("patience", 10)),
            min_lr=float(cfg["sched"].get("min_lr", 1e-7)),
            threshold=float(cfg["sched"].get("threshold", 1e-4)),
        ), monitor
    raise ValueError(f"Unknown scheduler {s}")


# ------------------- main -------------------
def main(cfg_path: str, resume_arg: Optional[str] = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    trainer_cfg = cfg.setdefault("trainer", {})
    seed = int(trainer_cfg.get("seed", 17))
    deterministic = bool(trainer_cfg.get("deterministic", False))
    log_interval = int(trainer_cfg.get("log_interval", 0))  # 0 disables per-step logging

    train_cfg = cfg.setdefault("train", {})
    model_family = str(train_cfg.get("model_family", "surrogate")).lower()
    train_cfg["model_family"] = model_family

    task_cfg = cfg.setdefault("task", {})
    task_name = str(task_cfg.get("name", "surrogate")).lower()
    task_cfg["name"] = task_name

    model_cfg = cfg.setdefault("model", {})
    backbone = model_cfg.get("backbone")
    if not backbone:
        backbone = _infer_backbone_name(model_cfg)
        model_cfg["backbone"] = backbone
    backbone = str(backbone)

    diffusion_cfg = cfg.setdefault("diffusion", {})
    noise_schedule = str(diffusion_cfg.get("noise_schedule", "linear"))
    diffusion_cfg["noise_schedule"] = noise_schedule
    timestep_sampler = str(diffusion_cfg.get("timestep_sampler", "uniform"))
    diffusion_cfg["timestep_sampler"] = timestep_sampler

    loss_cfg = cfg.setdefault("loss", {})
    weight_wavelet_loss = float(loss_cfg.get("weight_wavelet_loss", 0.0))
    loss_cfg["weight_wavelet_loss"] = weight_wavelet_loss

    flow_match_cfg = cfg.setdefault("flow_matching", {})
    fm_sigma = float(flow_match_cfg.get("sigma", 0.0))
    flow_match_cfg["sigma"] = fm_sigma

    adaptive_cfg = cfg.setdefault("adaptive", {})
    region_selector = str(adaptive_cfg.get("region_selector", "none"))
    adaptive_cfg["region_selector"] = region_selector
    adaptive_resolution = bool(adaptive_cfg.get("enable_adaptive_resolution", False))
    adaptive_cfg["enable_adaptive_resolution"] = adaptive_resolution

    schedule_kwargs = dict(diffusion_cfg.get("schedule_kwargs", {}))
    sampler_kwargs = dict(diffusion_cfg.get("sampler_kwargs", {}))
    region_kwargs = dict(adaptive_cfg.get("region_kwargs", {}))
    diff_dt = float(diffusion_cfg.get("dt", 1.0))
    diff_steps = int(diffusion_cfg.get("n_steps", 1))
    diff_bc = str(diffusion_cfg.get("thermal_bc", "dirichlet"))
    diff_record = bool(diffusion_cfg.get("record_trajectory", False))

    # MLflow setup
    mlflow_cfg = cfg.setdefault("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    mlflow_active = False
    mlflow_parent_run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    if mlflow_enabled and mlflow is not None and _rank0():
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            experiment_name = mlflow_cfg.get("experiment_name", "models")
            mlflow.set_experiment(experiment_name)
            default_run = os.environ.get("SLURM_JOB_ID")
            run_name = mlflow_cfg.get("run_name") or default_run or f"run-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            active = mlflow.start_run(run_name=str(run_name))
            mlflow_run_id = active.info.run_id
            mlflow_parent_run_id = mlflow_run_id
            mlflow_active = True
            try:
                mlflow.set_tag("mlflow.parentRunId", mlflow_parent_run_id)
            except Exception:
                pass
        except Exception as e:
            print(f"MLflow disabled: {e}", flush=True)

    if _rank0():
        desc_msg = (
            f"Descriptors → model_family={model_family}, task={task_name}, backbone={backbone}, "
            f"noise_schedule={noise_schedule}, timestep_sampler={timestep_sampler}, "
            f"diff_dt={diff_dt}, diff_steps={diff_steps}, "
            f"weight_wavelet_loss={weight_wavelet_loss}, region_selector={region_selector}, "
            f"adaptive_resolution={adaptive_resolution}"
        )
        if model_family == "flow_matching":
            desc_msg += f", flow_matching.sigma={fm_sigma}"
        print(desc_msg, flush=True)
        if mlflow_active:
            ml_params = {
                "model_family": model_family,
                "model.backbone": backbone,
                "task.name": task_name,
                "diffusion.noise_schedule": noise_schedule,
                "diffusion.timestep_sampler": timestep_sampler,
                "diffusion.dt": diff_dt,
                "diffusion.n_steps": diff_steps,
                "diffusion.thermal_bc": diff_bc,
                "diffusion.record_trajectory": diff_record,
                "loss.weight_wavelet_loss": weight_wavelet_loss,
                "adaptive.region_selector": region_selector,
                "adaptive.enable_adaptive_resolution": adaptive_resolution,
                "seed": seed,
            }
            if model_family == "flow_matching":
                ml_params["flow_matching.sigma"] = fm_sigma
            try:
                mlflow.log_params(ml_params)
                cfg_flat: Dict[str, Any] = {}
                _flatten_dict("", cfg, cfg_flat)
                mlflow.log_params(cfg_flat)
            except Exception as e:
                print(f"MLflow param logging failed: {e}", flush=True)


    # Enforce per-rank thread count to suppress default OMP hints
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    _seed_all(seed, deterministic)

    have_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and have_cuda:
        # Set device before process group init to avoid warnings and bind streams
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(cfg["trainer"].get("device", "cuda" if have_cuda else "cpu"))

    if _rank0(): _print_visible_gpus_once()

    # CUDA backend knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not deterministic:
            torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("medium")
        except Exception: pass

    # AMP config
    amp_cfg = cfg["trainer"].get("amp", {"enabled": device.type == "cuda", "dtype": "bf16"})
    amp_enabled = bool(amp_cfg.get("enabled", device.type == "cuda"))
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bf16")).lower() == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    # Channels-last
    use_chlast = bool(cfg["trainer"].get("channels_last", False))

    # ----- datasets -----
    DSClass = _load_symbol(cfg["dataloader"]["file"], cfg["dataloader"]["class"])
    ds_args = cfg["dataloader"].get("args", {})

    # train
    if isinstance(cfg["paths"]["h5"]["train"], dict):
        train_ds = DSClass(**cfg["paths"]["h5"]["train"], **ds_args)
    else:
        train_ds = DSClass(cfg["paths"]["h5"]["train"], **ds_args)

    # val
    use_val_flag = bool(cfg["trainer"].get("use_val", True)) and cfg["paths"]["h5"].get("val")
    val_ds = None
    if use_val_flag:
        if isinstance(cfg["paths"]["h5"]["val"], dict):
            val_ds = DSClass(**cfg["paths"]["h5"]["val"], **ds_args)
        else:
            val_ds = DSClass(cfg["paths"]["h5"]["val"], **ds_args)

    if cfg["loader"].get("overfit_n") is not None:
        n = int(cfg["loader"]["overfit_n"])
        train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))

    # Sample-driven sanity checks
    sample = train_ds[0]
    x0, y0 = sample["input"], sample["target"]
    assert x0.dim() == 3 and y0.dim() == 3  # [C,H,W]
    H, W = int(x0.shape[-2]), int(x0.shape[-1])

    # Wavelet weight configuration
    has_weight = "weight" in sample
    use_weight_loss = bool(cfg["trainer"].get("use_wavelet_weights", False))
    if use_weight_loss and not has_weight:
        raise RuntimeError(
            "Configuration sets trainer.use_wavelet_weights=True "
            "but dataset samples do not contain 'weight'."
        )
    surrogate_loss_fn = build_surrogate_loss(loss_cfg)
    diffusion_loss_fn = build_diffusion_loss(loss_cfg)

    noise_schedule_obj = None
    timestep_sampler_obj = None
    region_selector_obj = None
    if model_family == "diffusion":
        noise_schedule_obj = get_noise_schedule(noise_schedule, **schedule_kwargs)
        sampler_kwargs.setdefault("device", device)
        timestep_sampler_obj = get_timestep_sampler(
            timestep_sampler, schedule=noise_schedule_obj, **sampler_kwargs
        )
        region_selector_obj = build_region_selector(region_selector, **region_kwargs)

    # ----- conditioning -----
    cond_cfg = dict(cfg.get("conditioning", {}))
    cond_enabled = cond_cfg.get("enabled", True)
    cond_dim = int(cond_cfg.get("cond_dim", 2))
    cond_source = str(cond_cfg.get("source", "field")).lower()
    if cond_enabled:
        if cond_source == "channels":
            assert x0.size(0) >= cond_dim
        else:
            assert "cond" in sample and sample["cond"].dim() == 1 and sample["cond"].numel() == cond_dim

    # ----- model -----
    model_cfg = cfg["model"]
    use_legacy_model = (
        "file" in model_cfg
        and "class" in model_cfg
        and model_cfg.get("file")
        and model_cfg.get("class")
    )
    if use_legacy_model:
        ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
        base_model = ModelClass(**model_cfg.get("params", {}))
    else:
        model_cfg_local = dict(model_cfg)
        if model_family == "flow_matching":
            params = dict(model_cfg_local.get("params", model_cfg_local))
            if "cond_dim" in params:
                params["cond_dim"] = int(params["cond_dim"]) + 1  # append time conditioning
            model_cfg_local["params"] = params
        base_model = registry_build_model(model_family, backbone, model_cfg_local)

    base_model = base_model.to(device)
    model = base_model
    if task_name == "diffusion":
        model = DiffusionTask(
            base_model,
            DiffusionConfig(dt=diff_dt, n_steps=diff_steps, thermal_bc=diff_bc, record_trajectory=diff_record),
        ).to(device)
    if use_chlast and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Report parameters once
    if _rank0():
        tot, trainable = _count_params(model)
        ws = int(os.environ.get("WORLD_SIZE", "1")) if _is_dist() else 1
        eff_bsz = int(cfg["loader"]["batch_size"]) * ws
        print(f"World size={ws}  batch_per_rank={cfg['loader']['batch_size']}  effective_batch={eff_bsz}")
        print(f"Model parameters: total={_fmt_params(tot)}, trainable={_fmt_params(trainable)}", flush=True)

    # ----- DDP -----
    use_ddp = _is_dist()
    if use_ddp:
        ddp_kwargs = dict(device_ids=[device.index], output_device=device.index,
                          broadcast_buffers=False, find_unused_parameters=False)
        try:
            model = DDP(model, static_graph=True, **ddp_kwargs)
        except TypeError:
            model = DDP(model, **ddp_kwargs)

    # ----- loaders -----
    num_workers = int(cfg["loader"].get("num_workers", 4))
    batch_size  = int(cfg["loader"]["batch_size"])
    pin_mem     = bool(cfg["loader"].get("pin_memory", True))
    gen = torch.Generator().manual_seed(seed)

    train_sampler = DistributedSampler(
        train_ds, shuffle=True, drop_last=True, seed=seed
    ) if use_ddp else None
    val_sampler = DistributedSampler(
        val_ds, shuffle=False, drop_last=False, seed=seed
    ) if (use_ddp and val_ds is not None) else None

    def _dl_kwargs(nw:int):
        kw = dict(num_workers=nw, pin_memory=pin_mem,
                  persistent_workers=(nw>0),
                  worker_init_fn=_seed_worker, generator=gen, collate_fn=_collate)
        if nw>0: kw["prefetch_factor"]=4
        return kw

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not use_ddp), sampler=train_sampler,
        drop_last=True, **_dl_kwargs(num_workers)
    )
    val_dl = None
    if use_val_flag and val_ds is not None:
        val_dl = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
            drop_last=False, **_dl_kwargs(num_workers)
        )

    # ----- optim/sched -----
    optim = _make_optimizer(model.parameters(), cfg)
    sched, sched_monitor_cfg = _make_scheduler(optim, cfg)

    grad_clip = float(cfg["trainer"].get("grad_clip", 0.0))

    # ----- outputs -----
    out_dir = Path(cfg["trainer"].get("out_dir", "./results"))
    if _rank0(): out_dir.mkdir(parents=True, exist_ok=True)
    if _is_dist(): _barrier_safe(local_rank)
    tag_src = model.module if isinstance(model, DDP) else model
    run_dir = out_dir / tag_src.__class__.__name__
    if _rank0():
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config_snapshot.yaml", "w") as f: yaml.safe_dump(cfg, f)

    # ----- resume (optional) -----
    resume_cfg = cfg["trainer"].get("resume", None)
    resume_path = None
    resume_parent_run_id: Optional[str] = None
    if isinstance(resume_cfg, str) and len(resume_cfg) > 0:
        resume_path = resume_cfg
    elif resume_cfg is True:
        resume_path = str(run_dir / "checkpoint.last.pth")
    if resume_arg:  # CLI has priority
        resume_path = resume_arg

    start_epoch = 1
    best_metric = math.inf  # updated after monitor mode is finalised

    # ----- CSV logging -----
    metrics_cfg = cfg["trainer"].get("metrics", {"mae": True, "psnr": True})
    want_mae  = bool(metrics_cfg.get("mae", True))
    want_psnr = bool(metrics_cfg.get("psnr", True))

    csv_path = run_dir / "metrics.csv"
    if _rank0():
        header = ["epoch", "split", "mse", "rmse", "lr"]
        if want_mae:  header.append("mae")
        if want_psnr: header.append("psnr")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # ----- initial shape check and CUDA cache clear -----
    assert x0.shape[-2:] == (H, W) and y0.shape[-2:] == (H, W)
    if device.type == "cuda": torch.cuda.empty_cache()

    # ----- early stopping configuration -----
    default_split = "val" if (use_val_flag and val_dl is not None) else "train"
    default_mode  = "min"
    monitor_split_sched = (sched_monitor_cfg or {"split": default_split}).get("split", default_split)
    monitor_mode_sched  = (sched_monitor_cfg or {"mode": default_mode}).get("mode", default_mode)
    es_cfg = cfg["trainer"].get("early_stop", {})
    monitor_split_es = es_cfg.get("split", default_split)
    monitor_mode_es  = es_cfg.get("mode", default_mode)
    es_enabled  = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 30))
    es_min_delta= float(es_cfg.get("min_delta", 0.0))
    best_metric = math.inf if monitor_mode_es == "min" else -math.inf
    def _better(new, best, mode, delta):
        return (new < best - delta) if mode == "min" else (new > best + delta)
    es_bad = 0

    # ----- SIGINT handler to save interrupt checkpoint -----
    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    # ----- optional resume load (after run_dir exists) -----
    if resume_path and Path(resume_path).is_file():
        state = torch.load(resume_path, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        if sched and state.get("sched"): sched.load_state_dict(state["sched"])
        if state.get("scaler") and scaler.is_enabled(): scaler.load_state_dict(state["scaler"])
        best_metric = state.get("best_metric", best_metric)
        start_epoch = int(state.get("epoch", 0)) + 1
        if _rank0(): print(f"Resumed from {resume_path} at epoch {start_epoch}")
        resume_parent_run_id = state.get("mlflow_parent_run_id")
        if resume_parent_run_id:
            mlflow_parent_run_id = resume_parent_run_id
            if mlflow_active:
                try:
                    if mlflow_run_id and mlflow_run_id != resume_parent_run_id:
                        mlflow.set_tags({"mlflow.parentRunId": resume_parent_run_id})
                except Exception as e:
                    print(f"MLflow parent tag failed: {e}", flush=True)

    # ----- run-level JSON (initial write, rank-0 only) -----
    run_json_path = run_dir / "run.json"
    if _rank0():
        tot_params, trainable_params = _count_params(tag_src)
        env_blob = {
            "started_utc": _now_utc_iso(),
            "hostname": socket.gethostname(),
            "platform": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            },
            "distributed": {
                "enabled": _is_dist(),
                "backend": ("nccl" if _is_dist() else None),
                "world_size": int(os.environ.get("WORLD_SIZE", "1")) if _is_dist() else 1,
                "local_rank": int(os.environ.get("LOCAL_RANK","0"))
            },
            "device": str(device),
            "seed": seed,
            "deterministic": bool(deterministic),
            "model": {
                "class": tag_src.__class__.__name__,
                "total_params": int(tot_params),
                "trainable_params": int(trainable_params)
            },
            "data": {
                "paths": cfg["paths"],
                "image_shape": {"C": int(x0.shape[0]), "H": int(H), "W": int(W)},
            },
            "trainer": {
                "epochs": int(cfg["trainer"]["epochs"]),
                "amp": cfg["trainer"].get("amp", {}),
                "channels_last": bool(cfg["trainer"].get("channels_last", False)),
                "early_stop": cfg["trainer"].get("early_stop", {}),
                "metrics_cfg": cfg["trainer"].get("metrics", {})
            },
            "optim": cfg.get("optim", {}),
            "sched": cfg.get("sched", {}),
            "conditioning": cfg.get("conditioning", {}),
            "resume": {
                "requested": cfg["trainer"].get("resume", None),
                "cli": (resume_arg if resume_arg is not None else None),
                "loaded_path": (resume_path if resume_path and Path(resume_path).is_file() else None),
                "start_epoch": int(start_epoch)
            },
            "outputs": {
                "run_dir": str(run_dir),
                "csv_path": str((run_dir / "metrics.csv")),
                "plot_path": str((run_dir / "learning_curve.png")),
                "checkpoint_last": str((run_dir / "checkpoint.last.pth")),
                "checkpoint_best": str((run_dir / "checkpoint.best.pth"))
            },
            "mlflow": {
                "enabled": bool(mlflow_active),
                "run_id": mlflow_run_id,
                "parent_run_id": mlflow_parent_run_id,
            },
            "status": {"state": "running"}
        }
        run_json_path.write_text(json.dumps(env_blob, indent=2))

    # ----- per-trajectory totals for coverage (rank-0 computes; others keep None) -----
    def _pairs_per_gid_from_dataset(ds):
        if hasattr(ds, "items"):
            return Counter(g for g, _ in ds.items)
        if isinstance(ds, Subset) and hasattr(ds.dataset, "items"):
            base_items = ds.dataset.items
            idxs = ds.indices
            return Counter(base_items[i][0] for i in idxs)
        return None

    total_pairs_by_gid = _pairs_per_gid_from_dataset(train_ds) if _rank0() else None

    # ----- training loop -----
    epochs = int(cfg["trainer"]["epochs"])
    hist_epochs: List[int] = []; hist_train: List[float] = []; hist_val: List[float] = []
    flow_matcher = None
    if model_family == "flow_matching":
        if ConditionalFlowMatcher is None:
            raise ImportError("torchcfm is required for flow_matching but is not installed.")
        flow_matcher = ConditionalFlowMatcher(sigma=fm_sigma)

    for epoch in range(start_epoch, epochs + 1):
        if _is_dist():
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            if val_sampler   is not None: val_sampler.set_epoch(epoch)

        # Train
        model.train()
        se_sum, elem_count = 0.0, 0
        mae_sum = 0.0 if want_mae else None
        t0 = time.time()
        # per-step logging accumulators
        log_se_sum, log_elem_count = 0.0, 0
        log_mae_sum = 0.0 if want_mae else None
        num_steps = len(train_dl) if hasattr(train_dl, "__len__") else None

        # per-gid accumulators for this epoch
        gid_stats_local = defaultdict(lambda: [0.0, 0])  # gid -> [sum_mse, count]
        seen_pairs_local = set()                         # set of (gid, pair_index)
        train_task_metric_sum: Dict[str, float] = {}
        train_task_metric_count: Dict[str, int] = {}

        for step_idx, batch in enumerate(train_dl, start=1):
            x, y, cond = _prepare_batch(batch, device, cond_cfg, use_chlast)

            # Optional spatial weights from dataset
            weight = batch.get("weight", None)
            if weight is not None:
                weight = weight.to(device, non_blocking=True)
                if use_chlast and weight.dim() == 4:
                    weight = weight.contiguous(memory_format=torch.channels_last)

            optim.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                if model_family == "diffusion":
                    if noise_schedule_obj is None or timestep_sampler_obj is None:
                        raise RuntimeError("Diffusion schedule/sampler not initialised.")
                    t = timestep_sampler_obj.sample(batch_size=x.shape[0])
                    if not torch.is_tensor(t):
                        t = torch.tensor(t, device=device, dtype=torch.long)
                    else:
                        t = t.to(device=device, dtype=torch.long)
                    x_noisy, eps = _q_sample(y, t, noise_schedule_obj)
                    region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                    if cond is not None:
                        pred = model(x_noisy, t, cond, region_info=region_info)
                    else:
                        pred = model(x_noisy, t, region_info=region_info)
                    loss = diffusion_loss_fn(pred, eps, target=y, region_info=region_info)
                    metric_target = eps
                elif model_family == "flow_matching":
                    if flow_matcher is None:
                        raise RuntimeError("Flow matcher not initialised.")
                    t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x, x1=y)
                    t_match = t_match.to(device).view(-1, 1)
                    x_t = x_t.to(device)
                    u_t = u_t.to(device)
                    cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                    pred = model(x_t, cond_t)
                    loss = F.mse_loss(pred, u_t, reduction="mean")
                    metric_target = u_t
                else:
                    pred = (model(x, cond) if cond is not None else model(x))
                    dataset_weight = weight if (use_weight_loss and weight is not None) else None
                    loss = surrogate_loss_fn(pred, y, weight=dataset_weight)
                    metric_target = y
                    metric_owner = model.module if isinstance(model, DDP) else model
                    if hasattr(metric_owner, "compute_metrics"):
                        try:
                            task_metrics = metric_owner.compute_metrics(pred.detach(), metric_target.detach())  # type: ignore[arg-type]
                        except Exception:
                            task_metrics = {}
                        for name, val in task_metrics.items():
                            train_task_metric_sum[name] = train_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                            train_task_metric_count[name] = train_task_metric_count.get(name, 0) + 1

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")

            # Backward + optimiser step
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optim.step()

            # --- Metrics: always unweighted MSE/MAE for logging ---
            with torch.no_grad():
                mse_batch = F.mse_loss(pred, metric_target, reduction="mean")
                if want_mae:
                    mae_batch = F.l1_loss(pred, metric_target, reduction="mean")

                elems = metric_target.numel()
                se_sum += float(mse_batch.detach().cpu()) * elems
                elem_count += elems
                if want_mae:
                    mae_sum += float(mae_batch.detach().cpu()) * elems
                log_se_sum += float(mse_batch.detach().cpu()) * elems
                log_elem_count += elems
                if want_mae and log_mae_sum is not None:
                    log_mae_sum += float(mae_batch.detach().cpu()) * elems

                # per-sample unweighted MSE for gid tracking
                ps_mse = (pred.detach() - metric_target.detach())
                ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()

            for g, k, m in zip(batch["gid"], batch["pair_index"], ps_mse):
                gid_stats_local[g][0] += float(m)
                gid_stats_local[g][1] += 1
                seen_pairs_local.add((g, int(k)))

            # Optional per-step logging (rank-0 only)
            if log_interval > 0 and _rank0() and (step_idx % log_interval == 0):
                int_mse = log_se_sum / max(log_elem_count, 1)
                int_rmse = math.sqrt(max(int_mse, 0.0))
                msg = f"epoch={epoch} step={step_idx}"
                if num_steps:
                    msg += f"/{num_steps}"
                msg += f" train_rmse={int_rmse:.6f}"
                if want_mae and log_mae_sum is not None:
                    int_mae = log_mae_sum / max(log_elem_count, 1)
                    msg += f" train_mae={int_mae:.6f}"
                msg += f" lr={optim.param_groups[0]['lr']:.6g}"
                print(msg, flush=True)
                log_se_sum, log_elem_count = 0.0, 0
                if want_mae:
                    log_mae_sum = 0.0


        if device.type == "cuda": torch.cuda.synchronize()
        se_sum_g, elem_count_g = _allreduce_sum_count(device, se_sum, elem_count)
        train_mse = se_sum_g / max(elem_count_g, 1)
        train_rmse = math.sqrt(max(train_mse, 0.0))
        train_mae = None
        if want_mae:
            mae_sum_g, _ = _allreduce_sum_count(device, mae_sum, elem_count)
            train_mae = mae_sum_g / max(elem_count_g, 1)

        # Gather per-gid stats and seen pairs
        gid_stats_all = _all_gather_object(dict(gid_stats_local))
        seen_sets_all = _all_gather_object(seen_pairs_local)
        if _rank0():
            gid_stats_merged = defaultdict(lambda: [0.0, 0])
            for d in gid_stats_all:
                for g, (s, c) in d.items():
                    gid_stats_merged[g][0] += float(s)
                    gid_stats_merged[g][1] += int(c)
            seen_all = set().union(*seen_sets_all)

        # Validate
        val_mse: Optional[float] = None
        val_rmse: Optional[float] = None
        val_mae: Optional[float] = None
        val_gid_stats_merged = None
        val_task_metric_sum: Dict[str, float] = {}
        val_task_metric_count: Dict[str, int] = {}
        with torch.inference_mode():
            if use_val_flag and val_dl is not None:
                model.eval()
                v_se_sum, v_elem_count = 0.0, 0
                v_mae_sum = 0.0 if want_mae else None
                v_gid_stats_local = defaultdict(lambda: [0.0, 0]) if model_family != "diffusion" else None
                for batch in val_dl:
                    x, y, cond = _prepare_batch(batch, device, cond_cfg, use_chlast)
                    if model_family == "diffusion":
                        if noise_schedule_obj is None or timestep_sampler_obj is None:
                            raise RuntimeError("Diffusion schedule/sampler not initialised.")
                        t = timestep_sampler_obj.sample(batch_size=y.shape[0])
                        if not torch.is_tensor(t):
                            t = torch.tensor(t, device=device, dtype=torch.long)
                        else:
                            t = t.to(device=device, dtype=torch.long)
                        x_noisy, eps = _q_sample(y, t, noise_schedule_obj)
                        region_info = region_selector_obj(batch, t) if region_selector_obj is not None else None
                        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                            if cond is not None:
                                pred = model(x_noisy, t, cond, region_info=region_info)
                            else:
                                pred = model(x_noisy, t, region_info=region_info)
                            vloss = F.mse_loss(pred, eps)
                            if want_mae:
                                vmae = F.l1_loss(pred, eps)
                        elems = eps.numel()
                        v_se_sum += float(vloss.detach().cpu()) * elems
                        if want_mae and v_mae_sum is not None:
                            v_mae_sum += float(vmae.detach().cpu()) * elems
                        v_elem_count += elems
                    elif model_family == "flow_matching":
                        if flow_matcher is None:
                            raise RuntimeError("Flow matcher not initialised.")
                        t_match, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(x0=x, x1=y)
                        t_match = t_match.to(device).view(-1, 1)
                        x_t = x_t.to(device)
                        u_t = u_t.to(device)
                        cond_t = torch.cat([cond, t_match], dim=1) if cond is not None else t_match
                        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                            pred = model(x_t, cond_t)
                            vloss = F.mse_loss(pred, u_t)
                            if want_mae:
                                vmae = F.l1_loss(pred, u_t)
                        elems = u_t.numel()
                        v_se_sum += float(vloss.detach().cpu()) * elems
                        if want_mae and v_mae_sum is not None:
                            v_mae_sum += float(vmae.detach().cpu()) * elems
                        v_elem_count += elems
                        ps_mse = (pred.detach() - u_t.detach())
                        ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                        for g, m in zip(batch["gid"], ps_mse):
                            v_gid_stats_local[g][0] += float(m)
                            v_gid_stats_local[g][1] += 1
                    else:
                        with autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                            pred = (model(x, cond) if cond is not None else model(x))
                            vloss = F.mse_loss(pred, y)
                            if want_mae:
                                vmae = F.l1_loss(pred, y)
                            metric_owner = model.module if isinstance(model, DDP) else model
                            if hasattr(metric_owner, "compute_metrics"):
                                try:
                                    task_metrics = metric_owner.compute_metrics(pred.detach(), y.detach())  # type: ignore[arg-type]
                                except Exception:
                                    task_metrics = {}
                                for name, val in task_metrics.items():
                                    val_task_metric_sum[name] = val_task_metric_sum.get(name, 0.0) + float(val.detach().cpu())
                                    val_task_metric_count[name] = val_task_metric_count.get(name, 0) + 1
                        elems = y.numel()
                        v_se_sum += float(vloss.detach().cpu()) * elems
                        if want_mae and v_mae_sum is not None:
                            v_mae_sum += float(vmae.detach().cpu()) * elems
                        v_elem_count += elems

                        ps_mse = (pred.detach() - y.detach())
                        ps_mse = (ps_mse * ps_mse).flatten(1).mean(1).cpu().tolist()
                        for g, m in zip(batch["gid"], ps_mse):
                            v_gid_stats_local[g][0] += float(m)
                            v_gid_stats_local[g][1] += 1

                v_se_sum_g, v_elem_count_g = _allreduce_sum_count(device, v_se_sum, v_elem_count)
                val_mse = v_se_sum_g / max(v_elem_count_g, 1)
                val_rmse = math.sqrt(max(val_mse, 0.0))
                if want_mae and v_mae_sum is not None:
                    v_mae_sum_g, _ = _allreduce_sum_count(device, v_mae_sum, v_elem_count)
                    val_mae = v_mae_sum_g / max(v_elem_count_g, 1)

                if v_gid_stats_local is not None:
                    v_gid_stats_all = _all_gather_object(dict(v_gid_stats_local))
                    if _rank0():
                        val_gid_stats_merged = defaultdict(lambda: [0.0, 0])
                        for d in v_gid_stats_all:
                            for g, (s, c) in d.items():
                                val_gid_stats_merged[g][0] += float(s)
                                val_gid_stats_merged[g][1] += int(c)

        train_task_avg = {
            k: v / max(train_task_metric_count.get(k, 1), 1) for k, v in train_task_metric_sum.items()
        }
        val_task_avg = {
            k: v / max(val_task_metric_count.get(k, 1), 1) for k, v in val_task_metric_sum.items()
        } if val_task_metric_sum else {}

        # Scheduler step
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = train_mse if monitor_split_sched == "train" else (val_mse if val_mse is not None else train_mse)
                sched.step(metric)
            else:
                sched.step()


        lr_val = optim.param_groups[0]["lr"]

        # Epoch CSV logging (global means)
        if _rank0():
            row_train = [epoch, "train", f"{train_mse:.8f}", f"{train_rmse:.8f}", f"{lr_val:.6g}"]
            if want_mae:  row_train.append(f"{train_mae:.8f}" if train_mae is not None else "")
            if want_psnr: row_train.append("" if train_mse <= 0 else f"{(-10.0*math.log10(train_mse)):.4f}")  # PSNR from MSE
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(row_train)
                if val_mse is not None and val_rmse is not None:
                    row_val = [epoch, "val", f"{val_mse:.8f}", f"{val_rmse:.8f}", f"{lr_val:.6g}"]
                    if want_mae:  row_val.append(f"{val_mae:.8f}" if val_mae is not None else "")
                    if want_psnr: row_val.append("" if val_mse <= 0 else f"{(-10.0*math.log10(val_mse)):.4f}")
                    w.writerow(row_val)

                # Per-gid train means
                for g, (s, c) in sorted(gid_stats_merged.items()):
                    mean_mse_g = s / max(c, 1)
                    mean_rmse_g = math.sqrt(max(mean_mse_g, 0.0))
                    row = [epoch, f"train:{g}", f"{mean_mse_g:.8f}", f"{mean_rmse_g:.8f}", f"{lr_val:.6g}"]
                    if want_mae: row.append("")  # per-gid MAE not computed
                    if want_psnr: row.append("")
                    w.writerow(row)

                # Per-gid val means
                if val_gid_stats_merged is not None:
                    for g, (s, c) in sorted(val_gid_stats_merged.items()):
                        mean_mse_g = s / max(c, 1)
                        mean_rmse_g = math.sqrt(max(mean_mse_g, 0.0))
                        row = [epoch, f"val:{g}", f"{mean_mse_g:.8f}", f"{mean_rmse_g:.8f}", f"{lr_val:.6g}"]
                        if want_mae: row.append("")
                        if want_psnr: row.append("")
                        w.writerow(row)

                # Coverage per gid (fraction of pairs seen this epoch). Stored in MSE column.
                if total_pairs_by_gid is not None:
                    cov_counts = Counter(g for g, _ in seen_all)
                    for g, tot in sorted(total_pairs_by_gid.items()):
                        cov = cov_counts.get(g, 0)
                        frac = cov / max(tot, 1)
                        row = [epoch, f"coverage:{g}", f"{frac:.8f}", "", f"{lr_val:.6g}"]
                        if want_mae: row.append("")
                        if want_psnr: row.append("")
                        w.writerow(row)

            dt = time.time() - t0
            msg = f"epoch={epoch} train_mse={train_mse:.6f} train_rmse={train_rmse:.6f}"
            if val_mse is not None and val_rmse is not None:
                msg += f" val_mse={val_mse:.6f} val_rmse={val_rmse:.6f}"
            if train_task_avg:
                extras = " ".join(f"{k}={v:.6f}" for k, v in sorted(train_task_avg.items()))
                msg += f" task[{extras}]"
            if val_task_avg:
                extras = " ".join(f"{k}={v:.6f}" for k, v in sorted(val_task_avg.items()))
                msg += f" val_task[{extras}]"
            msg += f" dt={dt:.2f}s"
            print(msg, flush=True)
            if mlflow_active:
                metrics = {"train_rmse": train_rmse, "train_mse": train_mse}
                if train_mae is not None:
                    metrics["train_mae"] = train_mae
                if val_rmse is not None:
                    metrics["val_rmse"] = val_rmse
                    metrics["val_mse"] = val_mse
                if val_mae is not None:
                    metrics["val_mae"] = val_mae
                for k, v in train_task_avg.items():
                    metrics[f"train_{k}"] = v
                for k, v in val_task_avg.items():
                    metrics[f"val_{k}"] = v
                try:
                    mlflow.log_metrics(metrics, step=epoch)
                except Exception as e:
                    print(f"MLflow metric logging failed: {e}", flush=True)

        # Early stopping tracking
        metric_for_es = train_mse if monitor_split_es == "train" else (val_mse if val_mse is not None else train_mse)
        improved = _better(metric_for_es, best_metric, monitor_mode_es, es_min_delta)

        if improved:
            best_metric = metric_for_es
            es_bad = 0
        else:
            es_bad += 1

        # Checkpoints (rank-0)
        if _rank0():
            state = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optim": optim.state_dict(),
                "sched": (sched.state_dict() if sched is not None else None),
                "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
                "best_metric": best_metric,
                "config": cfg,
                "mlflow_parent_run_id": mlflow_parent_run_id,
            }
            torch.save(state, run_dir / "checkpoint.last.pth")
            if improved:
                torch.save(state, run_dir / "checkpoint.best.pth")

        # History for plot
        hist_epochs.append(epoch)
        hist_train.append(float(train_rmse))
        hist_val.append(float(val_rmse) if val_rmse is not None else float("nan"))

        # Early stop or interrupt
        if es_enabled and es_bad >= es_patience:
            if _rank0(): print(f"Early stopping at epoch {epoch} (patience {es_patience})", flush=True)
            break
        if interrupted["flag"]:
            if _rank0():
                torch.save({
                    "epoch": epoch,
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optim": optim.state_dict(),
                    "sched": (sched.state_dict() if sched is not None else None),
                    "scaler": (scaler.state_dict() if scaler.is_enabled() else None),
                    "best_metric": best_metric,
                    "config": cfg,
                    "mlflow_parent_run_id": mlflow_parent_run_id,
                }, run_dir / "checkpoint.interrupt.pth")
                print(f"Interrupted at epoch {epoch}. Saved checkpoint.interrupt.pth")
            break

    # Learning-curve plot
    if _rank0():
        png_path = run_dir / "learning_curve.png"
        plt.figure()
        plt.plot(hist_epochs, hist_train, label="train")
        if any(not math.isnan(v) for v in hist_val):
            plt.plot(hist_epochs, hist_val, label="val")
        plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout()
        plt.savefig(png_path, dpi=200); plt.close()
        print(f"CSV: {csv_path}")
        print(f"Plot: {png_path}")
        print(f"Checkpoints: {run_dir}")
        if mlflow_active:
            try:
                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(png_path))
                mlflow.log_artifact(str(run_dir / "config_snapshot.yaml"))
                mlflow.log_metrics({"best_metric": float(best_metric)})
            except Exception as e:
                print(f"MLflow artifact logging failed: {e}", flush=True)

    # ----- run-level JSON finalise (rank-0 only) -----
    if _rank0():
        try:
            blob = _load_json_safe(run_json_path) or {}
            if interrupted["flag"]:
                status = "interrupted"
            elif es_enabled and es_bad >= es_patience:
                status = "early_stopped"
            else:
                status = "completed"
            blob["ended_utc"] = _now_utc_iso()
            blob["status"] = {
                "state": status,
                "final_epoch": int(hist_epochs[-1] if len(hist_epochs)>0 else 0),
                "best_metric": float(best_metric),
                "monitor": {"split": monitor_split_es, "mode": monitor_mode_es}
            }
            blob["outputs"] = {
                "run_dir": str(run_dir),
                "csv_path": str(csv_path),
                "plot_path": str(run_dir / "learning_curve.png"),
                "checkpoint_last": str(run_dir / "checkpoint.last.pth"),
                "checkpoint_best": str(run_dir / "checkpoint.best.pth"),
                "checkpoint_interrupt": str(run_dir / "checkpoint.interrupt.pth") if interrupted["flag"] else None
            }
            blob["mlflow"] = {
                "enabled": bool(mlflow_active),
                "run_id": mlflow_run_id,
                "parent_run_id": mlflow_parent_run_id,
            }
            run_json_path.write_text(json.dumps(blob, indent=2))
        except Exception as e:
            print(f"run.json finalise failed: {e}", flush=True)

    if _is_dist():
        _barrier_safe(local_rank)
        dist.destroy_process_group()

    if mlflow_active and _rank0():
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"MLflow end_run failed: {e}", flush=True)


# ------------------- cli -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimal DDP trainer")
    ap.add_argument("-c","--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = ap.parse_args()
    main(args.config, resume_arg=args.resume)
