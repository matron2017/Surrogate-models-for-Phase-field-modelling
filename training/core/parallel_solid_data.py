#!/usr/bin/env python3
# DDP trainer for UAFNO_PreSkip_Full using PFPairDataset from pf_dataloader.py.
# Adds epoch timing, throughput, peak GPU memory, and CSV logging per run.
# Comments avoid second-person phrasing.

import os, sys, argparse, subprocess, time, importlib.util, csv, datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.backbones.uafno_cond import UAFNO_PreSkip_Full  # noqa: E402

# ------------------- utilities -------------------
def _import_symbol(py_path: str, symbol: str):
    spec = importlib.util.spec_from_file_location("pf_loader_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, symbol)

def _query_gpu(index: int):
    try:
        import pynvml as N
        N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(index)
        util = N.nvmlDeviceGetUtilizationRates(h).gpu
        mem  = N.nvmlDeviceGetMemoryInfo(h); name = N.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        N.nvmlShutdown()
        return {"idx": index, "util": float(util), "mem_gb": float(mem.used)/(1024**3), "name": name}
    except Exception:
        fields = "utilization.gpu,memory.used,name"
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu="+fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True
        ).stdout.strip().split(",")
        return {"idx": index, "util": float(out[0]), "mem_gb": float(out[1])/1024.0, "name": ",".join(out[2:]).strip()}

def _gather_and_print(local, world, rank):
    objs = [None] * world
    dist.all_gather_object(objs, local)
    if rank == 0:
        print(" rank | gpu | util% | mem(GB) | name")
        for r, s in enumerate(objs):
            print(f"{r:5d} | {s['idx']:3d} | {s['util']:5.1f} | {s['mem_gb']:.2f}   | {s['name']}")
        print("", flush=True)

def _count_params(m: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def _fmt_params(n: int) -> str:
    return f"{n:,d} ({n/1e6:.3f} M)"

def _script_dir() -> str:
    # Resolve directory where this script file resides. Fallback to CWD if __file__ is not set.
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    # run control
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)       # per-GPU
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-total", type=int, default=64)     # total items across all groups
    ap.add_argument("--limit-per-group", type=int, default=8)  # items per simulation group
    ap.add_argument("--lr", type=float, default=1e-3)

    # geometry for UAFNO
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--W", type=int, default=1024)

    # data paths and loader symbol
    ap.add_argument("--data-root", type=str,
        default="/scratch/project_2008261/rapid_solidification/data/rapid_solidification")
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--pf-loader", type=str,
        default="/scratch/project_2008261/rapid_solidification/training/core/pf_dataloader.py")
    ap.add_argument("--pf-class", type=str, default="PFPairDataset")

    # channel selection
    ap.add_argument("--input-ch", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--target-ch", type=int, nargs="+", default=[0, 1])

    args = ap.parse_args()
    assert args.H % 32 == 0 and args.W % 32 == 0

    # distributed
    dist.init_process_group(backend="nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # numerical modes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # model
    model = UAFNO_PreSkip_Full(
    n_channels=2, n_classes=2,
    in_factor=46,              # was 40 → must make hidden_size divisible by num_blocks=6
    cond_dim=2,
    afno_inp_shape=(args.H // 16, args.W // 16),
    afno_depth=6,
    afno_mlp_ratio=8.0,
    # afno_num_blocks=8,       # optional, only if the constructor supports it
    ).to(device)

    if rank == 0:
        tot, train = _count_params(model)
        print(f"Model parameters: total={_fmt_params(tot)}, trainable={_fmt_params(train)}", flush=True)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=True)

    # dataset
    h5_path = os.path.join(args.data_root, f"simulation_{args.split}.h5")
    PFCls = _import_symbol(args.pf_loader, args.pf_class)
    dataset = PFCls(
        h5_path=h5_path,
        input_channels=args.input_ch,
        target_channels=args.target_ch,
        limit_per_group=args.limit_per_group,
    )

    # trim to a small subset for a quick run
    if args.limit_total is not None and args.limit_total > 0:
        N = min(args.limit_total, len(dataset))
        dataset = Subset(dataset, list(range(N)))

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    # logging file (rank 0 only)
    log_dir = _script_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"uafno_ddp_{world}gpus_b{args.batch_size}_H{args.H}_W{args.W}_{ts}.csv"
    log_path = os.path.join(log_dir, log_name)
    if rank == 0:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","world","per_gpu_batch","global_batch","epochs",
                "H","W","limit_total","limit_per_group","num_workers",
                "epoch","steps","mean_loss","epoch_time_s","throughput_img_s","peak_mem_GB"
            ])

    # mapping and initial GPU stats
    mapping = f"[host={os.uname().nodename} rank={rank}] local_rank={local_rank} device={torch.cuda.current_device()} name={torch.cuda.get_device_name()}"
    objs = [None] * world; dist.all_gather_object(objs, mapping)
    if rank == 0: print("\n".join(objs), flush=True)
    _gather_and_print(_query_gpu(torch.cuda.current_device()), world, rank)

    # shape probe from first batch
    first_batch = next(iter(loader))
    x0 = first_batch["input"]; y0 = first_batch["target"]; c0 = first_batch["cond"]
    if rank == 0:
        print(f"Batch shapes: input={tuple(x0.shape)} target={tuple(y0.shape)} cond={tuple(c0.shape)}", flush=True)
        print("Meaning of cond: [Δt_norm, G_norm]", flush=True)
    assert x0.ndim == 4 and x0.shape[1] == 2 and x0.shape[2:] == (args.H, args.W)
    assert y0.ndim == 4 and y0.shape[1] == 2 and y0.shape[2:] == (args.H, args.W)
    assert c0.ndim == 2 and c0.shape[1] == 2

    # training
    model.train()
    global_batch = args.batch_size * world
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        # reset and time epoch
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); t0 = time.time()
        samples_global = 0
        mean_loss = torch.zeros(1, device=device)

        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            c = batch["cond"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=True):
                pred = model(x, c)
                loss = F.mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            mean_loss += loss.detach()
            samples_global += x.size(0) * world

        # average loss across ranks
        steps = torch.tensor(len(loader), device=device, dtype=torch.float32)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps, op=dist.ReduceOp.SUM)
        mean_loss = (mean_loss / steps).item()

        torch.cuda.synchronize(device)
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        if rank == 0:
            thr = samples_global / dt if dt > 0 else float("nan")
            print(f"epoch={epoch} mean_loss={mean_loss:.6f} time={dt:.2f}s throughput={thr:.1f} img/s", flush=True)
            # append row
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    world, args.batch_size, global_batch, args.epochs,
                    args.H, args.W, args.limit_total, args.limit_per_group, args.num_workers,
                    epoch, int(len(loader)), f"{mean_loss:.6f}", f"{dt:.3f}", f"{thr:.3f}", f"{peak_gb:.3f}"
                ])

        _gather_and_print(_query_gpu(torch.cuda.current_device()), world, rank)

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()
    if rank == 0:
        print(f"CSV saved: {log_path}", flush=True)
        print("OK", flush=True)

if __name__ == "__main__":
    main()
