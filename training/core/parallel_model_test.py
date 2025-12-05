#!/usr/bin/env python3
# DDP trainer for UAFNO_PreSkip_Full on synthetic 2×1024×1024 data.
# Layout: 2 GPUs, per-GPU batch = 1. No micro-batching. Epoch-only loop.

import os, sys, argparse, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models.backbones.uafno_cond import UAFNO_PreSkip_Full  # noqa: E402

# ------------------- synthetic dataset -------------------
class RandomPF2ChDataset(Dataset):
    # Two channels: [phase field, Cu concentration]. Conditioning: [thermal_gradient, delta_t].
    def __init__(self, length: int, H: int, W: int, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(length, 2, H, W, generator=g, dtype=torch.float32)
        self.y = torch.randn(length, 2, H, W, generator=g, dtype=torch.float32)
        self.c = torch.randn(length, 2, generator=g, dtype=torch.float32)

    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return {"input": self.x[i], "target": self.y[i], "cond": self.c[i]}

# ------------------- optional GPU stats -------------------
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


# ------------------- parameter counting -------------------
def _count_params(m: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def _fmt_params(n: int) -> str:
    return f"{n:,d} ({n/1e6:.3f} M)"

# ------------------- training -------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)   # per-GPU
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--dataset-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    assert args.H % 32 == 0 and args.W % 32 == 0

    dist.init_process_group(backend="nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = UAFNO_PreSkip_Full(
        n_channels=2, n_classes=2, in_factor=48, cond_dim=2,
        afno_inp_shape=(args.H // 16, args.W // 16), afno_depth=6, afno_mlp_ratio=8.0
    ).to(device)
    if rank == 0:
        tot, train = _count_params(model)
        print(f"Model parameters: total={_fmt_params(tot)}, trainable={_fmt_params(train)}", flush=True)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=True)

    dataset = RandomPF2ChDataset(args.dataset_length, args.H, args.W, seed=42)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    # one print of mapping and initial GPU stats
    mapping = f"[host={os.uname().nodename} rank={rank}] local_rank={local_rank} device={torch.cuda.current_device()} name={torch.cuda.get_device_name()}"
    objs = [None] * world; dist.all_gather_object(objs, mapping)
    if rank == 0: print("\n".join(objs), flush=True)
    _gather_and_print(_query_gpu(torch.cuda.current_device()), world, rank)

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
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

        # end-of-epoch light report
        _gather_and_print(_query_gpu(torch.cuda.current_device()), world, rank)

    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()
    if rank == 0:
        print("OK", flush=True)

if __name__ == "__main__":
    main()
