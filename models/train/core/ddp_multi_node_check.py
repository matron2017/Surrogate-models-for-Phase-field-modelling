#!/usr/bin/env python3
"""
Minimal NCCL rendezvous test for multi-node, multi-GPU DDP runs.
Launch via torch.distributed.run so every rank prints its hostname/local rank,
performs an all-reduce, and exits. Mirrors the CSC Puhti/Mahti documentation.
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices are required for this multi-node check.")

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(
        f"[rank {rank:02d}/{world}] host={hostname} local_rank={local_rank} "
        f"device={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    payload = torch.tensor([float(rank)], device=device)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    print(f"[rank {rank:02d}] all_reduce sum={payload.item():.1f}", flush=True)

    torch.cuda.synchronize(device)
    time.sleep(2.0)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
