#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch


def _read_epoch(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch checkpoint.last.pth and render endpoint plots on each update.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Model run directory (e.g. .../UNetFiLMAttn).")
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--indices", type=str, default="234,255,300")
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--train-pid", type=int, default=0, help="Optional PID; watcher exits once this PID is gone.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ckpt_last = run_dir / "checkpoint.last.pth"
    plot_script = Path(__file__).resolve().parent / "plot_bridge_endpoint_from_ckpt.py"
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    last_mtime = -1.0
    seen_epochs = set()
    print(f"[watch] run_dir={run_dir}", flush=True)
    print(f"[watch] waiting for {ckpt_last}", flush=True)

    while True:
        if args.train_pid > 0:
            try:
                Path(f"/proc/{args.train_pid}").exists()
            except Exception:
                pass
            if not Path(f"/proc/{args.train_pid}").exists():
                print(f"[watch] train pid {args.train_pid} exited; stopping watcher", flush=True)
                break

        if ckpt_last.exists():
            mtime = ckpt_last.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                try:
                    epoch = _read_epoch(ckpt_last)
                except Exception as e:
                    print(f"[watch] failed reading epoch from {ckpt_last}: {e}", flush=True)
                    time.sleep(max(1, int(args.poll_seconds)))
                    continue
                if epoch in seen_epochs:
                    time.sleep(max(1, int(args.poll_seconds)))
                    continue
                seen_epochs.add(epoch)
                epoch_out = out_root / f"epoch_{epoch:04d}"
                cmd = [
                    args.python,
                    str(plot_script),
                    "--bridge-ckpt",
                    str(ckpt_last),
                    "--ae-ckpt",
                    str(args.ae_ckpt),
                    "--indices",
                    str(args.indices),
                    "--bridge-nfe",
                    str(int(args.bridge_nfe)),
                    "--out-dir",
                    str(epoch_out),
                    "--device",
                    str(args.device),
                ]
                print(f"[watch] epoch={epoch} plotting to {epoch_out}", flush=True)
                rc = subprocess.call(cmd)
                print(f"[watch] epoch={epoch} plot exit code={rc}", flush=True)

        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
