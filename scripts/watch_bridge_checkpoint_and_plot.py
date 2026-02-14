#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def _read_epoch(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1))


def _epoch_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return -1


def _sync_latest_plots(run_root: Path, latest_root: Path, max_epochs: int = 5) -> None:
    if max_epochs <= 0:
        return

    latest_root.mkdir(parents=True, exist_ok=True)
    for child in latest_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    epochs = sorted([p for p in run_root.glob("epoch_*") if p.is_dir()], key=_epoch_sort_key)
    for src in epochs[-max_epochs:]:
        dst = latest_root / src.name
        shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch checkpoint.last.pth and render endpoint plots on each update.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Model run directory (e.g. .../UNetFiLMAttn).")
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--indices", type=str, default="234,255,300")
    ap.add_argument("--bridge-nfe", type=int, default=20)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--latest-max-epochs", type=int, default=5, help="Keep latest N epoch folders mirrored under <out_dir>/latest.")
    ap.add_argument(
        "--checkpoint-stale-seconds",
        type=int,
        default=900,
        help="Emit warning if checkpoint.last.pth is missing for this long.",
    )
    ap.add_argument("--train-pid", type=int, default=0, help="Optional PID; watcher exits once this PID is gone.")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ckpt_last = run_dir / "checkpoint.last.pth"
    plot_script = Path(__file__).resolve().parent / "plot_bridge_endpoint_from_ckpt.py"

    requested_out_root = args.out_dir.expanduser().resolve()
    if requested_out_root.exists() and any(requested_out_root.iterdir()):
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_root = requested_out_root / f"run_{ts}"
        print(f"[watch] requested out dir not empty; using run folder: {out_root}", flush=True)
    else:
        out_root = requested_out_root
    out_root.mkdir(parents=True, exist_ok=True)
    latest_root = out_root / "latest"
    print(f"[watch] latest plots mirror: {latest_root}", flush=True)

    if args.checkpoint_stale_seconds <= 0:
        args.checkpoint_stale_seconds = 900

    last_mtime = -1.0
    seen_epochs = set()
    last_checkpoint_seen: float | None = None
    last_stale_warn = 0.0
    print(f"[watch] run_dir={run_dir}", flush=True)
    print(f"[watch] waiting for {ckpt_last}", flush=True)
    if not run_dir.exists():
        print(f"[watch] run dir does not exist yet: {run_dir}", flush=True)

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
            now = time.time()
            last_checkpoint_seen = now
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
                _sync_latest_plots(out_root=out_root, latest_root=latest_root, max_epochs=int(args.latest_max_epochs))
        else:
            now = time.time()
            if (
                last_checkpoint_seen is not None
                and now - last_checkpoint_seen >= args.checkpoint_stale_seconds
                and now - last_stale_warn >= args.checkpoint_stale_seconds
            ):
                print(
                    f"[watch] warning: checkpoint file {ckpt_last} has not appeared for "
                    f"{int(now - last_checkpoint_seen)}s",
                    flush=True,
                )
                last_stale_warn = now

        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
