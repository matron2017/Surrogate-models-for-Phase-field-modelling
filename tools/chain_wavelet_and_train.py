#!/usr/bin/env python3
"""
Submit wavelet-weight precompute, wait for completion, then submit training.

Examples:
python tools/chain_wavelet_and_train.py \
  --wavelet-sbatch datapipes/wavelet_weights_a10000b50000.sh \
  --train-sbatch slurm/train_big_uafno.sh \
  --train-config configs/train/uafno_wavelet.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True)


def submit_sbatch(script: Path, extra_env: Optional[dict[str, str]] = None) -> str:
    env = None
    if extra_env:
        env = {**dict(Path.cwd().env if hasattr(Path.cwd(), "env") else {}), **extra_env}
    proc = _run(["sbatch", str(script)])
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    line = proc.stdout.strip()
    if not line.lower().startswith("submitted batch job"):
        raise RuntimeError(f"Unexpected sbatch output: {line}")
    return line.split()[-1]


def submit_sbatch_with_cfg(script: Path, cfg: Optional[Path]) -> str:
    if cfg is None:
        return submit_sbatch(script)
    env = {"CFG": str(cfg.resolve())}
    proc = subprocess.run(
        ["sbatch", str(script)],
        text=True,
        capture_output=True,
        env={**env, **dict(**{"PATH": subprocess.os.environ.get("PATH", "")})},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    line = proc.stdout.strip()
    if not line.lower().startswith("submitted batch job"):
        raise RuntimeError(f"Unexpected sbatch output: {line}")
    return line.split()[-1]


def active_state(jobid: str) -> Optional[str]:
    proc = _run(["squeue", "-h", "-j", jobid, "-o", "%T"])
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0]
    return None


def final_state(jobid: str) -> Optional[str]:
    proc = _run(["sacct", "-j", jobid, "--format=JobID,State", "--parsable2", "--noheader"])
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        if jid == jobid or jid.startswith(jobid + ".batch"):
            return state
    return None


def wait_for_job(jobid: str, interval: int = 30) -> str:
    last = None
    while True:
        act = active_state(jobid)
        if act:
            if act != last:
                print(f"[active] {jobid}: {act}")
                last = act
            time.sleep(interval)
            continue
        fin = final_state(jobid)
        if fin:
            print(f"[final] {jobid}: {fin}")
            return fin
        print(f"[waiting] {jobid} not in squeue/sacct, retrying...")
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Chain wavelet precompute then training via Slurm.")
    ap.add_argument("--wavelet-sbatch", required=True, type=Path, help="Path to wavelet sbatch script")
    ap.add_argument("--train-sbatch", required=True, type=Path, help="Path to training sbatch script")
    ap.add_argument("--train-config", type=Path, help="Training config to set via CFG env")
    ap.add_argument("--interval", type=int, default=60, help="Polling interval (seconds)")
    args = ap.parse_args()

    print(f"Submitting wavelet job: {args.wavelet_sbatch}")
    wave_jid = submit_sbatch(args.wavelet_sbatch)
    print(f"[wavelet] jobid={wave_jid}")

    fin = wait_for_job(wave_jid, interval=args.interval)
    if fin.split("+")[0] not in TERMINAL_STATES or fin.split("+")[0] != "COMPLETED":
        raise RuntimeError(f"Wavelet job {wave_jid} ended with state {fin}")

    print(f"Submitting training job: {args.train_sbatch}")
    train_jid = submit_sbatch_with_cfg(args.train_sbatch, args.train_config)
    print(f"[train] jobid={train_jid}")
    wait_for_job(train_jid, interval=args.interval)
    print("Done.")


if __name__ == "__main__":
    main()
