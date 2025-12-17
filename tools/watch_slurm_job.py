#!/usr/bin/env python3
"""Local Slurm watcher that polls Puhti/Mahti via SSH until a job finishes."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Optional, Tuple

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
}


def run_ssh_command(host: str, remote_cmd: str) -> Tuple[int, str, str]:
    """Run a command on the remote host via ssh."""
    proc = subprocess.run(["ssh", host, remote_cmd], text=True, capture_output=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def get_active_state(host: str, jobid: str) -> Optional[str]:
    """Return squeue state if job is active; None otherwise."""
    cmd = f'squeue -h -j {jobid} -o "%T"'
    rc, out, _ = run_ssh_command(host, cmd)
    if rc == 0 and out:
        return out.splitlines()[0].strip() or None
    return None


def get_final_state(host: str, jobid: str) -> Optional[str]:
    """Return sacct-reported final job state if available."""
    cmd = (
        f'sacct -j {jobid} '
        '--format=JobID,State '
        '--parsable2 --noheader'
    )
    rc, out, _ = run_ssh_command(host, cmd)
    if rc != 0 or not out:
        return None

    main_state: Optional[str] = None
    batch_state: Optional[str] = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        if jid == jobid:
            main_state = state
        elif jid.startswith(jobid + ".batch"):
            batch_state = state
    return main_state or batch_state


def watch_job(host: str, jobid: str, interval: int) -> None:
    print(f"Monitoring job {jobid} on {host} (interval {interval}s)")
    last_state = None
    while True:
        active = get_active_state(host, jobid)
        if active:
            if active != last_state:
                print(f"[active] {jobid}: {active}")
                last_state = active
            time.sleep(interval)
            continue

        final = get_final_state(host, jobid)
        if final is None:
            print(f"[final?] {jobid} absent from squeue and sacct")
            print("\a", end="", flush=True)
            break

        print(f"[final] {jobid}: {final}")
        print("\a", end="", flush=True)
        if final.split("+")[0] in TERMINAL_STATES:
            break
        time.sleep(interval)

    print("Watcher exiting.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a Slurm job via SSH.")
    parser.add_argument("host", help="SSH alias (puhti, mahti, ...)")
    parser.add_argument("jobid", help="Slurm job ID")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        watch_job(args.host, args.jobid, args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
