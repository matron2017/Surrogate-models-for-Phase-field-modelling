#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run bridge preflight + Slurm error scan before a run.")
    ap.add_argument("--config", type=Path, required=True, help="Bridge config yaml to run preflight checks against.")
    ap.add_argument("--logs-dir", type=Path, default=Path("logs/slurm"), help="Slurm logs directory to scan.")
    ap.add_argument("--hours", type=float, default=72.0, help="Lookback window in hours for error scan.")
    ap.add_argument(
        "--min-severity",
        choices=["INFO", "WARN", "ERROR"],
        default="WARN",
        help="Minimum severity to report from error scan.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARN as non-zero in preflight and abort on any non-pass.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cfg = str(args.config.expanduser().resolve())
    pre_cmd = [sys.executable, str(Path(__file__).resolve().parent / "bridge_preflight_check.py"), "--config", cfg]
    if args.strict:
        pre_cmd.append("--strict")
    print(f"[bridge_readiness] running preflight: {' '.join(pre_cmd)}")
    pre = subprocess.run(pre_cmd, check=False)
    if pre.returncode != 0:
        print(f"[bridge_readiness] preflight failed with status={pre.returncode}", flush=True)
        return pre.returncode

    report_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "report_bridge_errors.py"),
        "--logs-dir",
        str(args.logs_dir),
        "--hours",
        str(args.hours),
        "--min-severity",
        args.min_severity,
    ]
    print(f"[bridge_readiness] running error scan: {' '.join(report_cmd)}")
    rep = subprocess.run(report_cmd, check=False)
    if rep.returncode != 0:
        print(f"[bridge_readiness] error scan found blockers (status={rep.returncode})", flush=True)
        return rep.returncode

    print("[bridge_readiness] readiness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
