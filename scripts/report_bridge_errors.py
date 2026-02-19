#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ErrorHit:
    category: str
    severity: str
    pattern: str
    file: str
    line_no: int
    text: str


RULES = [
    {
        "category": "plotter",
        "severity": "ERROR",
        "label": "missing_timestep",
        "pattern": re.compile(r"Generative timestep is required"),
    },
    {
        "category": "watcher",
        "severity": "ERROR",
        "label": "watcher_kwarg_mismatch",
        "pattern": re.compile(r"_sync_latest_plots\(\) got an unexpected keyword"),
    },
    {
        "category": "watcher",
        "severity": "WARN",
        "label": "checkpoint_incomplete",
        "pattern": re.compile(r"PytorchStreamReader failed finding central directory"),
    },
    {
        "category": "infra",
        "severity": "WARN",
        "label": "runjson_incomplete_interrupt",
        "pattern": re.compile(r"SIGTERM|SIGINT|interrupted|interrupt"),
    },
    {
        "category": "runtime",
        "severity": "ERROR",
        "label": "traceback",
        "pattern": re.compile(r"Traceback \(most recent call last\):"),
    },
    {
        "category": "runtime",
        "severity": "ERROR",
        "label": "runtime_error",
        "pattern": re.compile(r"RuntimeError:"),
    },
    {
        "category": "runtime",
        "severity": "WARN",
        "label": "nan_guard",
        "pattern": re.compile(r"\[nan_guard\]"),
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan Slurm bridge logs for known failure signatures.")
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/scratch/project_2008261/pf_surrogate_modelling/logs/slurm"),
        help="Folder with .out/.err slurm logs.",
    )
    ap.add_argument(
        "--hours",
        type=float,
        default=72.0,
        help="Only include files modified within this many hours (0=all).",
    )
    ap.add_argument(
        "--include-out",
        action="store_true",
        help="Also scan .out files (default scans .err only).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output file for CI-friendly ingestion.",
    )
    ap.add_argument(
        "--min-severity",
        choices=["INFO", "WARN", "ERROR"],
        default="ERROR",
        help="Show matches at this severity or higher.",
    )
    return ap.parse_args()


def _recent_files(logs_dir: Path, hours: float, include_out: bool) -> List[Path]:
    if not logs_dir.exists():
        raise RuntimeError(f"Logs dir does not exist: {logs_dir}")
    suffixes = ["*.err"]
    if include_out:
        suffixes.append("*.out")
    files: List[Path] = []
    for suf in suffixes:
        files.extend(sorted(logs_dir.glob(suf)))
    if not hours or hours <= 0:
        return files
    cutoff = time.time() - hours * 3600.0
    return [f for f in files if f.stat().st_mtime >= cutoff]


def _scan_file(path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            out.append((i, line.rstrip("\n")))
    return out


def _severity_rank(sev: str) -> int:
    if sev == "ERROR":
        return 2
    if sev == "WARN":
        return 1
    return 0


def _passes_min_sev(sev: str, min_sev: str) -> bool:
    return _severity_rank(sev) >= _severity_rank(min_sev)


def _classify(text: str) -> Optional[ErrorHit]:
    for rule in RULES:
        if rule["pattern"].search(text):
            return ErrorHit(
                category=str(rule["category"]),
                severity=str(rule["severity"]),
                pattern=str(rule["label"]),
                file="",
                line_no=0,
                text=text,
            )
    return None


def main() -> int:
    args = parse_args()
    logs_dir = args.logs_dir.expanduser().resolve()
    files = _recent_files(logs_dir=logs_dir, hours=float(args.hours), include_out=bool(args.include_out))
    if not files:
        print(f"[bridge_error_report] no logs in {logs_dir} for the requested window")
        return 0

    hits: List[ErrorHit] = []
    for p in files:
        for line_no, line in _scan_file(p):
            hit = _classify(line)
            if hit is None:
                continue
            hit.file = str(p)
            hit.line_no = line_no
            if _passes_min_sev(hit.severity, args.min_severity):
                hits.append(hit)

    hits.sort(key=lambda x: (_severity_rank(x.severity), x.category, x.file, x.line_no), reverse=True)
    if not hits:
        print("[bridge_error_report] no matching issues found")
        return 0

    print(f"[bridge_error_report] scanned {len(files)} file(s) in {logs_dir}")
    print(f"[bridge_error_report] matches={len(hits)} min_severity={args.min_severity}")

    by_category: Dict[str, int] = {}
    for h in hits:
        by_category.setdefault(h.category, 0)
        by_category[h.category] += 1

    print("[bridge_error_report] summary:")
    for category, count in sorted(by_category.items()):
        print(f"  - {category}: {count}")

    for h in hits[:200]:
        print(f"[{h.severity}] {h.category}:{h.pattern} :: {Path(h.file).name}:{h.line_no} :: {h.text}")

    if args.json_out is not None:
        payload = {
            "logs_dir": str(logs_dir),
            "window_hours": float(args.hours),
            "min_severity": args.min_severity,
            "matches": [h.__dict__ for h in hits],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[bridge_error_report] wrote json: {args.json_out}")

    return 1 if any(h.severity == "ERROR" for h in hits) else 0


if __name__ == "__main__":
    sys.exit(main())
