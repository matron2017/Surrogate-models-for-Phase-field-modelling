#!/usr/bin/env python3
"""Make per-gradient shared cutoff using earliest touch frame.

Input: per-sim touch CSV (e.g., touch_frames_fixed34.csv)
Output:
  - grad-level CSV with earliest touch per thermal gradient
  - per-sim CSV with cutoff_frame set to earliest touch for that gradient
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _to_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared cutoff per gradient using earliest touch frame.")
    ap.add_argument("--touch-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--grad-csv", required=True, type=Path)
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with args.touch_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # collect earliest touch per gradient
    grad_best: Dict[str, Dict[str, object]] = {}
    for row in rows:
        grad = row.get("thermal_gradient", "")
        touch = _to_int(row.get("touch_frame"))
        if not grad:
            continue
        entry = grad_best.setdefault(grad, {"best_touch": None, "best_gid": None, "best_seed": None})
        if touch is None:
            continue
        if entry["best_touch"] is None or touch < entry["best_touch"]:
            entry["best_touch"] = touch
            entry["best_gid"] = row.get("gid")
            entry["best_seed"] = row.get("seed")

    # write grad-level CSV
    args.grad_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.grad_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["thermal_gradient", "best_touch", "best_gid", "best_seed"],
        )
        w.writeheader()
        for grad, entry in sorted(grad_best.items(), key=lambda kv: float(kv[0])):
            w.writerow(
                {
                    "thermal_gradient": grad,
                    "best_touch": entry["best_touch"] if entry["best_touch"] is not None else "",
                    "best_gid": entry["best_gid"] or "",
                    "best_seed": entry["best_seed"] or "",
                }
            )

    # write per-sim shared-cutoff CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    if "cutoff_frame" not in fieldnames:
        fieldnames.append("cutoff_frame")
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            grad = row.get("thermal_gradient", "")
            entry = grad_best.get(grad, {})
            shared = entry.get("best_touch")
            row = dict(row)
            row["cutoff_frame"] = shared if shared is not None else ""
            w.writerow(row)


if __name__ == "__main__":
    main()
