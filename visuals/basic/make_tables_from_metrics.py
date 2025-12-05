#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tables_from_metrics.py
Build compact CSV and Markdown tables from one or more metrics.json files
produced by the evaluator.

Outputs for each input directory (containing metrics.json):
- metrics_overall.csv / .md
- metrics_per_channel.csv / .md

If multiple input paths are given, rows are stacked with an identifier column.
Comments avoid second-person phrasing.
"""

import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def _load_metrics(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        path = path / "metrics.json"
    with open(path, "r") as f:
        return json.load(f)

def _to_overall_row(m: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    row = {
        "run_id": run_id,
        "overall_mse": m["overall"]["mse"],
        "overall_rmse": m["overall"]["rmse"],
        "overall_mae": m["overall"]["mae"],
        "relL2_global": m.get("rel_l2", {}).get("global_dataset", None),
        "relL2_mean_over_samples": m.get("rel_l2", {}).get("mean_over_samples", None),
        "channels": m.get("counts", {}).get("channels", None),
        "elements_per_channel": m.get("counts", {}).get("elements_per_channel", None),
    }
    # Optionally include phase/concentration summaries when present
    for key in ("phase_summary", "concentration_summary"):
        if key in m and m[key].get("channel", None) is not None:
            row[f"{key}_ch"]   = m[key]["channel"]
            row[f"{key}_mse"]  = m[key]["mse"]
            row[f"{key}_rmse"] = m[key]["rmse"]
            row[f"{key}_mae"]  = m[key]["mae"]
            row[f"{key}_bias"] = m[key].get("bias", None)
            row[f"{key}_relL2_global"] = m[key].get("rel_l2_global", None)
    return row

def _to_per_channel_rows(m: Dict[str, Any], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pc in m.get("per_channel", []):
        rows.append({
            "run_id": run_id,
            "channel": pc["channel"],
            "mse": pc["mse"],
            "rmse": pc["rmse"],
            "mae": pc["mae"],
        })
    # Attach bias and relL2_global when available
    for key in ("phase_summary", "concentration_summary"):
        if key in m and m[key].get("channel", None) is not None:
            ch = m[key]["channel"]
            for r in rows:
                if r["channel"] == ch:
                    r["bias"] = m[key].get("bias", None)
                    r["relL2_global_channel"] = m[key].get("rel_l2_global", None)
    return rows

def _write_tables(df_overall: pd.DataFrame, df_ch: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    df_overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    df_ch.to_csv(out_dir / "metrics_per_channel.csv", index=False)
    # Markdown
    (out_dir / "metrics_overall.md").write_text(df_overall.to_markdown(index=False))
    (out_dir / "metrics_per_channel.md").write_text(df_ch.to_markdown(index=False))

def main():
    ap = argparse.ArgumentParser(description="Build tables from evaluator metrics.json")
    ap.add_argument("paths", nargs="+", help="metrics.json files or directories that contain them")
    ap.add_argument("-o", "--out", type=str, default="tables_out", help="Output directory for tables")
    args = ap.parse_args()

    rows_overall, rows_ch = [], []
    for p in args.paths:
        pth = Path(p)
        run_id = pth.stem if pth.is_file() else pth.name
        m = _load_metrics(pth)
        rows_overall.append(_to_overall_row(m, run_id))
        rows_ch.extend(_to_per_channel_rows(m, run_id))

    df_overall = pd.DataFrame(rows_overall)
    df_ch = pd.DataFrame(rows_ch)

    _write_tables(df_overall, df_ch, Path(args.out))

if __name__ == "__main__":
    main()
