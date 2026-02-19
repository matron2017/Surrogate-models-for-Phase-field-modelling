#!/usr/bin/env python3
"""
Build one composite PNG per (frame, channel) from eval output folders and
optionally remove the old per-view individual PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


VIEW_ORDER: Sequence[Tuple[str, str]] = (
    ("input", "Input"),
    ("gt", "Ground Truth"),
    ("pred", "Prediction"),
    ("residual", "Residual"),
)


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _img_read(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _collect_panels(
    root: Path,
    out_item: Dict,
    ch: int,
    *,
    merge_input_gt: bool = False,
) -> Tuple[List[Tuple[str, str, Path]], List[Path]]:
    panels: List[Tuple[str, str, Path]] = []
    sources: List[Path] = []
    ch_key = str(ch)
    if merge_input_gt:
        p_input = None
        p_gt = None
        entry_in = out_item.get("input")
        if isinstance(entry_in, dict):
            rel = entry_in.get(ch_key)
            if rel:
                p = (root / rel).resolve()
                if p.exists():
                    p_input = p
                    sources.append(p)
        entry_gt = out_item.get("gt")
        if isinstance(entry_gt, dict):
            rel = entry_gt.get(ch_key)
            if rel:
                p = (root / rel).resolve()
                if p.exists():
                    p_gt = p
                    sources.append(p)
        if p_input is not None or p_gt is not None:
            panels.append(
                (
                    "input_gt",
                    "Input / GT\n(Autoencoder: same image in -> same image out)",
                    p_input if p_input is not None else p_gt,
                )
            )
        for key, label in (("pred_ae", "Output (AE pred)"), ("residual_ae", "Residual (Output - GT)")):
            src_key = "pred" if key == "pred_ae" else "residual"
            entry = out_item.get(src_key)
            if not isinstance(entry, dict):
                continue
            rel = entry.get(ch_key)
            if not rel:
                continue
            p = (root / rel).resolve()
            if p.exists():
                panels.append((key, label, p))
                sources.append(p)
        return panels, sources
    for key, label in VIEW_ORDER:
        entry = out_item.get(key)
        if not isinstance(entry, dict):
            continue
        rel = entry.get(ch_key)
        if not rel:
            continue
        p = (root / rel).resolve()
        if p.exists():
            panels.append((key, label, p))
            sources.append(p)
    return panels, sources


def _stats_text(field_stats: Dict, key: str, ch: int) -> str:
    ch_stats = field_stats.get(str(ch), {}) if isinstance(field_stats, dict) else {}
    cur = ch_stats.get(key, {}) if isinstance(ch_stats, dict) else {}
    if not isinstance(cur, dict):
        return ""
    try:
        vmin = float(cur["min"])
        vmax = float(cur["max"])
        mean = float(cur["mean"])
        return f"min={vmin:.4g}\nmax={vmax:.4g}\nmean={mean:.4g}"
    except Exception:
        return ""


def _render_one(
    out_path: Path,
    *,
    gid: str,
    frame: int,
    ch: int,
    panels: Sequence[Tuple[str, str, Path]],
    field_stats: Dict,
    dpi: int,
) -> None:
    stats_key_map = {
        "input_gt": "gt",
        "pred_ae": "pred",
        "residual_ae": "residual",
    }
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 5.1), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (key, title, p) in zip(axes, panels):
        arr = _img_read(p)
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = _stats_text(field_stats, stats_key_map.get(key, key), ch)
        if txt:
            ax.text(
                0.02,
                0.02,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.70),
            )
    fig.suptitle(
        f"sim={gid} | frame={frame} | channel={ch}",
        fontsize=14,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _metadata_dirs(eval_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(eval_root.iterdir()):
        if p.is_dir() and (p / "metadata.json").exists():
            out.append(p)
    return out


def _process_dir(
    subdir: Path,
    *,
    repo_root: Path,
    dpi: int,
    delete_sources: bool,
    merge_input_gt: bool = False,
) -> Dict:
    meta_path = subdir / "metadata.json"
    meta = _load_json(meta_path)
    outputs = list(meta.get("outputs", []))

    generated: List[str] = []
    all_sources: List[Path] = []
    for item in outputs:
        gid = str(item.get("gid", "unknown"))
        frame = int(item.get("frame", -1))
        channels = [int(c) for c in item.get("channels", [])]
        field_stats = item.get("field_stats", {}) if isinstance(item, dict) else {}
        for ch in channels:
            panels, sources = _collect_panels(repo_root, item, ch, merge_input_gt=merge_input_gt)
            if not panels:
                continue
            out_name = f"composite_{gid}_frame{frame:04d}_ch{ch}.png"
            out_path = subdir / out_name
            _render_one(out_path, gid=gid, frame=frame, ch=ch, panels=panels, field_stats=field_stats, dpi=dpi)
            generated.append(str(out_path))
            all_sources.extend(sources)

    removed: List[str] = []
    removed_pct: List[str] = []
    if delete_sources:
        seen = set()
        for p in all_sources:
            if p in seen:
                continue
            seen.add(p)
            # Keep only newly produced composites/metadata; delete old panel PNGs.
            if p.exists():
                p.unlink()
                removed.append(str(p))
        # Always remove percent-residual source files if present.
        for item in outputs:
            pct_entry = item.get("pct_residual")
            if not isinstance(pct_entry, dict):
                continue
            for rel in pct_entry.values():
                if not rel:
                    continue
                p = (repo_root / rel).resolve()
                if p.exists():
                    p.unlink()
                    removed_pct.append(str(p))

    manifest = {
        "dir": str(subdir),
        "generated_count": len(generated),
        "generated": generated,
        "removed_count": len(removed),
        "removed": removed,
        "removed_pct_count": len(removed_pct),
        "removed_pct": removed_pct,
        "dpi": int(dpi),
        "merge_input_gt": bool(merge_input_gt),
    }
    (subdir / "composite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Create per-frame per-channel composite eval PNGs.")
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/scratch/project_2008261/pf_surrogate_modelling/results/eval_best_adamw_vs_psgd_2026-02-11"),
    )
    ap.add_argument("--dpi", type=int, default=320)
    ap.add_argument("--keep-individual", action="store_true", default=False)
    ap.add_argument(
        "--merge-input-gt",
        action="store_true",
        default=False,
        help="Use 3-panel AE layout: Input/GT, Output, Residual.",
    )
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    # metadata.json stores paths like "results/..."; these are relative to
    # pf_surrogate_modelling root (parent of results)
    repo_root = eval_root.parent.parent
    subdirs = _metadata_dirs(eval_root)
    if not subdirs:
        raise SystemExit(f"No metadata subdirs found in {eval_root}")

    summary: Dict[str, object] = {
        "eval_root": str(eval_root),
        "dirs_processed": len(subdirs),
        "items": [],
    }
    for sub in subdirs:
        info = _process_dir(
            sub,
            repo_root=repo_root,
            dpi=int(args.dpi),
            delete_sources=not bool(args.keep_individual),
            merge_input_gt=bool(args.merge_input_gt),
        )
        summary["items"].append(info)
    (eval_root / "composite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(eval_root), flush=True)


if __name__ == "__main__":
    main()
