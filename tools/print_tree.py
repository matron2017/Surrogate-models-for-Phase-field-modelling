#!/usr/bin/env python3
"""
Print a directory tree with file sizes (bytes) up to a given depth.

Usage:
  python tools/print_tree.py --root rapid_solidification --max-depth 3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def walk(root: Path, max_depth: int):
    root = root.resolve()
    prefix_stack = []

    def recurse(path: Path, depth: int):
        if depth > max_depth:
            return
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for idx, p in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            prefix = "".join(prefix_stack) + connector
            if p.is_dir():
                print(f"{prefix}{p.name}/")
                prefix_stack.append("    " if is_last else "│   ")
                recurse(p, depth + 1)
                prefix_stack.pop()
            else:
                size = p.stat().st_size
                print(f"{prefix}{p.name} ({size} bytes)")

    print(f"{root}/")
    recurse(root, 1)


def main():
    ap = argparse.ArgumentParser(description="Print directory tree with file sizes.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root directory to walk")
    ap.add_argument("--max-depth", type=int, default=3, help="Maximum depth to descend")
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root does not exist: {args.root}")
    walk(args.root, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
