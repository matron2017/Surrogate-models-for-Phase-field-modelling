#!/usr/bin/env python3
"""Compatibility launcher for renamed script."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("eval_field_examples.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
