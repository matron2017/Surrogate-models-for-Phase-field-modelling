#!/usr/bin/env python3
"""
inspect_config_and_paths.py

Purpose: Inspect a training/eval YAML, list referenced Python files, symbols, and data
paths, check existence, resolve import roots, and optionally search for missing files.
No second-person phrasing in comments.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Optional imports gated at call sites to avoid hard failures if not installed.


def _guess_root(py_path: Path) -> Path:
    """Mirror the repository-root inference used by many helper loaders.

    Strategy: walk upwards until a directory containing any of {models, scripts, src}
    is encountered. Fallback to parent of the file.
    """
    for a in [py_path.parent, *py_path.parents]:
        if (a / "models").is_dir() or (a / "scripts").is_dir() or (a / "src").is_dir():
            return a
    return py_path.parent


def _norm_path_like(v: Any, base: Path) -> Optional[Path]:
    """Accept str or {path: str} and return an absolute Path if present."""
    if v is None:
        return None
    if isinstance(v, dict) and "path" in v:
        v = v["path"]
    if isinstance(v, str) and v.strip():
        p = Path(v)
        return p if p.is_absolute() else (base / p)
    return None


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _exists(p: Optional[Path]) -> Tuple[bool, Optional[Path]]:
    return (p is not None and p.exists(), p)


def _search_candidates(root: Path, target_name: str, limit: int = 25) -> List[Path]:
    """Shallow search for filenames matching target_name under root."""
    hits: List[Path] = []
    skip_dirs = {".git", "__pycache__", ".mamba", ".conda", "env", "venv", "site-packages"}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noisy dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if target_name in filenames:
            hits.append(Path(dirpath) / target_name)
            if len(hits) >= limit:
                break
    return hits


def _try_module_name(py_file: Path, repo_root: Path) -> str:
    rel = py_file.resolve().relative_to(repo_root.resolve())
    return ".".join(rel.with_suffix("").parts)


def _attempt_import(py_file: Path, symbol: str, repo_root: Path) -> Tuple[bool, str]:
    """Attempt import via sys.path insertion, then report success and strategy."""
    sys_ok = False
    note = ""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        note += f"prepended sys.path: {repo_root}\n"
    mod_name = _try_module_name(py_file, repo_root)
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        sys_ok = hasattr(mod, symbol)
        note += f"imported module '{mod_name}'; symbol present={sys_ok}"
    except Exception as e:
        note += f"import_module failed for '{mod_name}': {type(e).__name__}: {e}"
    return sys_ok, note


def _print_header(title: str):
    print("\n" + title)
    print("=" * len(title))


def inspect(config_path: Path, search_root: Optional[Path], dry_import: bool, print_sys_path: bool):
    cfg = _load_yaml(config_path)
    base = config_path.parent

    # Pull common fields
    dl_file = cfg.get("dataloader", {}).get("file")
    dl_class = cfg.get("dataloader", {}).get("class")
    model_file = cfg.get("model", {}).get("file")
    model_class = cfg.get("model", {}).get("class")

    h5_section = (cfg.get("paths", {}).get("h5") or {})
    h5_train = _norm_path_like(h5_section.get("train"), base)
    h5_val = _norm_path_like(h5_section.get("val"), base)
    h5_test = _norm_path_like(h5_section.get("test"), base)

    _print_header("YAML overview")
    print(json.dumps({
        "config": str(config_path.resolve()),
        "cwd": str(Path.cwd().resolve()),
        "python": sys.executable,
    }, indent=2))

    if print_sys_path:
        _print_header("sys.path (first 10)")
        for i, p in enumerate(sys.path[:10]):
            print(f"[{i}] {p}")

    _print_header("Dataset loader reference")
    print(f"file: {dl_file}")
    print(f"class: {dl_class}")
    dl_path = _norm_path_like(dl_file, base) if isinstance(dl_file, str) else None
    ok, p = _exists(dl_path)
    print(f"resolved: {p}")
    print(f"exists: {ok}")

    _print_header("Model reference")
    print(f"file: {model_file}")
    print(f"class: {model_class}")
    model_path = _norm_path_like(model_file, base) if isinstance(model_file, str) else None
    okm, pm = _exists(model_path)
    print(f"resolved: {pm}")
    print(f"exists: {okm}")

    _print_header("HDF5 paths")
    for name, pth in ("train", h5_train), ("val", h5_val), ("test", h5_test):
        okh, ph = _exists(pth)
        print(f"{name:5s}: {ph}  exists={okh}")

    # Root inference and import attempts for Python files
    for label, pth, sym in (
        ("dataloader", dl_path, dl_class),
        ("model", model_path, model_class),
    ):
        _print_header(f"{label} import probing")
        if not pth or not sym:
            print("missing path or symbol; skip")
            continue
        repo_root = _guess_root(pth)
        print(f"file: {pth}")
        print(f"repo_root_guess: {repo_root}")
        if pth.exists():
            mod_name = _try_module_name(pth, repo_root)
            print(f"module_name_if_pkg: {mod_name}")
        else:
            print("file does not exist on disk")
            # search assistance
            root = search_root or base
            candidates = _search_candidates(root, pth.name)
            if candidates:
                print("candidates:")
                for q in candidates:
                    print(f"  - {q}")
            else:
                print("no candidates found under search root")
        if dry_import and pth.exists():
            okimp, note = _attempt_import(pth, sym, repo_root)
            print(f"import_probe_ok={okimp}\n{note}")

    # Optional: peek into HDF5 root attributes to see if any embedded references exist
    if h5_test and h5_test.exists():
        _print_header("HDF5 root attributes (test)")
        try:
            import h5py  # type: ignore
            with h5py.File(h5_test, "r") as h5:
                attrs = {k: (str(v) if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}
                # Keep it compact
                keys = sorted(list(attrs.keys()))
                print("keys:", keys)
                for k in keys:
                    v = attrs[k]
                    sv = str(v)
                    if len(sv) > 120:
                        sv = sv[:117] + "..."
                    print(f"attr[{k}]= {sv}")
                # Show top-level groups to help correlate gid naming
                groups = [g for g in h5.keys()][:10]
                print("top_groups (first 10):", groups)
        except Exception as e:
            print(f"failed to read HDF5: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    ap.add_argument("--search-root", default=None, help="Directory under which to search for missing files")
    ap.add_argument("--dry-import", action="store_true", help="Attempt to import symbols for existence checks")
    ap.add_argument("--print-sys-path", action="store_true", help="Print the leading entries of sys.path")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    search_root = Path(args.search_root).resolve() if args.search_root else None

    inspect(config_path, search_root, args.dry_import, args.print_sys_path)


if __name__ == "__main__":
    main()
