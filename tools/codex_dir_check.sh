#!/usr/bin/env bash
# Quick context bootstrap for Codex sessions.
# Prints the repo root, top-level tree, and leading lines of the key guides so we remember to consult them.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/codex_dir_check.sh [--root <path>] [--depth <n>] [--doc-lines <n>]

Defaults:
  --root      auto-detected git root or current directory
  --depth     2      (tree depth passed to tools/print_tree.py if available)
  --doc-lines 32     (number of lines shown from each doc)

Example:
  tools/codex_dir_check.sh --depth 3 --doc-lines 24
EOF
}

ROOT=""
DEPTH=2
DOC_LINES=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --doc-lines)
      DOC_LINES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    ROOT="$(git rev-parse --show-toplevel)"
  else
    ROOT="$(pwd)"
  fi
fi

if [[ ! -d "$ROOT" ]]; then
  echo "Root does not exist: $ROOT" >&2
  exit 1
fi

echo "pwd: $(pwd)"
echo "repo root: $ROOT"
echo

echo "Top-level entries:"
ls -a "$ROOT"
echo

if [[ -f "$ROOT/tools/print_tree.py" ]]; then
  PY_BIN=""
  PY_CANDIDATES=(
    "/scratch/project_2008261/physics_ml/bin/python3"
    "$ROOT/../physics_ml/bin/python3"
    "$(command -v python3 || true)"
    "$(command -v python || true)"
  )
  for candidate in "${PY_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      PY_BIN="$candidate"
      break
    fi
  done

  if [[ -n "$PY_BIN" ]]; then
    PY_VERSION="$("$PY_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
    if "$PY_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
      echo "Directory tree (depth=$DEPTH) using $PY_BIN (python $PY_VERSION):"
      "$PY_BIN" "$ROOT/tools/print_tree.py" --root "$ROOT" --max-depth "$DEPTH" || true
      echo
    else
      echo "Directory tree: skipped (python <3.8 or unusable; found ${PY_BIN:-none} ${PY_VERSION:-})"
      echo
    fi
  else
    echo "Directory tree: skipped (no python binary on PATH)"
    echo
  fi
fi

docs=(
  "README.md"
  "docs/README.md"
  "docs/DIRECTORY_GUIDE.md"
  "docs/DEV_GUIDE.md"
  "docs/ENVIRONMENT.md"
  "docs/EXPERIMENT_STATUS.md"
  "docs/NEXT_STEPS.md"
  "docs/GPU_MONITORING.md"
  "docs/PUHTI_PARTITIONS.md"
  "docs/ARCHITECTURE.md"
)

for doc in "${docs[@]}"; do
  doc_path="$ROOT/$doc"
  if [[ -f "$doc_path" ]]; then
    echo "=== $doc (first $DOC_LINES lines) ==="
    head -n "$DOC_LINES" "$doc_path"
    echo
  fi
done
