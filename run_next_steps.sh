#!/usr/bin/env bash
# Helper to enqueue the quick smoke tests (datapipes, training, visuals).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

echo "[next] Submitting datapipes smoke build"
sbatch "$here/datapipes/rapid_solidification/smoke_build.sh"

echo "[next] Submitting training smoke run"
sbatch "$here/training/slurm/train_smoke.sh"

echo "[next] Submitting visuals smoke run"
sbatch "$here/visuals/basic/run_smoke.sh"

echo "Queued all smoke tests. Track them via 'squeue -u $USER'."
