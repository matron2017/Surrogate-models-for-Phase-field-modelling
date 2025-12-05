#!/bin/bash
#SBATCH --job-name=gradviz
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:v100:3
#SBATCH --mem=8G
#SBATCH --time=00:14:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Threading for BLAS
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

# NCCL settings suitable for short gputest runs
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING

#  --pairs 203 204 205 207 2 210 217 230 260 \

# Paths
SCRIPT_HQ="/scratch/project_2008261/rapid_solidification/visuals/hq/pf_hqviz.py"
CFG="/scratch/project_2008261/rapid_solidification/runs/uafno_wavelet3000_9000_longest/UAFNO_PreSkip_Full/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/uafno_wavelet3000_9000_longest/UAFNO_PreSkip_Full/checkpoint.best.pth"
OUT="/scratch/project_2008261/rapid_solidification/results/visuals_hq/uafno_wavelet3000_9000_longest/hqviz"

mkdir -p "${OUT}"

PYBIN="/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11"

# High-quality single-image visualisation for selected pair_index values
"$PYBIN" "$SCRIPT_HQ" \
  -c "$CFG" \
  -k "$CKPT" \
  -o "$OUT" \
  --pairs 225 230 240 245 \
  --channel 0 \
  --figwidth 16 \
  --figheight 4 \
  --dpi 600 \
  --optimize-png
