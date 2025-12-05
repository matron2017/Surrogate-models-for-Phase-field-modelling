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
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NCCL settings suitable for short gputest runs
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING

SCRIPT="/scratch/project_2008261/rapid_solidification/visuals/hq/pf_gradviz.py"
CFG="/scratch/project_2008261/rapid_solidification/runs/uafno_wavelet3000_9000_longest/UAFNO_PreSkip_Full/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/uafno_wavelet3000_9000_longest/UAFNO_PreSkip_Full/checkpoint.best.pth"
OUT="/scratch/project_2008261/rapid_solidification/results/visuals_hq/uafno_wavelet_newgradviz_longest"

mkdir -p "$OUT"

/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 "$SCRIPT" \
  -c "$CFG" \
  -k "$CKPT" \
  -o "$OUT" \
  --device cuda \
  --batch 1 \
  --figwidth 12.0 \
  --figheight 4.0 \
  --dpi 400 \
  --interp nearest \
  --optimize-png \
  --wavelet-vis \
  --wavelet-theta 0.30 \
  --wavelet-alpha 1.0 \
  --wavelet-beta 500.0 \
  --wavelet-J 1 \
  --wavelet-name haar \
  --wavelet-mode zero
