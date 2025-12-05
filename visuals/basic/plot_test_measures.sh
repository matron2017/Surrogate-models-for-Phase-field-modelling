#!/bin/bash
#SBATCH --job-name=eval_phys
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:v100:3
#SBATCH --mem=8G
#SBATCH --time=0:14:00
#SBATCH --output=%x_%j.out

set -euxo pipefail

export PYTHONPATH="/scratch/project_2008261/rapid_solidification:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=lo

PY="/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11"
SCRIPT="/scratch/project_2008261/rapid_solidification/visuals/basic/pf_eval_grids_individual_minmetrics.py"
CFG="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/checkpoint.best.pth"
OUT="/scratch/project_2008261/rapid_solidification/results/visuals_basic/eval_sim12_k190_202"

mkdir -p "$OUT"

echo "CFG sha256: $(sha256sum "$CFG" | awk '{print $1}')" || true
echo "CKPT path:  $CKPT"

srun "$PY" "$SCRIPT" \
  -c "$CFG" -k "$CKPT" -o "$OUT" \
  --transitions 201:202 \
  --select-gid sim_0012 \
  --error-map relative_percent \
  --cbar-fontsize 18 \
  --cbar-max-ticks 5
