#!/bin/bash
#SBATCH --job-name=eval_phys
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:v100:3
#SBATCH --time=0:14:00
#SBATCH --output=%x_%j.out

set -euxo pipefail

PY="/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11"
SCRIPT="/scratch/project_2008261/rapid_solidification/visuals/basic/diagnostics/test_results.py"
CFG="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/checkpoint.best.pth"


# Optional: load modules if required by the system environment
# module load pytorch hdf5

OUT=/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/io_pair_Extra_202_203_sim12
mkdir -p "$OUT"

"$PY" /scratch/project_2008261/rapid_solidification/visuals/basic/diagnostics/visuals.py \
  -c "$CFG" \
  -k "$CKPT" \
  -o "$OUT" \
  --select-gid sim_0012 \
  --k 202 \
  --device cuda \
  --dpi 180 \
  --tile-w-in 3.2 --tile-h-in 2.4 \
  --bar-w-in 5.0 --bar-h-in 2.8


