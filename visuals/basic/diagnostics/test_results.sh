#!/bin/bash
#SBATCH --job-name=eval_phys
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:v100:3
#SBATCH --time=0:14:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euxo pipefail

PY="/scratch/project_2008261/physics_ml/bin/python3.11"
SCRIPT="/scratch/project_2008261/rapid_solidification/visuals/basic/diagnostics/test_results.py"
CFG="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/checkpoint.best.pth"
OUT="/scratch/project_2008261/rapid_solidification/results/visuals_basic/eval_sim12_k0_50"

# Optional: load modules if required by the system environment
# module load pytorch hdf5

"$PY" "$SCRIPT" \
  -c /scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/config_snapshot.yaml \
  -k /scratch/project_2008261/rapid_solidification/runs/UAFNO_PreSkip_Full/checkpoint.best.pth \
  -o /scratch/project_2008261/rapid_solidification/results/visuals_basic/eval_std \
  --conc-ch 0 --phase-ch 1 --batch 8 --amp
