#!/bin/bash
#SBATCH --job-name=inspect_paths
#SBATCH --account=project_2008261
#SBATCH --partition=gputest           # GPU not required; test or small CPU partition also fine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1             # remove if running on a CPU partition
#SBATCH --time=0:05:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err
# Choose at most one memory mode; none set here to avoid conflicts.
# Example alternatives (uncomment exactly one if required):

set -euo pipefail

# Threading and I/O

# Paths
PY="/scratch/project_2008261/physics_ml/bin/python3.11"
CFG="/scratch/project_2008261/rapid_solidification/runs/UNet_SSA_PreSkip_Full_trained/config_snapshot.yaml"
ROOT="/scratch/project_2008261/rapid_solidification"
SCRIPT="/scratch/project_2008261/rapid_solidification/visuals/basic/inspect_paths.py"

# Run
"$PY" "$SCRIPT" \
  -c "$CFG" \
  --search-root "$ROOT" \
  --dry-import \
  --print-sys-path
