#!/bin/bash
# 5-hour latent flow/diffusion run (1 GPU), effective batch size 80.

#SBATCH --job-name=latent_uvit_thermal_5h_1gpu_b80
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi
CFG=${CFG:-${ROOT}/configs/train/train_flowmatch_uvit_thermal_sourceanchored_latentbest213_gpu5h_1gpu_b80.yaml}

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi

echo "[latent-5h-1gpu] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[latent-5h-1gpu] cfg=${CFG}"
nvidia-smi || true

cd "${ROOT}"
"${PY}" - "${CFG}" <<'PY'
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
batch = int(cfg.get("loader", {}).get("batch_size", 0))
acc = int(cfg.get("trainer", {}).get("accumulation_steps", 0))
eff = batch * acc
print(f"[latent-5h-1gpu] guard: batch={batch} accumulation={acc} effective_batch={eff}")
if eff != 80:
    raise SystemExit(f"Refusing run: expected effective batch 80 on 1 GPU, got {eff}.")
PY

"${PY}" -m models.train.core.train -c "${CFG}"
