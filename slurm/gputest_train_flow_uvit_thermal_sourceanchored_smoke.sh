#!/bin/bash
# Short gputest run for source-anchored flow matching + thermal conditioning.

#SBATCH --job-name=gputest_flow_uvit_thermal_sa_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
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

CFG=${CFG:-${ROOT}/configs/train/train_flowmatch_uvit_thermal_sourceanchored_latentdev_quick_gputest_smoke.yaml}

echo "[gputest-flow-smoke] host=$(hostname) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true
echo "[gputest-flow-smoke] cfg=${CFG}"

cd "${ROOT}"
"${PY}" - "${CFG}" <<'PY'
import sys, yaml

cfg_path = sys.argv[1]
cfg = yaml.safe_load(open(cfg_path, "r"))

tr = cfg.get("dataloader", {}).get("train_args", {}).get("max_items", None)
va = cfg.get("dataloader", {}).get("val_args", {}).get("max_items", None)
epochs = int(cfg.get("trainer", {}).get("epochs", 0))
steps = int(cfg.get("trainer", {}).get("steps_per_epoch", 0))

def _ok_frac(x):
    return isinstance(x, (int, float)) and float(x) > 0.0 and float(x) <= 0.4

if not _ok_frac(tr) or not _ok_frac(va):
    raise SystemExit(
        f"Refusing gputest run: max_items must be fraction <=0.4 (train={tr}, val={va})."
    )
if epochs > 2 or steps > 2:
    raise SystemExit(
        f"Refusing gputest run: require short schedule epochs<=2 and steps_per_epoch<=2 "
        f"(got epochs={epochs}, steps_per_epoch={steps})."
    )
print(f"[gputest-flow-smoke] guard OK: train_max={tr} val_max={va} epochs={epochs} steps={steps}")
PY

"${PY}" -m models.train.core.train -c "${CFG}"
