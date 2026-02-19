#!/bin/bash
#SBATCH --job-name=gputest_flow_unet_overfit
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export HDF5_USE_FILE_LOCKING=FALSE
export CUBLAS_WORKSPACE_CONFIG=:16:8

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

CFG_BASE=${CFG:-${ROOT}/configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8.yaml}
if [[ ! -f "$CFG_BASE" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

TMP_DIR=${SLURM_TMPDIR:-/tmp}
TMP_CFG=${TMP_DIR}/cfg_flow_unet_overfit_${SLURM_JOB_ID:-$$}.yaml

cat <<'PYCFG' > /tmp/build_gputest_flow_overfit_cfg.py
import os
import yaml
import sys

base_cfg, tmp_cfg = sys.argv[1], sys.argv[2]
overfit_n = int(os.environ.get("OVERFIT_N", "2"))

tmp_obj = int(os.environ.get("EPOCHS", "1"))
train_steps = int(os.environ.get("TRAIN_STEPS_PER_EPOCH", "2"))
val_steps = int(os.environ.get("VAL_STEPS_PER_EPOCH", "1"))
seed = int(os.environ.get("SEED", "1"))

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("train", {})
cfg.setdefault("trainer", {})
cfg.setdefault("loader", {})
cfg.setdefault("dataloader", {})
cfg["trainer"]["seed"] = seed
cfg["trainer"]["epochs"] = tmp_obj
cfg["trainer"]["steps_per_epoch"] = train_steps
cfg["trainer"]["val_steps_per_epoch"] = val_steps
cfg["trainer"]["accumulation_steps"] = 1
cfg["trainer"]["use_val"] = True
cfg["loader"]["batch_size"] = 1
cfg["loader"]["num_workers"] = 0
cfg["loader"]["overfit_n"] = overfit_n
cfg["dataloader"].setdefault("train_args", {})
cfg["dataloader"].setdefault("val_args", {})
cfg["dataloader"]["train_args"]["limit_per_group"] = 1
cfg["dataloader"]["train_args"]["max_items"] = 1
cfg["dataloader"]["val_args"]["limit_per_group"] = 1
cfg["dataloader"]["val_args"]["max_items"] = 1

out_dir = str(cfg["trainer"].get("out_dir", "")).rstrip("/")
if out_dir:
    cfg["trainer"]["out_dir"] = out_dir + "_gputest_overfit"

with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[gputest-flow-overfit] wrote cfg: {tmp_cfg}")
print(f"[gputest-flow-overfit] out_dir: {cfg['trainer'].get('out_dir')}")
print(f"[gputest-flow-overfit] overfit_n: {overfit_n}")
PYCFG

echo "[gputest-flow-overfit] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[gputest-flow-overfit] base_cfg=${CFG_BASE}"
$PY /tmp/build_gputest_flow_overfit_cfg.py "$CFG_BASE" "$TMP_CFG"

cd "$ROOT"
$PY -m models.train.core.train -c "$TMP_CFG"
