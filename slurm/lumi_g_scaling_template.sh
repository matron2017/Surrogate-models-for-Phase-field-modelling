#!/bin/bash
#SBATCH --job-name=pf_lumi_template
#SBATCH --account=project_462001306
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# Optional module setup for CSC-managed PyTorch on LUMI.
if [[ "${LOAD_MODULES:-1}" == "1" ]]; then
  module purge
  module load LUMI
  module load partition/G
  module load Local-CSC/default
  module load pytorch/2.5
fi

PROJECT_ROOT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
LUMI_ENV_DIR=${LUMI_ENV_DIR:-${PROJECT_ROOT}/.venv_physics_ml_lumi}
if [[ -d "${LUMI_ENV_DIR}" ]]; then
  # shellcheck disable=SC1090
  source "${LUMI_ENV_DIR}/bin/activate"
  PYTHON_BIN=${PYTHON_BIN:-${LUMI_ENV_DIR}/bin/python}
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi

CFG_BASE=${CFG_BASE:-${PROJECT_ROOT}/configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_afno8_stochastic.yaml}
RUN_TAG=${RUN_TAG:-lumi_template}

TRAIN_H5=${TRAIN_H5:-}
VAL_H5=${VAL_H5:-}
SIM_MAP=${SIM_MAP:-}

EPOCHS=${EPOCHS:-2}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-20}
BATCH_PER_GPU=${BATCH_PER_GPU:-1}
ACCUM_STEPS=${ACCUM_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
USE_VAL=${USE_VAL:-1}

RUN_TASKS=${RUN_TASKS:-${SLURM_NTASKS}}

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/tmp"
CFG_RUN="${PROJECT_ROOT}/tmp/${RUN_TAG}_${SLURM_JOB_ID}.yaml"
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/runs/${RUN_TAG}_${SLURM_JOB_ID}}

if [[ ! -f "${CFG_BASE}" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${CFG_BASE}" "${CFG_RUN}" "${PROJECT_ROOT}" "${OUT_DIR}" \
  "${TRAIN_H5}" "${VAL_H5}" "${SIM_MAP}" "${EPOCHS}" "${STEPS_PER_EPOCH}" \
  "${BATCH_PER_GPU}" "${ACCUM_STEPS}" "${NUM_WORKERS}" "${USE_VAL}" <<'PY'
import sys
import yaml
from pathlib import Path

(
    cfg_base,
    cfg_run,
    project_root,
    out_dir,
    train_h5,
    val_h5,
    sim_map,
    epochs,
    steps_per_epoch,
    batch_per_gpu,
    accum_steps,
    num_workers,
    use_val,
) = sys.argv[1:]

project_root = str(Path(project_root).resolve())
old_root = "/scratch/project_2008261/pf_surrogate_modelling"

with open(cfg_base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def replace_root(value):
    if isinstance(value, str) and value.startswith(old_root):
        return value.replace(old_root, project_root, 1)
    return value

cfg.setdefault("paths", {}).setdefault("h5", {})
cfg.setdefault("dataloader", {})
cfg.setdefault("loader", {})
cfg.setdefault("trainer", {})

for k in ("file",):
    if k in cfg.get("dataloader", {}):
        cfg["dataloader"][k] = replace_root(cfg["dataloader"][k])
for k in ("file",):
    if k in cfg.get("model", {}):
        cfg["model"][k] = replace_root(cfg["model"][k])

if "sim_map" in cfg.get("paths", {}):
    cfg["paths"]["sim_map"] = replace_root(cfg["paths"]["sim_map"])
if sim_map:
    cfg["paths"]["sim_map"] = sim_map

h5_cfg = cfg["paths"]["h5"]
for split in ("train", "val"):
    if split in h5_cfg:
        if isinstance(h5_cfg[split], str):
            h5_cfg[split] = replace_root(h5_cfg[split])
        elif isinstance(h5_cfg[split], dict):
            if "h5_path" in h5_cfg[split]:
                h5_cfg[split]["h5_path"] = replace_root(h5_cfg[split]["h5_path"])
            if "weight_h5" in h5_cfg[split] and isinstance(h5_cfg[split]["weight_h5"], str):
                h5_cfg[split]["weight_h5"] = replace_root(h5_cfg[split]["weight_h5"])

if train_h5:
    if isinstance(h5_cfg.get("train"), dict):
        h5_cfg["train"]["h5_path"] = train_h5
    else:
        h5_cfg["train"] = train_h5
if val_h5:
    h5_cfg["val"] = val_h5

cfg["trainer"]["out_dir"] = out_dir
cfg["trainer"]["epochs"] = int(epochs)
cfg["trainer"]["steps_per_epoch"] = int(steps_per_epoch)
cfg["trainer"]["accumulation_steps"] = int(accum_steps)
cfg["trainer"]["use_val"] = bool(int(use_val))
cfg["loader"]["batch_size"] = int(batch_per_gpu)
cfg["loader"]["num_workers"] = int(num_workers)

with open(cfg_run, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[lumi-template] wrote config: {cfg_run}")
print(
    f"[lumi-template] out_dir={out_dir} batch_per_gpu={cfg['loader']['batch_size']} "
    f"accum={cfg['trainer']['accumulation_steps']} epochs={cfg['trainer']['epochs']} "
    f"steps_per_epoch={cfg['trainer']['steps_per_epoch']}"
)
PY

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

srun --ntasks="${RUN_TASKS}" --cpu-bind=cores --gpu-bind=closest bash -lc "
  set -euo pipefail
  export RANK=\${SLURM_PROCID}
  export WORLD_SIZE=\${SLURM_NTASKS}
  export LOCAL_RANK=\${SLURM_LOCALID}
  export PYTHONPATH='${PROJECT_ROOT}'
  exec '${PYTHON_BIN}' -m models.train.core.train -c '${CFG_RUN}'
"
