#!/bin/bash
# Quick gputest smoke for diffusion-bridge UVit thermal with AFNO at 8x8 stage.

#SBATCH --job-name=gputest_diff_uvit_afno8x8_smoke
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
export HDF5_USE_FILE_LOCKING=FALSE

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

CFG_BASE=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80_afno12_stage8x8.yaml}
if [[ ! -f "${CFG_BASE}" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

TMP_DIR=${SLURM_TMPDIR:-/tmp}
TMP_CFG=${TMP_DIR}/cfg_afno8x8_smoke_${SLURM_JOB_ID:-$$}.yaml

echo "[gputest-diff-afno8x8] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[gputest-diff-afno8x8] base_cfg=${CFG_BASE}"
nvidia-smi || true

"${PY}" - "${CFG_BASE}" "${TMP_CFG}" <<'PY'
import sys
import yaml

base_cfg, tmp_cfg = sys.argv[1], sys.argv[2]
with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("trainer", {})
cfg.setdefault("loader", {})
cfg.setdefault("dataloader", {})
cfg["dataloader"].setdefault("train_args", {})
cfg["dataloader"].setdefault("val_args", {})

cfg["trainer"]["epochs"] = 2
cfg["trainer"]["steps_per_epoch"] = 2
cfg["trainer"]["accumulation_steps"] = 1
cfg["trainer"]["use_val"] = True
cfg["loader"]["batch_size"] = 1
cfg["loader"]["num_workers"] = 0
cfg["dataloader"]["train_args"]["limit_per_group"] = 2
cfg["dataloader"]["train_args"]["max_items"] = 0.2
cfg["dataloader"]["val_args"]["limit_per_group"] = 2
cfg["dataloader"]["val_args"]["max_items"] = 0.2
out_dir = str(cfg["trainer"].get("out_dir", "")).rstrip("/")
if out_dir:
    cfg["trainer"]["out_dir"] = out_dir + "_gputest_smoke"

with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[gputest-diff-afno8x8] wrote smoke cfg: {tmp_cfg}")
print(f"[gputest-diff-afno8x8] out_dir: {cfg['trainer'].get('out_dir')}")
PY

cd "${ROOT}"
"${PY}" -m models.train.core.train -c "${TMP_CFG}"
