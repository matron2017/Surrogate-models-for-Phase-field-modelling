#!/bin/bash
# 15-minute smoke test for latent UNet flow-matching with AFNO bottleneck.

#SBATCH --job-name=gputest_flow_unet_latent_sto
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

CFG_BASE=${CFG:-${ROOT}/configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic.yaml}
if [[ ! -f "${CFG_BASE}" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

TMP_DIR=${SLURM_TMPDIR:-/tmp}
TMP_CFG=${TMP_DIR}/cfg_flow_unet_smoke_${SLURM_JOB_ID:-$$}.yaml

echo "[gputest-flow-unet-smoke] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[gputest-flow-unet-smoke] base_cfg=${CFG_BASE}"
echo "[gputest-flow-unet-smoke-stochastic] Running flow stochastic smoke"
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

cfg["trainer"]["epochs"] = 1
cfg["trainer"]["steps_per_epoch"] = 2
cfg["trainer"]["accumulation_steps"] = 1
cfg["trainer"]["use_val"] = True
cfg["loader"]["batch_size"] = 1
cfg["loader"]["num_workers"] = 0
cfg["dataloader"]["train_args"]["limit_per_group"] = 2
cfg["dataloader"]["train_args"]["max_items"] = 0.1
cfg["dataloader"]["val_args"]["limit_per_group"] = 2
cfg["dataloader"]["val_args"]["max_items"] = 0.1

train_obj = cfg.setdefault("train", {})
train_obj["objective"] = str(train_obj.get("objective", "rectified_flow_source_anchored_concat"))

out_dir = str(cfg["trainer"].get("out_dir", "")).rstrip("/")
if out_dir:
    cfg["trainer"]["out_dir"] = out_dir + "_gputest_smoke"

with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[gputest-flow-unet-smoke] wrote smoke cfg: {tmp_cfg}")
print(f"[gputest-flow-unet-smoke] out_dir: {cfg['trainer'].get('out_dir')}")
print(f"[gputest-flow-unet-smoke] train_obj: {cfg['train'].get('objective')}")
PY

cd "${ROOT}"
"${PY}" -m models.train.core.train -c "${TMP_CFG}"
