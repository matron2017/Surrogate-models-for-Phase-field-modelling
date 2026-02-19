#!/bin/bash
# Resume AE latent PSGD run from latest checkpoint with a short 3h continuation:
# - preserve previous best checkpoint in a separate backup
# - start from latest epoch
# - enforce LR schedule 3e-6 -> 1e-6 over 70 additional epochs

#SBATCH --job-name=ae_latent_psgd_resume3h_lr3e6_70ep
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH=${ROOT}
PY=/scratch/project_2008261/physics_ml/bin/python3.11
TORCHRUN=${TORCHRUN:-$(dirname "$PY")/torchrun}

BASE_CFG=${ROOT}/configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml
SRC_BUNDLE=${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel
SRC_LAST=${SRC_BUNDLE}/checkpoint.last.pth
SRC_BEST=${SRC_BUNDLE}/checkpoint.best.pth

NEW_OUT=${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128_resume3h_lr3e6_to1e6_70ep
BOOTSTRAP_DIR=${NEW_OUT}/bootstrap_from_prev_run
TMP=${SLURM_TMPDIR:-/tmp}/ae_resume_${SLURM_JOB_ID}
mkdir -p "${TMP}" "${BOOTSTRAP_DIR}"

CFG_RESUME=${TMP}/config_ae_resume3h_lr3e6_to1e6_70ep.yaml
RESUME_CKPT=${BOOTSTRAP_DIR}/checkpoint.resume_lr3e6_reset_sched.pth
BACKUP_BEST=${BOOTSTRAP_DIR}/checkpoint.best.pre_resume_copy.pth
BACKUP_LAST=${BOOTSTRAP_DIR}/checkpoint.last.pre_resume_copy.pth
MANIFEST=${BOOTSTRAP_DIR}/resume_manifest.json

if [[ ! -f "${SRC_LAST}" ]]; then
  echo "Missing source checkpoint.last: ${SRC_LAST}" >&2
  exit 1
fi

echo "=== Preparing resume payload from ${SRC_LAST} ==="
"${PY}" - <<'PYCODE' "${BASE_CFG}" "${SRC_LAST}" "${SRC_BEST}" "${CFG_RESUME}" "${RESUME_CKPT}" "${BACKUP_BEST}" "${BACKUP_LAST}" "${MANIFEST}" "${NEW_OUT}"
import copy
import datetime as dt
import json
import os
import shutil
import sys
import time

import torch
import yaml

base_cfg_path, src_last, src_best, cfg_resume, resume_ckpt, backup_best, backup_last, manifest_path, new_out = sys.argv[1:10]

with open(base_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def _load_stable_checkpoint(path: str, dst_copy: str, retries: int = 20, sleep_s: float = 5.0):
    last_err = None
    for _ in range(retries):
        try:
            m0 = os.path.getmtime(path)
            s0 = os.path.getsize(path)
            shutil.copy2(path, dst_copy)
            m1 = os.path.getmtime(path)
            s1 = os.path.getsize(path)
            if m0 != m1 or s0 != s1:
                time.sleep(sleep_s)
                continue
            state_obj = torch.load(dst_copy, map_location="cpu")
            return state_obj
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to obtain stable checkpoint from {path}: {last_err}")


state = _load_stable_checkpoint(src_last, backup_last)
last_epoch = int(state.get("epoch", 0))
target_epoch = last_epoch + 70
resume_lr = 3.0e-6
eta_min = 1.0e-6

cfg = copy.deepcopy(cfg)
cfg["trainer"]["epochs"] = int(target_epoch)
cfg["trainer"]["out_dir"] = new_out
cfg["trainer"]["resume"] = False
cfg["optim"]["lr"] = float(resume_lr)
cfg["sched"]["name"] = "cosine"
cfg["sched"]["T_max"] = 70
cfg["sched"]["eta_min"] = float(eta_min)
cfg["sched"]["warmup_epochs"] = 0
cfg["sched"]["warmup_start_lr"] = 0.0
cfg["trainer"]["lr_warmup_steps"] = 0
cfg["trainer"]["lr_warmup_start_lr"] = 0.0
cfg["trainer"]["lr_warmup_epoch_phases"] = []

with open(cfg_resume, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

if os.path.isfile(src_best):
    shutil.copy2(src_best, backup_best)

resume_state = copy.deepcopy(state)
resume_state["sched"] = None
if "optim" in resume_state and "param_groups" in resume_state["optim"]:
    for pg in resume_state["optim"]["param_groups"]:
        pg["lr"] = float(resume_lr)
        pg["initial_lr"] = float(resume_lr)

torch.save(resume_state, resume_ckpt)

manifest = {
    "created_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "purpose": "3h AE continuation from latest epoch with controlled LR schedule",
    "source_last_checkpoint": src_last,
    "source_best_checkpoint": src_best if os.path.isfile(src_best) else None,
    "backup_best_checkpoint": backup_best if os.path.isfile(backup_best) else None,
    "backup_last_checkpoint": backup_last if os.path.isfile(backup_last) else None,
    "resume_checkpoint": resume_ckpt,
    "generated_config": cfg_resume,
    "source_epoch": int(last_epoch),
    "target_final_epoch": int(target_epoch),
    "extra_epochs": 70,
    "lr_start": float(resume_lr),
    "lr_end": float(eta_min),
    "scheduler": {
        "name": "cosine",
        "T_max": 70,
        "eta_min": float(eta_min),
        "state_reset": True,
    },
    "note": "Original best checkpoint is preserved separately before resume training.",
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
PYCODE

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
NODE_RANK=${SLURM_NODEID:-0}
export MASTER_ADDR MASTER_PORT

echo "=== Launching AE resume job (3h, +70 epochs max, LR 3e-6 -> 1e-6) ==="
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODE_RANK=${NODE_RANK}"
echo "Config: ${CFG_RESUME}"
echo "Resume checkpoint: ${RESUME_CKPT}"

srun --cpu-bind=cores --hint=nomultithread \
  "${TORCHRUN}" \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --node_rank=${NODE_RANK} \
    -m models.train.core.train -c "${CFG_RESUME}" --resume "${RESUME_CKPT}"
