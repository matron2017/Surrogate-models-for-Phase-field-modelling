#!/bin/bash
# DDP gputest launcher for latent UNet bridge with enforced effective batch size 80.
# Tries multi-GPU/multi-node depending on SLURM allocation.

#SBATCH --job-name=gputest_unet_bridge_ddp_b80
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export HDF5_USE_FILE_LOCKING=FALSE
export GIT_PYTHON_REFRESH=quiet
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

CFG_DEFAULT_BASE=${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint.yaml
CFG_DEFAULT_RDBM=${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_rdbmres_controlhint.yaml
if [[ -n "${CFG:-}" ]]; then
  CFG_BASE="${CFG}"
else
  JOB_NAME_LC=$(echo "${SLURM_JOB_NAME:-}" | tr '[:upper:]' '[:lower:]')
  if [[ "${JOB_NAME_LC}" == *"rdbm"* ]]; then
    CFG_BASE="${CFG_DEFAULT_RDBM}"
  else
    CFG_BASE="${CFG_DEFAULT_BASE}"
  fi
fi
if [[ ! -f "${CFG_BASE}" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WORLD_SIZE=$((NODES * GPUS_PER_NODE))
SMOKE_EPOCHS=${SMOKE_EPOCHS:-1}
SMOKE_STEPS_PER_EPOCH=${SMOKE_STEPS_PER_EPOCH:-1}
SMOKE_USE_VAL=${SMOKE_USE_VAL:-1}
SMOKE_BATCH_PER_RANK=${SMOKE_BATCH_PER_RANK:-1}
SMOKE_NUM_WORKERS=${SMOKE_NUM_WORKERS:-2}
SMOKE_LIMIT_PER_GROUP=${SMOKE_LIMIT_PER_GROUP:-2}
SMOKE_MAX_ITEMS=${SMOKE_MAX_ITEMS:-0.1}

TMP_DIR=${ROOT}/tmp/ddp_smoke_cfg
mkdir -p "${TMP_DIR}"
TMP_CFG=${TMP_DIR}/cfg_unet_bridge_ddp_smoke_${SLURM_JOB_ID:-$$}.yaml

echo "[gputest-unet-ddp-b80] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[gputest-unet-ddp-b80] base_cfg=${CFG_BASE}"
echo "[gputest-unet-ddp-b80] nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} world_size=${WORLD_SIZE}"
nvidia-smi || true

"${PY}" - "${CFG_BASE}" "${TMP_CFG}" "${WORLD_SIZE}" "${SMOKE_EPOCHS}" "${SMOKE_STEPS_PER_EPOCH}" "${SMOKE_USE_VAL}" "${SMOKE_BATCH_PER_RANK}" "${SMOKE_NUM_WORKERS}" "${SMOKE_LIMIT_PER_GROUP}" "${SMOKE_MAX_ITEMS}" <<'PY'
import sys
import yaml
from models.backbones.registry import build_backbone

base_cfg, tmp_cfg, world_size_s = sys.argv[1], sys.argv[2], sys.argv[3]
smoke_epochs_s = sys.argv[4]
smoke_steps_s = sys.argv[5]
smoke_use_val_s = sys.argv[6]
smoke_batch_s = sys.argv[7]
smoke_workers_s = sys.argv[8]
smoke_limit_s = sys.argv[9]
smoke_max_items_s = sys.argv[10]
world_size = int(world_size_s)
if world_size <= 0:
    raise SystemExit(f"Invalid world_size={world_size}")

smoke_epochs = int(smoke_epochs_s)
smoke_steps = int(smoke_steps_s)
smoke_use_val = bool(int(smoke_use_val_s))
smoke_batch = int(smoke_batch_s)
smoke_workers = int(smoke_workers_s)
smoke_limit = int(smoke_limit_s)
try:
    smoke_max_items = float(smoke_max_items_s)
except ValueError:
    smoke_max_items = smoke_max_items_s

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("trainer", {})
cfg.setdefault("loader", {})
cfg.setdefault("dataloader", {})
cfg["dataloader"].setdefault("train_args", {})
cfg["dataloader"].setdefault("val_args", {})

model_cfg = cfg.get("model", {})
backbone = model_cfg.get("backbone")
params = dict(model_cfg.get("params", {}))
model = build_backbone(backbone, params)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[gputest-unet-ddp-b80] trainable_params={trainable/1e6:.2f}M")
if trainable <= 150e6:
    raise SystemExit(
        f"Refusing run: expected >150M trainable params, got {trainable/1e6:.2f}M."
    )

cfg["trainer"]["epochs"] = smoke_epochs
cfg["trainer"]["steps_per_epoch"] = smoke_steps
cfg["trainer"]["use_val"] = smoke_use_val
cfg["loader"]["batch_size"] = smoke_batch
cfg["loader"]["num_workers"] = smoke_workers
cfg["dataloader"]["train_args"]["limit_per_group"] = smoke_limit
cfg["dataloader"]["train_args"]["max_items"] = smoke_max_items
cfg["dataloader"]["val_args"]["limit_per_group"] = smoke_limit
cfg["dataloader"]["val_args"]["max_items"] = smoke_max_items

batch = int(cfg["loader"]["batch_size"])
den = world_size * batch
if 80 % den != 0:
    raise SystemExit(
        f"Cannot realize effective batch 80 with world_size={world_size} batch_per_rank={batch}."
    )
cfg["trainer"]["accumulation_steps"] = 80 // den

out_dir = str(cfg["trainer"].get("out_dir", "")).rstrip("/")
if out_dir:
    cfg["trainer"]["out_dir"] = out_dir + f"_gputest_ddp_ws{world_size}_b80"

with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[gputest-unet-ddp-b80] wrote cfg={tmp_cfg}")
print(
    f"[gputest-unet-ddp-b80] epochs={cfg['trainer']['epochs']} steps_per_epoch={cfg['trainer']['steps_per_epoch']} "
    f"use_val={cfg['trainer']['use_val']} num_workers={cfg['loader']['num_workers']} "
    f"limit_per_group={cfg['dataloader']['train_args']['limit_per_group']} max_items={cfg['dataloader']['train_args']['max_items']}\n"
    f"[gputest-unet-ddp-b80] batch_per_rank={batch} world_size={world_size} "
    f"accum={cfg['trainer']['accumulation_steps']} effective_batch={batch*world_size*cfg['trainer']['accumulation_steps']}"
)
PY

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1{print $1}')
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR=$("${PY}" - "${MASTER_HOST}" <<'PY'
import socket
import sys

host = sys.argv[1]
try:
    print(socket.gethostbyname(host))
except Exception:
    print("")
PY
)
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="${MASTER_HOST}"
fi
MASTER_PORT=${MASTER_PORT:-29547}
export MASTER_ADDR MASTER_PORT
echo "[gputest-unet-ddp-b80] master_host=${MASTER_HOST} master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

cd "${ROOT}"
srun --ntasks="${NODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  node_rank=${SLURM_PROCID}
  '"${PY}"' -m torch.distributed.run \
    --nnodes='"${NODES}"' \
    --nproc_per_node='"${GPUS_PER_NODE}"' \
    --node_rank=${node_rank} \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    -m models.train.core.train -c '"${TMP_CFG}"'
'
