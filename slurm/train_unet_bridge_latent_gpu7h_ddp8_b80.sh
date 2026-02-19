#!/bin/bash
# 7-hour latent diffusion-bridge run (2 nodes x 4 GPUs), effective batch size 80.

#SBATCH --job-name=latent_unet_bridge7h_ddp8_b80
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=07:00:00
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

NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WORLD_SIZE=$((NODES * GPUS_PER_NODE))

TMP_DIR=${ROOT}/tmp/ddp_long_cfg
mkdir -p "${TMP_DIR}"
TMP_CFG=${TMP_DIR}/cfg_unet_bridge_ddp_long_${SLURM_JOB_ID:-$$}.yaml

echo "[latent-unet-bridge-7h-ddp8-b80] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[latent-unet-bridge-7h-ddp8-b80] base_cfg=${CFG_BASE}"
echo "[latent-unet-bridge-7h-ddp8-b80] nodes=${NODES} gpus_per_node=${GPUS_PER_NODE} world_size=${WORLD_SIZE}"
nvidia-smi || true

"${PY}" - "${CFG_BASE}" "${TMP_CFG}" "${WORLD_SIZE}" <<'PY'
import sys
import yaml

from models.backbones.registry import build_backbone

base_cfg, tmp_cfg, world_size_s = sys.argv[1], sys.argv[2], sys.argv[3]
world_size = int(world_size_s)
if world_size <= 0:
    raise SystemExit(f"Invalid world_size={world_size}")

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("trainer", {})
cfg.setdefault("loader", {})

model_cfg = cfg.get("model", {})
backbone = model_cfg.get("backbone")
params = dict(model_cfg.get("params", {}))
model = build_backbone(backbone, params)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[latent-unet-bridge-7h-ddp8-b80] trainable_params={trainable/1e6:.2f}M")
if trainable <= 150e6:
    raise SystemExit(
        f"Refusing run: expected >150M trainable params, got {trainable/1e6:.2f}M."
    )

batch = int(cfg["loader"].get("batch_size", 0))
epochs = int(cfg["trainer"].get("epochs", 0))
if batch <= 0:
    raise SystemExit(f"Invalid batch_size={batch}")
if epochs <= 0:
    raise SystemExit(f"Invalid epochs={epochs}")

den = world_size * batch
if 80 % den != 0:
    raise SystemExit(
        f"Cannot realize effective batch 80 with world_size={world_size} batch_per_rank={batch}."
    )
cfg["trainer"]["accumulation_steps"] = 80 // den

eff = batch * world_size * int(cfg["trainer"]["accumulation_steps"])
print(
    f"[latent-unet-bridge-7h-ddp8-b80] batch_per_rank={batch} world_size={world_size} "
    f"accum={cfg['trainer']['accumulation_steps']} effective_batch={eff}"
)
if eff != 80:
    raise SystemExit(f"Refusing run: expected effective batch 80, got {eff}.")

cond_cfg = cfg.get("conditioning", {})
dl_cfg = cfg.get("dataloader", {})
ds_args_base = dict(dl_cfg.get("args", {}))
ds_args_train = dict(dl_cfg.get("train_args", {}))
ds_args = {**ds_args_base, **ds_args_train}
if bool(cond_cfg.get("use_theta", False)):
    if not bool(ds_args.get("add_thermal", False)):
        raise SystemExit(
            "Refusing run: use_theta=true requires dataloader.args.add_thermal=true "
            "(or overridden in dataloader.train_args)."
        )
    if not bool(ds_args.get("thermal_require_precomputed", False)):
        raise SystemExit(
            "Refusing run: thermal_require_precomputed must be true in dataloader.args "
            "(or overridden in dataloader.train_args) for this run."
        )

channels = list(params.get("channels", []))
if channels:
    downsample_levels = max(len(channels) - 1, 0)
    expected_bottleneck = 64 // (2 ** downsample_levels)
    print(
        f"[latent-unet-bridge-7h-ddp8-b80] expected_bottleneck_hw={expected_bottleneck}x{expected_bottleneck}"
    )

with open(tmp_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[latent-unet-bridge-7h-ddp8-b80] wrote cfg={tmp_cfg}")
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
MASTER_PORT=${MASTER_PORT:-29549}
export MASTER_ADDR MASTER_PORT
echo "[latent-unet-bridge-7h-ddp8-b80] master_host=${MASTER_HOST} master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

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
