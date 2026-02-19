#!/bin/bash
# 7-hour latent diffusion-bridge run (1 node x 4 GPUs), effective batch size 80.

#SBATCH --job-name=latent_unet_bridge_2h_b80
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
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
TORCHRUN=${TORCHRUN:-$(dirname "$PY")/torchrun}
CFG=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m.yaml}

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi

echo "[latent-unet-bridge-7h-b80] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[latent-unet-bridge-7h-b80] cfg=${CFG}"
nvidia-smi || true

cd "${ROOT}"
"${PY}" - "${CFG}" <<'PY'
import sys
import yaml

from models.backbones.registry import build_backbone

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
model_cfg = cfg.get("model", {})
backbone = model_cfg.get("backbone")
params = dict(model_cfg.get("params", {}))
model = build_backbone(backbone, params)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[latent-unet-bridge-7h-b80] trainable_params={trainable/1e6:.2f}M")
if not (150e6 <= trainable <= 250e6):
    raise SystemExit(
        f"Refusing run: expected 150M-250M trainable params, got {trainable/1e6:.2f}M."
    )

batch = int(cfg.get("loader", {}).get("batch_size", 0))
acc = int(cfg.get("trainer", {}).get("accumulation_steps", 0))
epochs = int(cfg.get("trainer", {}).get("epochs", 0))
if batch <= 0 or acc <= 0:
    raise SystemExit(f"Invalid batch/accumulation: batch={batch}, accumulation_steps={acc}")
if epochs <= 0:
    raise SystemExit(f"Invalid epochs={epochs}")
eff = batch * 4 * acc
print(f"[latent-unet-bridge-7h-b80] guard OK: batch_per_rank={batch} world_size=4 accumulation={acc} effective_batch={eff}")
if eff != 80:
    raise SystemExit(f"Refusing run: expected effective batch 80, got {eff}.")

channels = list(params.get("channels", []))
if channels:
    downsample_levels = max(len(channels) - 1, 0)
    expected_bottleneck = 64 // (2 ** downsample_levels)
    print(
        f"[latent-unet-bridge-7h-b80] expected_bottleneck_hw={expected_bottleneck}x{expected_bottleneck}"
    )
PY

"${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  -m models.train.core.train -c "${CFG}"
