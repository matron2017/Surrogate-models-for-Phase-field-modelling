#!/bin/bash
# 2-hour latent diffusion-bridge run (1 GPU) for large UNet 8x8 bottleneck attention.

#SBATCH --job-name=latent_unet_bridge_2h_1gpu
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
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

CFG=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m.yaml}
if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi

echo "[latent-unet-bridge-2h] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[latent-unet-bridge-2h] cfg=${CFG}"
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
print(f"[latent-unet-bridge-2h] trainable_params={trainable/1e6:.2f}M")
if not (150e6 <= trainable <= 250e6):
    raise SystemExit(
        f"Refusing run: expected 150M-250M trainable params, got {trainable/1e6:.2f}M."
    )

batch = int(cfg.get("loader", {}).get("batch_size", 0))
acc = int(cfg.get("trainer", {}).get("accumulation_steps", 0))
if batch <= 0 or acc <= 0:
    raise SystemExit(
        f"Invalid batch/accumulation in config: batch={batch}, accumulation={acc}"
    )
print(f"[latent-unet-bridge-2h] effective_batch={batch*acc} (1 GPU)")

channels = list(params.get("channels", []))
if channels:
    downsample_levels = max(len(channels) - 1, 0)
    expected_bottleneck = 64 // (2 ** downsample_levels)
    print(
        f"[latent-unet-bridge-2h] expected_bottleneck_hw={expected_bottleneck}x{expected_bottleneck}"
    )
PY

"${PY}" -m models.train.core.train -c "${CFG}"
