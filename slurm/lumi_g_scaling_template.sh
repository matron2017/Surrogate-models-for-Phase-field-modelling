#!/bin/bash
#SBATCH --job-name=pf_lumi_scaling
#SBATCH --account=<PROJECT_ID>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Load your validated LUMI software stack/container before running.
# Example (adapt to your environment):
# module purge
# module load LUMI
# module load partition/G
# module load rocm

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Recommended for reproducibility in scaling tests.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# Replace with your actual training entrypoint and config.
srun python -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  models/train/core/train.py \
  --config configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8_smoke3ep.yaml
