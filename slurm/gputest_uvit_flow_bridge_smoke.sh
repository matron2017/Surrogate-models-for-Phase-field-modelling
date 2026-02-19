#!/bin/bash
# GPU smoke for latent flow/diffusion + thermal conditioning wiring.

#SBATCH --job-name=gputest_uvit_flow_bridge_smoke
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

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

echo "[gputest] host=$(hostname) job=${SLURM_JOB_ID:-na}"
nvidia-smi || true

cd "${ROOT}"

"${PY}" - <<'PY'
import torch
from models.backbones.uvit_film import UVitFiLMVelocity
from models.backbones.uvit_thermal import UVitThermalSurrogate
from models.latent_rf_unet_controlnet import LatentRFUNetControlNet

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in gputest job.")

device = torch.device("cuda:0")
torch.manual_seed(0)

# Diffusion-style thermal + time call
m_th = UVitThermalSurrogate(
    in_channels=64,
    out_channels=32,
    channels=[32, 64, 64],
    heads=4,
    film_dim=128,
    theta_in_ch=1,
).to(device).eval()
x_in = torch.randn(2, 64, 64, 64, device=device)
t = torch.randint(0, 1000, (2, 1), device=device).float()
theta = torch.randn(2, 1, 1024, 1024, device=device)
with torch.no_grad():
    y = m_th(x_in, t, theta=theta)
assert y.shape == (2, 32, 64, 64)
assert torch.isfinite(y).all()
print("uvit_thermal diffusion-style: OK", tuple(y.shape))

# Flow-style call on UViT thermal: model(x_t, t, x_cond, theta)
x_t = torch.randn(2, 32, 64, 64, device=device)
x_cond = torch.randn(2, 32, 64, 64, device=device)
with torch.no_grad():
    y2 = m_th(x_t, t, x_cond, theta)
assert y2.shape == (2, 32, 64, 64)
assert torch.isfinite(y2).all()
print("uvit_thermal flow-style: OK", tuple(y2.shape))

# UViT FiLM with cond_dim=0 should still respond to time and accept theta path.
m_film = UVitFiLMVelocity(
    in_channels=64,
    out_channels=32,
    cond_dim=0,
    channels=[32, 64, 64],
    heads=4,
    film_dim=128,
    use_theta=True,
    theta_in_ch=1,
).to(device).eval()
with torch.no_grad():
    z1 = m_film(x_in, t, theta=theta)
    z2 = m_film(x_in, t + 1.0, theta=theta)
assert z1.shape == (2, 32, 64, 64)
assert torch.isfinite(z1).all() and torch.isfinite(z2).all()
assert torch.max(torch.abs(z1 - z2)).item() > 0.0
print("uvit_film diffusion-style + theta + time-effect: OK", tuple(z1.shape))

# Latent RF controlnet style reference path.
m_rf = LatentRFUNetControlNet(
    Cz=32,
    channels=[64, 128, 128, 128],
    blocks_per_level=1,
    use_theta=True,
    theta_in_ch=1,
).to(device).eval()
theta_lat = torch.randn(2, 1, 64, 64, device=device)
with torch.no_grad():
    y_rf = m_rf(x_t, torch.rand(2, 1, device=device), x_cond, theta_lat)
assert y_rf.shape == (2, 32, 64, 64)
assert torch.isfinite(y_rf).all()
print("latent_rf_unet_controlnet flow-style: OK", tuple(y_rf.shape))

print("ALL_GPU_SMOKES_OK")
PY
