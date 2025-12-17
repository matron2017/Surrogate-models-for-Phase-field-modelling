## Puhti Conda/Apptainer Environment Notes

The packaged Tykkÿ container we copied from Mahti lives at:

- `ENV_ROOT=/scratch/project_2008261/physics_ml`
- `POST_INSTALL=/scratch/project_2008261/post-install.sh`

Whenever you need to refresh Python/pip or add packages, run the following Slurm
job (same recipe we used on Mahti, minor edits for Puhti paths):

```bash
#!/bin/bash
#SBATCH --job-name=prepare_env
#SBATCH --account=project_2008261
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --gres=nvme:50
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

module --force purge
module load tykky

PROJECT_ROOT="/scratch/project_2008261"
ENV_ROOT="${PROJECT_ROOT}/physics_ml"
POST_INSTALL="${PROJECT_ROOT}/post-install.sh"
PROJECT_SCRATCH="${PROJECT_ROOT}/tmp/${USER}"
JOB_SCRATCH="${PROJECT_SCRATCH}/tykky-${SLURM_JOB_ID:-$$}"

mkdir -p "${JOB_SCRATCH}"
chmod 700 "${JOB_SCRATCH}"
export TMPDIR="${JOB_SCRATCH}"
export CW_BUILD_TMPDIR="${JOB_SCRATCH}"
export CW_LOCAL_SCRATCH="${PROJECT_SCRATCH}"

chmod +x "${POST_INSTALL}"
echo "[info] Running: conda-containerize update"
set -x
conda-containerize update "${ENV_ROOT}" --post-install "${POST_INSTALL}"
set +x
```

The `post-install.sh` hook is where you keep the pip upgrades (e.g. `pip install -U pip`,
`pip install ruamel.yaml psutil PyWavelets aiida-core`) and local extras like
`/scratch/project_2008261/pytorch_wavelets`:

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "[post-install] Python: $(python -c 'import sys; print(sys.version)')"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir ruamel.yaml psutil PyWavelets aiida-core
python -m pip install --no-cache-dir /scratch/project_2008261/pytorch_wavelets
```

Rebuild the container via `conda-containerize update …` whenever you change
`post-install.sh` so the packaged environment stays consistent on Puhti.

## Installing Hugging Face Diffusion Dependencies

The rapid_solidification diffusion prototype uses Hugging Face's `diffusers`,
`transformers`, and `accelerate` packages. Install or update them inside the
Tykkÿ container with:

```bash
/scratch/project_2008261/physics_ml/bin/pip3 install --upgrade \
  'diffusers[torch]' transformers accelerate
```

The command defaults to the user site-packages path because the container image
is read-only; the environment automatically appends `~/.local` to `sys.path`, so
Python can still import the freshly installed packages. Re-run the command
whenever you rebuild the container so the baked image includes the latest HF
packages.

## Optional SBATCH Notifications

To avoid polling `squeue`, add mail notifications to any Slurm launcher:

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>@example.com
```

Alternatively, append `--comment notify` to any submission and watch for the job
transition via `sacct -X -j <jobid> --format=JobID,State,End`. For interactive
alerts without email, wrap the `sbatch` command with `squeue --wait` or
`squeue --job <jobid> --start` to block until the allocation begins/finishes.
