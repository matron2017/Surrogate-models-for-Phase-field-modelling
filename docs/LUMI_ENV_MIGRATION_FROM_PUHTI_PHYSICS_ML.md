# Puhti `physics_ml` -> LUMI Environment Migration (2026-02-19)

## Goal

Run the PF surrogate workflows on LUMI-G with a ROCm-compatible stack while
keeping the training command surface compatible with existing scripts.

## 1. Baseline Comparison

Puhti env sample (`/scratch/project_2008261/physics_ml`):
- `torch==2.8.0`
- `torchaudio==2.8.0+cu128`
- many CUDA/NVIDIA packages (`nvidia-*`)
- optional packages used in code paths: `heavyball`, `pytorch_wavelets`, `torchcfm`, `dgl`

LUMI module stack sample:
- `module load LUMI partition/G Local-CSC/default pytorch/2.5`
- `torch==2.5.1+rocm6.2`
- includes core scientific stack (numpy/scipy/h5py/matplotlib/mlflow)
- missing from module env: `heavyball`, `pytorch_wavelets`, `torchcfm`, `physicsnemo`, `vtk`, `cupy`, `pynvml`

## 2. Migration Strategy

Use a layered environment:

1. Base from LUMI modules (provides ROCm PyTorch).
2. Virtualenv with `--system-site-packages` for additional Python packages needed by this repo.
3. Keep legacy CUDA-specific packages out of the LUMI env.

## 3. Setup Script

Use:
- `scripts/lumi_setup_physics_ml_env.sh`

Default env path:
- `/scratch/project_462001306/pf_surrogate_modelling/.venv_physics_ml_lumi`

The script:
- loads the LUMI module stack,
- creates a virtualenv on top of module Python,
- installs extra dependencies from `env/lumi/requirements-extra.txt`,
- runs an import sanity check.

## 4. Script/Batch Changes Required on LUMI

For LUMI runs:

1. Use `slurm/lumi_g_*.sh` launchers (already added).
2. Avoid Puhti launchers with:
- `--gres=gpu:v100:*`
- `--account=project_2008261`
- hardcoded `/scratch/project_2008261/...`
- CUDA-specific assumptions (`CUBLAS_WORKSPACE_CONFIG`, etc.)

3. In LUMI launchers:
- `--ntasks-per-node=8`
- `--gpus-per-node=8`
- `--cpus-per-task=7`
- `--cpu-bind=cores --gpu-bind=closest`

4. Keep paths configurable via environment variables:
- `PROJECT_ROOT`, `TRAIN_H5`, `VAL_H5`, `SIM_MAP`, `PYTHON_BIN`

## 5. Recommended Validation Sequence

1. Environment sanity:
- `python -c "import torch; print(torch.__version__, torch.version.hip)"`

2. Single-node smoke:
- `sbatch slurm/lumi_g_ae_smoke.sh`
- `sbatch slurm/lumi_g_backbone_scaling_job.sh`

3. Multi-node short tests:
- strong and weak 1/2/4 node points first

4. Only then launch longer scaling jobs.

## 6. Notes on Optional Packages

- `physicsnemo`, `cupy`, `vtk`, `pynvml` are not required for the core LUMI scaling campaign.
- Add them only if a specific active code path requires them.
