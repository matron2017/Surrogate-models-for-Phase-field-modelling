## Rapid Solidification — Current Experiment Log

Latest checkpoints use the placeholder HDF5 dataset under `data/rapid_solidification` because `/scratch/project_2008261/alloy_solidification/data/pf_data` has not been copied from Mahti yet (see `README.md:56-64`). Keep using the smoke configs until the VTU bundle lands.

### Datapipes smoke (job 30877586)
- Launcher: `sbatch datapipes/smoke_build.sh`
- Log: `logs/slurm/rs_build_smoke_30877586.out:1`
- Status: **Failed** — `RuntimeError: No VTU in G1.2e6_V200` because `/scratch/project_2008261/alloy_solidification/data/pf_data` is missing on Puhti.
- Next step: create/copy the VTU bundle on Puhti and re-run the smoke build before chaining datapipes+train+visuals.

### Training smoke (job 30877587)
- Launcher: `sbatch slurm/train_smoke.sh`
- Log: `logs/slurm/rs_train_smoke_30877587.out:1-15`
- Status: **Completed** — ran UNet SSA with descriptors logged, produced checkpoints/plots under `runs_debug/smoke_train/UNet_SSA_PreSkip_Full`.
- Metrics: `epoch=1`, `val_rmse=1.574872`, `val_mse=2.480223`.

### Visuals smoke (job 30877633)
- Launcher: `sbatch visuals/basic/run_smoke.sh`
- Log: `logs/slurm/rs_visual_smoke_30877633.out`
- Status: **Completed** — silent log except for Slurm prelude, which is expected when the plotting script finishes without warnings.
- Outputs: plots and diagnostics stored under `results/rapid_solidification/smoke_visuals` (refer to the script config for exact paths).

### Diffusion smoke (job 30876242)
- Launcher: `sbatch slurm/train_diffusion.sh`
- Log: `logs/slurm/diffusion_smoke_30876242.out:1-5`
- Status: **Completed** — saved checkpoint `runs_debug/diffusion_smoke/epoch001.pt`.
- Metrics: `epoch=1 step=1 loss=1.041203`.

### Multi-node DDP test (job 30878234)
- Launcher: `sbatch slurm/test_parallel_multinode.sh`
- Log: `logs/slurm/rs_ddp_2x4_30878234.out:1-90`
- Status: **Completed** — DDP connectivity check succeeded across `r01g01` + `r02g01` (8× V100). NCCL warns about deprecated `NCCL_ASYNC_ERROR_HANDLING`; switch to `TORCH_NCCL_ASYNC_ERROR_HANDLING` in env when convenient.
- Metrics: `epoch=0 mean_loss=0.27118`, `time=36.89s`, `throughput=1.7 img/s` (see the Slurm log for details).

### Action items
1. Copy `/scratch/project_2008261/alloy_solidification/data/pf_data` from Mahti so datapipes can rebuild the smoke bundle.
2. Re-run `run_next_steps.sh` after the copy to validate the chained datapipes/train/visual workflow with real VTUs.
3. Update training/visual configs once the higher-resolution dataset is in place.
