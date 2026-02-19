# Bridge Workflow Diagnostics (2026-02-15)

## Current state snapshot

- Repo path: `/scratch/project_2008261/pf_surrogate_modelling`
- Latent bridge long run: `31841349`
- Main data: `data/latent_best_psgd_e279_dev/{train,val}_latent_experimental_midtrain.h5`
- Active bridge config: `configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8.yaml`
- Active run outputs: `runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8/UNetFiLMAttn`

## Data and model contract (what is on purpose)

- Train model reads latent pairs with thermal precomputed and conditioning via `use_theta=true`.
- AE source: `latent_best_psgd_e279_dev` (latest trained latent AE used as input to bridge).
- `normalize_images=false`, `normalize_source=file`, z-score dataset path assumes latent pre-normalized.
- Training objective: `diffusion_objective=unidb_predict_next`, `diffusion.noise_schedule=unidb_cosine`.
- Channel flow: `in_channels=64`, `out_channels=32`, `input_mode=delta_source_concat`.
- AFNO bottleneck path with ConvTranspose2d decoder and no bottleneck attention.
- Model family: `diffusion`, AFNO params are full trainable in the active run (`total=trainable=452,627,872`).

## Measured diagnostics (now)

- `31841349` running status: `RUNNING` (state from `squeue`).
- Active `run.json` `status.state = running` and run path above.
- Best bridge validation metrics so far (from `metrics.csv`, split=`val`):
  - best `endpoint_rmse = 0.07968549` at epoch `178`
  - best `mae = 0.02843919` at epoch `190`
  - best `vrmse = 0.13271703` at epoch `178`
  - `val_endpoint_spectral_rmse` best reached `40.95748` at epoch `172`
- Long-run residual concern remains: historical gputest snapshots show concentration prediction range collapse when compared to GT in fixed index plots (`results/bridge_best_afno8_plot/bridge_endpoint_pred_idx0234...png`).

## Gputest comparison status and blocker

- Existing Flow-matching smoke run (`31846052`) succeeded and shows very poor first-epoch endpoint quality:
  - `val_endpoint_rmse=0.395414`, `val_endpoint_spectral_rmse=48157.7168`, `val_spectral_rmse=1088112.046875`.
- AFNO bridge overfit gputest (`31846482`) succeeded with strict 1-epoch/2-steps, overfit_n=2 on gputest.
  - `val_endpoint_rmse=0.47347759`, `val_endpoint_spectral_rmse=355342.5625`, `val_spectral_rmse=355342.5625`.
- Flow-matching overfit gputest (`31846558`) succeeded with same data reduction.
  - `val_endpoint_rmse=0.54362558`, `val_endpoint_spectral_rmse=86143.15625`, `val_spectral_rmse=180811.1719`.
- Both overfit modes were weak and unstable in first-epoch physics-scale metrics, confirming need for stronger diagnostics (windowed rollout, multi-epoch stress, maybe schedule/objective review).
- New gputest blocker encountered and fixed: deterministic CuBLAS crash when `CUBLAS_WORKSPACE_CONFIG` missing; both bridge and flow overfit launchers now set it.

## Error history snapshot (auto-detected)

`python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity WARN --include-out`
returned WARN/ERROR rows dominated by historical issues:

- `TypeError: _sync_latest_plots() got an unexpected keyword argument 'out_root'`
- `ValueError: Generative timestep is required ...`
- `RuntimeError: Traceback` stack rows in watch/plot scripts

Both are fixed in committed bridge tooling updates (older versions still present in historical log files).

## Likely hypotheses to test next

1. **Residual path / objective mismatch in training vs plotting**
   - Ensure the same normalization, theta affine settings, and residual_mode are used in train and visualization.
2. **Data split and stochastic realism**
   - Confirm AE dataset and h5 attributes are consistent (`normalization_schema`).
3. **Model underfitting on concentration dynamics in tiny-batch runs**
   - Keep flow-matching baseline as comparison if bridge behavior remains unstable in small-batch tests.
4. **Overfit gputest needed**
   - Submit 1 overfit run per architecture family only after scheduler quota frees.

## How to run automated preflight + error checks before any launch

```bash
cd /scratch/project_2008261/pf_surrogate_modelling
PY=/scratch/project_2008261/physics_ml/bin/python3.11
$PY scripts/bridge_preflight_check.py --config <config.yaml>
$PY scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity WARN
```

Recommended full preflight: `scripts/check_bridge_readiness.py --config <config.yaml> --logs-dir logs/slurm --strict`.


## Quick gputest comparison (overfit_n=2, 1 epoch)

- `gputest_train_unet_bridge_latent_overfit.sh` (job `31846482`) now auto-generates:
  - `epochs=1`, `steps_per_epoch=2`, `loader.overfit_n=2`, `batch_size=1`, `num_workers=0`.
  - `val_endpoint_rmse=0.47347759`
  - `val_endpoint_spectral_rmse=355342.5625`

- `gputest_train_flow_unet_latent_overfit.sh` (job `31846558`) now auto-generates with same reduction and
  - `val_endpoint_rmse=0.54362558`
  - `val_endpoint_spectral_rmse=86143.15625`
  - `val_spectral_rmse=180811.1719`

Interpretation:
- Bridge overfit is slightly better on endpoint RMSE than flow-match under this constrained 1-epoch overfit regime, but both are far from production-quality and both show very large spectral residuals.

Next diagnostics to run next:
- run 2-3 epoch overfit to check if endpoint improves and if spectral collapses with extra time
- compare with `cfg['diffusion'].scheduler` and objective variants only after confirming train-script deterministic settings are consistent
