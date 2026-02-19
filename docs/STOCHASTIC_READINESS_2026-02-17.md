# Stochastic Readiness Note (2026-02-17)

## Scope

Goal: verify whether repeated predictions for the same input can produce meaningfully different outputs in practice, and whether that stochasticity helps on the rapid-solidification latent next-step task.

Canonical stack used:

- train/val/test latent H5 family: `data/latent_best_psgd_e279_dev/*_latent_experimental_midtrain.h5`
- AE decode checkpoint: `runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth`

## Jobs Run (gputest)

- `31921035` stochastic contract probe on main checkpoints
  - output: `results/diag_stochastic_contract_job31921035.json`
- `31921043` bridge overfit train on `[0,287,288,289,290]`
  - output dir: `runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8_gputest_overfit_bridge_288c/UNetFiLMAttn`
- `31921286` flow-stochastic overfit train on `[0,287,288,289,290]`
  - output dir: `runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic_gputest_overfit_flowstoch_288c/UNetFiLMAttn`
- `31921312` panel plots for bridge vs flow vs copy on exact requested times
  - output: `results/overfit_288c_bridge_flow_copy_panels_job31921312/`
- `31921319` stochastic contract probe on the new overfit checkpoints
  - output: `results/diag_stochastic_contract_overfit288c_job31921319.json`

## Requested 288k Overfit Window

Indices used: `[0, 287, 288, 289, 290]`

Mapped time pairs:

- `0`: `1 -> 1000`
- `287`: `287000 -> 288000`
- `288`: `288000 -> 289000`
- `289`: `289000 -> 290000`
- `290`: `290000 -> 291000`

## Main Findings

### 1) Global-checkpoint stochasticity

From `diag_stochastic_contract_job31921035.json`:

- `bridge_main` draw-to-draw decoded RMSE: `~9e-06`
- `flow_stochastic` draw-to-draw decoded RMSE: `~2.51e-01`

Interpretation:

- current bridge rollout path is effectively deterministic in deployment diagnostics,
- flow stochastic checkpoint can produce different samples from same input.

### 2) Overfit quality on requested window (deterministic rollout compare)

From `overfit_288c_bridge_flow_copy_panels_job31921312/manifest.json`:

Channel 0 RMSE mean:

- `bridge_overfit_288c`: `0.1686`
- `flow_overfit_288c`: `0.6806`
- `copy_past`: `1.6132`

Channel 1 RMSE mean:

- `bridge_overfit_288c`: `0.0462`
- `flow_overfit_288c`: `0.0887`
- `copy_past`: `0.1076`

Hard 288k transitions (`287000->291000`) are clearly better for bridge than flow in this short overfit setting.

### 3) Stochastic behavior on overfit checkpoints

From `diag_stochastic_contract_overfit288c_job31921319.json`:

- `bridge_overfit_288c` decoded draw-pair RMSE mean: `~5.34e-05`
- `flow_overfit_288c` decoded draw-pair RMSE mean: `~6.21e-04`

Interpretation:

- both overfit checkpoints are near-deterministic in practice on this tiny set,
- flow stochasticity collapsed strongly under this overfit regime.

## Practical Conclusion

- Task formulation is sound: `x_n + theta_n -> x_{n+1}` in latent space, decode with fixed AE.
- For stochastic claims, current bridge diagnostics are not yet reliable because `eta` has negligible effect in the current rollout formula.
- Flow can be stochastic on full checkpoints, but can still collapse under tiny overfit conditions.

## Recommended Next Steps Before Full Long Run

1. Keep deployment-style validation only (source-only rollout), never teacher-forced rollout metrics for model selection.
2. Use stochastic monitor for stochastic flow runs:
   - `val_num_samples >= 4`,
   - `val_deterministic = false`,
   - checkpoint monitor on `endpoint_crps` (with `endpoint_rmse` tracked alongside).
3. Keep deterministic side metric (`noise=0`) in reports, but do not use only single-sample RMSE to claim stochastic skill.
4. Treat bridge stochastic mode as open issue until `eta` variance path is fixed.
5. Run a medium-size (not tiny overfit) stochastic calibration sweep on `gputest` over `noise_stochastic_std` (e.g., `0.02, 0.05, 0.1`) and compare CRPS/spread/SSR.

## Related references (for method framing)

- Stochastic Interpolants / Follmer forecasting: https://arxiv.org/abs/2403.13724
- Forecasting with imperfect data via stochastic interpolants: https://arxiv.org/abs/2503.12273
- Schr"odinger bridge flow (unpaired translation): https://arxiv.org/abs/2409.09347
- Multi-marginal temporal SB matching: https://arxiv.org/abs/2510.01894
- Unified diffusion-bridge temporal framework (UniDB): https://arxiv.org/abs/2502.05749
- Fractional diffusion bridges: https://arxiv.org/abs/2511.01795
- Stochastic optimal control for diffusion bridges in function spaces: https://arxiv.org/abs/2405.20630
