# Flow-Matching Latent Debug Log (2026-02-16)

## Scope

This note tracks the current latent flow issue where rollout predictions appear copy-like and show grid-like artifacts.

Canonical stack enforced for all checks in this note:

- Train H5: `/scratch/project_2008261/pf_surrogate_modelling/data/latent_best_psgd_e279_dev/train_latent_experimental_midtrain.h5`
- Val H5: `/scratch/project_2008261/pf_surrogate_modelling/data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5`
- AE checkpoint: `/scratch/project_2008261/pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth`
- Flow checkpoint: `/scratch/project_2008261/pf_surrogate_modelling/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth`

Verified active training data paths in log:

- `logs/slurm/latent_unet_flowmatch24h_rdbm_afno8_b64_1n4g_sto_31855650.out` (`train_h5`/`val_h5` lines 33-36).

---

## Error List (One by One)

### F-001: Invalid old diagnostics due AE/H5 mismatch

- Status: Confirmed, isolated.
- Evidence:
  - `results/diag_flow_decoded_val_job31852644.json` -> `decoded_rmse=4.269053339958191` (mismatch).
  - `results/diag_flow_decoded_val_job31852655.json` -> `decoded_rmse=4.279776096343994` (mismatch).
  - Both mismatch `README_EXPERIMENTAL.json` checkpoint for `latent_best_psgd_e279_dev`.
- Impact:
  - These files should not be used for model quality conclusions.

### F-002: Rollout is worse than a copy baseline on canonical val subset

- Status: Confirmed, critical.
- Evidence:
  - Rollout eval (no noise): `results/diag_flow_decoded_val_job31867111.json`
    - `latent_rmse=0.28956607915461063`
    - `decoded_rmse=0.11649247165769339`
  - Copy baseline: `results/diag_copy_baseline_val_job31867644.json`
    - `latent_rmse=0.020369624966406263`
    - `decoded_rmse=0.09137287581688724`
- Impact:
  - Current rollout does not beat a trivial predictor (`y_hat = x`) on the tested subset.

### F-003: Validation objective is teacher-forced and optimistic vs rollout

- Status: Confirmed, root-cause class. Mitigation implemented in training loop.
- Evidence:
  - Validation code builds `x_t` with true target `y`:
    - `models/train/core/loops.py:1335` (`x_t = (1.0 - t_view) * z0 + t_view * y`)
    - `models/train/core/loops.py:1357` (`endpoint_pred = x_t + (1.0 - t_view) * pred`)
  - Rollout code integrates from source state:
    - `scripts/eval_latent_flow_bridge_decoded.py:318` (Euler rollout from `x_t(0)=x`)
  - Direct teacher-vs-rollout diagnostic:
    - `results/diag_flow_teacher_vs_rollout_job31867690.json`
    - Summary: `latent_rmse_rollout.mean=0.2788`, `latent_rmse_teacher.mean=0.1019`, `latent_rmse_copy.mean=0.02037`.
- Impact:
  - Good validation numbers can coexist with poor free rollout behavior.

### F-008: Flow checkpoint monitor could select objective-good but rollout-bad checkpoints

- Status: Confirmed, fixed by default monitor policy for flow runs.
- Evidence:
  - `trainer.early_stop.metric` was often left as `mae`, which tracks objective-space error.
  - Flow deployment target is endpoint rollout quality (`x_n -> x_{n+1}`), not objective-space velocity fit.
- Fix:
  - `models/train/core/train.py` now overrides monitor metric to `endpoint_rmse` for flow+val runs when endpoint metrics are enabled and an objective-like metric is requested.
  - Opt-out knob: `trainer.flow_monitor_force_endpoint: false`.

### F-004: Pairing bug hypothesis (identity targets) is not supported

- Status: Rejected hypothesis.
- Evidence:
  - Data loader uses explicit pair indices when `use_pairs_idx=True`:
    - `models/train/core/pf_dataloader.py:126-129`
    - `models/train/core/pf_dataloader.py:439-444`
  - Input/target read from `i` and `j`:
    - `models/train/core/pf_dataloader.py:456-464`
  - H5 stats check: `same_pairs=0`, `mean_abs_idx_gap=1.0`, `max_abs_idx_gap=1`.
  - Time deltas mostly +1000 Euler steps (as intended for next-step framing).
- Impact:
  - Dataset pairing itself is not accidentally identity-copy.

### F-005: Concentration checkerboard amplification appears in rollout, not in copy baseline

- Status: Confirmed on sampled indices, open for full-dataset quantification.
- Evidence:
  - Rollout-only checker diagnostic:
    - `results/diag_checkerboard_val_job31866389.json`
    - concentration `pred_over_gt` ratio mean `1.4803`, median `1.0238`.
  - Rollout vs copy checker diagnostic:
    - `results/diag_checkerboard_rollout_vs_copy_job31868296.json`
    - concentration:
      - `rollout_over_gt_mean=1.4183`
      - `copy_over_gt_mean=0.8415`
    - phase remains close to parity for both (`~0.99` rollout, `~0.97` copy means).
- Impact:
  - Concentration grid energy is more amplified by rollout prediction than by pure decode(copy), so this is not only a decoder baseline effect.

### F-006: Stochastic source perturbation is not the primary failure mode

- Status: Partially ruled out.
- Evidence:
  - With noise off:
    - `results/diag_flow_teacher_vs_rollout_noiseablate_job31868190.json`
    - `noise_std=0.0`, `noise_perturb_source=false`
    - `latent_rmse_rollout.mean=0.2896` (still poor).
  - With checkpoint noise settings:
    - `results/diag_flow_teacher_vs_rollout_job31867690.json`
    - `latent_rmse_rollout.mean=0.2788`.
- Impact:
  - Disabling stochastic perturbation does not remove the rollout failure.

### F-007: Copy baseline is strong because many next-step pairs are near-identity (with a hard tail)

- Status: Confirmed dataset characteristic.
- Evidence:
  - `results/diag_latent_pair_delta_stats_val_20260216.json` on full canonical val pairs (`n=3345`):
    - median latent pair delta RMSE (`p50`) = `0.02384`
    - low-motion quartile (`p25`) = `0.01661`
    - hard-tail (`p75`) = `0.10607`, (`p95`) = `0.15075`
  - `idx_gap_counts` confirms all pair gaps are `1`; `dt_euler_counts` are almost all `1000`.
- Impact:
  - A copy predictor can score well on aggregate RMSE, but it still fails on the high-change tail where forecasting quality matters most.

---

## What Is Still Open (Current Thinking)

1. Training/selection metric contract is misaligned with deployment metric.
   - Mitigated for flow runs with endpoint metrics enabled; still verify in long runs.
2. Model learns local velocity targets under mixed states but does not integrate to correct endpoint under source-only rollout.
3. Concentration branch is more sensitive to high-frequency amplification than phase, possibly from latent dynamics plus decoder upsampling (`ConvTranspose2d` in `models/backbones/unet_film_bottleneck.py:407`).
4. Metric scale consistency between scripts needs tighter control (raw decoded vs denormalized decoded reporting).

---

## Immediate Next Debug Steps

1. Add rollout-based validation metric collection in the training loop and use it for checkpoint selection.
   - Status: Done in `models/train/core/loops.py` and `models/train/core/train.py`.
2. Run a small `gputest` NFE sweep (`1, 5, 10, 20`) with identical subset and fixed seeds.
3. Add a concentration-focused spectral penalty/monitor in endpoint space and compare against copy baseline.
4. Expand checkerboard comparison from 8 indices to a larger deterministic slice to quantify tail behavior.

---

## Validation Contract Update (2026-02-16)

Flow validation endpoint metrics now use source-only rollout:

- rollout integration with configurable `flow_matching.val_rollout_nfe`,
- stochastic validation via ensemble mean (`flow_matching.val_num_samples`),
- deterministic validation mode (`flow_matching.val_deterministic=true`) for stable one-sample checkpointing.

Updated files:

- `models/train/core/loops.py`
- `models/train/core/train.py`
- `models/train/core/config.py`
- `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8*.yaml`

---

## Stochastic Validation Update (2026-02-16)

To align with SFM-style probabilistic evaluation for stochastic rollouts, validation now computes ensemble-aware endpoint metrics:

- `endpoint_crps` (ensemble CRPS),
- `endpoint_spread` (sqrt of ensemble variance),
- `endpoint_ssr` (spread-skill ratio),
- `endpoint_ssr_distance` (`|SSR-1|` calibration gap).

Behavior:

- computed on source-only rollout samples (`flow_matching.val_num_samples` members),
- enabled by default with `flow_matching.val_probabilistic_metrics=true`,
- flow monitor defaults to `flow_matching.val_monitor_metric_stochastic` (`endpoint_crps`) when stochastic validation is active (`val_num_samples>1` and `val_deterministic=false`).

Updated files:

- `models/train/core/loops.py`
- `models/train/core/train.py`
- `models/train/core/config.py`
- `models/train/core/logging.py`
- `tests/test_flow_objectives_rs.py`

---

## Implemented SFM-Style Latent Objective (2026-02-16)

Added a new flow objective to keep the existing UNet+AFNO backbone while adopting the SFM training contract in latent space:

- `train.objective: sfm_latent_source_denoise_concat`

Training behavior:

- deterministic encoder anchor uses source latent (`E(y_cond) := x_n` in current implementation),
- adaptive noise scale `sigma_z` (EMA of source-target RMSE, bounded by `[sfm_sigma_min, sfm_sigma_max]`),
- denoising perturbation `x_sigma = x_target + sigma * (e + eps)` with `e = (E - x_target)/sigma_z`,
- weighted denoising loss `(sigma_z/sigma)^2 * ||x_hat - x_target||^2 + lambda * ||e||^2`.
- `sigma_z` diagnostics are logged per epoch in task metrics as:
  - `sfm_sigma_z` (EMA value used by training),
  - `sfm_sigma_z_batch_rmse` (latest batch RMSE estimate).

Rollout behavior:

- start from `z0 = E + sigma_eval * eps`,
- model predicts `x_hat`,
- convert to velocity with `v = (x_hat - x_t)/(1 - t)`,
- integrate in flow-time with Euler.

Updated files:

- `models/train/core/loops.py`
- `models/train/core/train.py`
- `models/train/core/config.py`
- `scripts/eval_latent_flow_bridge_decoded.py`
- `scripts/plot_latent_flow_bridge_fields.py`
- `scripts/bridge_preflight_check.py`
- `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_sfm_latent.yaml`

---

## SFM Sanity Checks (2026-02-16, gputest)

1. Smoke train (`31887340`) completed with objective/monitor contract active:
   - objective `sfm_latent_source_denoise_concat`,
   - monitor metric auto-switched to `endpoint_crps`,
   - epoch log includes `sfm_sigma_z` and `sfm_sigma_z_batch_rmse`.

2. Tiny overfit run (`31886888`) shows the model can fit beyond pure copy under the new objective:
   - `val_endpoint_rmse`: `0.1669 -> 0.0272` over 6 epochs.

3. Copy-vs-rollout diagnostic (`31887065`, result file `results/diag_sfm_copy_vs_rollout_overfit.json`):
   - `copy_rmse_mean = 0.4648`,
   - `rollout_rmse_det_mean = 0.3078`,
   - `rollout_rmse_ensmean_n4_mean = 0.3078`.
   This confirms source-only rollout is better than identity copy for that stress subset.

4. Data-gap distribution check on latent pairs (first 1600 samples each split):
   - train: mean `0.0466`, median `0.0186`, p95 `0.1523`, max `0.4648`,
   - val: mean `0.0484`, median `0.0203`, p95 `0.1505`, max `0.4648`.
   Interpretation: most transitions are small, with a heavy-tail of hard transitions; copy-like behavior can look good on easy cases, so tail-sensitive metrics and rollout checks are required.

5. Deterministic vs stochastic rollout parity check (`gputest`, 2026-02-16):
   - stochastic eval job `31887414` -> `results/diag_flow_decoded_val_job31887414.json`,
   - deterministic eval job `31887436` -> `results/diag_flow_decoded_val_job31887436.json`.
   - both used the same SFM overfit checkpoint and canonical val latent H5.
   - metrics were nearly identical:
     - stochastic: `latent_rmse=0.3078507`, `decoded_rmse=0.6304327`, `flow_ensemble_latent_std=0.0010304`,
     - deterministic: `latent_rmse=0.3078158`, `decoded_rmse=0.6300870`.
   Interpretation: current stochasticity is still under-dispersed (ensemble spread too small to materially change endpoint error), so additional calibration/tuning is still required.

6. Overfit A/B comparison against past flow objective (`gputest`, fair replay):
   - old objective replay (same tiny overfit slice, 6 epochs): job `31887718`,
     run dir `runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_predictnext_nomass_afno8_gputest_overfit_replay6e`.
   - old objective decoded evals:
     - deterministic `31887755` -> `results/diag_flow_decoded_val_job31887755.json`
       (`latent_rmse=0.322289`, `decoded_rmse=2.043172`),
     - stochastic `31887776` (`noise_std=0.05`, `num_samples=4`) -> `results/diag_flow_decoded_val_job31887776.json`
       (`latent_rmse=0.321135`, `decoded_rmse=2.039792`, `flow_ensemble_latent_std=0.005012`).
   - SFM reference decoded evals (same slice):
     - deterministic `31887436`: `latent_rmse=0.307816`, `decoded_rmse=0.630087`,
     - stochastic `31887414`: `latent_rmse=0.307851`, `decoded_rmse=0.630433`, `flow_ensemble_latent_std=0.001030`.
   - copy baseline reference on this slice:
     - `results/diag_sfm_copy_vs_rollout_overfit.json`: `copy_rmse_mean=0.464805`.
   Interpretation:
   - both objectives are below copy baseline in latent RMSE on this slice (`0.32` / `0.31` vs `0.4648`), so neither is pure identity copy,
   - SFM is clearly less ill-posed than the old flow objective in decoded-space quality (`decoded_rmse ~0.63` vs `~2.04`),
   - stochastic spread remains small for both; this is still a calibration issue rather than a data-pairing contract bug.

---

## Implemented PBFM-Style Option (2026-02-16)

To align with the requested PBFM-like training style while keeping source conditioning for next-step forecasting, a new flow objective was added:

- `train.objective: rectified_flow_noise_source_concat`

Behavior:

- start state in flow-time is random noise (`z0 ~ N(0, I)`),
- model still receives past state context via concat (`[x_t, x_n]`),
- thermal field still enters through existing theta/control-branch path.

Files updated:

- `models/train/core/loops.py` (train/val objective routing)
- `models/train/core/train.py` (objective registration for no-`torchcfm` path)
- `scripts/eval_latent_flow_bridge_decoded.py` (rollout matches objective contract)
- `scripts/plot_latent_flow_bridge_fields.py` (visual rollout matches objective contract)
- `scripts/bridge_preflight_check.py` (objective allowed list)
- `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_pbfmstyle.yaml` (new config)

First smoke run:

- train smoke job `31870983` (`gputest`) completed with objective correctly logged as `rectified_flow_noise_source_concat`.
- decoded rollout eval job `31870999` completed and produced:
  - `results/diag_flow_decoded_val_job31870999.json`
  - very poor metrics expected from a 1-epoch/2-step smoke (not quality-significant yet).

---

## 2026-02-17 Follow-up (Requested 288k Consecutive Window)

Exact overfit subset used:

- indices `[0, 287, 288, 289, 290]`
- mapped pairs: `1->1000`, `287000->288000`, `288000->289000`, `289000->290000`, `290000->291000`

Jobs:

- bridge overfit: `31921043`
- flow-stochastic overfit: `31921286`
- side-by-side plots: `31921312` -> `results/overfit_288c_bridge_flow_copy_panels_job31921312/`
- stochastic probe on these overfit checkpoints: `31921319`

Deterministic panel summary (`manifest.json`):

- channel 0 RMSE mean:
  - bridge_overfit_288c: `0.1686`
  - flow_overfit_288c: `0.6806`
  - copy_past: `1.6132`
- channel 1 RMSE mean:
  - bridge_overfit_288c: `0.0462`
  - flow_overfit_288c: `0.0887`
  - copy_past: `0.1076`

Interpretation:

- On this stress subset, bridge overfit is substantially better than flow overfit on the concentration-like channel.
- Both beat copy baseline by a wide margin on hard 288k transitions.

Stochastic probe result on new overfit checkpoints (`diag_stochastic_contract_overfit288c_job31921319.json`):

- bridge draw-pair decoded RMSE mean: `5.34e-05`
- flow draw-pair decoded RMSE mean: `6.21e-04`
- deterministic repeat RMSE mean: `0.0` for both

Interpretation:

- In this tiny overfit regime both models are effectively near-deterministic (stochastic collapse), even when flow uses the stochastic config.
- This reinforces the requirement to judge stochasticity on larger uncapped validation subsets with ensemble metrics (`CRPS`, `spread`, `SSR`), not on tiny overfit-only checks.
