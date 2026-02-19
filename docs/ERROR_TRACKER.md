# Bridge Error Tracker

Status of known bridge workflow errors for PF surrogate training.

## Why

This file is the first place to check when a new bridge run fails.

Use this with:

```bash
python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity ERROR
```

Also run before new runs:

```bash
python scripts/bridge_preflight_check.py --config <config.yaml>
```

## Known Issues (2026-02-15)

| ID | Category | Symptom | Date seen | Status | Fix location | Notes |
| --- | --- | --- | --- | --- | --- |
| B-001 | Plotter contract | `ValueError: Generative timestep is required ...` in `plot_bridge_endpoint_from_ckpt.py` | 2026-02-15 | ✅ Fixed | `scripts/plot_bridge_endpoint_from_ckpt.py` | Passed `(cond, timestep)` to model in plotting rollout instead of timestep-only. |
| B-002 | Watcher bug | `TypeError: _sync_latest_plots() got an unexpected keyword argument 'out_root'` | 2026-02-15 | ✅ Fixed | `scripts/watch_bridge_checkpoint_and_plot.py` | Corrected argument name to `run_root=...`. |
| B-003 | Checkpoint IO race | `PytorchStreamReader failed finding central directory` while watcher reads `checkpoint.last.pth` | 2026-02-15 | ⚠️ Mitigated | `scripts/watch_bridge_checkpoint_and_plot.py` | Added retry on checkpoint read to avoid partial file windows. |
| B-004 | Interrupted run status | `run.json` can remain `running` after hard cancellation | 2026-02-15 | ✅ Fixed | `models/train/core/train.py` | `SIGTERM` handled same as `SIGINT`; finalizer writes `interrupted` state. |
| B-005 | Flow-matching preflight false alarm | `scripts/bridge_preflight_check.py` reported `schedule_kwargs.timesteps must be > 0` + thermal path fails for FM configs | 2026-02-15 | ✅ Fixed | `scripts/bridge_preflight_check.py` | Updated preflight to branch checks by `train.model_family`; flow-matching now checks `train.objective` instead of diffusion-only fields. |
| B-006 | Gputest submit blocked | `AssocMaxSubmitJobLimit` on scheduler submit (`sbatch`) with user-level limit reached | 2026-02-15 | ⚠️ Known | `slurm` / scheduler policy | Keep one GPU test queued/running before submitting a second; retry after quota window reset or coordinate with partition policy. |
| B-007 | AFNO bridge overfit smoke launch syntax | 31846322 launch failed before execution due heredoc quoting in `sbatch --wrap` command | 2026-02-15 | ✅ Fixed | shell command + temp script generation | Use a proper here-doc/temporary script launcher and submit only after quota check passes. |
| B-008 | Deterministic CuBLAS abort on gputest | `RuntimeError: Deterministic behavior was enabled ...` before first step | 2026-02-15 | ✅ Fixed | `slurm/gputest_train_unet_bridge_latent_overfit.sh` | Add `CUBLAS_WORKSPACE_CONFIG=:16:8` in gputest launchers using torch deterministic mode. |
| B-009 | Flow val/checkpoint contract mismatch | Objective-space val metrics looked good while source-rollout predictions were copy-like/gridy | 2026-02-16 | ✅ Fixed | `models/train/core/loops.py`, `models/train/core/train.py` | Flow endpoint validation now uses source-only rollout; flow monitor defaults to `endpoint_rmse` (opt-out: `trainer.flow_monitor_force_endpoint=false`). |
| B-010 | Stochastic checkpoint metric under-specified | Single-sample endpoint RMSE can rank stochastic checkpoints poorly (under/over-dispersed ensembles) | 2026-02-16 | ✅ Fixed | `models/train/core/loops.py`, `models/train/core/train.py`, `models/train/core/logging.py` | Added rollout ensemble metrics (`endpoint_crps`, `endpoint_spread`, `endpoint_ssr`, `endpoint_ssr_distance`) and stochastic monitor default `flow_matching.val_monitor_metric_stochastic=endpoint_crps` when `val_num_samples>1`. |
| B-011 | SFM training contract missing in latent FM | Prior objectives lacked encoder-anchored adaptive-noise denoising contract from SFM paper | 2026-02-16 | ✅ Fixed | `models/train/core/loops.py`, `models/train/core/train.py`, `scripts/eval_latent_flow_bridge_decoded.py`, `scripts/plot_latent_flow_bridge_fields.py` | Added `train.objective=sfm_latent_source_denoise_concat` with adaptive `sigma_z`, encoder regularization, denoising loss, and matching rollout rule `v=(x_hat-x_t)/(1-t)` in train/eval/plot paths. |
| B-012 | SFM stochastic spread too low in overfit sanity | Ensemble spread remained very small (`SSR << 1`) despite stochastic start, indicating under-dispersion on current setup | 2026-02-16 | ⚠️ Monitoring | `configs/train/*sfm*`, `models/train/core/loops.py` | Reconfirmed on `gputest` eval jobs `31887414` (SFM stochastic, `flow_ensemble_latent_std=0.00103`) and `31887436` (SFM deterministic): `decoded_rmse` nearly identical (`0.63043` vs `0.63009`). Old objective replay stochastic (`31887776`) had slightly larger spread (`0.00501`) but much worse decoded error (`2.03979`), so raw spread increase alone is not sufficient. Keep `endpoint_crps` + `endpoint_ssr` in monitor set; tune `sfm_sigma_z`, `sfm_sigma_ema_beta`, and `sfm_encoder_reg_lambda` on gputest before long gpu runs. Latest check (`31921035`) keeps same direction: flow stochastic shows spread, but bridge path remains near-deterministic. |
| B-013 | Diagnostics crash on tiny datasets | Probe run can fail with `generator raised StopIteration` when `dataset_size < batch_size` and `loader.drop_last=true` | 2026-02-16 | ✅ Fixed | `models/train/core/loops.py` | `_iter_train_batches` now raises a clear actionable `RuntimeError` explaining the drop_last/batch-size mismatch instead of surfacing a cryptic generator failure. |
| B-014 | Diffusion val/deploy mismatch | Diffusion validation endpoint metric is computed from teacher-forced noisy states (`x_t` built using true `y`) instead of source-only rollout from `x`; this can hide deployment failures | 2026-02-17 | ❌ Open | `models/train/core/loops.py` | In val loop, diffusion branch samples `x_noisy` from `(x,y,t)` and computes `endpoint_rmse` from that path. See `models/train/core/loops.py` around `1513-1592`. |
| B-015 | Bridge rollout `eta` has near-zero effect | UniDB diagnostic rollout uses `base_var = c_t^2 - (coeff_xs*c_s)^2` with `coeff_xs = c_t/c_s`, making added variance numerically ~0, so `eta>0` is effectively deterministic | 2026-02-17 | ❌ Open | `scripts/eval_latent_flow_bridge_decoded.py`, `scripts/plot_bridge_endpoint_from_ckpt.py`, `scripts/plot_latent_flow_bridge_fields.py` | Probe showed `base_var` ~ `0` for almost all steps; `eta=0` and `eta=1` gave identical metrics for predict-next bridge diagnostics. Reconfirmed on `31921035` (`bridge_draw_pair≈9e-06`) and `31921319` (`bridge_draw_pair≈5e-05`) even with `eta=1.0`. |
| B-016 | Residual-modulator near-zero instability | With `residual_mode=abs`, `residual_normalize=true`, `residual_pi_floor=0`, many pixels have `pi≈0`; score uses division by `pi^2`, which can explode updates and decoded fields | 2026-02-17 | ⚠️ Monitoring | `models/diffusion/scheduler_registry.py` | `get_score_from_noise` divides by `sigma*pi^2` (`models/diffusion/scheduler_registry.py:290-292`). On sample checks, `~85%` pixels had `pi<1e-3` for multiple pairs; residual-reverse visuals showed concentration ranges around `[-376, 251]`. |
| B-017 | Bridge eval/plot input contract mismatch | Diagnostics and plots used raw concat instead of configured `input_mode`, skipped theta normalization, and in one plot path fed unbatched sample tensors; this made strong checkpoints appear copy-like/gridy | 2026-02-17 | ✅ Fixed | `scripts/eval_latent_flow_bridge_decoded.py`, `scripts/plot_latent_flow_bridge_fields.py`, `scripts/plot_bridge_endpoint_from_ckpt.py` | Reused training `_prepare_batch` preprocessing and `_build_diffusion_model_input`; fixed single-sample batching in latent plot path. |
| B-018 | Overfit harness inherited long-run LR warmup | Bridge gputest overfit sanity used base config warmup (`warmup_epochs=150`), causing tiny LR and misleading weak fit in short overfit runs | 2026-02-17 | ✅ Fixed | `slurm/gputest_train_unet_bridge_latent_overfit.sh` | Overfit config builder now forces warmup off by default (`FORCE_WARMUP_ZERO=1`), keeping sanity checks representative. |
| B-019 | Cross-style compare fairness mismatch | SFM/ReFlow checkpoints carried reduced eval caps from training config (`limit_per_group=2`, `max_items=0.1`), so style-comparison metrics were computed on 2 samples while bridge/flow-main used uncapped subsets | 2026-02-17 | ✅ Fixed | `scripts/eval_latent_flow_bridge_decoded.py` | Added explicit eval overrides: `--dataset-clear-caps`, `--dataset-limit-per-group`, `--dataset-max-items`. Fair rerun `gputest_pde_compare_fair_31916273.out` shows all styles on matching uncapped subset (`samples=16`). |
| B-020 | ReFlow objective underfit on 1-sample overfit sanity | `rectified_flow_noise_source_concat` does not overfit even a one-sample latent next-step task under matched gputest settings; endpoint error decreases slowly but remains high | 2026-02-17 | ❌ Open | `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_reflow2506_long_v1.yaml`, `slurm/gputest_train_flow_unet_latent_overfit.sh` | Matched 20-epoch overfit jobs: source-anchored best `val_endpoint_rmse=0.04947` (`31917538`), SFM best `0.03014` (`31918142`), ReFlow best `0.43550` (`31918456`). Suggests random-noise-start objective is currently ill-conditioned for this one-step PDE mapping (at least with current concat/conditioning and short-horizon optimization). |
| B-021 | Flow stochastic overfit collapsed to near-deterministic outputs on 288k subset | Even with stochastic config (`cond_dim=2`, `noise_stochastic_std=0.05`), repeated draws stayed almost identical on the 5-sample overfit stress set | 2026-02-17 | ❌ Open | `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic.yaml`, `models/train/core/loops.py` | Overfit run `31921286` + stochastic probe `31921319`: `decoded_draw_pair_rmse_mean≈6.21e-4` and `decoded_ens_spread_mean_std≈1.92e-4`, while deterministic pair RMSE is `0.0`. This indicates practical stochastic collapse in the tiny-set regime. |

## Severity and action mapping

- **ERROR**: stop before rerun, requires explicit remediation.
- **WARN**: monitor; rerun is usually safe but capture new stack trace for trend.
- **INFO**: noise for long run diagnostics.

## How to extend

When a new issue appears:

1. Add the pattern to `scripts/report_bridge_errors.py`.
2. Add one row in the table above with:
   - date, first seen job id, and category
   - status (`New`, `Fixed`, `Mitigated`)
   - exact fix file/path
3. Add the failing job in `EXPERIMENT_STATUS` if it affects scheduling decisions.
