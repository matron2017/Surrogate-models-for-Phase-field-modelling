## Bridge Sanity Checklist

Use this checklist before submitting, and again after first validation epoch.

### 0) Preflight checks
- [ ] Run `python scripts/bridge_preflight_check.py --config <config.yaml>`.
- [ ] Run `python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity WARN`.

### 1) Objective vs Inference Contract
- [ ] `loss.diffusion_objective` matches inference math.
- [ ] If objective is `unidb_predict_next`/`predict_next`, model output is treated as endpoint prediction (not epsilon).
- [ ] If objective is epsilon/reverse-step, rollout uses epsilon inversion consistently.

### 2) Schedule and Timestep Sampling
- [ ] `diffusion.noise_schedule` and objective are compatible (UniDB objective only with `unidb_*` schedule).
- [ ] `sampler_kwargs.t_min >= 1` and `t_max <= timesteps`.
- [ ] For UniDB training, terminal sampling policy is explicit (`include_terminal_unidb` set intentionally).

### 3) Input Wiring and Channels
- [ ] For `input_mode: delta_source_concat`, `model.params.in_channels == 2 * model.params.out_channels`.
- [ ] Thermal conditioning path is explicit:
  - [ ] `conditioning.use_theta: true`
  - [ ] `dataloader.args.add_thermal: true`
  - [ ] `dataloader.args.thermal_require_precomputed: true`
- [ ] Control branch hint channels match thermal channels.

### 4) Residual Bridge Parameters
- [ ] `residual_scale > 0`
- [ ] `residual_power > 0`
- [ ] `residual_clip` positive (or null intentionally)
- [ ] `residual_pi_floor >= 0`

### 5) Data and Normalization
- [ ] H5 `normalization_schema` matches dataloader behavior.
- [ ] No accidental double-normalization.
- [ ] Phase/concentration decoding scale checked in visuals.

### 6) Metrics (Objective-space + Endpoint-space)
- [ ] Objective metric logged (loss-space).
- [ ] Endpoint metrics logged (`endpoint_rmse`, `spectral_rmse`, `vrmse`).
- [ ] Model selection criterion is endpoint metric for deployment decisions.

### 7) First-Epoch Runtime Checks
- [ ] Log confirms intended `base_cfg=...`.
- [ ] Log confirms expected trainable params and effective batch.
- [ ] No NaN/Inf warnings.
- [ ] Early predictions do not show obvious striping/saturation from pipeline mismatch.

### 8) GPU/Job Hygiene
- [ ] Smoke on `gputest` first.
- [ ] Memory headroom verified before long run.
- [ ] Use consistent env (`/scratch/project_2008261/physics_ml/bin/python3.11`).

### 9) Regression Check After Any Core Change
- [ ] Re-run preflight checker script.
- [ ] Re-run a tiny `gputest` eval and compare decoded endpoint visuals.
- [ ] Confirm no objective/inference mismatch regressions.
