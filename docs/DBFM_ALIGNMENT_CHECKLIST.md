# DBFM Alignment Checklist for `pf_surrogate_modelling`

## 1) Objective and domain (memory anchor)

- Domain: phase-field surrogate modelling with thermal-field-conditioned latent dynamics.
- Physical target: predict next-step latent state `(z_{t+1})` from current latent state `(z_t)` for rightclean stochastic simulations.
- Input physics context: full thermal field is injected as `theta` conditioning (`conditioning.use_theta: true`, `add_thermal: true` in dataloader).
- Comparison target repo: `/scratch/project_2008261/DBFM-3E8E` (latest chosen reference).

## 2) What DBFM actually does

### Flow Matching branch
- Paired latent interpolation: `x_t = t*x1 + (1-t)*x0`.
- Velocity objective: `v(x_t, t) -> (x1 - x0)`.
- Default source in training script is source latent `x0` (not Gaussian) for translation.
- Model call shape: `model(x_t, t)`.

### Diffusion Bridge branch
- UniDB/GOUB SDE with `mu = LQ latent`, `x0 = GT latent`.
- Typical settings in train options:
  - `T=100`
  - `lambda_square=30`
  - `gamma_inv=0` (GOUB limit)
  - `schedule=cosine`
  - `eps=0.005`
- Training objective is reverse-step matching:
  - `xt_1_expectation = reverse_sde_step_mean(...)`
  - `xt_1_optimum = reverse_optimum_step(...)`
  - loss between the two (often L1 in options).

## 3) Our current implementation status against DBFM

### Diffusion Bridge status: aligned at objective/schedule level
- Configs now use:
  - `diffusion.noise_schedule: unidb_cosine`
  - `loss.diffusion_objective: unidb_reverse_step`
  - `loss.diffusion_matching_loss: l1`
  - `schedule_kwargs.timesteps: 100`, `lambda_square: 30`, `gamma_inv: 0.0`, `eps: 0.005`
- Training path samples UniDB forward states and optimizes reverse-step matching.

### Flow Matching status: aligned at trajectory/objective level
- New objective mode: `train.objective: dbfm_source_anchored`.
- Training path now uses:
  - `z0 = source latent`
  - `x_t = (1-t)z0 + t*y`
  - target velocity `u_t = y - z0`
  - model call style `model(x_t, t, theta=...)` (theta kept for phase-field physics context)
- Config template added:
  - `configs/train/train_flowmatch_uvit_thermal_dbfm_latent_template.yaml`
  - optimizer/scheduler default in this template now follows DBFM FM script style:
    - `optim.name: adamw`
    - `sched.name: step`, `step_size: 50`, `gamma: 0.1`

## 4) Important differences that are still intentional (not bugs)

- Conditioning:
  - DBFM image tasks generally do not inject a thermal field.
  - We do inject `theta` to respect physics-conditioned phase-field setting.
- Backbone family:
  - DBFM uses large Transformer backbones (SiT/ConditionalDiT).
  - We currently use `uvit_thermal` with bottleneck attention.
- Scale:
  - DBFM FM/DB backbones: ~456.8M trainable params.
  - Our FM template backbone: 950,432 trainable params.
  - Our bridge backbone: 1,213,600 trainable params.

## 5) Correctness verdict right now

- If the requirement is "DBFM-style objective/process": current FM and bridge implementations are correct in the core math and schedule/objective structure.
- If the requirement is "DBFM-scale apples-to-apples capacity": not yet aligned (ours is ~380x to ~480x smaller depending on branch).

## 6) Next action to continue correctly

- Keep current branch/objective math.
- Decide whether to:
  - keep `uvit_thermal` (physics-conditioned, lightweight), or
  - add a DiT/SiT-class large latent backbone variant for a true capacity-controlled DBFM comparison.
