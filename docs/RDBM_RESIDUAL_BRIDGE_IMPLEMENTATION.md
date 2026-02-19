# RDBM-Style Residual Modulation (UniDB)

This project now supports an optional residual-modulated UniDB bridge schedule.

## Enabled Behavior

When `diffusion.schedule_kwargs.residual_mode != none`, forward sampling becomes:

`x_t = f_mean(x0, mu, t) + f_sigma(t) * (pi * eps)`

where:
- `x0` = target (next latent),
- `mu` = source (past latent),
- `eps ~ N(0, I)`,
- `pi` = residual modulator from `(x0 - mu)`.

Implemented in:
- `models/diffusion/scheduler_registry.py`

## Residual Modes

- `residual_mode: none` (default): standard UniDB (`pi=1`).
- `residual_mode: signed`: `pi = (x0 - mu)`.
- `residual_mode: abs`: `pi = |x0 - mu|`.

Optional controls:
- `residual_normalize` (bool): normalize `pi` by per-sample mean absolute magnitude.
- `residual_scale` (float): multiply `pi` by a constant.
- `residual_power` (float): power transform for `pi`.
- `residual_clip` (float or null): clip `pi`.
- `residual_eps` (float): numerical floor.

## Score Mapping

If model output is interpreted as `u = pi * eps`, score is computed as:

`score = -u / (sigma * pi^2)`

Implemented via:
- `UniDBSchedule.get_score_from_noise(noise, t, pi=...)`
- `models/train/loss_registry.py` (passes `pi` into score for `unidb_reverse_step` objective).

## Config Example

Use:
- `configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80_rdbmres.yaml`

This keeps all existing settings but enables:
- `residual_mode: abs`
- `residual_normalize: true`

## Notes

- Defaults remain backward-compatible; existing configs are unchanged.
- This is a practical RDBM-style adaptation in the current UniDB pipeline, not a full reproduction of the original paper training stack.
