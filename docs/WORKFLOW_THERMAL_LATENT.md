# Thermal-Latent Workflow (PDE -> PDE)

This is the current workflow for next-step prediction of phase field + concentration, conditioned by the **full thermal field**.

## 1) Data Stage (HDF5 pairs + thermal field)

Purpose: load `(x_t, x_{t+1})` frame pairs and append thermal map channel(s).

- Dataset class: `models/train/core/pf_dataloader.py` (`PFPairDataset`)
- Thermal map toggle: `dataloader.args.add_thermal: true`
- Scalar conditioning: disabled (`return_cond: false`)

Output sample contract:

- `input`: physical fields + thermal field channel(s)
- `target`: next-step physical fields (optionally thermal if configured)
- `cond`: not used in the thermal-field-only path

## 2) Latent Stage (Autoencoder)

Purpose: encode physical fields to latent space for cheaper surrogate learning.

- AE build/load: `models/train/core/latent.py`
- Encoder use inside training loop: `models/train/core/loops.py` via `_apply_latent_pipeline(...)`
- Current AE training jobs/logs are producing the latent shape that surrogate consumes.

## 3) Conditioning Stage (Thermal Field Injection)

Purpose: inject full thermal field into latent surrogate backbone.

- Batch split (state vs theta): `models/train/core/utils.py` (`_prepare_batch`)
- `theta` handling + resize-to-latent: `models/train/core/loops.py` (`_resize_theta_to_match`)
- Dedicated thermal backbone: `models/backbones/uvit_thermal.py`

Design rule:

- Thermal conditioning is spatial (map), not scalar.
- `conditioning.enabled: false`
- `conditioning.use_theta: true`
- `conditioning.theta_channels: 1` (or more if configured)

## 4) Surrogate Stage (Latent Next-Step Predictor)

Purpose: learn `z_t -> z_{t+1}` in latent space.

- Train/val loops: `models/train/core/loops.py`
- Surrogate forward wrapper: `_forward_surrogate(...)`
- Model registry entry: `models/backbones/registry.py` with backbone key `uvit_thermal`

## 5) Decode + Metrics Stage

Purpose: evaluate latent predictions against targets with reconstruction metrics.

- Loss/metrics orchestration: `models/train/core/loops.py`
- Optional additional tooling/plots: `visuals/basic/`, `visuals/hq/`

## Config Pattern (Recommended)

For thermal-latent surrogate configs, keep this pattern:

- `train.model_family: surrogate`
- `dataloader.args.return_cond: false`
- `dataloader.args.add_thermal: true`
- `conditioning.enabled: false`
- `conditioning.use_theta: true`
- `model.backbone: uvit_thermal`
- `latent.enabled: true`

This keeps the workflow consistent with thermal-field-only conditioning and avoids accidental scalar-cond paths.
