# Phase Field Surrogates

Compact extraction for the Puhti project `/scratch/project_2008261/pf_surrogate_modelling`.

Verified target packaged here:

- Run: `runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_predictnext_nomass_afno8_gputest_overfit_det_nonfrac_q1/UNetFiLMAttn`
- Config copy: `utils/deterministic_afno_bridge_config.yaml`
- Checkpoint reference: `/scratch/project_2008261/pf_surrogate_modelling/runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_predictnext_nomass_afno8_gputest_overfit_det_nonfrac_q1/UNetFiLMAttn/checkpoint.best.pth`
- Data reference: `/scratch/project_2008261/pf_surrogate_modelling/data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5`

Important: this verified deterministic run is latent-space, not pixel-space. The loader sample is `input=(33,64,64)` because it is `32 latent channels + 1 thermal channel`; the model contract is `model input=(1,64,64,64)`, `theta=(1,1,64,64)`, `output=(1,32,64,64)`.

Wavelet status from the copied config is explicit: `trainer.use_wavelet_weights: false` and `loss.weight_wavelet_loss: 0.0`. Run the test with `--require-wavelet` if you want this to fail closed when wavelet weighting is not enabled.

Run the smoke test on Puhti:

```bash
cd /scratch/project_2008261/pf_surrogate_modelling
/scratch/project_2008261/solidification_modelling/physics_ml/bin/python Phase_field_surrogates/tests/test_shapes_forward_wavelet.py
```

Run the stricter wavelet gate:

```bash
cd /scratch/project_2008261/pf_surrogate_modelling
/scratch/project_2008261/solidification_modelling/physics_ml/bin/python Phase_field_surrogates/tests/test_shapes_forward_wavelet.py --require-wavelet
```
