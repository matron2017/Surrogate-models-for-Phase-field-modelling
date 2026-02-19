# AE PSGD Debug Log

This doc tracks the failing LoLA AE PSGD runs, what was observed, and what was changed.
Update this file after each new run or fix attempt.

## Latest failures (Slurm logs reviewed)

- Job 31341786 (full data, PSGD, no wavemap): `logs/slurm/ae_latent_gpu_6n24g_lola_big_psgd_f1_64_1024_31341786.{out,err}`
  - Failed with `FloatingPointError: Non-finite loss at epoch 5`.
  - Training loss spiked by epoch 4 (train_mse ~37.7).
  - `nan_guard` shows `pred` contains NaNs while `target` is finite.
- Job 31341788 (wavemap p92 w500): `logs/slurm/ae_latent_gpu_6n24g_lola_big_psgd_f1_64_1024_wavemap_p92_w500_31341788.{out,err}`
  - Failed with `FloatingPointError: Non-finite loss at epoch 2`.
  - NCCL watchdog timeouts followed after ranks exited.
- Job 31343267 (6-stage r32): `logs/slurm/ae_latent_gpu_6n24g_lola_big_psgd_f1_64_1024_wavemap_p92_w500_6stage_r32_1024_31343267.{out,err}`
  - Failed with `FloatingPointError: Non-finite loss at epoch 1`.
  - NCCL teardown warnings followed.

## Config context (from run logs)

- Full-data PSGD run logs:
  - `ae_wavelet_enabled=True`, `weight_wavelet_loss=0`, `use_wavelet_weights=False`.
- Wavemap PSGD runs:
  - `ae_wavelet_enabled=False`, `weight_wavelet_loss=0.1`, `use_wavelet_weights=False`.
- All three are PSGD with the LoLA AE model; AMP fp16 enabled.

## Working hypothesis

- The NaNs are model/optimizer divergence (pred contains NaNs while target stays finite).
- PSGD updates appear too aggressive at LR=1e-4 with full-size preconditioners, especially on the wavemap/r32 variants.
- Wavelet-weighted loss may amplify gradients for w500, but the non-wavemap run also diverged.

## Changes applied (stability pass)

Reduced PSGD aggressiveness and gradient clipping in the configs used by the failing jobs:

- `configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g.yaml`
  - `optim.lr: 1.0e-4 -> 5.0e-5`
  - `optim.preconditioner_update_probability: 1.0 -> 0.25`
  - `trainer.grad_clip: 1.0 -> 0.5`
- `configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_wavemap_p92_w500.yaml`
  - Same adjustments as above.
- `configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_wavemap_p92_w500_r32_6stage_1024.yaml`
  - Same adjustments as above.

## Gputest coverage (max settings)

Prepared gputest launchers for full data (2 nodes x 4 GPUs, gputest max GPUs = 8):

- `slurm/train_ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024.sh`
- `slurm/train_ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024_wavemap_p92_w500.sh`
- `slurm/train_ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024_wavemap_p92_w500_6stage_r32_1024.sh`

Submit these before re-queueing the 6n24g jobs to confirm NaN behavior on the updated configs.

Submission status (latest):
- RUNNING: `ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024` job 31353518.
- CANCELED: `ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024_wavemap_p92_w500` job 31353519 (freed submit slot).
- PENDING: `ae_latent_gputest_2n8g_lola_big_psgd_f1_64_1024_wavemap_p92_w500_6stage_r32_1024` job 31353528 (AssocMaxJobsLimit; will start when slot frees).

## Next steps if NaNs persist

1. Reduce `optim.lr` further (e.g., `2e-5`) and/or `preconditioner_update_probability` to `0.1`.
2. Increase `optim.precondition_frequency` to `128` to reduce update cadence.
3. For baseline ablation, consider disabling AE wavelet loss:
   - Set `loss.ae_wavelet.enabled: false` (to align with "no wavelet" baseline).
4. Run a short gputest PSGD job with `trainer.nan_debug: true` to catch the first bad step.
5. If PSGD still diverges, switch the long run to AdamW for stability and keep PSGD as a separate ablation.

## Jobs (2026-01-29)
- SUBMITTED: 31482227 (5 nodes x 4 GPUs, wavelet, multistep110 schedule, effective batch 40).
  - Config: `train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent64_wavelet_multistep110_b40.yaml`
  - Scheduler: multistep milestones every 110 epochs (110, 220, 330, 440, 550, 660, 770, 880, 990), gamma=0.1.
  - Warmup: 20 epochs, start LR = 1e-6 -> base LR = 1e-5.
  - Architecture: 6-stage (~390M params), 16x compression (stage_strides [2,2,2,2,1]).
- SUBMITTED: 31482228 (5 nodes x 4 GPUs, wavelet, multistep110 schedule, effective batch 20).
  - Config: `train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent64_wavelet_multistep110_b20.yaml`
  - Scheduler: multistep milestones every 110 epochs (110, 220, 330, 440, 550, 660, 770, 880, 990), gamma=0.1.
  - Warmup: 20 epochs, start LR = 1e-6 -> base LR = 1e-5.
  - Architecture: 6-stage (~390M params), 16x compression (stage_strides [2,2,2,2,1]).
