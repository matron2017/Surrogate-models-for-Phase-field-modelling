# PF Surrogate Worklist (Personal Notes)

Date: 2026-02-05

## Priorities
1. Data integrity and right-edge cutoff (before any retrain).
2. Baseline AE training: non-wavelet vs wavelet (mild weighting).
3. Compression study: 1/8 and 1/16, compare wavelet vs non-wavelet.
4. Latent-space dynamics models (diffusion bridge / flow models).
5. Physics constraints (mass conservation AE).

## P0 — Right-edge diagnostics and data cleanup
- [ ] Implement right-edge buffer onset diagnostics (per-sim, per-split).
- [ ] Decide policy: remove time steps that touch right boundary (not whole sims).
- [ ] Apply removal to all splits (train/val/test).
- [ ] Summarize how many frames removed per split and per grad.

## P0 — Rightclean fixed34 grad-shared dataset (normalization + usage)
- New rightclean dataset is a truncated version of the old dataset (end frames removed only).
- Variable T per gradient is expected; all 5 seeds in a gradient share the same cutoff.
- Use file-level channel_mean/std computed from the rightclean train split for all splits (same style as old dataset).
- Stats recompute job (CPU small) completed: 31656127.
- Rightclean data has been re-normalized to new train-split stats; file/group attrs updated.
- Current rightclean file attrs (train/val/test):
  `channel_mean`: [-6.5235667, 0.9819305]
  `channel_std`:  [7.4176931, 0.5344091]
  `channel_mean_prev`: [-2.974396, 0.98999465]
  `channel_std_prev`:  [9.087654, 0.8249514]
  `normalization_schema`: zscore
  `normalization_source`: rightclean_fixed34_gradshared_train
  `normalization_timestamp`: 2026-02-07T00:25:11Z
- Reinversion formula (raw from normalized):
  `raw = norm * channel_std + channel_mean`
- Training should keep `normalize_force: false` to avoid double-normalization.

## P0 — AE baseline: non-wavelet vs wavelet
- [ ] Train non-wavelet AE baseline.
- [ ] Train wavelet AE with mild weighting.
- [ ] Verify wavelet weights include low-frequency content (not only sharp edges).
- [ ] Compare recon quality (RMSE + qualitative visuals, physical scale).

## P1 — Compression study
- [ ] Train AE at 1/8 compression, non-wavelet.
- [ ] Train AE at 1/8 compression, wavelet.
- [ ] Train AE at 1/16 compression, non-wavelet.
- [ ] Train AE at 1/16 compression, wavelet.
- [ ] Compare RMSE and qualitative results across all four.

## P1 — Latent dynamics
- [ ] Start latent diffusion bridge or flow models.
- [ ] Consider U-ViT backbone.
- [ ] Consider AFNO backbone.

## P2 — Physics constraint
- [ ] Add mass conservation regularization to non-wavelet AE.
- [ ] Evaluate impact on recon and stability.

## Current status / completed
- [x] Physical-scale AE visuals with bold/large labels and corrected residual stats.
- [x] Cleaned AE visuals to keep only key frames in `eval_visuals_phys_b40_best_test`.
- [x] Wavelet weight visuals for frames 1, 51, 100, 200, 245, 299.
- [x] Combo weights (highfreq + multiband) produced.
- [ ] Soft combo (low-freq included with mild weights) running on gputest (job 31587007).

## Pointers
- AE visuals: `pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent64_wavelet_multiband_beta150_multistep110_b40/LatentAELoLAModel/eval_visuals_phys_b40_best_test/`
- Wavelet weights: `pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent64_wavelet_multiband_beta150_multistep110_b40/LatentAELoLAModel/eval_wavelet_weights_b150_test/`
