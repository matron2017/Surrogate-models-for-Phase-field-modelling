# Deterministic Pixel Trace

Status: VERIFIED from source files in the old projects and the current workspace.

Purpose:
Rebuild the pixel-space deterministic surrogate in the current project area only:
/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates

Read-only reference projects:
1. /scratch/project_2008488/Simon_surrogate/pf_bridge_cleanstack
2. /scratch/project_2007141/simon_surrogate/pf_bridge_cleanstack

Use them only for reference, never as the target for new work.

Exact SSH access pattern:
AGENTS.md
AMD_vs_NVIDIA_and_REPORTING_GUIDE.py
AUTOENCODER_SMOKE_TESTS_QUICK_REF.md
CODEX.md
FINAL_MEASURED_VS_PREDICTED_REPORT.py
README.md
configs
data
docs
env
external_refs
logs
measure_2node_fullnode.sh
meta
models
runs
scripts
slurm
tmp
train_utils
visuals
workdirs
AGENTS.md
AMD_vs_NVIDIA_and_REPORTING_GUIDE.py
AUTOENCODER_SMOKE_TESTS_QUICK_REF.md
CODEX.md
FINAL_MEASURED_VS_PREDICTED_REPORT.py
README.md
configs
data
docs
env
external_refs
logs
measure_2node_fullnode.sh
meta
models
runs
scripts
slurm
tmp
train_utils
visuals
workdirs
README.md
__init__.py
__pycache__
autoencoder_dc_ae
deterministic_pixel
models
tests
utils

What is in the old reference projects:
1. Pixel-space deterministic surrogate config:
   - 
2. Pixel-space smoke config:
   - 
3. Plotting and eval helpers:
   - 
   - 
   - 
   - 
4. Training launcher:
   - 
5. Evidence of previous smoke runs and output directories under , , and 

What is already recreated in the current workspace:
1. Current pixel-space config:
   - 
2. Current GPU smoke scripts:
   - 
   - 
   - 
   - 
3. Current plotting helpers:
   - 
   - 
   - 

What was verified in the current workspace:
1. Forward GPU smoke passed.
2. Short train GPU smoke passed.
3. The recreated smoke path uses:
   - 
   - 
   - thermal conditioning through the third channel
   - wavelet weighting enabled with 

Core deterministic model facts from the old config:
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 
11. 
12. 
13. 
14. 
15. 
16. 
17. 
18. 
19. 

Where we left off:
1. The current deterministic pixel path is only a smoke/rebuild path so far.
2. The next step is to add a fuller current-project deterministic launcher and the sanity-check plotting/eval flow that mirrors the old project behavior.
3. No trained deterministic checkpoint has been recovered from the old projects yet.

Checkpoint availability:
UNVERIFIED as a trained deterministic checkpoint. Searches found config and profiling artifacts, but no deterministic training checkpoint under the old projects or the current workspace.

Suggested copyable workflow:
./TRACE.md
./logs/slurm/det_px_fwd_34386687.err
./logs/slurm/det_px_fwd_34386687.out
./logs/slurm/det_px_trn_34386691.err
./logs/slurm/det_px_trn_34386691.out
./metadata/last_forward_jobid.txt
./metadata/last_train_jobid.txt
./metadata/tree.txt
./plotting/plot_pair_style.py
./plotting/plot_past_next_pred_diff.py
./plotting/smoke_plot_det_pixel.py
./scripts/test_det_pixel_forward.py
./scripts/train_det_pixel_smoke.py
./slurm/det_pixel_forward_gputest.sh
./slurm/det_pixel_train_gputest.sh
./tmp/forward_gputest_34386687/forward_report.json
./tmp/train_gputest_34386691/metrics.csv
./tmp/train_gputest_34386691/summary.json
configs/pf_surrogates/README.md
configs/pf_surrogates/big/train_bridge_fdbm_frac_unet_afno_controlxs_pixel512_big.yaml
configs/pf_surrogates/big/train_bridge_flowmatch_rectified_sourceanchored_concat_unet_afno_controlxs_pixel512_452m_full_2008488.yaml
configs/pf_surrogates/big/train_bridge_flowmatch_rectified_sourceanchored_concat_unet_afno_controlxs_pixel512_452m_gputest_2n8g_2008488.yaml
configs/pf_surrogates/big/train_bridge_flowmatch_rectified_sourceanchored_concat_unet_afno_controlxs_pixel512_452m_puhti_2008488.yaml
configs/pf_surrogates/big/train_bridge_frac_unet_afno_controlxs_pixel512_big.yaml
configs/pf_surrogates/big/train_bridge_fractional_unet_afno_controlxs_pixel512_452m_from1024adapt_puhti_2008488.yaml
configs/pf_surrogates/big/train_bridge_unidb_unet_afno_controlxs_pixel512_big.yaml
configs/pf_surrogates/big/train_ddpm_x0_dpmsolver_unet_afno_controlxs_pixel512_big.yaml
configs/pf_surrogates/big/train_det_unet_afno_controlxs_wavelet_512_big.yaml
configs/pf_surrogates/smoke/train_bridge_fdbm_frac_unet_afno_controlxs_pixel512_overfit40.yaml
configs/pf_surrogates/smoke/train_bridge_unidb_unet_afno_controlxs_pixel512_overfit40.yaml
configs/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_overfit40.yaml
configs/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_smoke.yaml
configs/pf_surrogates/smoke/train_smoke_bridge_unet_afno_controlxs_predictnext_512.yaml
configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_eightpair.yaml
configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_onepair.yaml
configs/pf_surrogates/smoke/train_smoke_ddpm_x0_dpmsolver_eightpair.yaml
train_utils/__pycache__/profile_det_afno_forward.cpython-36.pyc
train_utils/analyze_scaling_bottleneck.py
train_utils/autoencoder/big/submit_ae_lola_3x512_lat96_big.sh
train_utils/autoencoder/big/submit_ae_lola_3x512_paperstyle_rb_big.sh
train_utils/autoencoder/smoke/submit_ae_lola_3x512_lat96_smoke.sh
train_utils/big/submit_bridge_fdbm_frac_unet_afno_controlxs_pixel512_big.sh
train_utils/big/submit_bridge_frac_unet_afno_controlxs_pixel512_big.sh
train_utils/big/submit_bridge_unidb_unet_afno_controlxs_pixel512_big.sh
train_utils/big/submit_det_unet_afno_controlxs_wavelet_big.sh
train_utils/build_final_gpu_report_bundle.py
train_utils/estimate_3d_from_2d_baseline.py
train_utils/plot_scaling_clarity.py
train_utils/print_scaling_summary.py
train_utils/profile_det_afno_forward.py
train_utils/report_measured_efficiencies.py
train_utils/run_multinode_test.sbatch
train_utils/run_profile_det_afno_ablation_dev_g.sh
train_utils/run_profile_det_afno_dev_g.sh
train_utils/run_profile_det_afno_fullnode_weakscaling.sh
train_utils/run_profile_det_afno_maxbatch_8gcd.sh
train_utils/run_profile_det_afno_multinode_direct.sh
train_utils/run_profile_det_afno_multinode_weakscaling.sh
train_utils/run_profile_det_afno_rocprof_dev_g.sh
train_utils/run_profile_det_afno_scaling_dev_g.sh
train_utils/run_profile_det_afno_weakscaling_2node.sh
train_utils/smoke/__pycache__/eval_overfit_det_suite.cpython-311.pyc
train_utils/smoke/__pycache__/eval_overfit_unidb_suite.cpython-311.pyc
train_utils/smoke/__pycache__/eval_overfit_unidb_suite.cpython-312.pyc
train_utils/smoke/__pycache__/eval_overfit_unidb_suite.cpython-36.pyc
train_utils/smoke/__pycache__/eval_theta_ablation_bridge.cpython-312.pyc
train_utils/smoke/__pycache__/eval_theta_ablation_bridge.cpython-36.pyc
train_utils/smoke/__pycache__/plot_pair_style.cpython-36.pyc
train_utils/smoke/__pycache__/plot_past_next_pred_diff.cpython-36.pyc
train_utils/smoke/__pycache__/smoke_plot_bridge_pixel.cpython-36.pyc
train_utils/smoke/__pycache__/smoke_plot_det_pixel.cpython-311.pyc
train_utils/smoke/__pycache__/smoke_plot_det_pixel.cpython-312.pyc
train_utils/smoke/__pycache__/smoke_plot_det_pixel.cpython-36.pyc
train_utils/smoke/eval_overfit_det_suite.py
train_utils/smoke/eval_overfit_unidb_suite.py
train_utils/smoke/eval_theta_ablation_bridge.py
train_utils/smoke/plot_pair_style.py
train_utils/smoke/plot_past_next_pred_diff.py
train_utils/smoke/run_ae_visuals_current_dev_g.sh
train_utils/smoke/run_current_flow_ae_panels_dev_g.sh
train_utils/smoke/run_overfit_det_overfit40_dev_g.sh
train_utils/smoke/run_overfit_unidb_eightpair_dev_g.sh
train_utils/smoke/run_overfit_unidb_onepair_dev_g.sh
train_utils/smoke/run_overfit_unidb_overfit40_dev_g.sh
train_utils/smoke/run_smoke_bridge_dev_g.sh
train_utils/smoke/run_smoke_ddpm_x0_dpmsolver_eightpair_dev_g.sh
train_utils/smoke/smoke_plot_bridge_pixel.py
train_utils/smoke/smoke_plot_det_pixel.py
train_utils/smoke/submit_det_unet_afno_controlxs_wavelet_smoke.sh
train_utils/summarize_fullnode_matrix.py
