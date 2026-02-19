# Smoke Test Slurm Notes

Date: 2026-02-10

Scope:
- `logs/slurm/gputest_flow_uvit_thermal_sa_smoke_*.out`
- `logs/slurm/gputest_flow_uvit_thermal_sa_smoke_*.err`

Summary:
- Smoke runs were short 2-epoch sanity checks (`gputest`) for flow-matching and diffusion-bridge latent configs.
- No critical training/runtime failures were found in these specific smoke logs.
- Common stderr lines were informational TensorFlow oneDNN notices and CPU feature messages.
- No OOM, CUDA crash, or traceback in these smoke files.

Key quick-check trend observed in smoke logs:
- Flow smoke variants showed expected loss decrease across 2 epochs.
- Bridge smoke variants also showed expected early decrease but remained much higher loss than flow (as expected in early sanity runs).

Action taken:
- Raw smoke slurm outputs were removed to reduce log clutter after extracting this note.
