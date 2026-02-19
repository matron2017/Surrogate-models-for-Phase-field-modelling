# Scripts

Project-specific analysis and dataset diagnostics live here.

Typical use:
- post-training evaluation
- right-buffer dataset filtering/inspection
- metric/plot generation helpers

For generic developer tools (job watch, tree print), use `tools/`.

Wavelet strategy prep (compact, AE-focused):
- `compare_ae_wavelet_strategies.py`
  - Compares quantile/bandpass/multiband weighting strategies on sampled dataset frames.
  - Produces compact outputs only: summary JSON, ranking CSV, and ready precompute commands.
  - Prioritizes low-frequency-aware strategies while penalizing overly bulk weighting.
