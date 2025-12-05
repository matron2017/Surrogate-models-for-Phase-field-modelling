## Reference hyper-parameters

### U-AFNO (Bonneville et al., 2024/2025)
- U-Net encoder–decoder with AFNO bottleneck.
- Input channels: 3 (ϕ, x_A, x_B).
- AFNO: 12 blocks, 16 heads, MLP 3072, GELU. Patch sizes B/1, B/2, B/4.
- Output: Sigmoid ([0,1]).
- Implementation notes: periodic padding in x; non-periodic in y.

Refs: arXiv:2406.17119; npj Comp. Mater. 2025.

### FourCastNet (Pathak et al., 2022)
- AFNO: n_b=8, depth=12, mlp_ratio=4, embed_dim=768.
- Patch: 8×8; sparsity λ=1e-2; activation GELU; dropout 0.
- Train: batch 64; cosine LR; 5e-4/1e-4/2.5e-4 (pre/fine/TP).

Ref: Appendix A, Table 3.

### AFNO (Guibas et al., ICLR 2022)
- Fourier-domain token mixer with block-diagonal channel mixing and soft-thresholding.
- Quasi-linear complexity in sequence length.

### Links
- U-AFNO (arXiv): https://arxiv.org/abs/2406.17119
- U-AFNO (npj Comput. Mater.): https://www.nature.com/articles/s41524-024-01488-z
- FourCastNet (paper PDF): https://arxiv.org/pdf/2202.11214
- FourCastNet (code): https://github.com/NVlabs/FourCastNet
- AFNO (arXiv): https://arxiv.org/abs/2111.13587
- AFNO (OpenReview PDF): https://openreview.net/pdf?id=EXHG-A3jlM
- U-NO: https://arxiv.org/abs/2204.11127
