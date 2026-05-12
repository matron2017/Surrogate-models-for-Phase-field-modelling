"""Custom convolutional autoencoder for 3-channel 512×512 PDE solidification fields.

Compression options
-------------------
  f8   : 8×  spatial (512 → 64×64), 32ch latent →  6× total volume compression
  f16  : 16× spatial (512 → 32×32), 32ch latent → 24× total volume compression  ← default
  f32  : 32× spatial (512 → 16×16), 32ch latent → 96× total volume compression

Decision: f16 is used.
  - f8  is too mild (6× total), wastes memory, defeats the purpose
  - f16 preserves dendritic arm structure (feature scale ~20–80px → 1–5px in latent)
        and smooth thermal gradients at 32×32 resolution
  - f32 = DC-AE baseline, proven to fail on theta channel (MAE 0.30 vs phi 0.008)

Architecture: ResNet-style encoder/decoder, no VAE bottleneck (deterministic AE).
  Encoder: stem + 4 DownBlocks  →  32×32 latent
  Decoder: 4 UpBlocks + head   →  512×512 recon
  Each Block: 2× [Conv-GN-SiLU] + shortcut
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Two-conv residual block with GroupNorm + SiLU."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(min(groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class DownBlock(nn.Module):
    """ResBlock followed by stride-2 downsample."""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2) -> None:
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch)]
        for _ in range(n_res - 1):
            blocks.append(ResBlock(out_ch, out_ch))
        self.res = nn.Sequential(*blocks)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        return self.down(x)


class UpBlock(nn.Module):
    """2× bilinear upsample followed by ResBlock."""
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        blocks = [ResBlock(out_ch, out_ch) for _ in range(n_res)]
        self.res = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PDEConvAE(nn.Module):
    """Pixel-space convolutional autoencoder for 3-channel 512×512 PDE data.

    Parameters
    ----------
    in_channels : int, default 3   (phi, c, theta)
    latent_channels : int, default 32
    base_ch : int, default 64      first encoder channel count
    ch_mult : tuple, default (1,2,4,4)  channel multipliers per stage
    n_res_per_block : int, default 2
    spatial_factor : int           32 for 16× compression (default),
                                   16 for 8× compression,
                                   64 for 32× compression
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        base_ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4, 4),
        n_res_per_block: int = 2,
    ) -> None:
        super().__init__()
        channels = [base_ch * m for m in ch_mult]  # [64, 128, 256, 256]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        enc_blocks = []
        ch_in = channels[0]
        for ch_out in channels[1:]:           # 3 DownBlocks (3 stages × ÷2 = 8×)
            enc_blocks.append(DownBlock(ch_in, ch_out, n_res=n_res_per_block))
            ch_in = ch_out
        # 4th DownBlock for 16× total
        enc_blocks.append(DownBlock(ch_in, ch_in, n_res=n_res_per_block))
        self.encoder = nn.Sequential(*enc_blocks)

        # bottleneck → latent
        self.to_latent = nn.Sequential(
            nn.GroupNorm(min(8, ch_in), ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, latent_channels, 1),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Mirror the encoder exactly: 4 UpBlocks bring (32×32→512×512)
        # dec_ch_in starts at channels[-1] and steps down to channels[0]
        self.from_latent = nn.Conv2d(latent_channels, ch_in, 1)
        dec_blocks = []
        # Mirror of 4th DownBlock (ch_in→ch_in, same channel count)
        dec_blocks.append(UpBlock(ch_in, ch_in, n_res=n_res_per_block))
        # Mirror of the channels[1:] DownBlocks in reverse order
        # reversed(channels[:-1]) = reversed([c0,c1,c2]) gives [c2,c1,c0]
        for ch_out in list(reversed(channels[:-1])):
            dec_blocks.append(UpBlock(ch_in, ch_out, n_res=n_res_per_block))
            ch_in = ch_out
        # ch_in is now channels[0]; head below expects channels[0]
        self.decoder = nn.Sequential(*dec_blocks)
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        )

        self.latent_channels = latent_channels
        self._log_compression()

    def _log_compression(self) -> None:
        H = W = 512
        lat_H = lat_W = H // 16      # 32×32 for 16× spatial
        lat_vol = self.latent_channels * lat_H * lat_W
        in_vol = 3 * H * W
        spatial = H // lat_H
        total = in_vol / lat_vol
        self._compression_info = {
            "spatial_factor": spatial,
            "latent_shape": (self.latent_channels, lat_H, lat_W),
            "total_compression": round(total, 1),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(self.stem(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.from_latent(z)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def compression_info(self) -> dict:
        return self._compression_info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    m = PDEConvAE()
    params = sum(p.numel() for p in m.parameters())
    print(f"PDEConvAE  params={params/1e6:.1f}M")
    print(f"Compression: {m.compression_info}")
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        z = m.encode(x)
        xr = m.decode(z)
    print(f"input={tuple(x.shape)}  latent={tuple(z.shape)}  recon={tuple(xr.shape)}")
