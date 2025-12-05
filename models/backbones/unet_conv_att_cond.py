# models/rapid_solidification/unet_ssa_preskip_full.py
# No second-person phrasing in comments

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData
from .unet_parts import DoubleConv, Down, Up, OutConv
from models.conditioning.skip_condition import ConditionalScaler


class SpatialSelfAttention2d(nn.Module):
    """
    Global spatial self-attention for feature maps.
    Input: x ∈ ℝ[B, C, H, W]
    Projections: 1×1 convs for q, k, v
    q/k dim = C//8 by default
    Uses fused scaled-dot-product attention for speed.
    """
    def __init__(self, channels: int, qk_ratio: float = 1/8, heads: int = 1,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, bias: bool = False,
                 use_norm: bool = False):
        super().__init__()
        assert channels % heads == 0
        qk_dim_total = max(heads, (int(channels * qk_ratio) // heads) * heads)
        self.heads = heads
        self.qk_each = qk_dim_total // heads
        self.v_each  = channels // heads

        self.q_proj   = nn.Conv2d(channels, qk_dim_total, 1, bias=bias)
        self.k_proj   = nn.Conv2d(channels, qk_dim_total, 1, bias=bias)
        self.v_proj   = nn.Conv2d(channels, channels,     1, bias=bias)
        self.out_proj = nn.Conv2d(channels, channels,     1, bias=bias)
        self.dropout  = nn.Dropout(proj_drop)
        self.norm     = nn.GroupNorm(32, channels) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        q = self.q_proj(x).view(B, self.heads, self.qk_each, N).transpose(2, 3)
        k = self.k_proj(x).view(B, self.heads, self.qk_each, N).transpose(2, 3)
        v = self.v_proj(x).view(B, self.heads, self.v_each,  N).transpose(2, 3)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        out = self.dropout(out)
        return self.norm(out + x)


@dataclass
class UNetSSAMetaData(ModelMetaData):
    name: str = "UNet_SSA_PreSkip_Full"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UNet_SSA_PreSkip_Full(Module):
    """
    U-Net + spatial self-attention bottleneck with pre-skip multiplicative conditioning.
    Encoder depth ×16, transposed-conv decoder.
    """
    def __init__(self,
                 n_channels: int = 2,
                 n_classes: int = 2,
                 in_factor: int = 78,
                 cond_dim: int = 2,
                 afno_inp_shape=(64, 64),
                 ssa_heads: int = 4,
                 ssa_qk_ratio: float = 1/8):
        super().__init__(meta=UNetSSAMetaData())
        C = in_factor

        # encoder
        self.inc   = DoubleConv(n_channels, C)
        self.down1 = Down(C,   2*C)
        self.down2 = Down(2*C, 4*C)
        self.down3 = Down(4*C, 8*C)
        self.down4 = Down(8*C, 16*C)

        # per-level scales (pre-skip)
        self.scaler = ConditionalScaler(
            cond_dim=cond_dim,
            widths=[C, 2*C, 4*C, 8*C, 16*C],
            hidden=128,
            identity_init=True
        )

        # bottleneck
        self.bot_pool = nn.MaxPool2d(2)
        self.bot_pre  = nn.Sequential(
            nn.Conv2d(16*C, 16*C, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.ssa_bottleneck = SpatialSelfAttention2d(
            channels=16*C, qk_ratio=ssa_qk_ratio, heads=ssa_heads,
            attn_drop=0.0, proj_drop=0.0, bias=False, use_norm=False
        )

        self.bot_scaler = ConditionalScaler(
            cond_dim=cond_dim,
            widths=[16*C],
            hidden=128,
            identity_init=True
        )

        self.bot_post = nn.Sequential(
            nn.Conv2d(16*C, 16*C, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.bot_up = nn.ConvTranspose2d(16*C, 16*C, kernel_size=2, stride=2)

        # decoder
        self.up1 = Up(16*C, 8*C, bilinear=False)
        self.up2 = Up(8*C,  4*C, bilinear=False)
        self.up3 = Up(4*C,  2*C, bilinear=False)
        self.up4 = Up(2*C,  C,   bilinear=False)
        self.outc = OutConv(C, n_classes)

        expected = (C, 2*C, 4*C, 8*C, 16*C)
        if tuple(self.scaler.widths) != expected:
            raise ValueError(f"ConditionalScaler widths {self.scaler.widths} do not match encoder {expected}.")

    @staticmethod
    def _apply_scale(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), s: (B,C)
        return x * s.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # pre-skip conditioning
        s1, s2, s3, s4, s5 = self.scaler(cond_vec)
        x1s = self._apply_scale(x1, s1)
        x2s = self._apply_scale(x2, s2)
        x3s = self._apply_scale(x3, s3)
        x4s = self._apply_scale(x4, s4)
        x5s = self._apply_scale(x5, s5)

        # bottleneck
        x5s = self.bot_pool(x5s)
        x5s = self.bot_pre(x5s)
        x5s = self.ssa_bottleneck(x5s)

        (sb,), = (self.bot_scaler(cond_vec),)
        x5s = self._apply_scale(x5s, sb)

        x5s = self.bot_post(x5s)
        x5s = self.bot_up(x5s)

        # decoder with conditioned skips
        x = self.up1(x5s, x4s)
        x = self.up2(x,   x3s)
        x = self.up3(x,   x2s)
        x = self.up4(x,   x1s)
        x = self.outc(x)
        return x
