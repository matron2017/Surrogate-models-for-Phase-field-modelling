# models/models/unet_ssa_preskip_full.py
# No second-person phrasing in comments

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData
from .unet_parts import DoubleConv, Down, Up, OutConv
from models.conditioning.skip_condition import ConditionalScaler, ConditionalFiLM
from models.conditioning.mixed_padding import MixedBCConv2d


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
                 ssa_qk_ratio: float = 1/8,
                 skip_film: bool = True):
        super().__init__(meta=UNetSSAMetaData())
        C = in_factor
        self.skip_film = bool(skip_film)

        # encoder
        self.inc   = DoubleConv(n_channels, C)
        self.down1 = Down(C,   2*C)
        self.down2 = Down(2*C, 4*C)
        self.down3 = Down(4*C, 8*C)
        self.down4 = Down(8*C, 16*C)

        widths = [C, 2*C, 4*C, 8*C, 16*C]

        # per-level conditioning (pre-skip)
        if self.skip_film:
            self.skip_cond = ConditionalFiLM(
                cond_dim=cond_dim,
                widths=widths,
                hidden=128,
                identity_init=True
            )
        else:
            self.skip_cond = ConditionalScaler(
                cond_dim=cond_dim,
                widths=widths,
                hidden=128,
                identity_init=True
            )

        # bottleneck
        self.bot_pool = nn.MaxPool2d(2)
        self.bot_pre  = nn.Sequential(
            MixedBCConv2d(16*C, 16*C, kernel_size=3, bias=True),
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
            MixedBCConv2d(16*C, 16*C, kernel_size=3, bias=True),
            nn.GELU(),
        )
        self.bot_up = nn.ConvTranspose2d(16*C, 16*C, kernel_size=2, stride=2)

        # decoder
        self.up1 = Up(16*C, 8*C, bilinear=False)
        self.up2 = Up(8*C,  4*C, bilinear=False)
        self.up3 = Up(4*C,  2*C, bilinear=False)
        self.up4 = Up(2*C,  C,   bilinear=False)
        self.outc = OutConv(C, n_classes)

        expected = tuple(widths)
        if tuple(widths) != expected:
            raise ValueError(f"Conditional widths {widths} do not match encoder {expected}.")

    @staticmethod
    def _apply_scale(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), s: (B,C)
        return x * s.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), gamma/beta: (B,C)
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b

    def _merge_cond_and_t(self, cond_vec: torch.Tensor | None, t: torch.Tensor | None) -> torch.Tensor:
        """
        Supports both call patterns:
          - flow matching: forward(x, cond_with_time)
          - diffusion:     forward(x, t, cond)  (t appended as extra scalar)
        """
        if t is None:
            if cond_vec is None:
                raise ValueError("Conditioning vector is required.")
            return cond_vec

        t = t.view(t.shape[0], -1).to(dtype=torch.float32, device=cond_vec.device if cond_vec is not None else t.device)
        if cond_vec is None:
            return t
        if cond_vec.dim() != 2:
            raise ValueError(f"Expected cond_vec shape (B, C), got {tuple(cond_vec.shape)}")
        return torch.cat([cond_vec, t], dim=1)

    def _split_args(self, cond_vec: torch.Tensor | None, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Disambiguate positional calls:
          - forward(x, cond)
          - forward(x, cond, t)
          - forward(x, t, cond)
        """
        if len(args) == 0:
            return cond_vec, None
        if len(args) == 1:
            extra = args[0]
            # If cond_vec looks like timestep (1D) and extra looks like cond (2D), swap.
            if cond_vec is not None and cond_vec.dim() <= 1 and extra.dim() == 2:
                return extra, cond_vec
            # Otherwise treat cond_vec as cond and extra as t.
            return cond_vec if cond_vec is not None else extra, extra if cond_vec is not None else None
        raise TypeError(f"UNet_SSA_PreSkip_Full.forward expected at most 3 positional args, got {2 + len(args)}")

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor | None = None, *args, region_info=None) -> torch.Tensor:
        cond_vec, timestep = self._split_args(cond_vec, args)

        cond = self._merge_cond_and_t(cond_vec, timestep)

        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # pre-skip conditioning
        if self.skip_film:
            gammas, betas = self.skip_cond(cond)
            (g1, g2, g3, g4, g5), (b1, b2, b3, b4, b5) = gammas, betas
            x1s = self._apply_film(x1, g1, b1)
            x2s = self._apply_film(x2, g2, b2)
            x3s = self._apply_film(x3, g3, b3)
            x4s = self._apply_film(x4, g4, b4)
            x5s = self._apply_film(x5, g5, b5)
        else:
            s1, s2, s3, s4, s5 = self.skip_cond(cond)
            x1s = self._apply_scale(x1, s1)
            x2s = self._apply_scale(x2, s2)
            x3s = self._apply_scale(x3, s3)
            x4s = self._apply_scale(x4, s4)
            x5s = self._apply_scale(x5, s5)

        # bottleneck
        x5s = self.bot_pool(x5s)
        x5s = self.bot_pre(x5s)
        x5s = self.ssa_bottleneck(x5s)

        (sb,), = (self.bot_scaler(cond),)
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
