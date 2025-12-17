# models/models/uafno_preskip_full.py
# No second-person phrasing in comments

from dataclasses import dataclass
import torch
import torch.nn as nn

from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.afno.afno import AFNO
from .unet_parts import DoubleConv, Down, Up, OutConv
from models.conditioning.skip_condition import ConditionalScaler


@dataclass
class UAFNOCondMetaData(ModelMetaData):
    name: str = "UAFNO_PreSkipConditioned_Full"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UAFNO_PreSkip_Full(Module):
    """
    U-Net + AFNO bottleneck with pre-skip multiplicative conditioning.
    Encoder depth ×16, transposed-conv decoder.
    """
    def __init__(self,
                 n_channels: int = 2,
                 n_classes: int = 2,
                 in_factor: int = 40,
                 cond_dim: int = 2,
                 afno_inp_shape=(64, 64),
                 afno_depth: int = 12,
                 num_blocks: int = 16,
                 afno_mlp_ratio: float = 12.0):
        super().__init__(meta=UAFNOCondMetaData())
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

        # bottleneck: pool → conv → AFNO(@/32) → scale → conv → up
        self.bot_pool = nn.MaxPool2d(2)
        self.bot_pre  = nn.Sequential(
            nn.Conv2d(16*C, 16*C, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        pooled_shape = (afno_inp_shape[0] // 2, afno_inp_shape[1] // 2)  # /32 relative to original grid
        self.afno_bottleneck = AFNO(
            inp_shape=list(pooled_shape),
            in_channels=16*C,
            out_channels=16*C,
            patch_size=(1, 1),
            embed_dim=16*C,
            depth=afno_depth,
            mlp_ratio=afno_mlp_ratio,
            drop_rate=0.0,
            num_blocks=num_blocks,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
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

    def _merge_cond_and_t(self, cond_vec: torch.Tensor | None, t: torch.Tensor | None) -> torch.Tensor:
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
        if len(args) == 0:
            return cond_vec, None
        if len(args) == 1:
            extra = args[0]
            if cond_vec is not None and cond_vec.dim() <= 1 and extra.dim() == 2:
                return extra, cond_vec  # swap: cond_vec was t, extra is cond
            return cond_vec if cond_vec is not None else extra, extra if cond_vec is not None else None
        raise TypeError(f"UAFNO_PreSkip_Full.forward expected at most 3 positional args, got {2 + len(args)}")

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
        s1, s2, s3, s4, s5 = self.scaler(cond)
        x1s = self._apply_scale(x1, s1)
        x2s = self._apply_scale(x2, s2)
        x3s = self._apply_scale(x3, s3)
        x4s = self._apply_scale(x4, s4)
        x5s = self._apply_scale(x5, s5)

        # bottleneck
        x5s = self.bot_pool(x5s)
        x5s = self.bot_pre(x5s)
        x5s = self.afno_bottleneck(x5s)

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
