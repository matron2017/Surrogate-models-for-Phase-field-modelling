import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["pad_mixed_bc", "MixedBCPad2d", "MixedBCConv2d"]


def pad_mixed_bc(x: torch.Tensor, pad_x: int, pad_y: int) -> torch.Tensor:
    """
    Applies mixed boundary padding for full-grid convs:
    - y (height) periodic -> circular
    - x-left replicate    -> copies edge outward
    - x-right reflect     -> mirrors interior (approx zero-normal-gradient)
    """
    if pad_y > 0:
        x = F.pad(x, (0, 0, pad_y, pad_y), mode="circular")  # (l,r,t,b)
    if pad_x > 0:
        x = F.pad(x, (pad_x, 0, 0, 0), mode="replicate")
        x = F.pad(x, (0, pad_x, 0, 0), mode="reflect")
    return x


class MixedBCPad2d(nn.Module):
    """Module wrapper for mixed BC padding (y circular, x-left replicate, x-right reflect)."""
    def __init__(self, pad_x: int, pad_y: int):
        super().__init__()
        self.pad_x = int(pad_x)
        self.pad_y = int(pad_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pad_mixed_bc(x, self.pad_x, self.pad_y)


class MixedBCConv2d(nn.Module):
    """
    Drop-in Conv2d that first applies mixed-BC padding, then a valid conv.
    Note: reflect padding requires pad < feature width; keep inputs wider than 2*pad.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int | tuple[int, int] = 3,
                 stride: int | tuple[int, int] = 1, dilation: int | tuple[int, int] = 1,
                 bias: bool = True, groups: int = 1):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kx, ky = kernel_size
        else:
            kx = ky = kernel_size
        if isinstance(dilation, tuple):
            dx, dy = dilation
        else:
            dx = dy = dilation
        eff_kx = dx * (kx - 1) + 1
        eff_ky = dy * (ky - 1) + 1
        pad_x = eff_kx // 2
        pad_y = eff_ky // 2

        self.pad = MixedBCPad2d(pad_x=pad_x, pad_y=pad_y)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            padding=0, dilation=dilation, bias=bias, groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))
