from __future__ import annotations

r"""Common layers and modules (vendored from LoLA)."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "RMSNorm",
    "SelfAttentionNd",
    "Patchify",
    "Unpatchify",
    "Shrink",
    "Dilate",
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Sequence, Union


class PeriodicPadConv2d(nn.Module):
    """Conv2d with circular padding on a single spatial axis and configurable other-axis padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        pad_axis: str = "y",
        pad_other_mode: str | tuple[str, str] = "zeros",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding

        pad_axis = str(pad_axis).lower()
        if pad_axis not in ("x", "y"):
            raise ValueError(f"pad_axis must be 'x' or 'y', got {pad_axis!r}")

        self.pad_axis = pad_axis
        self.pad_h = int(pad_h)
        self.pad_w = int(pad_w)
        if isinstance(pad_other_mode, (tuple, list)) and len(pad_other_mode) == 2:
            self.pad_other_left = str(pad_other_mode[0]).lower()
            self.pad_other_right = str(pad_other_mode[1]).lower()
        else:
            mode = str(pad_other_mode).lower()
            self.pad_other_left = mode
            self.pad_other_right = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.pad_axis == "y" and self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="circular")
            if self.pad_w > 0:
                x = F.pad(x, (self.pad_w, 0, 0, 0), mode=self.pad_other_left)
                x = F.pad(x, (0, self.pad_w, 0, 0), mode=self.pad_other_right)
        elif self.pad_axis == "x" and self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
            if self.pad_h > 0:
                x = F.pad(x, (0, 0, self.pad_h, 0), mode=self.pad_other_left)
                x = F.pad(x, (0, 0, 0, self.pad_h), mode=self.pad_other_right)
        return self.conv(x)


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels C_i.
        out_channels: The number of output channels C_o.
        spatial: The number of spatial dimensions N.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to torch.nn.Conv2d.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    padding_mode = kwargs.pop("padding_mode", "zeros")
    pad_other_mode = kwargs.pop("padding_mode_other", "zeros")
    if padding_mode in ("circular_y", "periodic_y", "circular_x", "periodic_x"):
        if spatial != 2:
            raise NotImplementedError("Axis-specific periodic padding is only implemented for 2D.")
        pad_axis = "y" if "y" in padding_mode else "x"
        padding = kwargs.pop("padding", 0)
        conv = PeriodicPadConv2d(
            in_channels,
            out_channels,
            pad_axis=pad_axis,
            pad_other_mode=pad_other_mode,
            padding=padding,
            **kwargs,
        )
        base_conv = conv.conv
    else:
        conv = Conv(in_channels, out_channels, padding_mode=padding_mode, **kwargs)
        base_conv = conv

    if identity_init:
        kernel_size = base_conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(base_conv.weight.data)

        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1

        base_conv.weight.data.mul_(1e-2)
        base_conv.weight.data.add_(eye)

    return conv


class ReLU2(nn.Module):
    r"""Creates a ReLU^2 activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.relu(x).square()


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension."""

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - mean) * torch.rsqrt(variance + self.eps)


class RMSNorm(nn.Module):
    r"""Creates a layer that normalizes features along a dimension."""

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(torch.mean(torch.square(x), dim=self.dim, keepdim=True) + self.eps)

        return x * rms


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer."""

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        y = rearrange(x, "B C ...  -> B (...) C")

        qkv = torch.nn.functional.linear(y, self.in_proj_weight, self.in_proj_bias)
        q, k, v = rearrange(qkv, "B L (n H C) -> n B H L C", n=3, H=self.num_heads)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        y = rearrange(y, "B H L C -> B L (H C)")
        y = torch.nn.functional.linear(y, self.out_proj.weight, self.out_proj.bias)

        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
        else:
            return Rearrange("... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) (Z z) -> ... L H W Z (C l h w z)", l=l, h=h, w=w, z=z)
        else:
            return Rearrange("... C (L l) (H h) (W w) (Z z) -> ... (C l h w z) L H W Z", l=l, h=h, w=w, z=z)
    else:
        raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        else:
            return Rearrange("... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange("... L H W Z (C l h w z) -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z)
        else:
            return Rearrange("... (C l h w z) L H W Z -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z)
    else:
        raise NotImplementedError()


class Shrink(nn.Module):
    def __init__(self, stride: Sequence[int]):
        super().__init__()

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        slices = [slice(None, None, stride) for stride in self.stride]
        idx = (Ellipsis, *slices)
        return x[idx]


class Dilate(nn.Module):
    def __init__(self, stride: Sequence[int]):
        super().__init__()

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        kernel = torch.zeros(self.stride, dtype=x.dtype, device=x.device)
        kernel = kernel.flatten()
        kernel[0] = 1.0
        kernel = kernel.reshape(self.stride)

        return torch.kron(x.contiguous(), kernel)
