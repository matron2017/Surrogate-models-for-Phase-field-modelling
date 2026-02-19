from __future__ import annotations

r"""Deep Compressed Auto-Encoder (DCAE) building blocks (vendored from LoLA)."""

__all__ = [
    "DCEncoder",
    "DCDecoder",
]

import math
import torch.nn as nn

from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Union

from models.latent.lola_ref_layers import (
    ConvNd,
    LayerNorm,
    Patchify,
    SelfAttentionNd,
    Unpatchify,
)


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module."""

    def __init__(
        self,
        channels: int,
        norm: str = "layer",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels),
                num_channels=channels,
                affine=False,
            )
        else:
            raise NotImplementedError()

        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            self.attn = Residual(
                SelfAttentionNd(channels, heads=attention_heads),
            )

            kwargs.update(kernel_size=1, padding=0)

        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        self.ffn[-1].weight.data.mul_(1e-2)

    def _forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.attn(y)
        y = self.ffn(y)

        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class DCEncoder(nn.Module):
    r"""Creates a deep-compressed (DC) encoder module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        stage_strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool | str = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        padding_mode_other = "zeros"
        if isinstance(periodic, str):
            p = periodic.strip().lower()
            if "reflect" in p or "mirror" in p:
                padding_mode_other = "reflect"
            elif "replicate" in p or "neumann" in p or "zero_grad" in p:
                padding_mode_other = "replicate"
            left_mode = None
            right_mode = None
            if "left_reflect" in p or "reflect_left" in p:
                left_mode = "reflect"
            if "left_replicate" in p or "replicate_left" in p or "left_neumann" in p:
                left_mode = "replicate"
            if "left_zero" in p or "left_zeros" in p:
                left_mode = "zeros"
            if "right_reflect" in p or "reflect_right" in p:
                right_mode = "reflect"
            if "right_replicate" in p or "replicate_right" in p or "right_neumann" in p:
                right_mode = "replicate"
            if "right_zero" in p or "right_zeros" in p:
                right_mode = "zeros"
            if left_mode is not None or right_mode is not None:
                if left_mode is None:
                    left_mode = str(padding_mode_other)
                if right_mode is None:
                    right_mode = str(padding_mode_other)
                padding_mode_other = (left_mode, right_mode)
            if p in ("y", "vertical", "height", "periodic_y", "circular_y", "y_replicate"):
                padding_mode = "circular_y"
            elif p in ("x", "horizontal", "width", "periodic_x", "circular_x", "x_replicate"):
                padding_mode = "circular_x"
            elif p in ("xy", "both", "all", "circular", "periodic", "true", "1"):
                padding_mode = "circular"
            else:
                padding_mode = "zeros"
        else:
            padding_mode = "circular" if periodic else "zeros"

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode=padding_mode,
            padding_mode_other=padding_mode_other,
        )
        if stage_strides is None:
            stage_stride_list = [tuple(stride)] * (len(hid_blocks) - 1)
        else:
            if len(stage_strides) != len(hid_blocks) - 1:
                raise ValueError("stage_strides must have length len(hid_blocks) - 1.")
            stage_stride_list = []
            for s in stage_strides:
                if isinstance(s, int):
                    stage_stride_list.append((s,) * spatial)
                else:
                    if len(s) != spatial:
                        raise ValueError("stage_strides entries must match spatial dims.")
                    stage_stride_list.append(tuple(int(v) for v in s))

        self.patch = Patchify(patch_size=patch_size)
        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                stage_stride = stage_stride_list[i - 1]
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stage_stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stage_stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stage_stride,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch(x)

        for blocks in self.descent:
            for block in blocks:
                x = block(x)

        return x


class DCDecoder(nn.Module):
    r"""Creates a deep-compressed (DC) decoder module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        stage_strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool | str = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        padding_mode_other = "zeros"
        if isinstance(periodic, str):
            p = periodic.strip().lower()
            if "reflect" in p or "mirror" in p:
                padding_mode_other = "reflect"
            elif "replicate" in p or "neumann" in p or "zero_grad" in p:
                padding_mode_other = "replicate"
            left_mode = None
            right_mode = None
            if "left_reflect" in p or "reflect_left" in p:
                left_mode = "reflect"
            if "left_replicate" in p or "replicate_left" in p or "left_neumann" in p:
                left_mode = "replicate"
            if "left_zero" in p or "left_zeros" in p:
                left_mode = "zeros"
            if "right_reflect" in p or "reflect_right" in p:
                right_mode = "reflect"
            if "right_replicate" in p or "replicate_right" in p or "right_neumann" in p:
                right_mode = "replicate"
            if "right_zero" in p or "right_zeros" in p:
                right_mode = "zeros"
            if left_mode is not None or right_mode is not None:
                if left_mode is None:
                    left_mode = str(padding_mode_other)
                if right_mode is None:
                    right_mode = str(padding_mode_other)
                padding_mode_other = (left_mode, right_mode)
            if p in ("y", "vertical", "height", "periodic_y", "circular_y", "y_replicate"):
                padding_mode = "circular_y"
            elif p in ("x", "horizontal", "width", "periodic_x", "circular_x", "x_replicate"):
                padding_mode = "circular_x"
            elif p in ("xy", "both", "all", "circular", "periodic", "true", "1"):
                padding_mode = "circular"
            else:
                padding_mode = "zeros"
        else:
            padding_mode = "circular" if periodic else "zeros"

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode=padding_mode,
            padding_mode_other=padding_mode_other,
        )
        if stage_strides is None:
            stage_stride_list = [tuple(stride)] * (len(hid_blocks) - 1)
        else:
            if len(stage_strides) != len(hid_blocks) - 1:
                raise ValueError("stage_strides must have length len(hid_blocks) - 1.")
            stage_stride_list = []
            for s in stage_strides:
                if isinstance(s, int):
                    stage_stride_list.append((s,) * spatial)
                else:
                    if len(s) != spatial:
                        raise ValueError("stage_strides entries must match spatial dims.")
                    stage_stride_list.append(tuple(int(v) for v in s))

        self.unpatch = Unpatchify(patch_size=patch_size)
        self.ascent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            if i > 0:
                stage_stride = stage_stride_list[i - 1]
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stage_stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                            Unpatchify(patch_size=stage_stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stage_stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
            else:
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for blocks in self.ascent:
            for block in blocks:
                x = block(x)

        x = self.unpatch(x)

        return x
