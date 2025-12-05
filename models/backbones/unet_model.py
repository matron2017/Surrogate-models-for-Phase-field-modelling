#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNet (PhysicsNeMo-native)
-------------------------
• Subclasses physicsnemo.models.module.Module
• Declares ModelMetaData (AMP/CUDA Graphs toggles)
• Stores JSON-serialisable _init_kwargs for PN checkpoint helpers
• Tanh output in [-1, 1]
"""

from dataclasses import dataclass
from typing import Dict, Any
import sys

import torch
import torch.nn as nn

# Local UNet parts
sys.path.append('/scratch/project_2008261/alloy_solidification/src/models')
from unet_parts import DoubleConv, Down, Up, OutConv  # noqa: E402

# PhysicsNeMo core
from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData


@dataclass
class UNetMetaData(ModelMetaData):
    """Runtime/optimisation capabilities advertised to PhysicsNeMo."""
    name: str = "UNet"
    # Enable only features supported by the implementation
    jit: bool = False          # set True only after verifying TorchScript
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class UNet(Module):
    """
    UNet for multi-channel inputs/outputs (PhysicsNeMo-native).

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output channels.
    bilinear : bool, optional
        Use bilinear upsampling in the decoder (default: True).
    in_factor : int, optional
        Base number of feature maps in the first encoder block (default: 48).
    use_small : bool, optional
        If True, a shallower variant where the first decoder skip starts from 8× (default: False).
    debug : bool, optional
        If True, prints tensor shapes during forward (default: False).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool = True,
        in_factor: int = 48,
        use_small: bool = False,
        debug: bool = False,
    ):
        super().__init__(meta=UNetMetaData())

        # Record constructor args (kept JSON-serialisable for PN checkpointing)
        self._init_kwargs: Dict[str, Any] = dict(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            in_factor=in_factor,
            use_small=use_small,
            debug=debug,
        )

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_small = use_small
        self.debug = debug

        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(n_channels, in_factor)              # -> in_factor
        self.down1 = Down(in_factor,     in_factor * 2)             # 1024 -> 512
        self.down2 = Down(in_factor * 2, in_factor * 4)             # 512  -> 256
        self.down3 = Down(in_factor * 4, in_factor * 8)             # 256  -> 128
        self.down4 = Down(in_factor * 8, in_factor * 16 // factor)  # 128  -> 64

        # Decoder
        self.up1  = Up(in_factor * 16,         in_factor * 8 // factor, bilinear)  # 64  -> 128
        self.up2  = Up(in_factor * 8,          in_factor * 4 // factor, bilinear)  # 128 -> 256
        self.up3  = Up(in_factor * 4,          in_factor * 2 // factor, bilinear)  # 256 -> 512
        self.up4  = Up(in_factor * 2,          in_factor,               bilinear)  # 512 -> 1024
        self.outc = OutConv(in_factor, n_classes)

        self.out_activation = nn.Tanh()

    # Optional rank-safe debug printer (kept silent during CUDA Graph capture)
    def _dprint(self, *args):
        if self.debug:
            print(*args, flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._dprint("UNet: input", x.shape)
        # Optional statistic (not used further)
        _ = x.sum(dim=[1, 2, 3])

        # Encoder
        x1 = self.inc(x);    self._dprint("inc:",   x1.shape)
        x2 = self.down1(x1); self._dprint("down1:", x2.shape)
        x3 = self.down2(x2); self._dprint("down2:", x3.shape)
        x4 = self.down3(x3); self._dprint("down3:", x4.shape)
        x5 = self.down4(x4); self._dprint("down4:", x5.shape)

        # Decoder
        if not self.use_small:
            x = self.up1(x5, x4); self._dprint("up1:", x.shape)
            x = self.up2(x,  x3); self._dprint("up2:", x.shape)
        else:
            x = self.up1(x4, x3); self._dprint("up1 (small):", x.shape)

        x = self.up3(x,  x2); self._dprint("up3:", x.shape)
        x = self.up4(x,  x1); self._dprint("up4:", x.shape)

        logits = self.outc(x); self._dprint("outc:", logits.shape)
        return self.out_activation(logits)


if __name__ == "__main__":
    # Smoke test + PN checkpoint round-trip (CPU)
    m = UNet(n_channels=2, n_classes=2, bilinear=False, in_factor=76, use_small=False, debug=True)
    x = torch.randn(1, 2, 1024, 1024)
    y = m(x)
    print("Output shape:", y.shape)

    # Save/load using PN helpers (constructor args must be JSON-serialisable)
    m.save("unet_pn_untrained.mdlus")
    m2 = Module.from_checkpoint("unet_pn_untrained.mdlus")
    m2.eval()
    with torch.inference_mode():
        y2 = m2(torch.zeros_like(x))
    print("Reloaded OK:", tuple(y2.shape))
