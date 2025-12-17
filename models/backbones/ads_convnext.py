"""
Autoregressive Deep Surrogate (ADS) with an isotropic ConvNeXt backbone.

Implements:
- DropPath (stochastic depth)
- ConvND with optional periodic padding
- ConvNeXt-style LayerNorm
- ConvNeXtBlock (depthwise + pointwise MLP + residual)
- ADSConvNeXt backbone (resolution preserving)
- ADSAutoregressiveSurrogate (one-step + rollout)
- PFSequenceDataset (state/conditioning trajectory pairs)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class ConvND(nn.Module):
    """2D/3D convolution with optional periodic padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dim: int = 2,
        periodic: bool = False,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert dim in (2, 3), "ConvND supports only 2D or 3D"
        self.dim = dim
        self.periodic = periodic
        self.kernel_size = int(kernel_size)

        padding = 0 if periodic else self.kernel_size // 2

        if dim == 2:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=padding,
                groups=groups,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.periodic and self.kernel_size > 1:
            pad = self.kernel_size // 2
            if self.dim == 2:
                x = F.pad(x, (pad, pad, pad, pad), mode="circular")
            else:
                x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="circular")
        return self.conv(x)


class LayerNorm(nn.Module):
    """LayerNorm for channels_last or channels_first."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
        DIM: int = 2,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.DIM = DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.DIM == 2:
            weight = self.weight[:, None, None]
            bias = self.bias[:, None, None]
        else:
            weight = self.weight[:, None, None, None]
            bias = self.bias[:, None, None, None]
        return weight * x + bias


class ConvNeXtBlock(nn.Module):
    """Resolution-preserving ConvNeXt block (isotropic)."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
        DIM: int = 2,
        periodic: bool = False,
    ):
        super().__init__()
        self.DIM = DIM
        self.channel_last = [0] + list(range(2, 2 + DIM)) + [1]
        self.channel_first = [0, DIM + 1] + list(range(1, 1 + DIM))

        self.dwconv = ConvND(dim, dim, kernel_size=kernel_size, dim=DIM, periodic=periodic, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last", DIM=DIM)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(*self.channel_last)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(*self.channel_first)
        return shortcut + self.drop_path(x)


class ADSConvNeXt(nn.Module):
    """Resolution-preserving ConvNeXt backbone (SI-ConvNeXt style)."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        depth: int = 7,
        dim: int = 128,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 0.0,
        head_init_scale: float = 1.0,
        DIM: int = 2,
        periodic: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.DIM = DIM

        self.stem = ConvND(in_chans, dim, kernel_size=3, dim=DIM, periodic=periodic)

        dp_rates = torch.linspace(0.0, drop_path_rate, depth).tolist()
        blocks = [
            ConvNeXtBlock(
                dim=dim,
                kernel_size=kernel_size,
                drop_path=dp_rates[i],
                layer_scale_init_value=layer_scale_init_value,
                DIM=DIM,
                periodic=periodic,
            )
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first", DIM=DIM)
        self.head = ConvND(dim, out_chans, kernel_size=1, dim=DIM, periodic=False)

        self._init_weights(head_init_scale)

    def _init_weights(self, head_init_scale: float):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if hasattr(self.head, "conv"):
            self.head.conv.weight.data.mul_(head_init_scale)
            if self.head.conv.bias is not None:
                self.head.conv.bias.data.mul_(head_init_scale)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)


class ADSAutoregressiveSurrogate(nn.Module):
    """Autoregressive surrogate using an ADSConvNeXt backbone."""

    def __init__(
        self,
        state_channels: int,
        cond_channels: int = 0,
        dim: int = 128,
        depth: int = 7,
        kernel_size: int = 3,
        periodic: bool = False,
        drop_path_rate: float = 0.0,
        noise_std: float = 1e-3,
        DIM: int = 2,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.cond_channels = cond_channels
        self.noise_std = noise_std

        in_chans = state_channels + cond_channels
        out_chans = state_channels

        self.backbone = ADSConvNeXt(
            in_chans=in_chans,
            out_chans=out_chans,
            depth=depth,
            dim=dim,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=0.0,
            head_init_scale=1.0,
            DIM=DIM,
            periodic=periodic,
            kernel_size=kernel_size,
        )

    def _prepare_cond(self, cond: Optional[torch.Tensor], spatial: Tuple[int, int]) -> Optional[torch.Tensor]:
        if cond is None:
            return None
        if cond.dim() == 2:
            # (B, C) -> (B, C, H, W)
            cond = cond[..., None, None].expand(-1, -1, spatial[0], spatial[1])
        elif cond.dim() == 4:
            assert cond.shape[-2:] == spatial, "Cond map must match spatial dims"
        else:
            raise ValueError("Conditioning must be (B,C) or (B,C,H,W)")
        return cond

    def _concat_state_cond(self, state: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return state
        return torch.cat([state, cond], dim=1)

    def forward_step(
        self,
        state: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        if self.training and add_noise and self.noise_std > 0.0:
            state = state + torch.randn_like(state) * self.noise_std
        cond_map = self._prepare_cond(cond, (state.shape[-2], state.shape[-1]))
        x = self._concat_state_cond(state, cond_map)
        return self.backbone(x)

    def forward(self, state: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        return self.forward_step(state, cond, add_noise=self.training)

    @torch.no_grad()
    def rollout(
        self,
        state0: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B, _, H, W = state0.shape
        if cond_seq is not None:
            assert cond_seq.dim() in (3, 5), "cond_seq should be (B,T,C) or (B,T,C,H,W)"
            if cond_seq.dim() == 3:
                cond_seq = cond_seq[..., None, None].expand(-1, -1, -1, H, W)
            Bc, T, _, Hc, Wc = cond_seq.shape
            assert Bc == B and Hc == H and Wc == W
            n_steps = T if n_steps is None else min(n_steps, T)
        else:
            assert n_steps is not None, "n_steps must be specified when cond_seq is None"

        states = [state0]
        state = state0
        for t in range(n_steps):
            cond_t = cond_seq[:, t] if cond_seq is not None else None
            state = self.forward_step(state, cond_t, add_noise=False)
            states.append(state)
        return torch.stack(states, dim=1)


class PFSequenceDataset(Dataset):
    """Phase-field trajectory dataset producing (u_t, u_{t+1}, cond_t) pairs."""

    def __init__(
        self,
        state_seq: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
        dt_stride: int = 1,
        noise_std: float = 0.0,
    ):
        super().__init__()
        assert state_seq.ndim == 5, "state_seq should be (N, T, C, H, W)"
        self.state_seq = state_seq
        self.cond_seq = cond_seq
        self.dt_stride = int(dt_stride)
        self.noise_std = float(noise_std)

        self.N, self.T, self.C_state, self.H, self.W = state_seq.shape
        self.num_pairs_per_sim = self.T - self.dt_stride
        if self.num_pairs_per_sim <= 0:
            raise ValueError("Not enough timesteps for the requested dt_stride.")

        if cond_seq is not None:
            assert cond_seq.shape[0] == self.N
            assert cond_seq.shape[1] == self.T
            assert cond_seq.shape[3] == self.H and cond_seq.shape[4] == self.W

    def __len__(self) -> int:
        return self.N * self.num_pairs_per_sim

    def __getitem__(self, idx: int):
        sim_idx = idx // self.num_pairs_per_sim
        t = idx % self.num_pairs_per_sim
        t_next = t + self.dt_stride

        state_t = self.state_seq[sim_idx, t]  # (C_state, H, W)
        state_tp1 = self.state_seq[sim_idx, t_next]

        cond_t = None
        if self.cond_seq is not None:
            cond_t = self.cond_seq[sim_idx, t]  # (C_cond, H, W)

        if self.noise_std > 0.0:
            state_t = state_t + self.noise_std * torch.randn_like(state_t)

        sample = {"state_t": state_t, "state_tp1": state_tp1}
        if cond_t is not None:
            sample["cond_t"] = cond_t
        return sample


__all__ = [
    "ADSConvNeXt",
    "ADSAutoregressiveSurrogate",
    "PFSequenceDataset",
    "ConvNeXtBlock",
    "ConvND",
    "LayerNorm",
    "DropPath",
]
