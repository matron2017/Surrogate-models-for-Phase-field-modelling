# models/models/fno_field.py
# No second-person phrasing in comments

from dataclasses import dataclass
import torch
import torch.nn as nn

from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.fno.fno import FNO
from models.conditioning.skip_condition import ConditionalScaler


@dataclass
class FNOFieldMetaData(ModelMetaData):
    name: str = "FNO_Field2D_Cond"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class FNO_Field2D(Module):
    """
    Single-stage FNO over 2D fields with scalar conditioning via multiplicative scaling.
    Input:  (B, n_channels, H, W)
    Output: (B, n_classes,  H, W)
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 2,
        cond_dim: int = 2,
        embed_channels: int = 256,
        fno_inp_shape=(64, 64),
        fno_depth: int = 12,
        fno_latent_channels: int = 0,
        fno_num_modes: int = 16,
        fno_decoder_layers: int = 2,
        fno_decoder_layer_size: int = 0,
        fno_padding: int = 0,
        fno_coord_features: bool = False,
    ):
        super().__init__(meta=FNOFieldMetaData())

        self.embed = nn.Conv2d(n_channels, embed_channels, kernel_size=1, bias=True)

        self.scaler = ConditionalScaler(
            cond_dim=cond_dim,
            widths=[embed_channels],
            hidden=128,
            identity_init=True,
        )

        latent_ch = fno_latent_channels or embed_channels
        decoder_size = fno_decoder_layer_size or latent_ch

        self.fno = FNO(
            in_channels=embed_channels,
            out_channels=embed_channels,
            decoder_layers=fno_decoder_layers,
            decoder_layer_size=decoder_size,
            dimension=2,
            latent_channels=latent_ch,
            num_fno_layers=fno_depth,
            num_fno_modes=fno_num_modes,
            padding=fno_padding,
            activation_fn="gelu",
            coord_features=fno_coord_features,
        )

        self.head = nn.Conv2d(embed_channels, n_classes, kernel_size=1, bias=True)

    @staticmethod
    def _apply_scale(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return x * s.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        (scale,) = self.scaler(cond_vec)
        x = self._apply_scale(x, scale)
        x = self.fno(x)
        x = self.head(x)
        return x
