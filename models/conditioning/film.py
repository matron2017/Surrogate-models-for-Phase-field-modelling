#!/usr/bin/env python3
"""
FiLM.py

This module defines FiLM conditioning modules in a style compatible with Nvidia Modulus.
It now assumes the thermal gradient (or other conditioning input) is already scaled
as needed, so there's no built-in gradient normalization here.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

#from modulus.models.meta import ModelMetaData
#from modulus.models.module import Module
from physicsnemo.models.module import Module
from physicsnemo.models.meta import ModelMetaData


@dataclass
class FiLMMetaData(ModelMetaData):
    """
    Metadata for the FiLM modules.
    Used by Modulus for JIT, AMP, and other runtime options.
    """
    name: str = "FiLMModule"
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class FiLMGenerator(Module):
    """
    A simplified FiLM generator for Nvidia Modulus.

    It uses a single linear layer to map the conditioning input (e.g., a scalar or
    vector feature) to a concatenated vector of length 2 * num_channels. That vector
    is split into two halves: delta_gamma, delta_beta.

    The final modulation parameters are computed as:
        gamma = 1.0 + gamma_scale * delta_gamma
        beta  = beta_scale * delta_beta

    The linear layer is initialized to zero so that initially gamma = 1.0, beta = 0,
    i.e. no modulation.
    """
    def __init__(self, cond_dim, num_channels, gamma_scale=1, beta_scale=1):
        super().__init__(meta=FiLMMetaData())
        # Single linear layer that outputs 2*num_channels values:
        self.fc = nn.Linear(cond_dim, num_channels * 2)
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        
        # Initialize weights/biases to zero => identity modulation at first
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, cond_input):
        """
        cond_input: shape [B, cond_dim]

        Returns:
          gamma, beta: shape [B, num_channels] each
        """
        out = self.fc(cond_input)  # shape [B, 2*num_channels]
        num_channels = out.size(1) // 2
        delta_gamma = out[:, :num_channels]
        delta_beta  = out[:,  num_channels:]
        gamma = 1.0 + self.gamma_scale * delta_gamma
        beta  =         self.beta_scale * delta_beta
        return gamma, beta

class FiLM(Module):
    """
    FiLM applies feature-wise affine modulation:
        output = gamma * x + beta

    x: shape [B, C, H, W]
    gamma, beta: shape [B, C]
    We unsqueeze them to [B, C, 1, 1] for broadcasting.
    """
    def __init__(self):
        super().__init__(meta=FiLMMetaData())

    def forward(self, x, gamma, beta):
        # shape: gamma, beta => [B, C], unsqueeze to [B, C, 1, 1]
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta  = beta.unsqueeze(2).unsqueeze(3)
        return gamma * x + beta
