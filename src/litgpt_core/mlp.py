# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""MLP layers for GPT models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import Config


class GptNeoxMLP(nn.Module):
    """GPT-NeoX style MLP with GELU activation."""
    
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    """LLaMA style MLP with SiLU activation and gating."""
    
    def __init__(self, config: Config, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    """Gemma style MLP with GELU activation and gating."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)

