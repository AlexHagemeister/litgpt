# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Embedding layers for GPT models."""

import torch
import torch.nn as nn
from typing import Optional


class GPTEmbedding(nn.Module):
    """Token embedding layer for GPT models."""
    
    def __init__(self, vocab_size: int, n_embd: int, scale_embeddings: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.scale_embeddings = scale_embeddings
        self.n_embd = n_embd
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layer."""
        x = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.scale_embeddings:
            x = x * torch.tensor(self.n_embd**0.5, dtype=x.dtype)
        return x

