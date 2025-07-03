# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Configuration for GPT models."""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Type, Union

import torch
import torch.nn as nn


@dataclass
class Config:
    """Configuration for GPT models."""
    
    # Model architecture
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    scale_embeddings: bool = False
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    head_size: Optional[int] = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    
    # Attention configuration
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_1: bool = True
    norm_2: bool = True
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    
    # MLP configuration
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    
    # Additional features
    attention_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    attention_scores_scalar: Optional[float] = None
    sliding_window_size: Optional[int] = None
    sliding_window_indices: Optional[list] = None
    rope_adjustments: Optional[dict] = None
    rope_indices: Optional[list] = None
    norm_qk: bool = False
    norm_qk_type: str = "default"
    
    # MoE configuration
    n_expert: int = 0
    n_expert_per_token: int = 0
    moe_intermediate_size: Optional[int] = None

    def __post_init__(self):
        # Compute derived values
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        
        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head
        
        if self.n_query_groups is None:
            self.n_query_groups = self.n_head
        
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError("LLaMAMLP requires intermediate_size to be set")
            self.intermediate_size = 4 * self.n_embd
        
        # Set up norm and MLP classes
        self.norm_class = getattr(nn, self.norm_class_name)
        
        # Import MLP classes
        if self.mlp_class_name == "GptNeoxMLP":
            from .mlp import GptNeoxMLP
            self.mlp_class = GptNeoxMLP
        elif self.mlp_class_name == "LLaMAMLP":
            from .mlp import LLaMAMLP
            self.mlp_class = LLaMAMLP
        elif self.mlp_class_name == "GemmaMLP":
            from .mlp import GemmaMLP
            self.mlp_class = GemmaMLP
        else:
            raise ValueError(f"Unknown MLP class: {self.mlp_class_name}")
        
        # Compute rope parameters
        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "Config":
        """Create config from model name."""
        # For now, return a basic config - this would normally load from a registry
        if "tiny" in name.lower():
            return cls(
                name=name,
                n_layer=6,
                n_head=6,
                n_embd=384,
                block_size=256,
                vocab_size=16000,
                **kwargs
            )
        else:
            return cls(name=name, **kwargs)


def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is >= n."""
    if n % k == 0:
        return n
    return n + k - n % k

