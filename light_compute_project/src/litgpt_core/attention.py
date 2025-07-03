# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Attention mechanisms for GPT models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    # x: (B, nh, T, hs)
    # cos, sin: (T, hs//2) or (1, 1, T, hs//2)
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs//2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs//2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build rotary position embedding cache."""
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if torch.get_default_dtype() == torch.float16:
        cos = cos.half()
        sin = sin.half()

    return cos, sin


class KVCache:
    """Key-Value cache for efficient inference."""
    
    def __init__(
        self,
        k_shape: Tuple[int, ...],
        v_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.k = torch.zeros(k_shape, device=device, dtype=dtype)
        self.v = torch.zeros(v_shape, device=device, dtype=dtype)

    def __call__(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v


class CausalSelfAttention(nn.Module):
    """Causal self-attention mechanism."""
    
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias,
        )
        # output projection
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        
        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - hs         | head size
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram in `lit_gpt/config.py::Config`).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, nh_v, hs)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
            if input_pos_maxp1 is not None:
                # Subselect along sequence dimension
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length

        # Grouped queries: balance the number of heads across all three matrices.
        # NOTE: flash attention requires it in training mode.
        # Multi-query: this step can be skipped since there is only 1 head, allowing us to use broadcasting.
        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        v_shape = (batch_size, self.config.n_query_groups, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.config.n_query_groups,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)

