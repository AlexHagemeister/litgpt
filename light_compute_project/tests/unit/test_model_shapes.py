"""Test shape correctness for GPT model components."""

import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from litgpt_core.model import GPT
from litgpt_core.config import Config


class TestModelShapes:
    """Test forward pass shapes for all model components."""
    
    def test_forward_shape_tiny(self):
        """Test forward pass shape for tiny model configuration."""
        cfg = Config(
            vocab_size=128, 
            n_layer=2, 
            n_head=2, 
            n_embd=64, 
            block_size=32,
            padded_vocab_size=128
        )
        model = GPT(cfg)
        B, T = 4, 32
        x = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(x)
        assert logits.shape == (B, T, cfg.padded_vocab_size), f"Expected {(B, T, cfg.padded_vocab_size)}, got {logits.shape}"
    
    def test_forward_shape_small(self):
        """Test forward pass shape for small model configuration."""
        cfg = Config(
            vocab_size=1000,
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=64,
            padded_vocab_size=1024  # Padded to multiple of 512
        )
        model = GPT(cfg)
        B, T = 2, 64
        x = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(x)
        assert logits.shape == (B, T, cfg.padded_vocab_size), f"Expected {(B, T, cfg.padded_vocab_size)}, got {logits.shape}"
    
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 16), (3, 24), (8, 32)])
    def test_forward_shape_parametrized(self, batch_size, seq_len):
        """Test forward pass shapes with different batch sizes and sequence lengths."""
        cfg = Config(
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=64,
            padded_vocab_size=256
        )
        model = GPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        logits = model(x)
        expected_shape = (batch_size, seq_len, cfg.padded_vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    def test_attention_output_shape(self):
        """Test attention layer output shape."""
        cfg = Config(
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embd=64,
            block_size=32,
            padded_vocab_size=100
        )
        model = GPT(cfg)
        B, T = 2, 16
        
        # Get intermediate attention output
        x = torch.randint(0, cfg.vocab_size, (B, T))
        embeddings = model.transformer.wte(x)
        
        # Test first block attention
        block = model.transformer.h[0]
        cos, sin = model.rope_cache()
        cos = cos[:T]
        sin = sin[:T]
        
        x_normed = block.norm_1(embeddings)
        attn_out = block.attn(x_normed, cos, sin)
        
        assert attn_out.shape == (B, T, cfg.n_embd), f"Expected {(B, T, cfg.n_embd)}, got {attn_out.shape}"
    
    def test_mlp_output_shape(self):
        """Test MLP layer output shape."""
        cfg = Config(
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embd=64,
            block_size=32,
            padded_vocab_size=100
        )
        model = GPT(cfg)
        B, T = 2, 16
        
        # Test MLP output shape
        block = model.transformer.h[0]
        x = torch.randn(B, T, cfg.n_embd)
        mlp_out = block.mlp(x)
        
        assert mlp_out.shape == (B, T, cfg.n_embd), f"Expected {(B, T, cfg.n_embd)}, got {mlp_out.shape}"
    
    def test_embedding_shape(self):
        """Test embedding layer output shape."""
        cfg = Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=256,
            block_size=128,
            padded_vocab_size=1024
        )
        model = GPT(cfg)
        B, T = 3, 64
        
        x = torch.randint(0, cfg.vocab_size, (B, T))
        embeddings = model.transformer.wte(x)
        
        assert embeddings.shape == (B, T, cfg.n_embd), f"Expected {(B, T, cfg.n_embd)}, got {embeddings.shape}"


if __name__ == "__main__":
    pytest.main([__file__])

