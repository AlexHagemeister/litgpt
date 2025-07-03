"""Test gradient correctness for GPT model components."""

import pytest
import torch
from torch.autograd import gradcheck
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from litgpt_core.model import GPT
from litgpt_core.config import Config


class TestGradients:
    """Test gradient correctness using PyTorch's gradcheck."""
    
    def test_gradcheck_tiny_model(self):
        """Test analytical gradient check for tiny model."""
        cfg = Config(
            vocab_size=17, 
            n_layer=1, 
            n_head=1, 
            n_embd=8, 
            block_size=3,
            padded_vocab_size=17
        )
        model = GPT(cfg).double()
        
        # Create input
        inp = torch.randint(0, 17, (2, 3)).long()
        
        # For embedding layers, we can't use gradcheck directly on integer inputs
        # Instead, we'll test the model with continuous inputs to the embedding layer
        # Create a continuous approximation by using embedding weights directly
        with torch.no_grad():
            # Get one-hot encoded version
            inp_onehot = torch.zeros(inp.shape[0], inp.shape[1], cfg.vocab_size, dtype=torch.double)
            inp_onehot.scatter_(2, inp.unsqueeze(-1), 1.0)
            inp_onehot.requires_grad_(True)
        
        def func_continuous(inp_oh):
            # Manual embedding lookup using matrix multiplication
            emb_weights = model.transformer.wte.weight.double()
            x = torch.matmul(inp_oh, emb_weights)
            
            # Continue with the rest of the model
            if model.config.scale_embeddings:
                x = x * torch.tensor(model.config.n_embd**0.5, dtype=x.dtype)
            
            for block in model.transformer.h:
                cos, sin = model.rope_cache()
                cos = cos[:x.size(1)].double()
                sin = sin[:x.size(1)].double()
                x = block(x, cos, sin)
            x = model.transformer.ln_f(x)
            return model.lm_head(x)[0].sum()
        
        # Run gradcheck on continuous input
        assert gradcheck(func_continuous, (inp_onehot,), fast_mode=True), "Gradient check failed for tiny model"
    
    def test_gradcheck_small_model(self):
        """Test analytical gradient check for small model."""
        cfg = Config(
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            block_size=8,
            padded_vocab_size=32
        )
        model = GPT(cfg).double()
        
        # Create input
        inp = torch.randint(0, 32, (1, 4)).long()
        
        # Use the same continuous approach as the tiny model test
        with torch.no_grad():
            # Get one-hot encoded version
            inp_onehot = torch.zeros(inp.shape[0], inp.shape[1], cfg.vocab_size, dtype=torch.double)
            inp_onehot.scatter_(2, inp.unsqueeze(-1), 1.0)
            inp_onehot.requires_grad_(True)
        
        def func_continuous(inp_oh):
            # Manual embedding lookup using matrix multiplication
            emb_weights = model.transformer.wte.weight.double()
            x = torch.matmul(inp_oh, emb_weights)
            
            # Continue with the rest of the model
            if model.config.scale_embeddings:
                x = x * torch.tensor(model.config.n_embd**0.5, dtype=x.dtype)
            
            for block in model.transformer.h:
                cos, sin = model.rope_cache()
                cos = cos[:x.size(1)].double()
                sin = sin[:x.size(1)].double()
                x = block(x, cos, sin)
            x = model.transformer.ln_f(x)
            return model.lm_head(x).sum()
        
        # Run gradcheck on continuous input
        assert gradcheck(func_continuous, (inp_onehot,), fast_mode=True), "Gradient check failed for small model"
    
    def test_gradcheck_attention_only(self):
        """Test gradient check for attention mechanism only."""
        cfg = Config(
            vocab_size=16,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_size=4,
            padded_vocab_size=16
        )
        model = GPT(cfg).double()
        
        # Test just the attention computation
        B, T = 1, 3
        x = torch.randn(B, T, cfg.n_embd, dtype=torch.double, requires_grad=True)
        
        def attn_func(x_input):
            block = model.transformer.h[0]
            cos, sin = model.rope_cache()
            cos = cos[:T].double()
            sin = sin[:T].double()
            x_normed = block.norm_1(x_input)
            return block.attn(x_normed, cos, sin).sum()
        
        assert gradcheck(attn_func, (x,), fast_mode=True), "Gradient check failed for attention"
    
    def test_gradcheck_mlp_only(self):
        """Test gradient check for MLP mechanism only."""
        cfg = Config(
            vocab_size=16,
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_size=4,
            padded_vocab_size=16
        )
        model = GPT(cfg).double()
        
        # Test just the MLP computation
        B, T = 1, 3
        x = torch.randn(B, T, cfg.n_embd, dtype=torch.double, requires_grad=True)
        
        def mlp_func(x_input):
            block = model.transformer.h[0]
            return block.mlp(x_input).sum()
        
        assert gradcheck(mlp_func, (x,), fast_mode=True), "Gradient check failed for MLP"
    
    def test_parameter_gradients_exist(self):
        """Test that all parameters receive gradients during backprop."""
        cfg = Config(
            vocab_size=50,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            padded_vocab_size=64
        )
        model = GPT(cfg)
        
        # Forward pass
        B, T = 2, 8
        x = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(x)
        loss = logits.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has infinite gradients"
    
    def test_gradient_flow_through_layers(self):
        """Test that gradients flow properly through all layers."""
        cfg = Config(
            vocab_size=100,
            n_layer=3,
            n_head=4,
            n_embd=64,
            block_size=32,
            padded_vocab_size=128
        )
        model = GPT(cfg)
        
        # Forward pass
        B, T = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(x)
        loss = logits.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradient magnitudes are reasonable
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Gradient norm should be positive and finite
        assert total_grad_norm > 0, "Total gradient norm is zero"
        assert torch.isfinite(torch.tensor(total_grad_norm)), "Total gradient norm is not finite"
        assert total_grad_norm < 1000, f"Total gradient norm is too large: {total_grad_norm}"


if __name__ == "__main__":
    pytest.main([__file__])

