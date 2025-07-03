"""Behavioral unit tests for GPT model components."""

import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from litgpt_core.model import GPT
from litgpt_core.config import Config


class TestBehavioral:
    """Test behavioral properties of the model."""
    
    def test_causal_mask_attention(self):
        """Test that attention respects causal masking (no future token attention)."""
        cfg = Config(
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=16,
            padded_vocab_size=64
        )
        model = GPT(cfg)
        model.eval()
        
        B, T = 1, 8
        x = torch.randint(0, cfg.vocab_size, (B, T))
        
        # Get attention weights by hooking into the attention mechanism
        attention_weights = []
        
        def attention_hook(module, input, output):
            # This is a simplified check - in practice, we'd need to access internal attention weights
            # For now, we'll test the causal property indirectly
            pass
        
        # Test causal property: changing future tokens shouldn't affect past predictions
        x1 = x.clone()
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # Change last token
        
        with torch.no_grad():
            logits1 = model(x1)
            logits2 = model(x2)
        
        # Predictions for all tokens except the last should be identical
        torch.testing.assert_close(
            logits1[0, :-1], logits2[0, :-1], 
            msg="Causal masking violated: future tokens affect past predictions"
        )
    
    def test_dropout_determinism(self):
        """Test that model.eval() produces identical outputs."""
        cfg = Config(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=32,
            padded_vocab_size=128
        )
        model = GPT(cfg)
        model.eval()  # Set to evaluation mode
        
        B, T = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, T))
        
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)
        
        torch.testing.assert_close(
            logits1, logits2,
            msg="Model outputs are not deterministic in eval mode"
        )
    
    def test_parameter_count(self):
        """Test that parameter count matches expected values."""
        cfg = Config(
            vocab_size=1000,
            n_layer=4,
            n_head=8,
            n_embd=512,
            block_size=128,
            padded_vocab_size=1024
        )
        model = GPT(cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate expected parameters
        # Embedding: padded_vocab_size * n_embd
        # LM head: n_embd * padded_vocab_size (+ bias if enabled)
        # Each layer: attention (qkv + proj) + mlp (fc + proj) + norms
        
        embedding_params = cfg.padded_vocab_size * cfg.n_embd
        lm_head_params = cfg.n_embd * cfg.padded_vocab_size
        
        # Attention per layer: qkv + proj
        qkv_params = cfg.n_embd * (cfg.n_head + 2 * cfg.n_query_groups) * cfg.head_size
        attn_proj_params = cfg.head_size * cfg.n_head * cfg.n_embd
        
        # MLP per layer: fc + proj  
        mlp_fc_params = cfg.n_embd * cfg.intermediate_size
        mlp_proj_params = cfg.intermediate_size * cfg.n_embd
        
        # Norms per layer: 2 layer norms + final norm
        norm_params_per_layer = 2 * cfg.n_embd  # norm_1 and norm_2
        final_norm_params = cfg.n_embd
        
        layer_params = (qkv_params + attn_proj_params + mlp_fc_params + 
                       mlp_proj_params + norm_params_per_layer)
        
        expected_params = (embedding_params + lm_head_params + 
                          cfg.n_layer * layer_params + final_norm_params)
        
        # Allow some tolerance for bias terms and other small differences
        assert abs(total_params - expected_params) < expected_params * 0.1, \
            f"Parameter count mismatch: got {total_params}, expected ~{expected_params}"
    
    @pytest.mark.parametrize("n_layer,n_head", [(2, 2), (4, 4), (6, 8)])
    def test_config_sweep(self, n_layer, n_head):
        """Test that different configurations work correctly."""
        cfg = Config(
            vocab_size=256,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_head * 16,  # Ensure n_embd is divisible by n_head
            block_size=64,
            padded_vocab_size=256
        )
        model = GPT(cfg)
        
        B, T = 2, 32
        x = torch.randint(0, cfg.vocab_size, (B, T))
        
        # Should not raise any errors
        logits = model(x)
        assert logits.shape == (B, T, cfg.padded_vocab_size)
        
        # Test backward pass
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
    
    def test_sequence_length_handling(self):
        """Test that model handles different sequence lengths correctly."""
        cfg = Config(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=128,
            padded_vocab_size=128
        )
        model = GPT(cfg)
        
        # Test different sequence lengths
        for seq_len in [1, 16, 32, 64, 128]:
            B = 2
            x = torch.randint(0, cfg.vocab_size, (B, seq_len))
            
            logits = model(x)
            assert logits.shape == (B, seq_len, cfg.padded_vocab_size)
    
    def test_max_sequence_length_error(self):
        """Test that model raises error for sequences longer than block_size."""
        cfg = Config(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=64,
            block_size=32,  # Small block size
            padded_vocab_size=128
        )
        model = GPT(cfg)
        
        # Try to forward a sequence longer than block_size
        B, T = 2, 64  # T > block_size
        x = torch.randint(0, cfg.vocab_size, (B, T))
        
        with pytest.raises(ValueError, match="Cannot forward sequence of length"):
            model(x)
    
    def test_model_device_consistency(self):
        """Test that model handles device placement correctly."""
        cfg = Config(
            vocab_size=50,
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=16,
            padded_vocab_size=64
        )
        model = GPT(cfg)
        
        # Test CPU
        x_cpu = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_cpu = model(x_cpu)
        assert logits_cpu.device == x_cpu.device
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            logits_cuda = model_cuda(x_cuda)
            assert logits_cuda.device == x_cuda.device


if __name__ == "__main__":
    pytest.main([__file__])

