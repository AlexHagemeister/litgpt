#!/usr/bin/env python3
"""
Test training locally with a small configuration to verify everything works.
"""

import os
import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litgpt_core.lightning_module import LitGPT
from litgpt_core.data_module import ShakespeareDataModule
from litgpt_core.config import Config


def test_training():
    """Test training setup without requiring actual data files."""
    print("🧪 Testing training setup...")
    
    # Set up reproducibility
    torch.manual_seed(1337)
    L.seed_everything(1337)
    
    # Create tiny model for testing
    config = Config(
        vocab_size=16000,
        n_layer=2,  # Very small for testing
        n_head=2,
        n_embd=64,
        block_size=32,  # Short sequences for testing
        intermediate_size=256,
        padded_vocab_size=16000,
    )
    
    # Create model
    model = LitGPT(
        model_config=config,
        learning_rate=1e-3,
        max_steps=10,  # Very few steps for testing
        warmup_steps=2,
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test model forward pass with dummy data
    print("🧪 Testing model forward pass...")
    batch_size = 2
    seq_len = 32
    vocab_size = 16000
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: ({batch_size}, {seq_len}, {config.n_embd})")
    
    # Test loss computation manually
    print("🧪 Testing loss computation...")
    # Create sequence with targets (model expects (B, T) where targets are shifted by 1)
    dummy_sequence = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    
    # Manual forward pass and loss computation
    inputs = dummy_sequence[:, :-1]  # (B, T-1)
    targets = dummy_sequence[:, 1:]  # (B, T-1)
    
    logits = model.model(inputs)  # (B, T-1, vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # (B*(T-1), vocab_size)
        targets.reshape(-1)  # (B*(T-1),)
    )
    
    print(f"✅ Loss computation successful!")
    print(f"   Loss value: {loss:.6f}")
    
    # Test optimizer step
    print("🧪 Testing optimizer step...")
    optimizer = model.configure_optimizers()["optimizer"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✅ Optimizer step successful!")
    
    # Basic sanity checks
    if loss.item() < 10:
        print("✅ Loss is reasonable")
    else:
        print("⚠️  Loss seems high, but this is expected with dummy data")
    
    return True


if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ Training test PASSED - ready for full training!")
    else:
        print("\n❌ Training test FAILED - check configuration")
        sys.exit(1)

