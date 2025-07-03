#!/usr/bin/env python3
"""
Smoke benchmark test - overfit one batch to verify gradient flow.
Based on the Phase 1A plan requirement.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from litgpt_core.model import GPT
from litgpt_core.config import Config


def create_tiny_dataset(vocab_size=128, seq_len=32, batch_size=4):
    """Create a tiny synthetic dataset for overfitting test."""
    # Create a small, repeatable dataset
    torch.manual_seed(1337)
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    return data


def smoke_benchmark():
    """Run smoke benchmark - overfit one batch."""
    print("🔥 Starting Smoke Benchmark Test")
    print("=" * 50)
    
    # Set seeds for reproducibility
    torch.manual_seed(1337)
    
    # Create tiny model configuration
    config = Config(
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        padded_vocab_size=128,
        intermediate_size=256  # 4 * n_embd
    )
    
    print(f"Model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    
    # Create model
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Create tiny dataset
    batch_size = 4
    data = create_tiny_dataset(config.vocab_size, config.block_size, batch_size)
    print(f"Data shape: {data.shape}")
    
    # Training loop - should overfit quickly
    model.train()
    initial_loss = None
    
    print("\nTraining to overfit one batch:")
    print("Iter | Loss     | Improvement")
    print("-" * 30)
    
    for iteration in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data)
        
        # Compute loss (next token prediction)
        # Shift targets: predict next token
        targets = data[:, 1:].contiguous()
        logits_shifted = logits[:, :-1, :].contiguous()
        
        loss = F.cross_entropy(
            logits_shifted.view(-1, logits_shifted.size(-1)),
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track progress
        if initial_loss is None:
            initial_loss = loss.item()
        
        improvement = initial_loss - loss.item()
        print(f"{iteration:4d} | {loss.item():.6f} | {improvement:+.6f}")
        
        # Early stopping if loss is very low
        if loss.item() < 0.1:
            print(f"\n✅ Converged at iteration {iteration}!")
            break
    
    final_loss = loss.item()
    total_improvement = initial_loss - final_loss
    
    print("\n" + "=" * 50)
    print("📊 Smoke Benchmark Results:")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Improvement:  {total_improvement:.6f}")
    print(f"Reduction:    {(total_improvement/initial_loss)*100:.1f}%")
    
    # Success criteria
    success = total_improvement > 1.0  # Should reduce loss by at least 1.0
    
    if success:
        print("✅ SMOKE TEST PASSED - Model can overfit!")
        print("   Gradient flow is working correctly.")
    else:
        print("❌ SMOKE TEST FAILED - Model cannot overfit!")
        print("   Check gradient flow and model implementation.")
    
    return success, {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'improvement': total_improvement,
        'iterations': iteration + 1,
        'total_params': total_params
    }


if __name__ == "__main__":
    success, metrics = smoke_benchmark()
    
    # Save results
    import json
    with open('smoke_test_results.json', 'w') as f:
        json.dump({
            'success': success,
            'metrics': metrics,
            'config': {
                'vocab_size': 128,
                'n_layer': 2,
                'n_head': 2,
                'n_embd': 64,
                'block_size': 32
            }
        }, f, indent=2)
    
    print(f"\n📁 Results saved to smoke_test_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

