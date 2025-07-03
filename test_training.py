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
sys.path.insert(0, str(Path(__file__).parent / "src"))

from litgpt_core.lightning_module import LitGPT
from litgpt_core.data_module import ShakespeareDataModule
from litgpt_core.config import Config


def test_training():
    """Test training with minimal configuration."""
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
    
    # Create data module
    data_module = ShakespeareDataModule(
        data_dir="data",
        tokenizer_path="data/raw/sp16k.model",
        seq_len=32,  # Short sequences
        batch_size=2,  # Small batch
        num_workers=0,
    )
    data_module.setup("fit")
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=10,
        precision="32-true",
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        val_check_interval=5,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,  # Disable logging for test
    )
    
    # Test training
    print("🔥 Running test training...")
    trainer.fit(model, datamodule=data_module)
    
    # Check final metrics
    if trainer.logged_metrics:
        val_loss = trainer.logged_metrics.get("val_loss", None)
        train_loss = trainer.logged_metrics.get("train_loss", None)
        
        print(f"✅ Test completed!")
        if train_loss:
            print(f"   Final train loss: {train_loss:.6f}")
        if val_loss:
            print(f"   Final val loss: {val_loss:.6f}")
        
        # Basic sanity checks
        if train_loss and train_loss < 10:
            print("✅ Training loss is reasonable")
        if val_loss and val_loss < 10:
            print("✅ Validation loss is reasonable")
        
        return True
    else:
        print("❌ No metrics logged")
        return False


if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ Training test PASSED - ready for full training!")
    else:
        print("\n❌ Training test FAILED - check configuration")
        sys.exit(1)

