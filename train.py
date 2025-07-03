#!/usr/bin/env python3
"""
Main training script for LitGPT using Hydra configuration.
Usage: python train.py +model=tiny +data=tinyshake +optim=adamw
"""

import os
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from litgpt_core.lightning_module import LitGPT, create_model_from_config
from litgpt_core.data_module import ShakespeareDataModule
from litgpt_core.config import Config


def setup_reproducibility(seed: int = 1337):
    """Set up reproducibility settings."""
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["GLOBAL_SEED"] = str(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set Lightning seed
    L.seed_everything(seed, workers=True)
    
    print(f"🎲 Reproducibility set up with seed: {seed}")


def create_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename="litgpt-{step:06d}",
        every_n_train_steps=cfg.val_check_interval,
        save_top_k=3,  # Keep only latest 3 checkpoints
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Timer for performance tracking
    timer = Timer()
    callbacks.append(timer)
    
    return callbacks


def create_loggers(cfg: DictConfig) -> list:
    """Create training loggers."""
    loggers = []
    
    # CSV logger (always enabled)
    csv_logger = CSVLogger(
        save_dir="logs",
        name="litgpt_training",
        version=None,
    )
    loggers.append(csv_logger)
    
    # WandB logger (optional, for Phase 1B metric logging)
    try:
        wandb_logger = WandbLogger(
            project="litgpt-baseline",
            name=f"tiny-shake-{cfg.seed}",
            save_dir="logs",
            offline=True,  # Start offline to avoid requiring API key
        )
        loggers.append(wandb_logger)
        print("📊 WandB logger enabled (offline mode)")
    except Exception as e:
        print(f"⚠️  WandB logger disabled: {e}")
    
    return loggers


@hydra.main(version_base=None, config_path="configs", config_name="default")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print("🚀 Starting LitGPT training")
    print("=" * 50)
    
    # Print configuration
    print("📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set up reproducibility
    setup_reproducibility(cfg.seed)
    
    # Create data module
    print("📚 Setting up data module...")
    data_module = ShakespeareDataModule(
        data_dir=cfg.data_dir,
        tokenizer_path=cfg.tokenizer_path,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.get("num_workers", 0),
    )
    data_module.setup("fit")
    
    # Create model configuration
    model_config_dict = {
        **cfg.model,
        "vocab_size": data_module.vocab_size,
        **cfg.optim,
    }
    
    # Create model
    print("🧠 Creating model...")
    model = create_model_from_config(model_config_dict)
    
    # Print model info
    param_count = model.get_parameter_count()
    print(f"📊 Model parameters: {param_count['total_params']:,}")
    print(f"   Trainable: {param_count['trainable_params']:,}")
    
    # Create callbacks and loggers
    callbacks = create_callbacks(cfg)
    loggers = create_loggers(cfg)
    
    # Create trainer
    print("⚡ Setting up trainer...")
    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        precision=cfg.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        deterministic=cfg.trainer.get("deterministic", False),
        benchmark=cfg.trainer.get("benchmark", True),
    )
    
    # Start training
    print("🔥 Starting training...")
    print(f"   Max steps: {cfg.trainer.max_steps}")
    print(f"   Validation every: {cfg.val_check_interval} steps")
    print(f"   Precision: {cfg.precision}")
    
    trainer.fit(model, datamodule=data_module)
    
    # Training complete
    print("\n" + "=" * 50)
    print("✅ Training completed!")
    
    # Save final metrics
    if trainer.logged_metrics:
        final_metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
        
        # Save to CSV for easy access
        import csv
        metrics_file = "metrics/baseline.csv"
        os.makedirs("metrics", exist_ok=True)
        
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in final_metrics.items():
                writer.writerow([k, v])
        
        print(f"📊 Final metrics saved to: {metrics_file}")
        
        # Print key metrics
        val_loss = final_metrics.get("val_loss", None)
        if val_loss:
            print(f"🎯 Final validation loss: {val_loss:.6f}")
            if val_loss <= 1.55:
                print("✅ SUCCESS: Validation loss ≤ 1.55 target achieved!")
            else:
                print("⚠️  WARNING: Validation loss > 1.55 target")


if __name__ == "__main__":
    train()

