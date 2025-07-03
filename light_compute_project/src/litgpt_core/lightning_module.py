# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""PyTorch Lightning module for GPT training."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import GPT
from .config import Config


class LitGPT(L.LightningModule):
    """Lightning module for GPT training."""
    
    def __init__(
        self,
        model_config: Config,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 200,
        max_steps: int = 3000,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = GPT(model_config)
        self.model_config = model_config
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        
        # Metrics tracking
        self.train_step_outputs = []
        self.val_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # batch shape: (B, T)
        # For language modeling, we predict the next token
        inputs = batch[:, :-1]  # (B, T-1)
        targets = batch[:, 1:]  # (B, T-1)
        
        # Forward pass
        logits = self.model(inputs)  # (B, T-1, vocab_size)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # (B*(T-1), vocab_size)
            targets.reshape(-1)  # (B*(T-1),)
        )
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True)
        
        # Store for epoch-level metrics
        self.train_step_outputs.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Same as training step but without gradient computation
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = self.model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True)
        
        # Store for epoch-level metrics
        self.val_step_outputs.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self.train_step_outputs:
            avg_loss = torch.stack(self.train_step_outputs).mean()
            self.log("train_loss_epoch", avg_loss)
            self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        if self.val_step_outputs:
            avg_loss = torch.stack(self.val_step_outputs).mean()
            self.log("val_loss_epoch", avg_loss)
            self.val_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=1e-8
        )
        
        # Create learning rate scheduler
        min_lr = self.learning_rate * self.min_lr_ratio
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine annealing
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return min_lr / self.learning_rate + (1 - min_lr / self.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params
        }
    
    def estimate_tokens_per_second(self, batch_size: int, seq_len: int, step_time: float) -> float:
        """Estimate tokens processed per second."""
        tokens_per_batch = batch_size * seq_len
        return tokens_per_batch / step_time if step_time > 0 else 0.0


def create_model_from_config(config_dict: Dict[str, Any]) -> LitGPT:
    """Create LitGPT model from configuration dictionary."""
    # Extract model config
    model_config = Config(
        vocab_size=config_dict.get("vocab_size", 16000),
        n_layer=config_dict.get("n_layer", 6),
        n_head=config_dict.get("n_head", 6),
        n_embd=config_dict.get("n_embd", 384),
        block_size=config_dict.get("block_size", 256),
        intermediate_size=config_dict.get("d_ff", 1536),
        padded_vocab_size=config_dict.get("padded_vocab_size", 16000),
    )
    
    # Extract training config
    training_config = {
        "learning_rate": config_dict.get("lr", 3e-4),
        "weight_decay": config_dict.get("weight_decay", 0.1),
        "betas": config_dict.get("betas", (0.9, 0.95)),
        "warmup_steps": config_dict.get("warmup_steps", 200),
        "max_steps": config_dict.get("max_steps", 3000),
        "min_lr_ratio": config_dict.get("min_lr_ratio", 0.1),
    }
    
    return LitGPT(model_config, **training_config)

