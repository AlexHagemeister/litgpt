# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Data module for text datasets with SentencePiece tokenization."""

import os
from typing import Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import lightning as L


class TextDataset(Dataset):
    """Dataset for text data with SentencePiece tokenization."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        seq_len: int = 256,
        stride: Optional[int] = None
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.stride = stride or seq_len  # Non-overlapping by default
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load and tokenize text
        print(f"📖 Loading text from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"🔤 Tokenizing text ({len(text):,} chars)")
        self.tokens = self.tokenizer.encode(text)
        print(f"✅ Tokenized to {len(self.tokens):,} tokens")
        
        # Calculate number of sequences
        self.num_sequences = max(1, (len(self.tokens) - self.seq_len) // self.stride + 1)
        print(f"📊 Dataset: {self.num_sequences:,} sequences of length {self.seq_len}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        # Handle edge case where we don't have enough tokens
        if end_idx > len(self.tokens):
            # Pad with EOS token
            sequence = self.tokens[start_idx:] + [self.tokenizer.eos_id()] * (end_idx - len(self.tokens))
        else:
            sequence = self.tokens[start_idx:end_idx]
        
        return torch.tensor(sequence, dtype=torch.long)
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()


class ShakespeareDataModule(L.LightningDataModule):
    """Lightning data module for Shakespeare dataset."""
    
    def __init__(
        self,
        data_dir: str = "data",
        tokenizer_path: str = "data/raw/sp16k.model",
        seq_len: int = 256,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.vocab_size = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if stage == "fit" or stage is None:
            # Training dataset
            train_path = os.path.join(self.data_dir, "train.txt")
            self.train_dataset = TextDataset(
                data_path=train_path,
                tokenizer_path=self.tokenizer_path,
                seq_len=self.seq_len
            )
            
            # Validation dataset
            val_path = os.path.join(self.data_dir, "val.txt")
            self.val_dataset = TextDataset(
                data_path=val_path,
                tokenizer_path=self.tokenizer_path,
                seq_len=self.seq_len
            )
            
            # Set vocab size
            self.vocab_size = self.train_dataset.vocab_size
            print(f"📚 Vocabulary size: {self.vocab_size:,}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def test_data_module():
    """Test the data module functionality."""
    print("🧪 Testing data module...")
    
    # Create data module
    dm = ShakespeareDataModule(
        data_dir="data",
        tokenizer_path="data/raw/sp16k.model",
        seq_len=32,
        batch_size=2
    )
    
    # Setup
    dm.setup("fit")
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"✅ Train batch shape: {batch.shape}")
    
    # Test val dataloader
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    print(f"✅ Val batch shape: {batch.shape}")
    
    print(f"✅ Vocab size: {dm.vocab_size}")
    print("✅ Data module test passed!")


if __name__ == "__main__":
    test_data_module()

