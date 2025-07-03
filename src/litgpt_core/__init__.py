"""
LitGPT Core - Modular GPT implementation for architecture verification and training.

This package contains the core components of the GPT model split into dedicated files
for better testability and maintainability.
"""

from .model import GPT
from .config import Config

__all__ = ["GPT", "Config"]

