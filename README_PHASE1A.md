# LitGPT Core - Phase 1A Architecture Verification

[![CI](https://github.com/AlexHagemeister/litgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexHagemeister/litgpt/actions/workflows/ci.yml)

**Modular GPT implementation for architecture verification and baseline training.**

This is a fork of [Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt) reorganized into a modular architecture for the LLM Light Compute project Phase 1A.

## 🎯 Phase 1A Objectives

- ✅ **Architecture Verification**: Modular design with single-responsibility components
- ✅ **Mathematical Correctness**: Comprehensive gradient and shape verification
- ✅ **Training Capability**: Smoke test demonstrates successful overfitting
- ✅ **CI/CD Pipeline**: Automated testing with <5 minute runtime

## 🏗️ Architecture

### Modular Design (`src/litgpt_core/`)

```
src/litgpt_core/
├── __init__.py          # Package initialization
├── config.py            # Model configuration classes
├── model.py             # Main GPT model and Block classes
├── attention.py         # CausalSelfAttention, KVCache, RoPE
├── mlp.py              # GptNeoxMLP, LLaMAMLP, GemmaMLP
└── embedding.py        # Token embedding layers
```

### Key Features

- **Single Responsibility**: Each component in dedicated file
- **Clean Dependencies**: Minimal imports, clear interfaces
- **Type Safety**: Comprehensive type hints throughout
- **Test Coverage**: 23 unit tests covering all components

## 🧪 Verification Results

### Environment
- **Python**: 3.11.13
- **PyTorch**: 2.7.1+cu126
- **Platform**: Linux-6.1.102-x86_64-with-glibc2.35
- **CUDA**: 12.6

### Test Results
```
✅ 23/23 tests passed
├── Shape Tests (7): Forward pass shapes verified
├── Gradient Tests (6): Analytical gradcheck passed
└── Behavioral Tests (10): Causal masking, determinism verified
```

### Smoke Benchmark
```
🔥 Smoke Test Results:
├── Model: 2 layers, 2 heads, 64 embd (116,480 params)
├── Initial Loss: 5.039 → Final Loss: 2.586
├── Improvement: 2.45 (48.7% reduction in 20 iterations)
└── Status: ✅ PASSED - Gradient flow verified
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexHagemeister/litgpt.git
cd litgpt

# Create conda environment
conda create -n litgpt python=3.11 -y
conda activate litgpt

# Install dependencies
pip install -e .
pip install pytest
```

### Run Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run smoke benchmark
python smoke_test.py
```

### Basic Usage

```python
from litgpt_core.model import GPT
from litgpt_core.config import Config

# Create tiny model
config = Config(
    vocab_size=1000,
    n_layer=4,
    n_head=4,
    n_embd=256,
    block_size=128
)

model = GPT(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
import torch
x = torch.randint(0, config.vocab_size, (2, 64))
logits = model(x)
print(f"Output shape: {logits.shape}")  # (2, 64, vocab_size)
```

## 📊 Model Configurations

| Config | Layers | Heads | Embd | Params | Use Case |
|--------|--------|-------|------|--------|----------|
| Tiny   | 2      | 2     | 64   | 116K   | Testing  |
| Small  | 4      | 4     | 256  | 2.3M   | Development |
| Base   | 6      | 6     | 384  | 15M    | Baseline |

## 🔬 Testing Framework

### Shape Correctness
- Forward pass shapes for all model sizes
- Attention output verification
- MLP output verification
- Embedding layer verification

### Gradient Correctness
- Analytical gradient checking via PyTorch's `gradcheck`
- Component-wise gradient verification
- End-to-end gradient flow validation

### Behavioral Testing
- Causal masking enforcement
- Dropout determinism in eval mode
- Parameter counting accuracy
- Device consistency

## 🏃‍♂️ CI/CD Pipeline

The GitHub Actions pipeline runs in <5 minutes and includes:

1. **Environment Setup**: Python 3.11, dependency installation
2. **Unit Tests**: All 23 tests with verbose output
3. **Smoke Test**: Training capability verification
4. **Reproducibility**: Fixed seeds (PYTHONHASHSEED=1337)

## 📈 Performance Metrics

### Smoke Test Benchmarks
- **Convergence**: <20 iterations to significant loss reduction
- **Memory**: <1GB for tiny model training
- **Speed**: ~100ms per iteration on CPU

### Parameter Efficiency
- **Tiny Model**: 116K parameters, suitable for rapid iteration
- **Modular Design**: Easy to scale up/down components
- **Memory Optimized**: Efficient attention and MLP implementations

## 🔧 Development

### Adding New Components

1. Create new module in `src/litgpt_core/`
2. Add corresponding tests in `tests/unit/`
3. Update `__init__.py` exports
4. Run test suite to verify integration

### Configuration System

The `Config` class supports:
- **Model Architecture**: layers, heads, embedding dimensions
- **Attention Settings**: grouped queries, RoPE parameters
- **MLP Variants**: GptNeox, LLaMA, Gemma styles
- **Training Options**: bias, normalization, activation functions

## 📚 References

- **Original LitGPT**: [Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt)
- **nanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **GPT-NeoX**: [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

**Phase 1A Status**: ✅ **COMPLETE** - Architecture verified, ready for baseline training.

