# Changelog

All notable changes to the LitGPT Core project will be documented in this file.

## [Phase 1A] - 2025-07-03

### Added
- **Modular Architecture**: Reorganized LitGPT into single-responsibility modules
  - `src/litgpt_core/config.py` - Configuration classes
  - `src/litgpt_core/model.py` - Main GPT model and Block classes
  - `src/litgpt_core/attention.py` - CausalSelfAttention, KVCache, RoPE utilities
  - `src/litgpt_core/mlp.py` - MLP variants (GptNeox, LLaMA, Gemma)
  - `src/litgpt_core/embedding.py` - Token embedding layers

- **Comprehensive Test Suite**: 23 unit tests covering all components
  - Shape correctness tests for all model configurations
  - Analytical gradient verification using PyTorch's gradcheck
  - Behavioral tests (causal masking, determinism, parameter counting)

- **Smoke Benchmark**: Training capability verification
  - Overfit test with 48.7% loss reduction in 20 iterations
  - 116K parameter tiny model for rapid testing
  - Gradient flow validation

- **CI/CD Pipeline**: GitHub Actions workflow
  - <5 minute runtime requirement met
  - Automated testing on Python 3.11
  - Reproducibility with fixed seeds

- **Documentation**: Comprehensive README and environment logging
  - Phase 1A specific documentation
  - Environment version tracking
  - Usage examples and configuration guide

### Environment
- **Python**: 3.11.13
- **PyTorch**: 2.7.1+cu126
- **Platform**: Linux-6.1.102-x86_64-with-glibc2.35
- **CUDA**: 12.6

### Test Results
- **Unit Tests**: 23/23 passed
- **Smoke Test**: ✅ PASSED (loss: 5.039 → 2.586)
- **CI Pipeline**: ✅ Ready for deployment

### Git Tags
- `phase1A-complete` - Architecture verification complete

---

**Confidence Level**: 0.88 (as specified in Phase 1A plan)
**Status**: Ready for Phase 1B baseline training

