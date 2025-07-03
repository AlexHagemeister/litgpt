# Changelog

All notable changes to the LitGPT Light Compute project are documented here.

## [Phase 1B] - 2025-01-07 - IN PROGRESS

### Added
- **Multi-Region Deployment Strategy**: Enhanced GPU availability
  - Support for US-ORD1, US-IAD1, EU-FRA1 regions
  - A100-40GB, A100-80GB, L40S GPU type fallback
  - Automated retry logic with 10-minute intervals
  - 60-minute wall time limit for deployment attempts

- **Complete Training Infrastructure**: Production-ready pipeline
  - PyTorch Lightning module with proper loss computation
  - Hydra configuration system for modular hyperparameters
  - SentencePiece tokenization with 16k vocabulary
  - Tiny-Shakespeare dataset processing and validation

- **Guard-Rails Implementation**: All Phase 1B requirements
  - fp32 precision with Flash-Attention disabled
  - val_loss ≤ 1.55 target monitoring
  - $20 budget hard cap with 5-minute cost polling
  - Latest 3 checkpoints retention policy
  - Fixed seeds (PYTHONHASHSEED=1337, GLOBAL_SEED=1337)
  - CSV metrics logging and comprehensive audit trail

- **Automated Deployment**: Single-script execution
  - Startup script with embedded training pipeline
  - GitHub integration for results and tagging
  - Cost monitoring and auto-shutdown
  - Real-time status reporting and error handling

- **Documentation**: Comprehensive guides and status tracking
  - `PROJECT_STATUS.md` - Complete project overview
  - `DEPLOYMENT_GUIDE.md` - Deployment procedures and troubleshooting
  - Environment specifications and requirements
  - Performance metrics and success criteria

### Changed
- Simplified deployment from complex SSH/exec to startup script approach
- Enhanced error handling and retry logic for robust deployment
- Improved project structure with cleanup of temporary files
- Updated documentation to reflect current status and next steps

### In Progress
- **Baseline Training Execution**: Awaiting GPU availability
- **Target Achievement**: val_loss ≤ 1.55 on A100/L40S hardware
- **Results Collection**: metrics/baseline.csv and baseline-v0 tag creation
- **Cost Optimization**: Expected ~$0.20-0.30 total spend

### Technical Specifications
- **Model**: 6 layers, 6 heads, 384 embedding, ~23M parameters
- **Dataset**: Tiny-Shakespeare (253k train tokens, 13k val tokens)
- **Training**: 3000 steps, validation every 500 steps
- **Optimizer**: AdamW (lr=3e-4, cosine schedule, 200 warmup)
- **Hardware**: A100 40GB/80GB or L40S 48GB target

## [Phase 1A] - 2025-01-07 - COMPLETE ✅

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
- `phase1A-complete` - Architecture verification complete ✅

### Confidence Level
0.88 (as specified in Phase 1A plan)

## [Initial Setup] - 2025-01-07

### Added
- Project initialization and planning
- Repository fork from Lightning-AI/lit-gpt
- Environment setup with conda and dependencies
- Secrets management and git configuration
- Initial project structure and documentation

---

**Repository**: https://github.com/AlexHagemeister/litgpt  
**Current Status**: Phase 1B baseline training in progress  
**Next Milestone**: baseline-v0 tag upon successful GPU training

