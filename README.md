# LitGPT Light Compute Project

**A research project implementing modular GPT training with comprehensive testing and automated deployment.**

> 🔗 **Based on [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)** - This is a specialized fork focused on modular architecture, comprehensive testing, and production-ready training pipelines.

## 🎯 Project Overview

This project demonstrates production-ready LLM training with:
- **Modular Architecture**: Clean, testable, single-responsibility components
- **Comprehensive Testing**: 23/23 unit tests with analytical gradient verification
- **Automated Deployment**: Multi-region GPU deployment with cost controls
- **Reproducibility**: Fixed seeds, environment capture, and audit trails
- **Engineering Best Practices**: CI/CD, documentation, and error handling

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+ with conda
conda create -n litgpt python=3.11 -y
conda activate litgpt

# Install dependencies
pip install -e .
pip install hydra-core wandb sentencepiece
```

### Local Testing
```bash
# Run comprehensive test suite
python -m pytest tests/unit/ -v

# Verify training pipeline
python test_training.py

# Run smoke test
python smoke_test.py
```

### GPU Training Deployment
```bash
# Configure credentials in secrets.env
cp secrets.env.template secrets.env
# Edit with your GitHub PAT and RunPod API key

# Deploy baseline training
python multi_region_deploy.py
```

## 📊 Project Status

### ✅ Phase 1A: Architecture Verification - COMPLETE
- **Tag**: `phase1A-complete`
- **Achievement**: Modular GPT implementation with 23/23 tests passing
- **Verification**: Smoke test with 48.7% loss reduction
- **Infrastructure**: CI/CD pipeline and comprehensive documentation

### 🔄 Phase 1B: Baseline Training - IN PROGRESS
- **Target**: val_loss ≤ 1.55 on A100/L40S GPU
- **Infrastructure**: Complete training pipeline with all guard-rails
- **Deployment**: Multi-region strategy with automated cost controls
- **Expected**: baseline-v0 tag upon successful completion

## 🏗️ Architecture

### Modular Components
```
src/litgpt_core/
├── config.py              # Configuration management
├── model.py               # Core GPT model implementation  
├── attention.py           # Attention mechanisms and KV cache
├── mlp.py                 # Feed-forward network variants
├── embedding.py           # Token embedding layers
├── lightning_module.py    # PyTorch Lightning wrapper
└── data_module.py         # Data loading and processing
```

### Training Pipeline
- **Model**: 6 layers, 6 heads, 384 embedding (~23M parameters)
- **Dataset**: Tiny-Shakespeare with SentencePiece tokenization (16k vocab)
- **Training**: 3000 steps, fp32 precision, Flash-Attention disabled
- **Validation**: Every 500 steps targeting val_loss ≤ 1.55
- **Hardware**: A100 40GB/80GB or L40S 48GB

## 🛡️ Guard-Rails and Safety

### Training Requirements
- ✅ **Precision**: fp32 with Flash-Attention disabled
- ✅ **Target Metric**: val_loss ≤ 1.55 monitoring
- ✅ **Budget Control**: $20 hard cap with 5-minute polling
- ✅ **Reproducibility**: Fixed seeds (PYTHONHASHSEED=1337)
- ✅ **Checkpoints**: Latest 3 only retention policy
- ✅ **Logging**: Complete CSV metrics and audit trails

### Deployment Safety
- **Cost Monitoring**: Real-time budget tracking
- **Auto-Shutdown**: Pod termination after completion
- **Multi-Region**: Improved GPU availability
- **Error Recovery**: Robust retry logic and cleanup

## 📁 Key Files

### Core Implementation
- `src/litgpt_core/` - Modular GPT implementation
- `tests/unit/` - Comprehensive test suite (23 tests)
- `configs/` - Hydra configuration files

### Training and Deployment
- `train.py` - Main training script
- `multi_region_deploy.py` - Automated GPU deployment
- `test_training.py` - Local verification script

### Documentation
- `PROJECT_STATUS.md` - Detailed project status and progress
- `DEPLOYMENT_GUIDE.md` - Deployment procedures and troubleshooting
- `CHANGELOG.md` - Version history and achievements

## 🧪 Testing and Verification

### Test Coverage
```bash
# Shape correctness (all model components)
python -m pytest tests/unit/test_model_shapes.py -v

# Gradient verification (analytical gradcheck)  
python -m pytest tests/unit/test_gradients.py -v

# Behavioral tests (causal masking, determinism)
python -m pytest tests/unit/test_behavioral.py -v
```

### Performance Validation
- **Smoke Test**: 48.7% loss reduction in 20 iterations
- **Gradient Flow**: Verified through all layers
- **Reproducibility**: Consistent results with fixed seeds
- **CI/CD**: <5 minute pipeline runtime

## 🎯 Success Criteria

### Phase 1A ✅
- [x] All unit tests pass (23/23)
- [x] Gradcheck verification successful  
- [x] Smoke test achieves >1.0 loss reduction
- [x] CI pipeline <5 minutes
- [x] Tag `phase1A-complete` pushed

### Phase 1B 🔄
- [ ] Real GPU training execution
- [ ] val_loss ≤ 1.55 achieved
- [ ] All guard-rails maintained
- [ ] metrics/baseline.csv committed
- [ ] Tag `baseline-v0` pushed

## 🔗 Related Work

This project builds upon [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) with focus on:
- **Modular Architecture**: Enhanced component separation
- **Testing Infrastructure**: Comprehensive verification suite
- **Production Readiness**: Automated deployment and monitoring
- **Research Reproducibility**: Fixed seeds and environment capture

## 📞 Support

- **Repository**: https://github.com/AlexHagemeister/litgpt
- **Issues**: GitHub Issues for bug reports and questions
- **Documentation**: Comprehensive guides in repository
- **CI Status**: GitHub Actions for automated testing

---

**License**: Apache 2.0 (inherited from Lightning-AI/litgpt)  
**Maintainer**: Research project by Manus AI Agent  
**Last Updated**: January 7, 2025

