# Light Compute Project

**Modular GPT training implementation with comprehensive testing and automated deployment.**

This directory contains the Light Compute Project implementation - a specialized fork of Lightning-AI/litgpt focused on modular architecture, comprehensive testing, and production-ready training pipelines.

## 🎯 Project Overview

The Light Compute Project demonstrates production-ready LLM training with:

- **Modular Architecture**: Clean, testable, single-responsibility components. _Motivation: Isolate components for granular unit-testing and easier debugging._
- **Comprehensive Testing**: 23/23 unit tests with analytical gradient verification. _Motivation: Burn down all correctness uncertainty before experimenting with new architectural ideas._
- **Automated Deployment**: Multi-region GPU deployment with cost controls. _Motivation: Create a "push-button" system to run reproducible experiments with minimal manual intervention._
- **Reproducibility**: Fixed seeds, environment capture, and audit trails. _Motivation: Ensure that baseline results form a reliable control arm for future experiments._
- **Engineering Best Practices**: CI/CD, documentation, and error handling.

## Vision and Execution

This project's execution remained highly faithful to its "north-star" vision: building a modular, easily tinkerable GPT-style decoder that could be validated and trained in minutes. The core architecture, comprehensive testing, and configuration-driven design all directly reflect this original goal.

The most significant change from the original plan was an intentional evolution in the deployment strategy. While the initial idea was a simple, single-script launcher, the project evolved to use a sophisticated, multi-region, multi-GPU deployment system (`multi_region_deploy.py`). This provides greater resilience, cost-efficiency, and automation, moving from a simple playbook to a production-ready deployment system.

## 📁 Project Structure

```
light_compute_project/
├── README.md                     # This file
├── environment_info.md           # Environment specifications
├── docs/                         # All project documentation
│   ├── PROJECT_STATUS.md         # Detailed project status
│   ├── DEPLOYMENT_GUIDE.md       # Deployment procedures
│   ├── CHANGELOG.md              # Version history
│   ├── baseline_summary.md       # Training results
│   ├── README_PHASE1A.md         # Phase 1A documentation
│   └── COMPREHENSIVE_REORGANIZATION_SUMMARY.md
├── scripts/                      # All project scripts
│   ├── train.py                  # Main training script
│   ├── test_training.py          # Training verification
│   ├── multi_region_deploy.py    # GPU deployment
│   ├── runpod_exec_training.py   # Alternative deployment
│   ├── reproducibility_audit.py  # Environment capture
│   └── smoke_test.py             # Architecture verification
├── configs/                      # Project configurations
├── src/                          # Project source code
│   └── litgpt_core/              # Modular GPT implementation
├── tests/                        # Project test suite
│   └── unit/                     # Unit tests (23 tests)
├── metrics/                      # Training metrics and results
└── data/                         # Dataset and processing
```

## 🚀 Quick Start

> **Note**: For complete setup instructions, see the main [README.md](../README.md) in the repository root.

### Environment Setup

```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install hydra-core wandb sentencepiece pytest
```

### Project-Specific Commands

```bash
# Run project tests
python -m pytest tests/unit/ -v

# Run smoke test
python scripts/smoke_test.py

# Run training
python scripts/train.py

# Deploy to GPU
python scripts/multi_region_deploy.py
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

---

**Validation Status (2025-01-08):**

- All unit tests (23/23) passing after latest refactor
- Smoke test: 48.7% loss reduction (5.04 → 2.59)
- Training test: PASSED (forward, loss, optimizer step)
- Codebase verified ready for Phase 1B baseline training

---

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

---

**Latest Validation (2025-01-08):**

- All unit tests: PASSED (23/23)
- Smoke test: PASSED (48.7% loss reduction)
- Training test: PASSED (integration)

---

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

## 📚 Documentation

- **Project Status**: `docs/PROJECT_STATUS.md` - Detailed project status and progress
- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md` - Deployment procedures and troubleshooting
- **Changelog**: `docs/CHANGELOG.md` - Version history and achievements
- **Baseline Summary**: `docs/baseline_summary.md` - Training results and analysis
- **Environment Info**: `environment_info.md` - Environment specifications

## 🔗 Related Work

This project builds upon [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) with focus on:

- **Modular Architecture**: Enhanced component separation
- **Testing Infrastructure**: Comprehensive verification suite
- **Production Readiness**: Automated deployment and monitoring
- **Research Reproducibility**: Fixed seeds and environment capture

---

**For complete repository information and setup instructions, see the main [README.md](../README.md) in the repository root.**

### Performance Validation

- **Smoke Test**: 48.7% loss reduction in 20 iterations. _Verifies that gradient flow is working correctly end-to-end._
- **Gradient Flow**: Verified through all layers via analytical `gradcheck`. _Catches mathematical errors in the backward pass._
- **Reproducibility**: Consistent results with fixed seeds.
- **CI/CD**: <5 minute pipeline runtime.
