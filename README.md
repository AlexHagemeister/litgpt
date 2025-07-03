# LitGPT Light Compute Project

**A research project implementing modular GPT training with comprehensive testing and automated deployment.**

> 🔗 **Based on [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt)** - This is a specialized fork focused on modular architecture, comprehensive testing, and production-ready training pipelines.

## 🚀 Quick Start (Recommended: uv)

### 1. Install [uv](https://github.com/astral-sh/uv)

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
# or
brew install astral-sh/uv/uv
```

### 2. Create and Activate a Virtual Environment

```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install All Dependencies

```bash
uv pip install -e .
uv pip install hydra-core wandb sentencepiece pytest
```

### 4. Run Tests and Scripts

```bash
python -m pytest light_compute_project/tests/unit/ -v
python light_compute_project/scripts/smoke_test.py
```

### 5. Configure Deployment (Optional)

```bash
# For GPU training deployment
cp secrets.env.template secrets.env
# Edit secrets.env with your RunPod API key and GitHub PAT
```

---

## 🎯 Project Overview

This repository contains:

- **Original LitGPT**: Lightning-AI's litgpt implementation (unchanged)
- **Light Compute Project**: Our modular implementation in `light_compute_project/`

The Light Compute Project demonstrates production-ready LLM training with:

- **Modular Architecture**: Clean, testable, single-responsibility components
- **Comprehensive Testing**: 23/23 unit tests with analytical gradient verification
- **Automated Deployment**: Multi-region GPU deployment with cost controls
- **Reproducibility**: Fixed seeds, environment capture, and audit trails
- **Engineering Best Practices**: CI/CD, documentation, and error handling

## 📁 Repository Structure

```
litgpt/                           # Original Lightning-AI code (unchanged)
├── litgpt/                       # Original implementation
├── config_hub/                   # Original model configs
├── extensions/                   # Original extensions
├── tutorials/                    # Original tutorials
├── tests/                        # Original test suite
├── .github/                      # CI/CD (upstream)
├── .azure/                       # Azure config (upstream)
├── .devcontainer/                # Dev container (upstream)
├── LICENSE                       # Apache 2.0 (upstream)
├── CITATION.cff                  # Citation (upstream)
├── .gitignore                    # Git ignore (upstream)
├── .pre-commit-config.yaml       # Pre-commit (upstream)
├── pyproject.toml                # Project config (upstream)
└── README.md                     # This file

light_compute_project/            # 🆕 Our project namespace
├── README.md                     # Project-specific README
├── environment_info.md           # Environment specifications
├── docs/                         # All project documentation
│   ├── PROJECT_STATUS.md         # Detailed project status
│   ├── DEPLOYMENT_GUIDE.md       # Deployment procedures
│   ├── CHANGELOG.md              # Version history
│   ├── baseline_summary.md       # Training results
│   ├── README_PHASE1A.md         # Phase 1A documentation
│   └── STRUCTURE_MIGRATION_SUMMARY.md
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

## 📝 Environment Management

- **Preferred**: [uv](https://github.com/astral-sh/uv) (fast, modern, pyproject.toml-native)
- **Legacy**: pip/conda also supported, but uv is recommended for all workflows

## 📊 Project Status

### ✅ Phase 1A: Architecture Verification - COMPLETE

- **Tag**: `phase1A-complete`
- **Achievement**: Modular GPT implementation with 23/23 tests passing
- **Verification**: Smoke test with 48.7% loss reduction

### 🔄 Phase 1B: Baseline Training - IN PROGRESS

- **Target**: val_loss ≤ 1.55 on A100/L40S GPU
- **Infrastructure**: Complete training pipeline with all guard-rails
- **Expected**: baseline-v0 tag upon successful completion

> **For detailed project information, architecture, and testing procedures, see [`light_compute_project/README.md`](light_compute_project/README.md)**

## 📁 Key Files

### Original LitGPT (Unchanged)

- `litgpt/` - Original Lightning-AI implementation
- `config_hub/` - Original model configurations
- `extensions/` - Original extensions
- `tutorials/` - Original tutorials
- `tests/` - Original test suite

### Light Compute Project

- `light_compute_project/src/litgpt_core/` - Modular GPT implementation
- `light_compute_project/tests/unit/` - Comprehensive test suite (23 tests)
- `light_compute_project/configs/` - Hydra configuration files
- `light_compute_project/scripts/` - Training and deployment scripts
- `light_compute_project/docs/` - Project documentation

### Documentation

- `light_compute_project/docs/PROJECT_STATUS.md` - Detailed project status and progress
- `light_compute_project/docs/DEPLOYMENT_GUIDE.md` - Deployment procedures and troubleshooting
- `light_compute_project/docs/CHANGELOG.md` - Version history and achievements
- `light_compute_project/environment_info.md` - Environment specifications

## 🔗 Related Work

This project builds upon [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) with focus on:

- **Modular Architecture**: Enhanced component separation
- **Testing Infrastructure**: Comprehensive verification suite
- **Production Readiness**: Automated deployment and monitoring
- **Research Reproducibility**: Fixed seeds and environment capture

## 📞 Support

- **Repository**: https://github.com/AlexHagemeister/litgpt
- **Issues**: GitHub Issues for bug reports and questions
- **Documentation**: Comprehensive guides in `light_compute_project/docs/`
- **CI Status**: GitHub Actions for automated testing

---

**License**: Apache 2.0 (inherited from Lightning-AI/litgpt)

**Maintainer**: Research project by Alex Hagemeister

**Last Updated**: January 7, 2025
