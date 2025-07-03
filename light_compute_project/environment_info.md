# Environment Information

**Date:** 2025-07-03  
**Phase:** 1A/1B - uv Environment Setup

## System Environment

- **OS:** Darwin (macOS) 24.5.0 (Apple Silicon)
- **Python:** 3.11.6 (pyenv)
- **PyTorch:** 2.7.1
- **CUDA:** N/A (CPU or MPS for local dev)

## Environment Management

- **Preferred:** [uv](https://github.com/astral-sh/uv) (fast, modern, pyproject.toml-native)
- **Virtual Env:** `.venv` (created via `uv venv .venv`)
- **Activation:** `source .venv/bin/activate`
- **Install:**
  - `uv pip install -e .`
  - `uv pip install hydra-core wandb sentencepiece pytest`

## Key Dependencies

- lightning>=2.5
- torch>=2.5
- torchmetrics<3.0,>=0.7.0
- jsonargparse[signatures]>=4.37
- huggingface-hub<0.33,>=0.23.5
- hydra-core
- wandb
- sentencepiece
- pytest

## Repository Information

- **Original:** Lightning-AI/lit-gpt
- **Fork:** AlexHagemeister/litgpt
- **Clone URL:** https://github.com/AlexHagemeister/litgpt.git

---

**Note:**

- All development and testing should use the uv-managed `.venv` for reproducibility and speed.
- Legacy pip/conda workflows are still supported but not recommended.
