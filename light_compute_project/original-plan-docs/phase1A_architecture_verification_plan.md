## Phase 1A – Architecture & Verification  
*A meticulous development plan: commands, rationale, and reference links.*

---

### 1  Initial environment bootstrap (≈ 20 min)

| Step | Command / Action | Rationale | Confidence |
|------|------------------|-----------|-----------|
| **1.1** | `git clone https://github.com/Lightning-AI/lit-gpt.git` → `cd lit-gpt` | Pull the leanest, recipe‑rich scaffold. | 0.90 |
| **1.2** | `conda create -n litgpt python=3.11 -y && conda activate litgpt` | Isolate deps; PyTorch 2.3 prefers Python ≥ 3.10. | 0.85 |
| **1.3** | `pip install -e .[extra]` | Installs Lightning, Flash‑Attention wheels, etc. | 0.85 |
| **1.4** | ```python - <<'PY'
import torch, platform, sys; print(torch.__version__, platform.platform(), sys.version)
PY``` | Snapshot versions for the README. | 0.70 |

---

### 2  Re‑organise modules (≈ 40 min)

1. Create `src/litgpt_core/` and copy **only**  
   `model.py`, `attention.py`, `mlp.py`, `embedding.py`, `config.py`.
2. Ensure each block lives in a dedicated file; update imports.  
   *Motivation:* single‑responsibility files are unit‑test friendly.

---

### 3  Shape & gradient correctness tests (≈ 30 min)

#### 3.1  Shape test
```python
def test_forward_shape():
    from litgpt_core.model import GPT
    cfg = GPT.Config(vocab_size=128, n_layer=2, n_head=2, n_embd=64, block_size=32)
    model = GPT(cfg)
    B, T = 4, 32
    x = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, cfg.vocab_size)
```

#### 3.2  Analytical gradient check
```python
def test_gradcheck():
    import torch
    from torch.autograd import gradcheck
    from litgpt_core.model import GPT
    cfg = GPT.Config(vocab_size=17, n_layer=1, n_head=1, n_embd=8, block_size=3)
    m = GPT(cfg).double()
    inp = torch.randint(0, 17, (2, 3)).long()
    def func(i): return m(i)[0].sum()
    inp_d = inp.clone().detach().double().requires_grad_(True)
    assert gradcheck(func, (inp_d,), fast_mode=True)
```

---

### 4  Behavioural unit‑test suite with **pytest** (≈ 40 min)

| Test | What it checks |
|------|----------------|
| **Causal mask** | No attention to future tokens (`torch.triu(attn, 1)==0`). |
| **Dropout determinism** | `model.eval()` → identical outputs. |
| **Parameter count** | `sum(p.numel() …) == expected`. |
| **Config sweep** | `@pytest.mark.parametrize("layers,heads", …)` to ensure scaling works. |

---

### 5  Smoke benchmark (≈ 15 min)

```bash
python litgpt/scripts/overfit_one_batch.py \
  --config configs/tiny_shakespeare.yaml \
  optim.max_iters=20
```
Expect loss → ≈ 0 within 20 iters (CPU okay). Failure ⇒ gradient flow broken.

---

### 6  Continuous Integration (GitHub Actions) (≈ 25 min)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"
      - run: pytest -q
```
Target runtime < 5 min.

---

### 7  RunPod sanity loop (≈ 15 min)

1. Launch **Secure Cloud** A100 40 GB (image `runpod/pytorch`).  
2. SSH in → `git clone`, `pytest -q`.  
3. Run smoke benchmark on GPU (< 30 s); record tokens/s & memory.

---

### 8  Documentation & tagging (≈ 10 min)

* Update `README.md` with environment versions and CI badge.  
* `git tag phase1A-complete`.

---

### External reference links

* Lit‑GPT repo — <https://github.com/Lightning-AI/lit-gpt>  
* PyTorch `gradcheck` docs — <https://pytorch.org/docs/stable/autograd.html#gradcheck>  
* `pytest.mark.parametrize` — <https://docs.pytest.org/en/stable/example/parametrize.html>  
* GitHub Actions Python template — <https://docs.github.com/actions>  
* RunPod Secure Cloud — <https://www.runpod.io/docs>  
* Tiny‑Shakespeare corpus — <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

---

### Success checklist

- [ ] All unit tests & `gradcheck` pass locally and on CI.  
- [ ] Smoke benchmark over‑fits (CPU & GPU).  
- [ ] CI green in < 5 min.  
- [ ] Environment logged; tag `phase1A-complete` pushed.

*(Estimated confidence that this roadmap yields a mathematically sound baseline: 0.88.)*
