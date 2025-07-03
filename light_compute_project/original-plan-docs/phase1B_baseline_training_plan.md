## Phase 1B – Baseline Training  
*Detailed implementation guide for turning the verified skeleton from Phase 1A into a reproducible reference run.*

---

### 0  Quick orientation  
Train the **“tiny” GPT decoder** on a public‑domain corpus (Tiny‑Shakespeare or TinyStories) until it reaches the same validation loss Lit‑GPT reports (`≈ 1.47` on Tiny‑Shakespeare). Every knob lives in Hydra YAML so later experiments are config‑only diffs.

---

### 1  Dataset acquisition & tokenisation (≈ 45 min)

| Step | Command / Action | Rationale | Reference |
|------|------------------|-----------|-----------|
| **1.1** | `curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` → `data/raw/` | Tiny, licence‑free corpus (~1 MB). | [Tiny‑Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) |
| **1.2** | **SentencePiece** – `pip install sentencepiece` → `spm_train --input=input.txt --model_prefix=sp16k --vocab_size=16000 --character_coverage=1.0` | 16 k merges; extensible later. | [SentencePiece docs](https://github.com/google/sentencepiece) |
| **1.3** | Split 95 %/5 % into `train.txt`, `val.txt`. | Matches Lit‑GPT recipe. | — |
| **1.4** | Implement `TextDataset` in `data_module.py` that wraps the SentencePiece model and yields fixed‑length chunks (`seq_len = 256`). | Data becomes a Lightning plug‑in. | — |

---

### 2  Hydra configuration scaffolding (≈ 40 min)

```
configs/
│─ default.yaml
│
├─ model/
│   └─ tiny.yaml
├─ data/
│   └─ tinyshake.yaml
└─ optim/
    └─ adamw.yaml
```

*Expose*: `precision`, `flash.enabled`, `trainer.max_steps`, `seed`.  
Example:

```bash
python train.py +model=tiny +data=tinyshake optim.lr=3e-4 trainer.precision=bfloat16
```

Hydra tutorial → <https://hydra.cc>.

---

### 3  Baseline hyper‑parameters (≈ 15 min)

| Hyper‑parameter | Value |
|-----------------|-------|
| `n_layer` | 6 |
| `n_head` | 6 |
| `d_model` | 384 |
| `d_ff` | 1536 |
| `block_size` | 256 |
| `vocab_size` | 16 000 |
| optimiser | AdamW (β₁ 0.9, β₂ 0.95) |
| LR schedule | cosine, 200 warm‑up steps |
| batch tokens | 8192 |
| precision | **fp32** |

Matches nanoGPT “tiny” + Lit‑GPT tiny Shakespeare config.

---

### 4  Training script (PyTorch Lightning) (≈ 30 min)

```python
import lightning as L
from litgpt_core.model import GPT
from data_module import ShakespeareData
from omegaconf import DictConfig

def train(cfg: DictConfig):
    dm = ShakespeareData(cfg.data)
    model = GPT(cfg.model)
    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=500,
        callbacks=[L.callbacks.ModelCheckpoint(every_n_train_steps=500)]
    )
    trainer.fit(model, datamodule=dm)
```

Flash‑Attention and bf16 **off** for baseline.

Lightning docs → <https://lightning.ai/docs/pytorch/stable/quickstart.html>.

---

### 5  Run on RunPod A100 40 GB (≈ 20 min)

```bash
python train.py +model=tiny +data=tinyshake +optim=adamw   trainer.max_steps=3000 seed=1337
```

*Expected*:

| Metric | Value |
|--------|-------|
| Wall‑clock | ≈ 3 min |
| `val_loss` | ~ 1.47 |
| Tokens/s | ~ 75 k |
| GPU RAM | < 4 GB |

Loss > 1.55 ⇒ check tokeniser/LR; RAM > 6 GB ⇒ check batch size.

---

### 6  Metric logging (≈ 15 min)

```python
from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project="litgpt-baseline", name="tiny-shake")
```

Log `val_loss`, `tokens_per_sec`, `gpu_memory`; export to `metrics/baseline.csv`.

---

### 7  Optional toggles (compile ✓, default ✗)

| Toggle | Compile | Enable later |
|--------|---------|--------------|
| Flash-Attention 2 | `pip install flash-attn` | After baseline lock |
| bf16 | `trainer.precision=bfloat16` | When extra head‑room needed |

---

### 8  Reproducibility audit (≈ 20 min)

```bash
for seed in 1337 2025; do
  python train.py ... seed=$seed
done
```

Accept variance < 1 %. Larger drift ⇒ seed the dataloader generator too.

---

### 9  Tag & freeze

```bash
git add metrics/baseline.csv configs README.md
git commit -m "Baseline tiny Shakespeare ±5% achieved"
git tag baseline-v0 && git push --tags
```

README snippet:

```
val_loss = 1.473  (3 min, RunPod A100 40 GB)
tokens/s = 74.8k
GPU RAM  = 3.9 GB
Env: torch‑2.3.0 + CUDA 12.1
```

---

### 10  Success checklist

- [ ] Val‑loss ≤ 1.55.  
- [ ] Second seed reproduces within 1 %.  
- [ ] `baseline-v0` tag builds on CI.  
- [ ] README + metrics committed.

*(Confidence that this yields a reproducible baseline: 0.85.)*
