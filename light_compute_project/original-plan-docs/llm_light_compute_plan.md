### End-to-end plan for a *light-compute, highly-tinkerable* transformer LLM project
---

#### 0 · Project north-star  
Build a **modular GPT-style decoder** that:

* compiles and trains end-to-end on a toy corpus in minutes;
* matches the reference loss of a known implementation (`≈ 1.47` on Tiny‑Shakespeare);
* externalises every hyper‑parameter so architectural ideas are one‑line config changes;
* can later be scaled or adapted (Flash‑Attention, LoRA, multi‑GPU) without touching core code.

---

#### 1 · Tech stack snapshot

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Skeleton** | **Lit‑GPT** fork | Single‑file modules, no deep abstractions, yet ships recipes for Flash‑Attention, FSDP, LoRA, QLoRA. |
| **Training loop** | **PyTorch Lightning** `Trainer` | Mixed precision, gradient accumulation and multi‑GPU via flags; stays out of forward pass. |
| **Config** | **Hydra** | YAML‑driven overrides so experiments never require editing Python. |
| **Efficiency levers** | Flash‑Attention 2 · LoRA/QLoRA |
| **Cloud** | **RunPod Secure Cloud** (A100 40 GB @ ≈ $1.19 h) for dev and baseline runs. |

---

#### 2 · Phase 1 → split into *1A* (verification) and *1B* (baseline)

| Sub‑stage | Objective | Exit criteria |
|-----------|-----------|---------------|
| **1A — Architecture & Verification** | Prove the math and plumbing are sound without doing real training. | *pytest* suite passes; `gradcheck` OK; one Tiny‑Shakespeare batch over‑fits to loss ≈ 0 in < 5 min. |
| **1B — Baseline Training** | Show the model learns at parity with the reference. | Validation loss within ±5 % of 1.47; metrics + env tagged `baseline-v0`. |

---

##### 2.1  Stage 1A — architecture & verification

1. **Fork and trim**  
   Clone `Lightning‑AI/litgpt`; move each module (`Embedding`, `Attention`, `MLP`) into its own file for unit‑test granularity.  
2. **Shape‑only forward test**  
   Assert logits come out `[B, T, V]`. Catches most mis‑indexing.  
3. **Analytical gradient check**  
   `torch.autograd.gradcheck` on a 3‑token toy graph with `double` precision. Failure ⇒ backward wrong.  
4. **Pytest battery**  
   * causal mask blocks future tokens;  
   * dropout invariance under `model.eval()`;  
   * parameter count matches spec.  
5. **Smoke benchmark**  
   Over‑fit **one** 256‑token batch until loss ≈ 0. Activation flow confirmed.  
6. **CI hook**  
   GitHub Actions runs the whole pytest + smoke stack on every commit.  

**Deliverable 1A:** green CI, grad‑check pass, smoke loss ≈ 0.

---

##### 2.2  Stage 1B — baseline training

1. **Hydra YAML**  
   Surface `n_layer`, `d_model`, optimiser, dataset path, precision, flash toggle.  
2. **Reference run**  
   Tiny‑Shakespeare baseline on RunPod A100; expected val‑loss ≈ 1.47 in ≈ 3 min.  
3. **Metric harness**  
   Log val loss, tokens/s, peak GPU RAM to wandb or CSV.  
4. **Optional toggles (off by default)**  
   Flash‑Attention 2 and bf16 compile but are disabled so baseline is clean.  
5. **Repro check**  
   Two different seeds → final loss variance < 1 %.  
6. **Tag & freeze**  
   Commit `requirements.txt` and CUDA/PyTorch versions; tag `baseline-v0`.

**Deliverable 1B:** single command reproduces baseline loss ± 5 %, config‑only tweaks, tag reproducible on CI.

---

#### 3 · Why this Phase 1 matters
* All correctness uncertainty is burned down **before** you invent new blocks.  
* Baseline numbers form a control arm; any Phase 2 tweak that wins does so *fairly*.  
* Flash‑Attention and mixed precision are “escape valves” — not hidden variables — because they’re absent from the control run.

---

#### 4 · RunPod workflow snapshot

1. Launch Secure‑Cloud A100 40 GB; paste SSH key.  
2. `git clone`, create env, run smoke + baseline.  
3. Checkpoints go to an attached volume (cheap; survives instance stop).  
4. Shut the box — pay only for active minutes.  

---

#### 5 · Most common omissions & fixes

| Risk | Mitigation |
|------|------------|
| **Dataset licence / safety filters** | Use public‑domain toy corpus; document licence in README. |
| **“Works on GPU X but not Y”** | Record torch, CUDA, driver versions in tag. |
| **Session dies → lost progress** | Checkpoint every N minutes; store weights on volume. |

---

### Ready‑to‑execute checklist

- [ ] Fork **Lit‑GPT** and trim modules.  
- [ ] Write shape test, grad‑check, pytest suite; wire CI.  
- [ ] Over‑fit single batch → loss ≈ 0. *(Stage 1A done)*  
- [ ] Add Hydra configs; run Tiny‑Shakespeare baseline; log metrics.  
- [ ] Tag repo `baseline-v0`, store env + results. *(Stage 1B done)*  

*(Confidence that this plan gets you to a reproducible baseline on < $20 of cloud time: ~85 %.)*
