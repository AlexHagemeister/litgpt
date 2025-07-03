## Supplemental instructions for Manus  
*Loose ends and operational guard‑rails to lock down before execution.*

---

### 1  Randomness hygiene
```bash
export PYTHONHASHSEED=1337
export GLOBAL_SEED=1337
```
Set NumPy, PyTorch, Hydra seeds at runtime; log the seed in `metrics/baseline.csv`.

---

### 2  Dataset licence checksum
```
sha256sum input.txt  > data/README.md
sha256sum sp16k.model >> data/README.md
```
Manus halts if checksum drift detected.

---

### 3  Disk housekeeping
* Warn if workspace > 5 GB (`du -sh .`).
* Keep only the latest **3** checkpoints; delete older `.pt` files.

---

### 4  Dependency pin
```bash
pip freeze > requirements.lock
```
CI installs from `requirements.lock`; regenerate only on explicit instruction.

---

### 5  Documentation stub
Append to `CHANGELOG.md` after each successful phase:

```
YYYY‑MM‑DD  git:<hash>
Phase 1A complete  – tests green
Phase 1B baseline – val_loss 1.473, tokens/s 74.8k
```

---

### 6  Failure snapshot
On fatal error:

```
tar czf failures/$(date +%s).tgz     stderr.log configs/* training_last_100.log
```

---

### 7  Cost telemetry
Record to `metrics/cost_log.csv` every 300 s:

```
timestamp, runtime_seconds, cost_per_second, cumulative_usd
```

---

### 8  Next‑phase readiness flag
On human approval to proceed:

```bash
touch PHASE2_READY
git add PHASE2_READY && git commit -m "Ready for Phase 2"
git push
```

---

### Kick‑off reminder

* **Secrets already provided** (`RUNPOD_API_KEY`, etc.).  
* HARD_CAP_USD = 20 – terminate and pause if breached.  
* Pause after Phase 1A tag; pause after baseline metrics; require explicit approval to create `baseline-v0`.

Once all bullet‑points above are in place, Manus runs Phase 1 deterministically and hands back artefacts with minimal human touch.  
