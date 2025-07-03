# Clarified Instructions for Manus

All outstanding questions answered—follow these directives exactly.

| Topic | Decision | Action for Manus |
|-------|----------|------------------|
| **Repository fork** | Fork `Lightning-AI/lit-gpt` into my GitHub account **`AlexHagemeister`** and work exclusively in that fork. | Use the GitHub API (`POST /repos/Lightning-AI/lit-gpt/forks`). Set the fork as `origin`; push every commit and tag there. |
| **RunPod automation** | Implement as **`runpod_launcher.py`** (Python) that issues GraphQL calls exactly as in the playbook. | Include the script in the repo so the automation is version‑controlled. |
| **Git repo for deliverables** | The fork **is** the canonical repo. No local‑only repo needed. | Commit code, plans, metrics, and tags directly to the fork. |
| **Cost monitoring** | **Real‑time polling** every 5 min; terminate pod if projected spend ≥ `$MAX_GPU_COST` (default $20). | Implement as a background thread or async task. |
| **Checkpoint strategy** | Automatic pruning: keep only the **newest three** `.ckpt` files after each validation cycle. | Simple filesystem cleanup callback. |
| **Phase approvals** | Pause at **major milestones only**:<br>1. after `phase1A-complete` tag;<br>2. after baseline metrics, before tagging `baseline-v0`. | On pause, print status & next action; await my explicit **“Approved”**. |

---

## Secrets‑file workflow

1. Manus creates **`secrets.env.template`** in repo root:

   ```ini
   # ----------------- fill and rename to secrets.env -----------------
   GITHUB_USER=AlexHagemeister
   GITHUB_PAT=<PASTE_TOKEN_HERE>
   RUNPOD_API_KEY=<PASTE_TOKEN_HERE>
   MAX_GPU_COST=20
   # Optional: if using SSH push
   GIT_SSH_KEY=/work/id_ed25519
   -------------------------------------------------------------------
   ```

2. Manus **commits and pauses**.  
3. I will duplicate to **`secrets.env`**, fill real tokens, then tell Manus to continue.

---

### Kick‑off

Begin execution now, following the steps above.
