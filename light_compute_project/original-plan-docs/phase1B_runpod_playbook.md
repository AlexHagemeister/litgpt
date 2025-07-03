## Phase 1B – RunPod Operational Playbook  
*All-in-one instructions Manus can execute to complete the baseline training run, stay under budget, and hand back metrics.*

---

### 0  Prerequisite environment variables

```bash
export RUNPOD_API_KEY=<your_key>         # already obtained
export GIT_REPO=https://github.com/your-fork/lit-gpt.git
export GIT_BRANCH=baseline-setup         # or main
# Optional SSH clone:
export GIT_SSH_KEY=/work/id_ed25519      # path inside container
export POD_GPU="A100"                    # default 40 GB model
export POD_TEMPLATE="runpod/pytorch"     # CUDA 12.1 + PyTorch 2.3
export MAX_GPU_COST=20                   # USD cost ceiling
```

If you skip SSH keys, `GIT_REPO` can be an HTTPS URL with an embedded token.

---

### 1  Launch an A100 pod via GraphQL

```python
import os, json, requests, time
API   = "https://api.runpod.ai/graphql"
HEAD  = {"Authorization": os.environ["RUNPOD_API_KEY"]}

launch_q = '''
mutation Launch($in: LaunchInput!){
  launchPod(input:$in){ id }
}
'''
vars = {"in": {
  "gpuTypeId": os.environ["POD_GPU"],
  "templateId": os.environ["POD_TEMPLATE"],
  "volumeInGb": 0,
  "containerDiskInGb": 20,
  "env": {}
}}

pod_id = requests.post(API, headers=HEAD,
           json={"query": launch_q, "variables": vars}).json()["data"]["launchPod"]["id"]

# Wait until RUNNING
while True:
    q = f'{{pod(podId:"{pod_id}"){{status publicIp}}}}'
    pod = requests.post(API, headers=HEAD, json={"query": q}).json()["data"]["pod"]
    if pod["status"] == "RUNNING":
        host = pod["publicIp"]; break
    time.sleep(10)
print("Pod ready:", host)
```

---

### 2  Remote bootstrap & training (run over SSH)

```bash
ssh -o StrictHostKeyChecking=no ubuntu@$host <<'REMOTE'
  set -e
  git clone $GIT_REPO --branch $GIT_BRANCH code
  cd code
  conda env create -f environment.yml
  source activate litgpt
  # Baseline run (fp32, Flash‑Attention off)
  python train.py +model=tiny +data=tinyshake +optim=adamw       trainer.max_steps=3000 seed=1337
  # Bundle metrics
  tar czf /tmp/baseline_metrics.tgz metrics
REMOTE
```

Retrieve metrics locally:

```bash
scp ubuntu@$host:/tmp/baseline_metrics.tgz ./baseline_metrics.tgz
```

---

### 3  Automated cost guard

```python
q = '{"query":"{pod(podId:\"%s\"){runtimeSeconds costPerSecond}}"}' % pod_id
usage = requests.post(API, headers=HEAD, data=q).json()["data"]["pod"]
if usage["runtimeSeconds"] * usage["costPerSecond"] > float(os.environ["MAX_GPU_COST"]):
    requests.post(API, headers=HEAD,
        json={"query": 'mutation{terminatePod(podId:"%s")}' % pod_id})
```

Run this poll every few minutes.

---

### 4  Terminate pod when done

```bash
requests.post(API, headers=HEAD,
    json={"query": 'mutation{terminatePod(podId:"%s")}' % pod_id})
```

---

### 5  Commit & tag results locally

```bash
tar xzf baseline_metrics.tgz
git add metrics/baseline.csv
git commit -m "Baseline tiny Shakespeare achieved (val_loss ≤1.55)"
git tag baseline-v0 && git push --tags
```

---

### Success criteria checklist

- [ ] `val_loss` ≤ 1.55 in `metrics/baseline.csv`.  
- [ ] `baseline-v0` tag pushed; CI green.  
- [ ] Pod terminated; total cost ≤ `$MAX_GPU_COST`.

Once these boxes are ticked, Manus can report *“Phase 1 complete”* and hand back `baseline_metrics.tgz`.
