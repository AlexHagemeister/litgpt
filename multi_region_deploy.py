#!/usr/bin/env python3
"""
Multi-region, multi-GPU A100/L40S deployment for baseline training.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime


# Deployment configuration
REGIONS = ["US-ORD1", "US-IAD1", "EU-FRA1"]
GPUS = [
    "NVIDIA A100",                    # 40GB
    "NVIDIA A100 PCIe 80GB",         # 80GB  
    "NVIDIA A100-SXM4-80GB",         # 80GB alternative
    "NVIDIA L40S"                    # 48GB fallback
]

MAX_WALL_TIME = 3600  # 60 minutes
RETRY_INTERVAL = 600   # 10 minutes


def load_credentials():
    """Load credentials from secrets file."""
    credentials = {}
    
    try:
        with open("../secrets.env", "r") as f:
            content = f.read()
            for line in content.strip().split('\n'):
                line = line.strip()
                if line.startswith("GITHUB_USER="):
                    credentials["github_user"] = line.split("=", 1)[1].strip()
                elif line.startswith("GITHUB_PAT="):
                    credentials["github_pat"] = line.split("=", 1)[1].strip()
                elif line.startswith("RUNPOD_API_KEY="):
                    credentials["runpod_key"] = line.split("=", 1)[1].strip()
                elif line.startswith("MAX_GPU_COST="):
                    credentials["max_cost"] = float(line.split("=", 1)[1].strip())
    except Exception as e:
        print(f"❌ Failed to load credentials: {e}")
        return None
    
    required = ["github_user", "github_pat", "runpod_key"]
    missing = [k for k in required if k not in credentials]
    if missing:
        print(f"❌ Missing credentials: {missing}")
        return None
    
    if "max_cost" not in credentials:
        credentials["max_cost"] = 20.0
    
    return credentials

def create_startup_script(github_user: str, github_pat: str) -> str:
    """Create the startup script for baseline training."""
    
    script = """bash -lc '
set -e
echo "🚀 LitGPT Baseline Training - Multi-Region Deployment"
echo "====================================================="

# Environment setup
export PYTHONHASHSEED=1337
export GLOBAL_SEED=1337
export GITHUB_USER=""" + github_user + """
export GITHUB_PAT=""" + github_pat + """

echo "🔒 Seeds: PYTHONHASHSEED=$PYTHONHASHSEED, GLOBAL_SEED=$GLOBAL_SEED"
echo "🎯 Target: val_loss ≤ 1.55"
echo "🔧 Precision: fp32, Flash-Attention OFF"

# Verify GPU
echo "🔍 Verifying GPU access..."
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f\\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\"None\\"}\\")"

# Clone repository
echo "📥 Cloning repository..."
cd /workspace
git clone https://$GITHUB_PAT@github.com/$GITHUB_USER/litgpt.git repo
cd repo

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e . --quiet
pip install hydra-core wandb sentencepiece --quiet

# Run baseline training
echo "🔥 Starting baseline training..."
python -c "
import os,sys,time,json,csv,torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
sys.path.insert(0,\\"/workspace/repo/src\\")
from litgpt_core.lightning_module import LitGPT
from litgpt_core.data_module import ShakespeareDataModule
from litgpt_core.config import Config

# GUARD-RAIL 5: Reproducibility
os.environ[\\"PYTHONHASHSEED\\"] = \\"1337\\"
os.environ[\\"GLOBAL_SEED\\"] = \\"1337\\"
torch.manual_seed(1337)
L.seed_everything(1337)

print(\\"🚀 LitGPT Baseline Training - Multi-Region Deployment\\")
print(\\"🎯 Target: val_loss ≤ 1.55\\")
print(\\"🔧 Precision: fp32, Flash-Attention OFF\\")

# Verify GPU
if not torch.cuda.is_available():
    raise RuntimeError(\\"CUDA not available - training must run on GPU\\")

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f\\"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)\\")

# Model configuration (Phase 1B spec)
config = Config(
    vocab_size=16000,
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,
    intermediate_size=1536,
    padded_vocab_size=16000,
)

model = LitGPT(
    model_config=config,
    learning_rate=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    warmup_steps=200,
    max_steps=3000,
    min_lr_ratio=0.1,
)

print(f\\"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}\\")

# Data module
data_module = ShakespeareDataModule(
    data_dir=\\"data\\",
    tokenizer_path=\\"data/raw/sp16k.model\\",
    seq_len=256,
    batch_size=32,
    num_workers=4,
)
data_module.setup(\\"fit\\")

# Create directories
os.makedirs(\\"metrics\\", exist_ok=True)
os.makedirs(\\"checkpoints\\", exist_ok=True)

# GUARD-RAIL 4: Keep latest 3 checkpoints only
checkpoint_callback = ModelCheckpoint(
    dirpath=\\"checkpoints\\",
    filename=\\"baseline-{step:06d}\\",
    every_n_train_steps=500,
    save_top_k=3,
    monitor=\\"val_loss\\",
    mode=\\"min\\",
    save_last=True,
)

# GUARD-RAIL 6: CSV logging
csv_logger = CSVLogger(
    save_dir=\\"metrics\\",
    name=\\"baseline\\",
    version=\\"\\",
)

# GUARD-RAIL 1: fp32 precision, Flash-Attention OFF
trainer = L.Trainer(
    max_steps=3000,
    precision=\\"32-true\\",
    accelerator=\\"gpu\\",
    devices=1,
    gradient_clip_val=1.0,
    log_every_n_steps=50,
    val_check_interval=500,
    callbacks=[checkpoint_callback],
    logger=csv_logger,
    enable_progress_bar=True,
)

# Start training
start_time = time.time()
print(f\\"🔥 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\\")

try:
    trainer.fit(model, datamodule=data_module)
    
    # Get final metrics
    if trainer.logged_metrics:
        final_metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
        val_loss = final_metrics.get(\\"val_loss\\", float('inf'))
        
        # GUARD-RAIL 2: Check target achievement
        success = val_loss <= 1.55
        
        # Calculate performance
        end_time = time.time()
        duration = end_time - start_time
        total_tokens = 3000 * 32 * 256
        tokens_per_second = total_tokens / duration
        
        # Save results
        results = {
            \\"timestamp\\": time.strftime('%Y-%m-%d %H:%M:%S'),
            \\"duration_seconds\\": duration,
            \\"final_val_loss\\": val_loss,
            \\"target_achieved\\": success,
            \\"tokens_per_second\\": tokens_per_second,
            \\"total_tokens\\": total_tokens,
            \\"model_parameters\\": sum(p.numel() for p in model.parameters()),
            \\"seeds\\": {
                \\"PYTHONHASHSEED\\": os.environ[\\"PYTHONHASHSEED\\"],
                \\"GLOBAL_SEED\\": os.environ[\\"GLOBAL_SEED\\"]
            },
            \\"final_metrics\\": final_metrics,
            \\"precision\\": \\"fp32\\",
            \\"flash_attention\\": False,
            \\"gpu_name\\": gpu_name,
            \\"gpu_memory_gb\\": gpu_memory,
            \\"deployment_method\\": \\"multi_region_startup_script\\"
        }
        
        with open(\\"baseline_results.json\\", \\"w\\") as f:
            json.dump(results, f, indent=2)
        
        # GUARD-RAIL 6: Create baseline.csv
        with open(\\"metrics/baseline.csv\\", \\"w\\", newline=\\"\\") as f:
            writer = csv.writer(f)
            writer.writerow([\\"metric\\", \\"value\\"])
            for key, value in results.items():
                if key != \\"final_metrics\\":
                    writer.writerow([key, value])
        
        print(f\\"\\\\n🎉 TRAINING COMPLETE:\\")
        print(f\\"   Duration: {duration:.1f}s ({duration/60:.1f} min)\\")
        print(f\\"   Tokens/s: {tokens_per_second:,.0f}\\")
        print(f\\"   Final val_loss: {val_loss:.6f}\\")
        print(f\\"   Target achieved: {success}\\")
        print(f\\"   GPU: {gpu_name} ({gpu_memory:.1f}GB)\\")
        
        if success:
            print(f\\"✅ SUCCESS: val_loss {val_loss:.6f} ≤ 1.55 target!\\")
        else:
            print(f\\"❌ FAILED: val_loss {val_loss:.6f} > 1.55 target\\")
            exit(1)
    else:
        print(\\"❌ No metrics logged\\")
        exit(1)
        
except Exception as e:
    print(f\\"❌ Training failed: {e}\\")
    import traceback
    traceback.print_exc()
    exit(1)
"

# Git configuration and commit
echo "📋 Configuring git..."
git config user.email "manus@litgpt.ai"
git config user.name "Manus"

# Add and commit results
echo "📤 Committing results..."
git add metrics/baseline.csv baseline_results.json
git commit -m "Add Phase 1B baseline training results - Multi-Region Deployment"

# Create and push tag
echo "🏷️  Creating baseline-v0 tag..."
git tag baseline-v0 -m "Phase 1B baseline training complete - Multi-Region Deployment"

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push origin HEAD --tags

echo "🎉 PHASE 1B COMPLETE - baseline-v0 tagged and pushed!"

# Shutdown pod to prevent billing
echo "🧹 Shutting down pod..."
shutdown -h now
'"""
    
    return script

def deploy_pod(credentials: dict, region: str, gpu_type: str) -> dict:
    """Deploy pod using working mutation with region and GPU specification."""
    print(f"🚀 Deploying {gpu_type} in {region}...")
    
    github_user = credentials["github_user"]
    github_pat = credentials["github_pat"]
    runpod_key = credentials["runpod_key"]
    
    # Create startup script
    startup_script = create_startup_script(github_user, github_pat)
    
    # GraphQL mutation using working format
    mutation = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        machineId
      }
    }
    """
    
    variables = {
        "input": {
            "gpuTypeId": gpu_type,
            "cloudType": "SECURE",
            "region": region,
            "imageName": "runpod/pytorch:2.3.0-py3.11-cuda12.1",
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "name": f"baseline-train-{region.lower()}",
            "dockerArgs": startup_script
        }
    }
    
    headers = {
        "Authorization": f"Bearer {runpod_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers=headers,
            json={"query": mutation, "variables": variables},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "errors" in data:
                error_msg = data["errors"][0]["message"]
                return {"success": False, "error": error_msg}
            
            pod_data = data["data"]["podFindAndDeployOnDemand"]
            if pod_data:
                pod_id = pod_data["id"]
                image_name = pod_data.get("imageName")
                machine_id = pod_data.get("machineId")
                
                print(f"✅ Pod deployed successfully!")
                print(f"   Pod ID: {pod_id}")
                print(f"   Region: {region}")
                print(f"   GPU: {gpu_type}")
                print(f"   Image: {image_name}")
                print(f"   Machine ID: {machine_id}")
                
                return {
                    "success": True,
                    "pod_id": pod_id,
                    "region": region,
                    "gpu_type": gpu_type,
                    "image_name": image_name,
                    "machine_id": machine_id
                }
            else:
                return {"success": False, "error": "No pod data returned"}
        else:
            error_text = response.text
            return {"success": False, "error": error_text}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def monitor_pod_cost(pod_id: str, runpod_key: str, max_cost: float) -> dict:
    """Monitor pod cost and status."""
    headers = {
        "Authorization": f"Bearer {runpod_key}",
        "Content-Type": "application/json"
    }
    
    query = """
    query GetPod($podId: String!) {
      pod(podId: $podId) {
        status
        runtimeSeconds
        costPerSecond
      }
    }
    """
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers=headers,
            json={"query": query, "variables": {"podId": pod_id}},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "errors" in data:
                return {"success": False, "error": data["errors"][0]["message"]}
            
            pod_data = data["data"]["pod"]
            if pod_data:
                status = pod_data["status"]
                runtime_seconds = pod_data.get("runtimeSeconds", 0)
                cost_per_second = pod_data.get("costPerSecond", 0)
                
                total_cost = runtime_seconds * cost_per_second
                projected_cost = total_cost  # Conservative estimate
                
                return {
                    "success": True,
                    "status": status,
                    "runtime_seconds": runtime_seconds,
                    "cost_per_second": cost_per_second,
                    "total_cost": total_cost,
                    "projected_cost": projected_cost,
                    "over_budget": projected_cost >= max_cost
                }
            else:
                return {"success": False, "error": "No pod data returned"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def terminate_pod(pod_id: str, runpod_key: str) -> bool:
    """Terminate the pod."""
    headers = {
        "Authorization": f"Bearer {runpod_key}",
        "Content-Type": "application/json"
    }
    
    mutation = """
    mutation TerminatePod($podId: String!) {
      podTerminate(podId: $podId) {
        id
      }
    }
    """
    
    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers=headers,
            json={"query": mutation, "variables": {"podId": pod_id}},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "errors" not in data:
                print(f"✅ Pod {pod_id} terminated successfully")
                return True
        
        print(f"❌ Failed to terminate pod: {response.text}")
        return False
        
    except Exception as e:
        print(f"❌ Termination error: {e}")
        return False


def check_github_for_baseline() -> bool:
    """Check GitHub for baseline-v0 tag."""
    try:
        response = requests.get(
            "https://api.github.com/repos/AlexHagemeister/litgpt/tags",
            timeout=10
        )
        
        if response.status_code == 200:
            tags = response.json()
            baseline_tag = next((tag for tag in tags if tag["name"] == "baseline-v0"), None)
            return baseline_tag is not None
        
        return False
        
    except Exception:
        return False


def main():
    """Main multi-region deployment loop."""
    print("🌍 MULTI-REGION A100/L40S DEPLOYMENT")
    print("=" * 50)
    print(f"🎯 Regions: {', '.join(REGIONS)}")
    print(f"🔧 GPUs: {', '.join(GPUS)}")
    print(f"⏰ Max wall time: {MAX_WALL_TIME//60} minutes")
    print(f"🔄 Retry interval: {RETRY_INTERVAL//60} minutes")
    
    # Load credentials
    credentials = load_credentials()
    if not credentials:
        return False
    
    runpod_key = credentials["runpod_key"]
    max_cost = credentials["max_cost"]
    
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < MAX_WALL_TIME:
        attempt += 1
        elapsed_minutes = (time.time() - start_time) / 60
        
        print(f"\n🔄 Deployment cycle {attempt} (elapsed: {elapsed_minutes:.1f}m)")
        
        # Try all region/GPU combinations
        deployed = False
        for region in REGIONS:
            for gpu_type in GPUS:
                print(f"\n📍 Trying {gpu_type} in {region}...")
                
                result = deploy_pod(credentials, region, gpu_type)
                
                if result["success"]:
                    pod_id = result["pod_id"]
                    deployed_region = result["region"]
                    deployed_gpu = result["gpu_type"]
                    
                    print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
                    print(f"   Pod ID: {pod_id}")
                    print(f"   Region: {deployed_region}")
                    print(f"   GPU: {deployed_gpu}")
                    print(f"🚀 Startup script executing...")
                    print(f"⏱️  Expected completion: ~7-8 minutes")
                    print(f"💰 Expected cost: ~$0.20-0.30")
                    
                    deployed = True
                    break
                else:
                    error = result["error"]
                    if "no longer any instances available" in error.lower():
                        print(f"   ⏳ No {gpu_type} available in {region}")
                    else:
                        print(f"   ❌ Error: {error}")
            
            if deployed:
                break
        
        if deployed:
            # Monitor deployment
            print(f"\n💰 Starting monitoring...")
            print(f"   Cost monitoring: every 5 minutes")
            print(f"   GitHub monitoring: every 30 seconds")
            print(f"   Budget cap: ${max_cost}")
            
            monitor_start = time.time()
            last_cost_check = 0
            github_success = False
            
            while time.time() - monitor_start < 1800:  # 30 minute timeout
                current_time = time.time()
                
                # Cost monitoring every 5 minutes
                if current_time - last_cost_check >= 300:  # 5 minutes
                    cost_result = monitor_pod_cost(pod_id, runpod_key, max_cost)
                    
                    if cost_result["success"]:
                        status = cost_result["status"]
                        runtime = cost_result["runtime_seconds"]
                        total_cost = cost_result["total_cost"]
                        
                        print(f"💰 Pod status: {status}, Runtime: {runtime}s, Cost: ${total_cost:.4f}")
                        
                        # Check budget
                        if cost_result["over_budget"]:
                            print(f"🚨 BUDGET EXCEEDED: ${total_cost:.4f} ≥ ${max_cost}")
                            print(f"🛑 Terminating pod...")
                            terminate_pod(pod_id, runpod_key)
                            return False
                        
                        # Check if pod stopped (training complete)
                        if status in ["STOPPED", "TERMINATED"]:
                            print(f"✅ Pod stopped - training likely complete")
                            break
                    else:
                        print(f"⚠️  Cost check failed: {cost_result['error']}")
                    
                    last_cost_check = current_time
                
                # Check GitHub for baseline-v0 tag
                if check_github_for_baseline():
                    print(f"\n🎉 SUCCESS! baseline-v0 tag found!")
                    print(f"🔗 https://github.com/AlexHagemeister/litgpt/releases/tag/baseline-v0")
                    github_success = True
                    break
                
                time.sleep(30)  # Check every 30 seconds
            
            # Final status
            if github_success:
                print(f"\n🎉 PHASE 1B BASELINE TRAINING COMPLETE!")
                print(f"✅ baseline-v0 tag successfully created")
                print(f"📊 Metrics: https://github.com/AlexHagemeister/litgpt/blob/main/metrics/baseline.csv")
                print(f"🌍 Deployed in: {deployed_region}")
                print(f"🔧 GPU: {deployed_gpu}")
                
                # Ensure pod is terminated
                final_cost = monitor_pod_cost(pod_id, runpod_key, max_cost)
                if final_cost["success"]:
                    print(f"💰 Final cost: ${final_cost['total_cost']:.4f}")
                    if final_cost["status"] not in ["STOPPED", "TERMINATED"]:
                        print(f"🧹 Ensuring pod is terminated...")
                        terminate_pod(pod_id, runpod_key)
                
                return True
            else:
                print(f"\n⚠️  Training may have failed or taken longer than expected")
                print(f"🧹 Terminating pod to prevent further costs...")
                terminate_pod(pod_id, runpod_key)
                return False
        
        # No deployment successful, wait and retry
        remaining_time = MAX_WALL_TIME - (time.time() - start_time)
        if remaining_time > RETRY_INTERVAL:
            print(f"\n⏳ No GPU available in any region. Waiting {RETRY_INTERVAL//60} minutes...")
            print(f"   Remaining wall time: {remaining_time//60:.1f} minutes")
            time.sleep(RETRY_INTERVAL)
        else:
            print(f"\n⏰ Wall time limit reached ({MAX_WALL_TIME//60} minutes)")
            break
    
    print(f"\n❌ Deployment unsuccessful after {MAX_WALL_TIME//60} minutes")
    print(f"💡 Consider:")
    print(f"   - Trying again later when GPU availability improves")
    print(f"   - Using alternative GPU providers (Lambda Labs, Paperspace)")
    print(f"   - Manual deployment via RunPod web interface")
    return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 PHASE 1B COMPLETE - baseline-v0 tagged!")
        print(f"🔗 Repository: https://github.com/AlexHagemeister/litgpt")
        print(f"📋 Ready for review and next phase planning")
    else:
        print(f"\n⚠️  Multi-region deployment incomplete")
        print(f"📋 Infrastructure ready - waiting for GPU availability")
    
    sys.exit(0 if success else 1)

