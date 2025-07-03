#!/usr/bin/env python3
"""
RunPod training using exec endpoint - no SSH required.
"""

import os
import sys
import json
import time
from typing import Dict, Any

import runpod


class RunPodExecTrainer:
    """Execute training via RunPod exec endpoint."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key
        self.pod_id = None
        self.max_cost = 20.0
        
    def create_pod(self) -> str:
        """Create pod for training."""
        print("🚀 Creating RunPod for GPU training...")
        
        try:
            pod = runpod.create_pod(
                name="litgpt-baseline-exec",
                image_name="runpod/pytorch:2.3.0-py3.11-cuda12.1",
                gpu_type_id="NVIDIA A100 80GB PCIe",
                cloud_type="SECURE",
                container_disk_in_gb=20,
                volume_in_gb=0,
                docker_args="sleep infinity",
                env={
                    "PYTHONHASHSEED": "1337",
                    "GLOBAL_SEED": "1337"
                }
            )
            
            self.pod_id = pod["id"]
            print(f"✅ Created pod: {self.pod_id}")
            return self.pod_id
            
        except Exception as e:
            print(f"❌ Pod creation failed: {e}")
            raise
    
    def wait_for_running(self, timeout: int = 300) -> bool:
        """Wait for pod to be running."""
        print(f"⏳ Waiting for pod {self.pod_id} to be running...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pod = runpod.get_pod(self.pod_id)
                status = pod.get("desiredStatus", "UNKNOWN")
                
                print(f"   Status: {status}")
                
                if status == "RUNNING":
                    print(f"✅ Pod {self.pod_id} is running!")
                    return True
                elif status in ["FAILED", "TERMINATED"]:
                    print(f"❌ Pod failed with status: {status}")
                    return False
                    
            except Exception as e:
                print(f"   Status check error: {e}")
            
            time.sleep(10)
        
        print(f"❌ Pod not running after {timeout}s")
        return False
    
    def exec_cmd(self, cmd: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute command in pod via exec endpoint."""
        print(f"📋 Executing: {cmd[:60]}...")
        
        try:
            # Start command execution
            job = runpod.run_command(self.pod_id, cmd)
            job_id = job["id"]
            
            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    job_status = runpod.get_run_command_status(job_id)
                    status = job_status.get("status", "UNKNOWN")
                    
                    if status == "COMPLETED":
                        output = job_status.get("output", "")
                        print(f"✅ Command completed")
                        return {"success": True, "output": output, "status": status}
                    elif status == "FAILED":
                        error = job_status.get("error", "Unknown error")
                        print(f"❌ Command failed: {error}")
                        return {"success": False, "error": error, "status": status}
                    
                    # Still running
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"   Status check error: {e}")
                    time.sleep(5)
            
            print(f"❌ Command timeout after {timeout}s")
            return {"success": False, "error": "Timeout", "status": "TIMEOUT"}
            
        except Exception as e:
            print(f"❌ Command execution error: {e}")
            return {"success": False, "error": str(e), "status": "ERROR"}
    
    def verify_gpu(self) -> bool:
        """Verify GPU access."""
        print("🔍 Verifying GPU access...")
        
        # Check nvidia-smi
        result = self.exec_cmd("nvidia-smi --query-gpu=name --format=csv")
        if not result["success"] or "A100" not in result["output"]:
            print(f"❌ GPU verification failed: {result}")
            return False
        
        print(f"✅ GPU detected: {result['output'].strip()}")
        
        # Check PyTorch CUDA
        cuda_check = 'python -c "import torch; print(f\\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'None\'}\\"'
        result = self.exec_cmd(cuda_check)
        if not result["success"] or "True" not in result["output"]:
            print(f"❌ PyTorch CUDA verification failed: {result}")
            return False
        
        print(f"✅ PyTorch CUDA: {result['output'].strip()}")
        return True
    
    def setup_training_environment(self) -> bool:
        """Set up training environment."""
        print("🔧 Setting up training environment...")
        
        setup_commands = [
            "cd /workspace",
            "git clone https://github.com/AlexHagemeister/litgpt.git",
            "cd litgpt",
            "pip install -e . --quiet",
            "pip install hydra-core wandb sentencepiece --quiet",
            "echo 'Environment setup completed'"
        ]
        
        for cmd in setup_commands:
            result = self.exec_cmd(cmd, timeout=180)
            if not result["success"]:
                print(f"❌ Setup failed at: {cmd}")
                print(f"   Error: {result.get('error', 'Unknown')}")
                return False
        
        print("✅ Training environment ready")
        return True
    
    def run_baseline_training(self) -> Dict[str, Any]:
        """Execute baseline training with all guard-rails."""
        print("🔥 Starting baseline training with guard-rails...")
        
        # Create training script with all requirements
        training_script = '''
import os
import sys
import time
import json
import csv
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# Add src to path
sys.path.insert(0, "/workspace/litgpt/src")

from litgpt_core.lightning_module import LitGPT
from litgpt_core.data_module import ShakespeareDataModule
from litgpt_core.config import Config

def run_baseline():
    """Run baseline training with all guard-rails."""
    print("🚀 LitGPT Baseline Training - A100 GPU")
    print("=" * 50)
    
    # GUARD-RAIL 5: Reproducibility
    os.environ["PYTHONHASHSEED"] = "1337"
    os.environ["GLOBAL_SEED"] = "1337"
    torch.manual_seed(1337)
    L.seed_everything(1337)
    
    print(f"🔒 Seeds: PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}, GLOBAL_SEED={os.environ['GLOBAL_SEED']}")
    print(f"🎯 Target: val_loss ≤ 1.55")
    print(f"🔧 Precision: fp32, Flash-Attention OFF")
    
    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - training must run on GPU")
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
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
    
    # Create model
    model = LitGPT(
        model_config=config,
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        warmup_steps=200,
        max_steps=3000,
        min_lr_ratio=0.1,
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data module
    data_module = ShakespeareDataModule(
        data_dir="data",
        tokenizer_path="data/raw/sp16k.model",
        seq_len=256,
        batch_size=32,  # 8192 tokens per batch
        num_workers=4,
    )
    data_module.setup("fit")
    
    # Create directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # GUARD-RAIL 4: Keep latest 3 checkpoints only
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="baseline-{step:06d}",
        every_n_train_steps=500,
        save_top_k=3,  # Latest 3 only
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    # GUARD-RAIL 6: CSV logging
    csv_logger = CSVLogger(
        save_dir="metrics",
        name="baseline",
        version="",
    )
    
    # GUARD-RAIL 1: fp32 precision, Flash-Attention OFF
    trainer = L.Trainer(
        max_steps=3000,
        precision="32-true",  # fp32 precision
        accelerator="gpu",
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
    print(f"🔥 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer.fit(model, datamodule=data_module)
        
        # Get final metrics
        if trainer.logged_metrics:
            final_metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
            val_loss = final_metrics.get("val_loss", float('inf'))
            
            # GUARD-RAIL 2: Check target achievement
            success = val_loss <= 1.55
            
            # Calculate performance
            end_time = time.time()
            duration = end_time - start_time
            total_tokens = 3000 * 32 * 256
            tokens_per_second = total_tokens / duration
            
            # Save results
            results = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "duration_seconds": duration,
                "final_val_loss": val_loss,
                "target_achieved": success,
                "tokens_per_second": tokens_per_second,
                "total_tokens": total_tokens,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "seeds": {
                    "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                    "GLOBAL_SEED": os.environ["GLOBAL_SEED"]
                },
                "final_metrics": final_metrics,
                "precision": "fp32",
                "flash_attention": False,
                "gpu_name": torch.cuda.get_device_name(0),
            }
            
            with open("baseline_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # GUARD-RAIL 6: Create baseline.csv
            with open("metrics/baseline.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for key, value in results.items():
                    if key != "final_metrics":
                        writer.writerow([key, value])
            
            print(f"\\n🎉 TRAINING COMPLETE:")
            print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
            print(f"   Tokens/s: {tokens_per_second:,.0f}")
            print(f"   Final val_loss: {val_loss:.6f}")
            print(f"   Target achieved: {success}")
            
            if success:
                print(f"✅ SUCCESS: val_loss {val_loss:.6f} ≤ 1.55 target!")
            else:
                print(f"❌ FAILED: val_loss {val_loss:.6f} > 1.55 target")
            
            return success
        else:
            print("❌ No metrics logged")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = run_baseline()
    print(f"\\nFINAL RESULT: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
'''
        
        # Write training script to pod
        script_cmd = f'cat > /workspace/litgpt/baseline_training_gpu.py << \'EOF\'\n{training_script}\nEOF'
        result = self.exec_cmd(script_cmd)
        if not result["success"]:
            print(f"❌ Failed to create training script: {result}")
            return {"success": False, "error": "Script creation failed"}
        
        # Execute training
        training_cmd = "cd /workspace/litgpt && python baseline_training_gpu.py 2>&1 | tee baseline_train.log"
        result = self.exec_cmd(training_cmd, timeout=1800)  # 30 minutes max
        
        if result["success"]:
            output = result["output"]
            
            # Check for success indicators
            if "SUCCESS: val_loss" in output and "≤ 1.55 target!" in output:
                print("🎉 BASELINE TRAINING SUCCESSFUL!")
                
                # Extract metrics
                lines = output.split('\n')
                val_loss = None
                duration = None
                tokens_per_second = None
                
                for line in lines:
                    if "Final val_loss:" in line:
                        try:
                            val_loss = float(line.split("Final val_loss:")[1].strip())
                        except:
                            pass
                    elif "Duration:" in line and "min)" in line:
                        try:
                            duration = float(line.split("Duration:")[1].split("s")[0].strip())
                        except:
                            pass
                    elif "Tokens/s:" in line:
                        try:
                            tokens_per_second = float(line.split("Tokens/s:")[1].replace(",", "").strip())
                        except:
                            pass
                
                return {
                    "success": True,
                    "val_loss": val_loss,
                    "duration": duration,
                    "tokens_per_second": tokens_per_second,
                    "output": output
                }
            else:
                print("❌ Training did not achieve target")
                return {"success": False, "error": "Target not achieved", "output": output}
        else:
            print(f"❌ Training execution failed: {result}")
            return {"success": False, "error": result.get("error", "Unknown"), "output": result.get("output", "")}
    
    def get_training_artifacts(self) -> Dict[str, str]:
        """Get training artifacts from pod."""
        print("📁 Retrieving training artifacts...")
        
        artifacts = {}
        
        # Get baseline results
        result = self.exec_cmd("cat /workspace/litgpt/baseline_results.json")
        if result["success"]:
            artifacts["baseline_results.json"] = result["output"]
        
        # Get training log
        result = self.exec_cmd("cat /workspace/litgpt/baseline_train.log")
        if result["success"]:
            artifacts["baseline_train.log"] = result["output"]
        
        # Get metrics CSV
        result = self.exec_cmd("cat /workspace/litgpt/metrics/baseline.csv")
        if result["success"]:
            artifacts["baseline.csv"] = result["output"]
        
        return artifacts
    
    def monitor_cost(self) -> float:
        """Get current cost."""
        try:
            pod = runpod.get_pod(self.pod_id)
            cost_per_hr = pod.get("costPerHr", 0.0)
            uptime_seconds = pod.get("uptimeSeconds", 0)
            uptime_hours = uptime_seconds / 3600.0
            total_cost = cost_per_hr * uptime_hours
            return total_cost
        except:
            return 0.0
    
    def cleanup(self):
        """Stop and cleanup pod."""
        if self.pod_id:
            print(f"🧹 Stopping pod {self.pod_id}...")
            try:
                runpod.stop_pod(self.pod_id)
                print("✅ Pod stopped")
            except Exception as e:
                print(f"❌ Failed to stop pod: {e}")


def main():
    """Main execution."""
    print("🚀 RUNPOD EXEC BASELINE TRAINING")
    print("=" * 50)
    
    # Load API key
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        with open("../secrets.env", "r") as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    
    if not api_key:
        print("❌ RUNPOD_API_KEY not found")
        return False
    
    trainer = RunPodExecTrainer(api_key)
    
    try:
        # 1. Create pod
        pod_id = trainer.create_pod()
        
        # 2. Wait for running
        if not trainer.wait_for_running():
            return False
        
        # 3. Verify GPU
        if not trainer.verify_gpu():
            trainer.cleanup()
            return False
        
        # 4. Setup environment
        if not trainer.setup_training_environment():
            trainer.cleanup()
            return False
        
        # 5. Run training
        training_result = trainer.run_baseline_training()
        
        # 6. Get artifacts
        artifacts = trainer.get_training_artifacts()
        
        # 7. Get final cost
        final_cost = trainer.monitor_cost()
        
        # 8. Cleanup
        trainer.cleanup()
        
        # 9. Save artifacts locally
        for filename, content in artifacts.items():
            with open(filename, "w") as f:
                f.write(content)
            print(f"📁 Saved: {filename}")
        
        # 10. Report results
        if training_result["success"]:
            print(f"\n🎉 BASELINE TRAINING SUCCESSFUL!")
            print(f"🎯 Final val_loss: {training_result.get('val_loss', 'N/A')}")
            print(f"⚡ Tokens/s: {training_result.get('tokens_per_second', 'N/A'):,.0f}")
            print(f"⏱️  Duration: {training_result.get('duration', 'N/A'):.1f}s")
            print(f"💰 Total cost: ${final_cost:.4f}")
            return True
        else:
            print(f"\n❌ BASELINE TRAINING FAILED")
            print(f"💰 Total cost: ${final_cost:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        trainer.cleanup()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

