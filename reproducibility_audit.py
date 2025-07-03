#!/usr/bin/env python3
"""
Reproducibility audit script for LitGPT training.
Captures environment state, model checksums, and training metrics.
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
import lightning as L

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from litgpt_core.config import Config
from litgpt_core.model import GPT


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        # Get current commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=Path(__file__).parent,
            text=True
        ).strip()
        
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent,
            text=True
        ).strip()
        
        # Check if working directory is clean
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent,
            text=True
        ).strip()
        
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_clean": len(status) == 0,
            "status": status if status else "clean"
        }
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information."""
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 entries
        },
        "environment_variables": {
            "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED"),
            "GLOBAL_SEED": os.getenv("GLOBAL_SEED"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "lightning": {
            "version": L.__version__,
        }
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i),
            })
        env_info["pytorch"]["gpus"] = gpu_info
    
    return env_info


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = [
        "torch", "lightning", "hydra-core", "sentencepiece", 
        "numpy", "omegaconf", "wandb"
    ]
    
    versions = {}
    for package in packages:
        try:
            result = subprocess.check_output(
                [sys.executable, "-c", f"import {package.replace('-', '_')}; print({package.replace('-', '_')}.__version__)"],
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            versions[package] = result
        except (subprocess.CalledProcessError, ImportError):
            versions[package] = "not_installed"
    
    return versions


def compute_model_checksum(config: Config) -> str:
    """Compute checksum of model architecture."""
    # Create model
    model = GPT(config)
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Compute checksum of model structure (not weights)
    model_structure = {
        "config": {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                  for k, v in config.__dict__.items()},
        "parameter_names": list(state_dict.keys()),
        "parameter_shapes": {k: list(v.shape) for k, v in state_dict.items()},
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # Convert to JSON and compute hash
    model_json = json.dumps(model_structure, sort_keys=True)
    return hashlib.sha256(model_json.encode()).hexdigest()


def compute_data_checksums() -> Dict[str, str]:
    """Compute checksums of data files."""
    checksums = {}
    
    data_files = [
        "data/raw/input.txt",
        "data/raw/sp16k.model",
        "data/train.txt",
        "data/val.txt",
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                content = f.read()
                checksums[file_path] = hashlib.sha256(content).hexdigest()
        else:
            checksums[file_path] = "file_not_found"
    
    return checksums


def audit_reproducibility(output_file: str = "reproducibility_audit.json") -> Dict[str, Any]:
    """Perform comprehensive reproducibility audit."""
    print("🔍 Performing reproducibility audit...")
    
    audit_data = {
        "audit_info": {
            "timestamp": datetime.now().isoformat(),
            "script_version": "1.0.0",
            "purpose": "Phase 1B baseline training reproducibility audit"
        },
        "git_info": get_git_info(),
        "environment": get_environment_info(),
        "packages": get_package_versions(),
    }
    
    # Model architecture checksum
    try:
        config = Config(
            vocab_size=16000,
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=256,
            intermediate_size=1536,
            padded_vocab_size=16000,
        )
        audit_data["model_checksum"] = compute_model_checksum(config)
        print(f"✅ Model architecture checksum: {audit_data['model_checksum'][:16]}...")
    except Exception as e:
        audit_data["model_checksum"] = f"error: {e}"
        print(f"❌ Model checksum failed: {e}")
    
    # Data checksums
    try:
        audit_data["data_checksums"] = compute_data_checksums()
        print("✅ Data checksums computed")
    except Exception as e:
        audit_data["data_checksums"] = f"error: {e}"
        print(f"❌ Data checksums failed: {e}")
    
    # Save audit data
    with open(output_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    
    print(f"📁 Audit saved to: {output_file}")
    
    # Print summary
    print("\n📋 Reproducibility Summary:")
    print(f"   Git commit: {audit_data['git_info'].get('commit_hash', 'unknown')[:12]}...")
    print(f"   Git clean: {audit_data['git_info'].get('is_clean', 'unknown')}")
    print(f"   Python: {audit_data['environment']['platform']['python_version']}")
    print(f"   PyTorch: {audit_data['environment']['pytorch']['version']}")
    print(f"   CUDA: {audit_data['environment']['pytorch']['cuda_available']}")
    print(f"   Seeds: PYTHONHASHSEED={audit_data['environment']['environment_variables']['PYTHONHASHSEED']}")
    
    return audit_data


def compare_audits(audit1_file: str, audit2_file: str) -> Dict[str, Any]:
    """Compare two reproducibility audits."""
    print(f"🔍 Comparing audits: {audit1_file} vs {audit2_file}")
    
    with open(audit1_file, "r") as f:
        audit1 = json.load(f)
    
    with open(audit2_file, "r") as f:
        audit2 = json.load(f)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "audit1_file": audit1_file,
        "audit2_file": audit2_file,
        "differences": {},
        "identical_fields": [],
    }
    
    # Compare key fields
    key_fields = [
        ("git_info", "commit_hash"),
        ("model_checksum",),
        ("data_checksums",),
        ("environment", "pytorch", "version"),
        ("environment", "environment_variables", "PYTHONHASHSEED"),
    ]
    
    for field_path in key_fields:
        # Navigate nested dictionaries
        val1 = audit1
        val2 = audit2
        
        try:
            for key in field_path:
                val1 = val1[key]
                val2 = val2[key]
            
            field_name = ".".join(field_path)
            if val1 == val2:
                comparison["identical_fields"].append(field_name)
            else:
                comparison["differences"][field_name] = {
                    "audit1": val1,
                    "audit2": val2
                }
        except KeyError as e:
            comparison["differences"][".".join(field_path)] = f"KeyError: {e}"
    
    # Print comparison results
    print(f"\n📊 Comparison Results:")
    print(f"   Identical fields: {len(comparison['identical_fields'])}")
    print(f"   Different fields: {len(comparison['differences'])}")
    
    if comparison["differences"]:
        print("\n❌ Differences found:")
        for field, diff in comparison["differences"].items():
            print(f"   {field}: {diff}")
    else:
        print("\n✅ All key fields identical - perfect reproducibility!")
    
    return comparison


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        if len(sys.argv) < 4:
            print("Usage: python reproducibility_audit.py compare <audit1.json> <audit2.json>")
            sys.exit(1)
        
        comparison = compare_audits(sys.argv[2], sys.argv[3])
        
        # Save comparison
        comparison_file = f"audit_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"📁 Comparison saved to: {comparison_file}")
    else:
        # Perform audit
        audit_data = audit_reproducibility()
        
        # Check for potential issues
        issues = []
        
        if not audit_data["git_info"].get("is_clean", True):
            issues.append("Git working directory is not clean")
        
        if audit_data["environment"]["environment_variables"]["PYTHONHASHSEED"] != "1337":
            issues.append("PYTHONHASHSEED is not set to 1337")
        
        if issues:
            print(f"\n⚠️  Potential reproducibility issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ No reproducibility issues detected")

