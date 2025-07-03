# LitGPT Deployment Guide

This guide documents the deployment process for LitGPT baseline training on GPU infrastructure.

## 🚀 Quick Start

### Prerequisites
1. **Credentials**: RunPod API key and GitHub PAT in `secrets.env`
   ```bash
   # Copy template and configure
   cp secrets.env.template secrets.env
   # Edit secrets.env with your actual credentials
   ```
2. **Environment**: Python 3.11 with dependencies installed
3. **Repository**: Clean working directory with committed changes

### One-Command Deployment
```bash
python multi_region_deploy.py
```

This script handles:
- Multi-region GPU availability checking
- Automated pod creation and configuration
- Complete training pipeline execution
- Results collection and GitHub integration
- Cost monitoring and auto-shutdown

## 📋 Deployment Process

### 1. Environment Setup
```bash
# Activate environment
conda activate litgpt

# Verify credentials
cat secrets.env
# Should contain:
# GITHUB_USER=AlexHagemeister
# GITHUB_PAT=ghp_...
# RUNPOD_API_KEY=rpa_...
# MAX_GPU_COST=20
```

### 2. Pre-Deployment Checklist
- [ ] All code committed and pushed to GitHub
- [ ] Unit tests passing (23/23)
- [ ] Secrets file configured correctly
- [ ] Budget limits set appropriately
- [ ] Target metrics defined (val_loss ≤ 1.55)

### 3. Deployment Execution
```bash
# Start deployment
python multi_region_deploy.py

# Monitor progress
# - Console output for deployment status
# - GitHub for baseline-v0 tag creation
# - RunPod dashboard for cost tracking
```

### 4. Success Indicators
- ✅ Pod deployment successful
- ✅ GPU verification passed
- ✅ Training completed in ~7-8 minutes
- ✅ val_loss ≤ 1.55 achieved
- ✅ baseline-v0 tag created on GitHub
- ✅ Pod auto-shutdown completed

## 🌍 Multi-Region Strategy

### Supported Regions
1. **US-ORD1** (Chicago) - Primary
2. **US-IAD1** (Virginia) - Secondary  
3. **EU-FRA1** (Frankfurt) - Tertiary

### GPU Types (Priority Order)
1. **NVIDIA A100** (40GB) - Primary target
2. **NVIDIA A100 PCIe 80GB** - High-memory variant
3. **NVIDIA A100-SXM4-80GB** - Alternative 80GB
4. **NVIDIA L40S** (48GB) - Fallback option

### Deployment Logic
```python
for region in REGIONS:
    for gpu_type in GPUS:
        try_deploy(region, gpu_type)
        if success:
            break
    if success:
        break
```

## 🛡️ Guard-Rails and Safety

### Cost Protection
- **Hard Cap**: $20 maximum spend
- **Monitoring**: Every 5 minutes during training
- **Auto-Termination**: If projected cost ≥ budget
- **Expected Cost**: ~$0.20-0.30 for baseline run

### Training Requirements
- **Precision**: fp32 (Flash-Attention disabled)
- **Target**: val_loss ≤ 1.55
- **Reproducibility**: Fixed seeds (PYTHONHASHSEED=1337)
- **Checkpoints**: Keep latest 3 only
- **Logging**: Complete CSV metrics

### Failure Handling
- **Retry Logic**: 10-minute intervals, 60-minute wall time
- **Graceful Degradation**: A100 → L40S fallback
- **Error Recovery**: Automatic pod cleanup on failure
- **Status Reporting**: Real-time progress updates

## 📊 Monitoring and Validation

### Real-Time Monitoring
```bash
# Cost tracking (every 5 minutes)
# - Pod status and runtime
# - Current and projected costs
# - Budget remaining

# GitHub monitoring (every 30 seconds)  
# - Check for baseline-v0 tag
# - Verify metrics commit
# - Confirm CI status
```

### Success Validation
1. **Training Metrics**: val_loss ≤ 1.55 achieved
2. **Performance**: ~75k tokens/second throughput
3. **Duration**: 7-8 minutes wall-clock time
4. **Cost**: Under $0.30 total spend
5. **Artifacts**: metrics/baseline.csv committed
6. **Tagging**: baseline-v0 tag created

## 🔧 Troubleshooting

### Common Issues

#### GPU Unavailability
```
Error: "No instances available"
Solution: Wait for retry cycle (10 minutes)
Alternative: Try different time of day
```

#### Authentication Errors
```
Error: "HTTP 401 Unauthorized"
Solution: Verify RUNPOD_API_KEY in secrets.env
Check: GitHub PAT permissions for repository access
```

#### Training Failures
```
Error: "CUDA not available"
Solution: Verify GPU allocation in pod
Check: Container image includes CUDA support
```

#### Budget Exceeded
```
Error: "Projected spend ≥ $20"
Solution: Increase MAX_GPU_COST if appropriate
Check: Training efficiency and duration
```

### Debug Commands
```bash
# Test credentials
python -c "
import os
with open('secrets.env') as f:
    for line in f:
        if 'API_KEY' in line:
            print(f'{line.split(\"=\")[0]}={line.split(\"=\")[1][:10]}...')
"

# Verify environment
python -c "
import torch, lightning
print(f'PyTorch: {torch.__version__}')
print(f'Lightning: {lightning.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test training locally
python test_training.py
```

## 📁 File Structure

### Deployment Files
```
litgpt/
├── multi_region_deploy.py     # Main deployment script
├── secrets.env               # Credentials (git-ignored)
├── train.py                  # Training entry point
└── test_training.py          # Local verification
```

### Generated During Deployment
```
/workspace/repo/              # Pod working directory
├── metrics/baseline.csv      # Training metrics
├── baseline_results.json     # Detailed results
├── checkpoints/              # Model checkpoints (latest 3)
└── logs/                     # Training logs
```

## 🔄 Deployment Workflow

### Phase 1: Pod Creation
1. Load credentials from secrets.env
2. Generate startup script with embedded training code
3. Try GPU deployment across regions/types
4. Verify pod creation and status

### Phase 2: Training Execution
1. Pod starts with startup script
2. Environment setup (conda, dependencies)
3. Repository clone and data preparation
4. Model training with guard-rails
5. Metrics collection and validation

### Phase 3: Results Collection
1. Training completion verification
2. Metrics commit to repository
3. baseline-v0 tag creation
4. GitHub push with results
5. Pod auto-shutdown

### Phase 4: Validation
1. GitHub tag verification
2. Cost calculation and reporting
3. Metrics validation (val_loss ≤ 1.55)
4. Cleanup and status reporting

## 🎯 Expected Timeline

### Typical Deployment
- **Pod Creation**: 1-2 minutes
- **Environment Setup**: 2-3 minutes  
- **Training Execution**: 7-8 minutes
- **Results Upload**: 1 minute
- **Total Duration**: ~12-15 minutes

### With Retries
- **GPU Availability Wait**: 0-60 minutes
- **Retry Cycles**: 10-minute intervals
- **Maximum Wall Time**: 60 minutes
- **Success Rate**: >90% within 30 minutes

## 📞 Support and Maintenance

### Regular Maintenance
- **Credential Rotation**: Update API keys quarterly
- **Dependency Updates**: Monitor for security patches
- **Cost Optimization**: Review GPU pricing and availability
- **Performance Tuning**: Optimize training hyperparameters

### Monitoring Setup
- **GitHub Actions**: CI/CD pipeline status
- **RunPod Dashboard**: Cost and usage tracking
- **Alert Thresholds**: Budget and performance limits
- **Log Retention**: Training history and metrics

---

**Last Updated**: January 7, 2025  
**Version**: 1.0 (Phase 1B)  
**Maintainer**: Manus AI Agent

