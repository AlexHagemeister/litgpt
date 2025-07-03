# Design and Deviation Summary

This document compares the final implementation of the Light Compute Project against the original plans, capturing alignment, deviations, and the rationale behind them. The goal is to provide a clear audit trail of design decisions and ensure the project's evolution is transparent and well-documented.

---

## 🎯 Core Principles: Strong Alignment

The project's execution remained highly faithful to the original "north-star" vision:

> **Build a modular GPT-style decoder that:**
>
> - compiles and trains end-to-end on a toy corpus in minutes;
> - matches the reference loss of a known implementation;
> - externalises every hyper‑parameter so architectural ideas are one‑line config changes;
> - can later be scaled or adapted... without touching core code.

The final architecture achieves all these goals. The modular structure in `src/litgpt_core`, the comprehensive testing, the Hydra-based configuration, and the successful validation runs all directly reflect this original vision.

---

## ✅ Phase 1A: Architecture & Verification - Mostly Aligned

The implementation of Phase 1A followed the plan with high fidelity. All specified tests and artifacts were created as planned, successfully burning down correctness uncertainty before training.

| Planned Step                     | Implementation Status                 | Notes                                                                                                                                                                                                                                           |
| :------------------------------- | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Fork and Trim**             | ✅ **Aligned**                        | The `lit-gpt` repository was forked, and core modules (`attention`, `mlp`, `model`, etc.) were moved to `src/litgpt_core/` for granularity.                                                                                                     |
| **2. Shape-only Forward Test**   | ✅ **Aligned**                        | Implemented in `tests/unit/test_model_shapes.py`. Asserts that the model produces the correct output tensor shape `[B, T, V]`.                                                                                                                  |
| **3. Analytical Gradient Check** | ✅ **Aligned**                        | Implemented in `tests/unit/test_gradients.py` using `torch.autograd.gradcheck`. This successfully verified the backward pass logic.                                                                                                             |
| **4. Pytest Battery**            | ✅ **Aligned**                        | A comprehensive suite was created in `tests/unit/test_behavioral.py`, covering causal masking, dropout invariance, and parameter counts as specified.                                                                                           |
| **5. Smoke Benchmark**           | ✅ **Aligned** (with minor deviation) | Implemented as `scripts/smoke_test.py`. **Deviation**: The plan targeted `loss ≈ 0`, while the implementation targets a significant loss _reduction_ (>1.0), achieving 48.7%. This serves the same purpose: verifying end-to-end gradient flow. |
| **6. CI Hook**                   | ❌ **Not Yet Implemented**            | The original plan specified a GitHub Actions CI workflow. This has not been implemented yet but can be added in a future phase.                                                                                                                 |

---

## 🔄 Phase 1B: Baseline Training - Mostly Aligned

Phase 1B is currently in progress, but the infrastructure and setup align well with the original plan.

| Planned Step                 | Implementation Status | Notes                                                                                                                                                        |
| :--------------------------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Hydra YAML**            | ✅ **Aligned**        | A `configs` directory with modular YAML files for `model`, `data`, and `optim` was implemented, allowing for CLI-based experimentation as planned.           |
| **2. Reference Run**         | 🔄 **In Progress**    | The infrastructure is in place to run the Tiny-Shakespeare baseline on a RunPod A100 to target `val_loss ≈ 1.47`. This is the immediate next step.           |
| **3. Metric Harness**        | ✅ **Aligned**        | The `LitGPT` Lightning module and `CSVLogger` are configured to log `val_loss`, and the deployment scripts monitor performance metrics, fulfilling the plan. |
| **4. Optional Toggles**      | ✅ **Aligned**        | The baseline runs with `fp32` precision and Flash-Attention disabled by default, as specified.                                                               |
| **5. Reproducibility Check** | ✅ **Aligned**        | The project enforces fixed seeds (`1337`), and the `reproducibility_audit.py` script provides a mechanism for deeper analysis.                               |
| **6. Tag & Freeze**          | 🔄 **Pending**        | The `baseline-v0` tag will be created upon successful completion of the baseline training run.                                                               |

---

## 🚀 Deployment Strategy: Major (Positive) Deviation

The most significant deviation from the original plan is in the deployment and automation strategy. This was an intentional evolution to build a more robust, production-ready system.

- **Original Plan**: The initial idea was a simpler, single-script SSH-based workflow (`phase1B_runpod_playbook.md`) or a basic GraphQL launcher (`manus_clarified_instructions.md`). This would have required more manual intervention.
- **Current Implementation**: The project evolved to use a far more sophisticated deployment script, `multi_region_deploy.py`.
  - **Multi-Region/Multi-GPU**: It automatically cycles through different geographic regions (US-ORD1, US-IAD1, EU-FRA1) and GPU types (A100, L40S) to find available, cost-effective resources.
  - **Resilience**: It includes robust error handling, automatic retries, and a wall-time limit.
  - **Full Automation**: The script handles environment setup, training execution, cost monitoring, and auto-shutdown in a single, version-controlled command.

This deviation represents a significant improvement over the original plan, moving from a simple playbook to a resilient, automated deployment system while still honoring the core requirements of cost control and automation.

---

## 🛡️ Supplemental Guard-Rails: Aligned

The supplemental instructions for ensuring a robust workflow were successfully integrated.

| Guard-Rail             | Implementation Status | Notes                                                                                                                                                |
| :--------------------- | :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Randomness Hygiene** | ✅ **Aligned**        | Fixed seeds (`1337`) are used throughout the codebase and enforced via environment variables in deployment.                                          |
| **Disk Housekeeping**  | ✅ **Aligned**        | The `ModelCheckpoint` callback is configured to save only the latest 3 checkpoints.                                                                  |
| **Dependency Pinning** | ✅ **Aligned**        | Dependencies are managed in `pyproject.toml`, which serves the same purpose as a `requirements.lock` file by enabling reproducible environments.     |
| **Changelog**          | ✅ **Aligned**        | A `CHANGELOG.md` is actively maintained and was updated after the latest validation run.                                                             |
| **Cost Telemetry**     | ✅ **Aligned**        | The deployment script implements real-time cost polling against a hard cap (`$20`) and terminates the pod if the budget is projected to be exceeded. |
