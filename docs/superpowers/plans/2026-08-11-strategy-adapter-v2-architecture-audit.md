# Strategy Adapter V2 Architecture Audit Plan

> **For agentic workers:** Execute this audit inline. Do not delegate model or data interpretation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct and empirically verify the current 0040 Policy/Value/Meta architecture before any Strategy Adapter V2 implementation.

**Architecture:** Keep production code unchanged. Add isolated diagnostics that load the strict paired-0809 checkpoints, compile public replay observations through the existing 0040 feature path, inspect parameter/optimizer identity, and run independent backward probes. Reuse the 0036 validation contract for offline Value/Meta baselines and publish the evidence in one audit report.

**Tech Stack:** Python 3.11, PyTorch, existing 0036/0040 packages and materialized datasets, pytest/unittest-compatible diagnostics.

## Global Constraints

- Do not modify `engine/source/` or any production model/training semantics.
- Do not implement Strategy Adapter V2 in this work.
- Treat 0040 `PRIZE` as the active formal configuration and distinguish it from dormant Meta presets.
- Use only public, deployable observation tensors for leakage conclusions.
- Preserve all existing checkpoints and run artifacts.

---

### Task 1: Static architecture and identity map

**Files:**
- Read: `train/0040_dragapult_0809_action_boundary_rl/policy/actor_critic.py`
- Read: `train/0040_dragapult_0809_action_boundary_rl/semantic_policy/model/*.py`
- Read: `train/0040_dragapult_0809_action_boundary_rl/training/ppo_full_semantic.py`
- Read: `train/0040_dragapult_0809_action_boundary_rl/training/storage_full_semantic.py`

- [x] Trace the real encoder, option, decoder, Value and optional Meta calls.
- [x] Trace strict pretrained loading and model-only RL overlay behavior.
- [x] Map every optimizer group to concrete `Parameter` objects.

### Task 2: Runtime architecture and gradient diagnostic

**Files:**
- Create: `train/0040_dragapult_0809_action_boundary_rl/diagnostics/strategy_adapter_v2_audit.py`
- Create: `train/0040_dragapult_0809_action_boundary_rl/tests/test_strategy_adapter_v2_audit.py`

- [x] Compile a real public observation using the existing worker compiler and collator.
- [x] Instantiate the strict 0040 `PRIZE` model and PPO optimizer on CPU.
- [x] Export tensor shapes, module parameter counts, optimizer ownership, duplicate/shared parameter IDs, and frozen parameters held by the optimizer.
- [x] Run policy-only, Value-only, and isolated pretrained-Meta-only backward passes from fresh graphs and record nonzero gradients by audited module.
- [x] Verify the two-step zero-gated residual startup invariant in an isolated toy module.
- [x] Run the focused regression test.

### Task 3: Value and pretrained-Meta validation baseline

**Files:**
- Reuse: `train/0036_dedicated_action_value_network/training/*`
- Reuse: `rl_runs/0036_dedicated_action_value_network/` materialized validation artifacts
- Output: `.tmp/strategy_adapter_v2_audit/`

- [x] Identify the exact V9 validation dataset and loaded epoch-9 checkpoint provenance.
- [x] Evaluate Value BCE/Brier/ECE/AUROC, explained variance, correlation, and probability calibration.
- [x] Evaluate the checkpoint's 15-class archetype head with overall/per-class accuracy, confusion matrix, and turn/progress entropy buckets.
- [x] Audit the compiler/schema for opponent deck identity, hidden hand/deck/prize, exact-deck, team/source, and training-only labels.

### Task 4: Report and future implementation plan

**Files:**
- Create: `docs/reports/STRATEGY_ADAPTER_V2_PRE_IMPLEMENTATION_ARCHITECTURE_AUDIT.md`

- [x] Publish architecture graph and tensor/module mapping with source references.
- [x] Publish parameter/optimizer and measured gradient matrices.
- [x] Publish Value/Meta baselines with limitations and leakage findings.
- [x] Rate every proposed adapter integration point `SAFE`, `NEEDS CARE`, or `BLOCKER`.
- [x] Specify, but do not execute, the exact future file/class/test changes.
- [x] Review the report against every requested audit section and inspect the final git diff.
