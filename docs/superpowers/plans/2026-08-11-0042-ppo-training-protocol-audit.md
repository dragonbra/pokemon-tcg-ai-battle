# 0042 PPO Training Protocol Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct and runtime-verify the current effective 0042 PPO protocol, with special emphasis on rollout decision coverage and reuse, without changing training code or hyperparameters.

**Architecture:** Trace the formal CLI through rollout, target construction, sampling, loss, optimizer, checkpoint, and evaluation. Cross-check static control flow against current configuration and existing run artifacts, then use read-only/minimal diagnostic execution to measure sample-use histograms where logs are insufficient.

**Tech Stack:** Python 3.11, PyTorch, CUDA resident official-engine rollout, JSON/JSONL run artifacts.

## Global Constraints

- Do not modify formal training code or hyperparameters.
- Do not start or resume PPO training while the audit is active.
- Treat runtime values as authoritative when defaults, docs, and runtime differ.
- Preserve official engine sources and formal run/evaluation assets.
- Report every major conclusion with file/function/line evidence.

---

### Task 1: Trace Effective Runtime Path

**Files:**
- Read: `train/0042_full_model_design/training/run_full_semantic.py`
- Read: `train/0042_full_model_design/rollout/cuda_collector.py`
- Read: `train/0042_full_model_design/training/batch_full_semantic.py`
- Read: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Read: `train/0042_full_model_design/training/storage_full_semantic.py`

**Interfaces:**
- Consumes: formal CLI arguments and `RunConfig`/`PPOConfig`.
- Produces: evidence map from entry point to checkpoint/evaluation.

- [ ] Read the CLI construction, overrides, validation, and formal loop with numbered lines.
- [ ] Trace CUDA rollout capture and every rollout-to-batch filtering boundary.
- [ ] Trace frozen target tensors, sampler indices, loss computation, optimizer steps, and KL stop.
- [ ] Trace model-only checkpoint and Frozen evaluation scheduling.

### Task 2: Verify Runtime Model and Effective Configuration

**Files:**
- Read: `train/0042_full_model_design/integrated/presets.py`
- Read: `train/0042_full_model_design/policy/actor_critic.py`
- Read: current 0042 and predecessor run artifacts under `rl_runs/`.

**Interfaces:**
- Consumes: instantiated `FULL_MODEL` model/trainer and formal CLI defaults.
- Produces: complete effective hyperparameter and trainability inventory.

- [ ] Instantiate the exact formal 0042 model and trainer without starting rollout.
- [ ] Record total/trainable counts, optimizer group membership, learning rates, and precision.
- [ ] Compare dataclass defaults, CLI defaults, validation constraints, and persisted runtime configs.

### Task 3: Measure Decision Utilization

**Files:**
- Read: existing `training_metrics.jsonl`, rollout summaries, and smoke artifacts.
- Diagnostic output: `.tmp/0042_ppo_protocol_audit/`

**Interfaces:**
- Consumes: real prepared batches or existing per-update decision counts and sampler algorithm.
- Produces: valid decisions, actual sample slots, unique use count, coverage, both reuse ratios, and 0/1/2/3/4+ histogram.

- [ ] Locate usable real update artifacts and extract multiple updates with differing episode lengths.
- [ ] If usage counts are absent, replay the exact sampler index generation in an isolated read-only diagnostic process using recorded decision counts and fixed seeds.
- [ ] Quantify how fixed 32-step budget changes coverage/reuse as decisions per rollout vary.
- [ ] Distinguish measured historical values from deterministic sampler calculations and projections.

### Task 4: Audit Correctness and Monitoring

**Files:**
- Read: PPO/batch/logger implementation and current metrics artifacts.

**Interfaces:**
- Consumes: evidence from Tasks 1-3.
- Produces: target-freezing proof, filtering inventory, logging coverage, P0/P1/P2 findings.

- [ ] Verify rollout old logprob, advantage, return, value target, bootstrap, and normalization lifetimes.
- [ ] Enumerate all invalid/truncated/chance-boundary/perspective/capacity filtering with observed counts where available.
- [ ] Classify each requested health metric as logged, computed-not-output, or absent.
- [ ] Produce the Current Effective PPO Protocol and answer the ten core questions.

### Task 5: Deliver Audit

**Files:**
- Create: `.tmp/0042_ppo_protocol_audit/report.json`

**Interfaces:**
- Consumes: verified tables and findings.
- Produces: structured machine-readable evidence plus a Chinese user report.

- [ ] Cross-check all numerical claims against runtime output or cited artifacts.
- [ ] Run a placeholder scan and ensure no proposed parameter change is presented as implemented.
- [ ] Deliver findings ordered P0, P1, P2, followed by the complete protocol and ten direct answers.
