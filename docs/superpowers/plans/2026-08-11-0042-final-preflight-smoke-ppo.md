# 0042 Final Preflight and Smoke PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the implemented 0042 architecture contracts against real code and gradients, then run an isolated ten-update PPO smoke with fixed validation evidence.

**Architecture:** Keep production model and PPO semantics unchanged except for audit plumbing that makes the existing Meta coefficient and diagnostics explicit. Run the smoke through the real policy resolver, CUDA-resident rollout collector, batch preparation, and PPO trainer, but write only non-candidate artifacts below `.tmp/0042_smoke_ppo/`.

**Tech Stack:** Python 3.11, PyTorch, official policy identity resolver, CUDA-resident official-rule rollout runtime, unittest, JSON diagnostics.

## Global Constraints

- Do not run Frozen evaluation, promotion, formal model selection, or a long training job.
- Stop before PPO if any architecture preflight contract fails.
- Use exactly 15 own-archetype classes and distinct own/opponent semantic types.
- Preserve PPO, GAE, reward, opponent schedule, and action semantics.
- Smoke artifacts are `NON-CANDIDATE` and `SMOKE ONLY`; do not write to `rl_runs/` or W&B.
- Do not modify `engine/source/`.

---

### Task 1: Make preflight semantics explicit

**Files:**
- Modify: `train/0042_full_model_design/policy/own_archetype.py`
- Modify: `train/0042_full_model_design/integrated/config.py`
- Modify: `train/0042_full_model_design/integrated/presets.py`
- Modify: `train/0042_full_model_design/training/run_full_semantic.py`
- Modify: `train/0042_full_model_design/tests/test_strategy_architecture.py`

**Interfaces:**
- Produces: distinct `OwnArchetypeId` and `OpponentArchetypeTarget` values.
- Produces: preset-owned `meta_anchor_coef` checked against `PPOConfig`.

- [ ] Add regression tests for the two semantic types, 15-entry embeddings, independent parameters, and focal-deck-derived runtime index.
- [ ] Add the opponent target wrapper and unwrap it only while constructing training labels.
- [ ] Put `meta_anchor_coef=0.10` in both explicit 0042 presets and validate PPO/preset equality at run entry.
- [ ] Rename zero-gate test variables and documentation from ambiguous legacy language to 0042 Base.
- [ ] Run focused architecture/config tests.

### Task 2: Complete PPO diagnostics without changing optimization

**Files:**
- Modify: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Test: `train/0042_full_model_design/tests/test_strategy_architecture.py`

**Interfaces:**
- Produces: raw/effective gates, residual mean/p50/p95/max, adapter MLP and embedding gradient norms, PPO ratio statistics, and effective Meta/critic trunk gradient comparison.

- [ ] Add pure gradient-norm and distribution-stat helpers with unit tests.
- [ ] Record diagnostic values immediately after the existing backward and before clipping/step.
- [ ] Scale isolated Meta and critic trunk norms by their actual loss coefficients and report their ratio.
- [ ] Run PPO and architecture tests.

### Task 3: Add fixed-set and isolated smoke runner

**Files:**
- Modify: `train/0042_full_model_design/diagnostics/value_meta_baseline.py`
- Create: `train/0042_full_model_design/diagnostics/smoke_ppo.py`
- Create: `train/0042_full_model_design/tests/test_smoke_ppo_diagnostics.py`

**Interfaces:**
- Consumes: real update-0 model, Policy-0806 resolver, CUDA rollout collector, `prepare_episodes`, and `PPOTrainer`.
- Produces: `.tmp/0042_smoke_ppo/<run>/report.json`, JSONL update metrics, and model-only checkpoints marked `NON-CANDIDATE_SMOKE_ONLY`.

- [ ] Make the fixed validation evaluator load an optional 0042 checkpoint and evaluate the adapted Value path.
- [ ] Add exact raw-turn-0 plus early/middle/late fixed-set metrics.
- [ ] Implement fail-closed A1-A5 preflight and exact frozen-module tensor checks.
- [ ] Implement fixed-seed ten-update CUDA PPO with one bounded optimizer step per update and no formal side effects.
- [ ] Save before/after fixed-set Meta/Value and strategy sensitivity evidence.
- [ ] Add artifact-boundary and report-schema tests.

### Task 4: Remove CUDA trajectory feature round-trip and improve progress metrics

**Files:**
- Modify: `train/0042_full_model_design/rollout/cuda_collector.py`
- Modify: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Modify: `train/0042_full_model_design/training/run_full_semantic.py`
- Test: `train/0042_full_model_design/tests/test_cuda_collector.py`
- Test: `train/0042_full_model_design/tests/test_smoke_ppo_diagnostics.py`

**Interfaces:**
- Produces: CUDA-resident trajectory feature tensors through PPO collation, with only compact scalar/action metadata materialized on CPU.
- Produces: rollout progress tagged with source update, throughput, and cumulative win rate; PPO progress tagged with update, optimizer iteration, and iterations/second.
- Produces: per-update W&B-ready `trainer/update`, `rollout/win_rate`, adapter, gradient, transfer, and timing metrics.

- [ ] Add a regression proving CUDA trajectory features remain on CUDA until PPO consumption.
- [ ] Remove full feature `.cpu()` materialization while preserving exact rollout/replay tensors and actions.
- [ ] Record feature D2H bytes, scalar D2H bytes, materialization time, and device-resident status.
- [ ] Add update-aware CUDA rollout progress with cumulative W/L/D and games/second.
- [ ] Add bounded PPO optimizer progress with iterations/second and no extra forward/backward.
- [ ] Benchmark the pre/post materialization path and retain correctness evidence.

### Task 5: Execute and document

**Files:**
- Modify: `docs/rl/0042_full_model_design.md`
- Modify: `experiments/0042_full_model_design/DESIGN.md`
- Modify: `experiments/0042_full_model_design/DESIGN.html`
- Modify: `experiments/0042_full_model_design/DECISIONS.md`

- [ ] Run compileall, focused tests, full 0042 tests, and empirical A1-A5 audit.
- [ ] Abort if preflight is not fully passing.
- [ ] Run the isolated ten-update Smoke PPO and inspect every update for finite metrics.
- [ ] Compare fixed validation and strategy sensitivity before/after.
- [ ] Verify frozen modules are tensor-identical and no formal registry/run/evaluation paths changed.
- [ ] Synchronize only architecture naming/plumbing corrections and record Smoke evidence as diagnostic, not model-selection evidence.
