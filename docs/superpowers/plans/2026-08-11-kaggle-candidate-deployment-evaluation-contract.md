# Kaggle Candidate Deployment Evaluation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Kaggle-facing Frozen evaluation measure the FP16-storage, FP32-runtime candidate that will actually be submitted, while PPO rollout and optimization remain FP32.

**Architecture:** Extend the canonical Promote Champion V1 protocol with a candidate deployment identity that is separate from the immutable opponent identity. A single 0040 materializer exports the exact Kaggle payload, strict-loads it as FP32, audits both source and deployment hashes, and marks the evaluation model. Frozen evaluation collectors reject candidates without that passing audit; PPO sampling continues to consume the live FP32 training model.

**Tech Stack:** Python 3.11, PyTorch, JSON manifests, `unittest`, repository Markdown contracts.

## Global Constraints

- Frozen opponent identity remains independently resolved and immutable.
- Kaggle-facing candidate evaluation uses FP16 stored weights loaded into FP32 parameters.
- PPO rollout, loss computation, optimizer state, and trainable master weights remain FP32.
- Conversion order is full base plus decoder/LoRA/heads, then FP16 storage rounding, then FP32 runtime materialization.
- Missing or mismatched candidate deployment identity is fatal; warning-and-continue is forbidden.
- Existing historical FP32 evaluation artifacts remain unchanged and must not be relabeled.

---

### Task 1: Canonical protocol and agent rule

**Files:**
- Modify: `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: existing Promote Champion V1 opponent identity and seed contracts.
- Produces: one canonical `kaggle_fp16_storage_fp32_runtime_v1` candidate evaluation contract.

- [x] **Step 1: Add the candidate deployment identity definition**

Document source checkpoint identity, portable artifact hash, FP16 tensor inventory, FP32 runtime tensor inventory, conversion order, and hard-fail behavior.

- [x] **Step 2: Bind the contract to required evaluation stages**

State that every five-update RL Frozen evaluation and formal Frozen CUDA-2048/CPU-256 Kaggle-strength result must use the deployment identity; raw FP32 evaluations are diagnostics only.

- [x] **Step 3: Preserve PPO FP32 semantics**

State explicitly that PPO sampling, optimization, and checkpoints remain FP32 and are outside this deployment conversion.

- [x] **Step 4: Add the root-agent short invariant**

Link the canonical section and require future agents to enforce the candidate contract before running or accepting Kaggle-facing evaluations.

### Task 2: Candidate deployment identity gate

**Files:**
- Create: `train/0040_dragapult_0809_action_boundary_rl/candidate_deployment.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_candidate_deployment.py`

**Interfaces:**
- Consumes: `export_candidate(source: Path, checkpoint: Path, output: Path)` and `PortableCompoundSemanticPolicy.from_checkpoint(path, deck)`.
- Produces: `materialize_kaggle_evaluation_candidate(...) -> tuple[nn.Module, CandidateDeploymentAudit]` and `require_kaggle_candidate_deployment(model) -> CandidateDeploymentAudit`.

- [x] **Step 1: Write failing audit tests**

Cover rejection of FP32 stored tensors, rejection of a non-FP32 runtime model, source-checkpoint mismatch, and missing audit on a formal evaluation model.

- [x] **Step 2: Implement exact package materialization**

Export to an isolated temporary package, verify manifest and tensor dtypes, strict-load package weights, move the loaded model to FP32 on the evaluation device, compute the deployment-effective hash, and return an immutable PASS audit.

- [x] **Step 3: Implement the collector preflight**

Reject absent, failed, or wrong-contract candidate audits with a fatal candidate identity violation.

- [x] **Step 4: Run focused tests**

Run `python3 -m unittest -v train.0040_dragapult_0809_action_boundary_rl.tests.test_candidate_deployment` and require PASS.

### Task 3: Every-five-update and formal Frozen wiring

**Files:**
- Modify: `train/0040_dragapult_0809_action_boundary_rl/training/run_full_semantic.py`
- Modify: `train/0040_dragapult_0809_action_boundary_rl/tests/test_accelerated_transfer.py`

**Interfaces:**
- Consumes: `materialize_kaggle_evaluation_candidate` and `CandidateDeploymentAudit.to_manifest()`.
- Produces: formal Frozen result records containing `candidate_deployment_identity_audit`.

- [x] **Step 1: Write failing cadence/preflight tests**

Assert greedy strength evaluation rejects the live FP32 model and accepts a deployment-audited model, while sample PPO collection continues to accept the live FP32 model.

- [x] **Step 2: Materialize update 0 and every fifth update**

After the immutable checkpoint is saved, construct the deployment candidate and use it only for greedy Frozen evaluation. Keep sample rollout bound to the live FP32 training model.

- [x] **Step 3: Persist candidate deployment evidence**

Include source checkpoint SHA-256, portable checkpoint SHA-256, deployment effective SHA-256, storage/runtime dtypes, contract ID, and PASS status in per-checkpoint Frozen results and metrics metadata.

- [x] **Step 4: Run focused training tests**

Run the candidate deployment and accelerated-transfer unit tests and require PASS.

### Task 4: Verification

**Files:**
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_candidate_deployment.py`
- Test: `train/0040_dragapult_0809_action_boundary_rl/tests/test_accelerated_transfer.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: checked repository diff with no whitespace errors.

- [x] **Step 1: Run policy identity regressions**

Run `python3 -m unittest -v train.0040_dragapult_0809_action_boundary_rl.tests.test_policy_identity`.

- [x] **Step 2: Run candidate and cadence regressions**

Run both new/updated focused test modules.

- [x] **Step 3: Validate syntax and diff**

Run `python3 -m py_compile` on changed Python files and `git diff --check`.

- [x] **Step 4: Review historical-result boundary**

Confirm no existing evaluation report or checkpoint was modified or relabeled.

### Task 5: Standalone CPU-256 deployment path

**Files:**
- Modify: `evaluation/frozen_0806_full_evaluation.py`
- Test: `tests/test_evaluation_frozen_0806_full_evaluation.py`

**Interfaces:**
- Consumes: canonical Full-0806 source package and immutable Frozen opponent package.
- Produces: a temporary FP16-storage candidate package, an independent canonical
  opponent server, and formal CPU-256 candidate deployment audit evidence.

- [x] **Step 1: Materialize a temporary FP16 candidate package**

Convert every floating candidate state tensor to FP16 storage without modifying the
source package, then load it through the existing FP32 inference-server contract.

- [x] **Step 2: Keep the opponent runtime independent**

Start a separate inference server from the canonical Frozen opponent root even when
candidate source and opponent source are both Policy-0806.

- [x] **Step 3: Version CPU evidence and reject legacy payloads**

Write future results under a deployment-contract-specific directory and require the
candidate deployment PASS audit in report validation without moving old reports.

- [x] **Step 4: Run CPU contract tests and a real materialization smoke**

Verify raw FP32 evidence fails, temporary stored tensors are FP16, the source package
mtime is unchanged, and all CPU-256 contract tests pass.
