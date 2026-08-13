# 0044 Benchmark V2 Common Seeds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and integrate a CUDA-2048 Benchmark V2 that evaluates focal deck 007 against complete immutable Policy-0809 over the Meta-balanced 001–067 pool with candidate-independent common random seeds.

**Architecture:** Keep the Benchmark V1 Meta-first deck allocation and official context-41 seat choice, but create a new V2 schedule and report schema. Derive engine, Search, policy, and coin-winner seeds only from the V2 contract, fixed master seed, immutable opponent identity, exact deck identity, and schedule slot; retain focal deployment identity in the manifest and hard gates but exclude it from seed derivation.

**Tech Stack:** Python 3, PyTorch, CUDA Engine 2.0, pytest, JSON protocol manifests.

## Global Constraints

- Do not modify `engine/source/`.
- Opponent must resolve to the complete immutable `Policy-0809` identity before routing.
- Focal and opponent effective weights and tensor storage remain isolated.
- Exactly 2,048 official-engine greedy games are required, with 0 error and 0 unfinished.
- Candidate evidence remains `kaggle_fp16_storage_fp32_runtime_v1`.
- Benchmark V1 remains immutable historical evidence.
- The current training process remains stopped while code and protocol semantics change.

---

### Task 1: Publish the Benchmark V2 protocol and deterministic schedule

**Files:**
- Create: `docs/rl/0044_Benchmark_V2_Protocol.md`
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/benchmark_v2_schedule.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py`

**Interfaces:**
- Produces: `materialize(project_root: Path, *, focal_deck_id: str, focal_deployment_identity: str) -> dict[str, Any]`.
- Guarantees: changing only `focal_deployment_identity` changes schedule provenance/hash but not any job, seed, deck, or toss winner.

- [ ] Write tests asserting the 2,048-game 001–067 Meta-balanced composition, Policy-0809 opponent identity, seed uniqueness, and candidate-independent job equality.
- [ ] Run the focused schedule tests and confirm they fail before implementation.
- [ ] Implement the V2 schedule with fixed common random numbers and explicit `seed_derivation_excludes = ["focal_deployment_identity"]` metadata.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Add the Benchmark V2 runner and hard report gates

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/run_benchmark_v2.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_report.py`

**Interfaces:**
- Consumes: Benchmark V2 schedule from Task 1 and project-local Policy-0809 resolver.
- Produces: `run(...)->dict[str, Any]` and `validate_report(report)->None` with Benchmark-V2 report schema.

- [ ] Write tests that reject Champion-G2 opponents, candidate-dependent seed metadata, invalid deployment audits, storage aliasing, routing failures, and incomplete game sets.
- [ ] Run the focused report tests and confirm they fail before implementation.
- [ ] Implement the Policy-0809 CUDA-2048 runner by adapting the project-local V1 runner without importing another numbered project.
- [ ] Record every per-game engine/Search/policy seed, toss winner, context-41 choice, actual seat, and all identity audits.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Switch 0044 periodic evaluation from V1 to V2

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/config.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/tests/test_run_v1.py`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`

**Interfaces:**
- Consumes: `run_benchmark_v2.run`.
- Produces: W&B `eval/*` rows labeled `eval/benchmark_v2=1`, with `eval/checkpoint_update` and no V1 ambiguity.

- [ ] Write tests requiring every ten-update evaluation to invoke Benchmark V2 and mirror `eval/benchmark_v2`.
- [ ] Update config and runner wiring while leaving old V1 reports untouched.
- [ ] Update both authoritative design documents with the new opponent and common-seed contract.
- [ ] Run focused training/evaluation integration tests.

### Task 4: Validate reproducibility and prepare the U10 uniform-training fork

**Files:**
- Create: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/<next-version>/artifact/training_config.json` at launch time only.
- Create: a new versioned training entry point for the U10 pure-uniform fork.

**Interfaces:**
- Consumes: immutable V3 `update-000010.pt` and Benchmark V2 implementation.
- Produces: a new W&B run with fresh optimizer, pure-uniform 001–067 sampling, and periodic Benchmark V2.

- [ ] Run schedule materialization twice with different synthetic focal hashes and assert byte-identical jobs and different provenance schedule hashes.
- [ ] Run all 0044 schedule/report/training tests and a small CUDA smoke without W&B.
- [ ] Verify V3 remains stopped and U10 source checkpoint SHA-256 is recorded.
- [ ] Before the first PPO rollout, resolve or run a PASS Benchmark V2 CUDA-2048 report for the parent U10 deployment identity and mirror it to the new W&B run at `eval/checkpoint_update=10`.
- [ ] If a future runner cannot preserve the global update axis, record `trainer/local_update=0` separately while keeping strength curves keyed by `eval/checkpoint_update=10`.
- [ ] Launch the new formal version only after all gates pass.
