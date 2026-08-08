# 0038 Turn-Limit Draw Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 0038 CPU and CUDA rollout paths share one explicit 50-full-round terminal-draw contract without changing official engine source, policy observations, action semantics, or primitive `select` results.

**Architecture:** `RolloutJob.full_round_draw_limit` remains the single per-Episode contract. The CUDA adapter validates that a resident batch has one limit, converts 50 full rounds to official engine turn index 99, passes it to the resident scheduler, and materializes scheduler draws as valid reward-0 terminal Episodes. PPO keeps one final `done=True`, `next_value=0`, reward-0 transition, while diagnostics and aggregate metrics retain the distinct `turn_limit_draw` reason.

**Tech Stack:** Python 3.11, PyTorch, CUDA resident official-engine clone, `unittest`, JSON/checkpoint metadata.

## Global Constraints

- Do not modify `engine/source/`, official ABI, observation schema, or official `select` return shape.
- The 50-full-round rule is a project scheduler truncation contract, not a general official Pokémon TCG victory rule.
- Do not launch formal training or a formal Frozen evaluation in this task.
- Preserve Action Boundary, joint logprob, reward, GAE, and policy-transition semantics.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Define and test the CUDA conversion contract

**Files:**
- Modify: `train/0038_action_boundary_rl/rollout/protocol.py`
- Modify: `train/0038_action_boundary_rl/rollout/cuda_collector.py`
- Create: `train/0038_action_boundary_rl/tests/test_cuda_collector.py`

**Interfaces:**
- Consumes: `RolloutJob.full_round_draw_limit: int`.
- Produces: `_engine_turn_draw_limit(jobs: list[RolloutJob]) -> int`, where 50 maps to engine turn 99 and 0 disables the guard.

- [ ] Add failing tests for 50-to-99 conversion, disabled limit, mixed-limit rejection, and negative-limit rejection.
- [ ] Run `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_cuda_collector` and verify failure.
- [ ] Add validation and the pure conversion helper; pass its value explicitly to `run_resident_greedy_jobs`.
- [ ] Rerun the focused tests and verify success.

### Task 2: Preserve terminal reason and PPO draw semantics

**Files:**
- Modify: `train/0038_action_boundary_rl/rollout/cuda_collector.py`
- Modify: `train/0038_action_boundary_rl/tests/test_cuda_collector.py`

**Interfaces:**
- Consumes: `ResidentRunResult.turn_limit_draw_schedule_indices`.
- Produces: `_termination_status(...) -> str`, Episode diagnostic `turn_limit_draw`, aggregate `rollout/cuda_turn_limit_draws`.

- [ ] Add failing tests that repeat-forfeit wins precedence, turn-limit draw is distinct, and the chunk metric is additive.
- [ ] Implement per-Episode classification and include the metric in chunk aggregation.
- [ ] Verify a draw remains `reward=0`, final `done=True`, and `next_value=0` through existing trajectory tests.

### Task 3: Audit schedules, config, and design documentation

**Files:**
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Modify: `train/0038_action_boundary_rl/evaluation/frozen_jobs.py`
- Modify: `train/0038_action_boundary_rl/tests/test_project_identity.py`
- Modify: `train/0038_action_boundary_rl/tests/test_scaling_and_frozen_panel.py`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Modify: `ACTION_BOUNDARY_0038_STATUS.md`

**Interfaces:**
- Consumes: the common 50-full-round setting.
- Produces: schedule rows and checkpoint config that explicitly record `full_round_draw_limit=50` and CUDA engine turn index 99.

- [ ] Add assertions that rollout and canonical Frozen jobs both carry limit 50.
- [ ] Include the limit in schedule serialization/hash inputs so a future limit change cannot masquerade as the same environment contract.
- [ ] Document official-rule versus project-truncation evidence boundaries and the paused-training state.
- [ ] Run the focused 0038 suite, CUDA resident CPU tests, `py_compile`, and `git diff --check`; do not launch training.

