# 0038 Post-Fix CPU/CUDA Seeded-2048 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit the 0038 semantic-parity repair, then compare Policy-0806 deck 007 on the same Frozen-0806 fixed 2,048-game panel using the official CPU engine and the repaired CUDA engine.

**Architecture:** Both evaluations consume `frozen_0806_seeded_2048_v2`, evaluation seed `341512806`, the exact 256-slot frequency schedule repeated with eight independent seed replicas, balanced seats, Policy-0806 on both sides, greedy FP32 inference, repeat limit 20, and the 50-full-round draw contract. CPU uses the unmodified official seeded runtime; CUDA uses the repaired resident engine and exact same per-slot seed derivation. A postprocessor compares per-game outcomes and aggregate seat/matchup/shard distributions without treating aggregate win-rate agreement as proof of semantic parity.

**Tech Stack:** Python 3.11, official CPU engine ctypes runtime, PyTorch CUDA inference service, native CUDA resident engine, JSON/HTML evaluation artifacts, Git.

## Global Constraints

- Do not modify `engine/source/`, official ABI, observation schema, model weights, reward, or PPO settings.
- Do not start RL training or submit to Kaggle.
- Use exact deck 007 `dragapult_ex_07bedfffbfad` and immutable Frozen-0806 schedule SHA-256 `98b58bced460c1a2e622ae4b39bf506294fcb0230bb42e4117aaa6efc73c9ce9`.
- Both backends use 2,048 unique fixed seeds and balanced 1,024/1,024 seats.
- Preserve unrelated dirty-worktree changes and stage only semantic-parity/evaluation deliverables.

---

### Task 1: Contract preflight

**Files:**
- Inspect: `evaluation/frozen_0806_contract.py`
- Inspect: `evaluation/frozen_0806_full_evaluation.py`
- Inspect: `engine_cuda/tools/evaluate_policy_0806_cuda.py`

**Interfaces:**
- Consumes: committed Frozen-0806 catalog and Policy-0806 checkpoint.
- Produces: exact CPU/CUDA command lines with identical seed/schedule/seat/model contracts.

- [ ] Confirm both paths call `evaluation_game_seed(...)` with identical focal/opponent identity, slot and replica.
- [ ] Confirm both paths use opponent policy `0806`, greedy FP32, repeat limit 20 and turn limit 100.
- [ ] Confirm GPU and memory are idle before starting.

### Task 2: Commit semantic-parity repair

**Files:**
- Modify: `engine_cuda/` semantic feature/rule adapter files.
- Modify: `train/0038_action_boundary_rl/` Action Boundary, package and parity files.
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Create: `SEMANTIC_PARITY_AUDIT.md`

**Interfaces:**
- Consumes: Gate A/B/C/D evidence and 66 passing regression tests.
- Produces: one pushed commit identifying the exact code under evaluation.

- [ ] Run `git diff --check` and the release-focused 66-test suite.
- [ ] Stage only relevant repair, tests, fixtures, design and audit files.
- [ ] Review the staged diff and verify `engine/source/` is absent.
- [ ] Commit and push `dev/cyd_main`.

### Task 3: Official CPU seeded-2048

**Files:**
- Generate: `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_seeded_2048_v2/reports/007_dragapult_ex.html`

**Interfaces:**
- Consumes: `Policy-0806`, official seeded CPU engine, Frozen-0806 fixed schedule.
- Produces: 2,048 per-game records plus aggregate W-L-D/seat/matchup metrics.

- [ ] Run `python3 -m evaluation.frozen_0806_full_evaluation --deck 007 --opponent-policy 0806 --workers 8 --engine-pool-size 1 --candidate-concurrency 1 --batch-size 128 --batch-wait-ms 1 --inference-dtype fp32`.
- [ ] Require 2,048/2,048 complete, zero error and zero unfinished.
- [ ] Extract schedule/model/runtime hashes and per-game outcomes.

### Task 4: Repaired CUDA seeded-2048

**Files:**
- Generate: an isolated repaired-runtime result/report without overwriting the historical pre-fix CUDA report.

**Interfaces:**
- Consumes: the same Policy-0806 checkpoint and fixed per-slot schedule as Task 3.
- Produces: 2,048 CUDA outcomes and resident throughput/guard metadata.

- [ ] Run the CUDA evaluator for deck 007 using a new output namespace.
- [ ] Require 2,048/2,048 terminal, zero engine error and zero unfinished.
- [ ] Record CUDA extension/rule/feature-schema hashes and per-game outcomes.

### Task 5: Paired distribution analysis

**Files:**
- Create: `0038_007_CPU_CUDA_2048_PARITY_REPORT.md`

**Interfaces:**
- Consumes: CPU and CUDA per-game outcomes keyed by opponent, slot, replica and seat.
- Produces: evidence-backed conclusion about practical outcome distribution differences.

- [ ] Verify the two schedules contain the same 2,048 unique keys and no duplicates.
- [ ] Report pooled W-L-D, Wilson intervals, seat split, eight-replica mean/variance and per-matchup deltas.
- [ ] Report paired outcome agreement/flips and McNemar or paired bootstrap result.
- [ ] Explain that different engine RNG consumption can prevent exact game-by-game outcomes even with identical initial seeds; aggregate similarity does not replace Gates A–D.
- [ ] State whether yesterday's pre-fix CUDA strength reports remain usable as historical evidence only.

### Task 6: Publish evidence

**Files:**
- Commit: new CPU/CUDA report artifacts and comparison Markdown.

**Interfaces:**
- Consumes: validated reports from Tasks 3–5.
- Produces: pushed, reproducible evaluation evidence.

- [ ] Run report integrity assertions and `git diff --check`.
- [ ] Stage only the new evaluation evidence and comparison report.
- [ ] Commit and push `dev/cyd_main`.
