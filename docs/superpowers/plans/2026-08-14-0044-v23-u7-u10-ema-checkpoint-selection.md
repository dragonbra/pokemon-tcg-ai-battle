# 0044 V23 U7–U10 + EMA Checkpoint Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare four promising V23 checkpoints and a finite-window EMA under one deployment-audited greedy CUDA-2048 common-random Benchmark V2 schedule.

**Architecture:** Treat V23 U7/U8/U9/U10 as immutable source checkpoints, build a decay-0.5 FP32 EMA with explicit provenance, and materialize every evaluation candidate through `kaggle_fp16_storage_fp32_runtime_v1`. Reuse the project-local Benchmark V2 official-engine runner and publish one fail-closed comparison with overall, seat, Meta, and exact-deck results; this selects an observed checkpoint but never promotes it automatically.

**Tech Stack:** Python 3.11, PyTorch, CUDA Engine 2.0, pytest, JSON, static HTML.

## Global Constraints

- Candidate checkpoints are exactly V23 U7, U8, U9, and U10; their subsequent sampled rollout rates are 62.3047%, 63.8672%, 63.8672%, and 61.7188% respectively.
- V23 U11 is excluded because its subsequent complete 512-game rollout was 294-216-2 (57.4219%) and the following U12 optimizer update was partial and discarded.
- The derived EMA uses normalized oldest-to-newest weights `1/15, 2/15, 4/15, 8/15`; it is not a true optimizer update and cannot be promoted implicitly.
- Every candidate uses greedy selection, exact focal deck 069, independently resolved full Policy-0809 opponent weights, the same 2,048-game Benchmark V2 common schedule, and the official CUDA engine runtime.
- Every candidate must pass FP16 storage → FP32 runtime deployment identity, opponent identity, lane routing, terminal completion, and zero-error hard gates before it enters the comparison.
- No official engine source file may be modified, and no existing report or checkpoint may be overwritten.

---

### Task 1: Audited V23 finite EMA

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/build_v23_u7_u10_ema.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/tests/test_v23_checkpoint_selection.py`

**Interfaces:**
- Consumes: immutable V23 U7/U8/U9/U10 model-only checkpoints and their exact SHA-256 values.
- Produces: `build(...) -> dict[str, Any]`, a derived model-only checkpoint, and an immutable provenance manifest.

- [x] **Step 1: Write the EMA contract test**

Assert exact source version, updates, hashes, normalized weights, FP32 accumulation, newest-source non-floating tensors, derived metadata, and model-only schema.

- [x] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_v23_checkpoint_selection.py`

Expected: FAIL because the V23 EMA builder does not exist.

- [x] **Step 3: Implement and materialize the EMA**

Create the builder with fail-closed source identity/schema/key/shape/dtype checks and atomic checkpoint/manifest writes, then run:

```bash
python3 -m train.0044_g2_dragapult_policy_option_lora.evaluation.build_v23_u7_u10_ema
```

- [x] **Step 4: Re-run the focused test**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_v23_checkpoint_selection.py`

Expected: PASS.

### Task 2: V23 comparison renderer

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/render_v23_checkpoint_selection.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/tests/test_v23_checkpoint_selection.py`

**Interfaces:**
- Consumes: ordered reports `U7`, `U8`, `U9`, `U10`, and `EMA U7–U10`, plus the existing deck-069 Champion-G3/V18-U0 Benchmark V2 baseline manifest.
- Produces: a fail-closed aggregate manifest and static HTML comparison under a new immutable Benchmark V2 slug.

- [x] **Step 1: Add renderer contract tests**

Assert ordered labels, 2,048 valid jobs per candidate, identical common job identities, deployment audit PASS, overall selection, per-Meta rows, exact-deck rows, and no automatic promotion.

- [x] **Step 2: Implement the renderer**

Render overall W-L-D/Wilson/seat splits, all 16 Benchmark V2 Meta groups, all observed exact decks, rollout-selection provenance, source/deployment hashes, and explicit diagnostic-only/manual-decision language.

- [x] **Step 3: Run renderer and existing Benchmark V2 tests**

Run: `pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_v23_checkpoint_selection.py train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_report.py train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py`

Expected: PASS.

### Task 3: Common-seed CUDA-2048 execution

**Files:**
- Create: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V23_v22_u9_final_entropy_0015/artifact/benchmark_v2/<candidate>/report.json`
- Create: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V23_v22_u9_final_entropy_0015/artifact/benchmark_v2/<candidate>/materialization/model.bin`

**Interfaces:**
- Consumes: U7/U8/U9/U10/EMA checkpoints and `evaluation.run_benchmark_v2`.
- Produces: five immutable, deployment-audited greedy official-engine CUDA-2048 reports.

- [x] **Step 1: Verify preflight identities and unused outputs**

Require all five checkpoint inputs, exact source hashes, CUDA availability, Policy-0809 identity PASS, and nonexistent candidate output directories.

- [x] **Step 2: Run U7, U8, U9, and U10**

Invoke `python3 -m train.0044_g2_dragapult_policy_option_lora.evaluation.run_benchmark_v2` with deck 069, each exact checkpoint/update, and its unique V23 artifact output root.

- [x] **Step 3: Run EMA U7–U10**

Invoke the same runner with the derived checkpoint and checkpoint-update 10; retain the derived identity in its manifest rather than presenting it as a true U10 optimizer checkpoint.

- [x] **Step 4: Validate all five reports**

Require 10,240 total valid terminal games, zero error/unfinished, common job identities, candidate/opponent/CUDA identity PASS, and five distinct deployment-effective hashes.

### Task 4: Publish and interpret

**Files:**
- Create: `docs/evaluation/combat_mat/benchmark_v2/0044_v23_deck069_u7_u10_ema_checkpoint_selection_core16_policy0809_cuda2048_v2/index.html`
- Create: `docs/evaluation/combat_mat/benchmark_v2/0044_v23_deck069_u7_u10_ema_checkpoint_selection_core16_policy0809_cuda2048_v2/manifest.json`
- Modify: `docs/evaluation/combat_mat/benchmark_v2/index.html`

**Interfaces:**
- Consumes: five validated reports and the existing comparable baseline manifest.
- Produces: a clickable immutable comparison and a factual checkpoint-selection recommendation without promotion.

- [x] **Step 1: Publish the comparison atomically**

Archive each source report beside the manifest, render the HTML, and add the new report to the Benchmark V2 index without altering prior reports.

- [x] **Step 2: Verify report integrity**

Check embedded JSON, source report hashes, common schedule hash, 16 Meta rows, exact-deck rows, focal/opponent identity, and deployment contract.

- [x] **Step 3: Summarize the result**

Report overall and seat-split ranking, Meta 00/01/02/03/05/27 results, exact decks 007/003/002/067, confidence limits, and whether EMA helps. Keep the human decision at `HOLD` unless the user explicitly issues `PROMOTE`.
