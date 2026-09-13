# 0045 PinHaoLong V3 Policy-0814 Benchmark V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the final eight V14/V19 Policy-0814-descendant checkpoints with exact focal deck 067 against complete immutable Policy-0809 under the standard common-seed Benchmark V2 CUDA-2048 contract.

**Architecture:** Add a project-local evaluation adapter that retains the historical Benchmark V2 opponent and schedule but materializes the focal candidate from the immutable Policy-0814 actor/value base. Keep the old `run_benchmark_v2.py` unchanged because its Policy-0809 focal base is part of existing evidence. Store all eight immutable reports under a new evaluation-only V20 version and render one authoritative comparison page.

**Tech Stack:** Python 3, PyTorch, project CUDA engine 2.0 collector, JSON reports, pytest, standalone HTML.

## Global Constraints

- Do not modify `engine/source/` or any official engine source.
- Focal candidates use `kaggle_fp16_storage_fp32_runtime_v1`: Policy-0814 FP32 base plus checkpoint delta, then FP16 storage, strict FP32 runtime reload.
- Opponent is the complete immutable `Policy-0809` with identity audit `PASS` and zero focal/opponent tensor-storage aliases.
- Benchmark contract remains `0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2`, 2,048 terminal games per candidate, greedy selection, zero errors and zero unfinished games.
- Final candidates are exactly `V14/U5`, `V14/U25`, `V19/U125`, `V19/U140`, `V19/U145`, `V19/U160`, `V19/U165`, and `V19/U170`.
- `V14/U40`, `V14/U75`, `V19/U135`, and `V19/U155` are explicitly excluded.
- Existing V14/V19 checkpoints and earlier evaluation reports are immutable and must not be overwritten.

---

### Task 1: Policy-0814 Focal / Policy-0809 Opponent Adapter

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_policy0814_candidate_benchmark_v2.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_candidate_benchmark_v2.py`

**Interfaces:**
- Consumes: `candidate.materialize`, `DEFAULT_0814_ACTOR_CHECKPOINT`, `DEFAULT_0814_VALUE_CHECKPOINT`, `benchmark_v2_schedule.materialize`, and the existing CUDA rollout collector.
- Produces: `run(...) -> dict[str, Any]`, `validate_report(report) -> None`, and a CLI accepting source checkpoint, source version, update, deck 067, and an unused output root.

- [ ] **Step 1: Write a failing identity-contract test**

Assert that the new adapter declares Policy-0814 focal base paths, complete Policy-0809 opponent identity, the historical Benchmark V2 schedule ID, and a lineage-qualified candidate policy ID that cannot collide across V14/V19.

- [ ] **Step 2: Run the focused test and verify failure**

Run `python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_candidate_benchmark_v2.py -q`; expect import failure before the adapter exists.

- [ ] **Step 3: Implement the adapter**

Copy only the stable orchestration shape of `run_benchmark_v2.py`; replace focal materialization with `candidate.materialize(... DEFAULT_0814_ACTOR_CHECKPOINT, DEFAULT_0814_VALUE_CHECKPOINT ...)`, let checkpoint metadata select legacy or expanded adaptation, pass exact opponent own-archetype IDs to the collector, and hard-fail all candidate/opponent/CUDA/schedule/completion identity mismatches.

- [ ] **Step 4: Run focused and nearby tests**

Run the focused test plus `test_v19_u125_ffn_lora_expansion.py` and the Benchmark V2 contract test in `test_minimal_optimizer_and_tiny_v2.py`.

### Task 2: Execute the Eight Immutable CUDA-2048 Reports

**Files:**
- Existing: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/evaluation_config.json`
- Generate: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/candidates/<version>-u<update>/report.json`
- Modify: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/status.json`

**Interfaces:**
- Consumes: the adapter CLI and eight source checkpoint SHA-256 identities from `evaluation_config.json`.
- Produces: eight validated, non-overwriting JSON reports with a shared common-random schedule and independent deployment-effective hashes.

- [ ] **Step 1: Preflight the first candidate**

Run V14/U5 through the formal adapter; require candidate, opponent, CUDA, schedule, exact-deck and completion gates to pass.

- [ ] **Step 2: Run the remaining seven candidates sequentially**

Use one GPU process at a time to avoid memory and CUDA-context interference. Stop immediately on the first failed report rather than continuing with incomplete evidence.

- [ ] **Step 3: Cross-report audit**

Validate all reports, assert identical 2,048 game IDs/opponent IDs/random seeds/toss outcomes, assert distinct expected candidate identities, and summarize overall and per-Meta results.

### Task 3: Render and Register the Formal Comparison

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/render_pinhaolong_v3_candidate_benchmark_v2.py`
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V20_pinhaolong_v3_067_candidate_benchmark_v2.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/evaluation/index.html`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/evaluation.json`
- Modify: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V20_pinhaolong_v3_067_candidate_benchmark_v2/artifact/status.json`

**Interfaces:**
- Consumes: all eight passing reports.
- Produces: one authoritative HTML heatmap with overall, first/second, Core-16 Meta rows, candidate identities and evidence boundaries, plus an immutable reverse link from V20 artifacts.

- [ ] **Step 1: Implement fail-closed renderer input validation**

Require exactly the configured eight labels, PASS reports, deck 067, complete Policy-0809, common Benchmark V2 contract, identical random schedule inputs, and 16,384 total terminal games.

- [ ] **Step 2: Render the comparison HTML**

Use row-relative heat colors, exact percentages and wins/games, row-best outlines, sticky headers, and explicit separation between the 60.15625% Eval-512 exact-meta oracle upper bound and deployable Benchmark V2 evidence.

- [ ] **Step 3: Refresh the evaluation index and status**

Add V20 to the project evaluation index, write `evaluation.json`, and set status to `completed` only after all identity and report validations pass.

- [ ] **Step 4: Final verification**

Run `git diff --check`, parse the HTML, rerun report validation over all eight JSON files, and report exact W-L-D/win rate rankings to the user.
