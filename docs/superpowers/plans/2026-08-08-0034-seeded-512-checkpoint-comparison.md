# 0034 Seeded 512-Game Checkpoint Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare zero-shot and five representative 0034 V6 decoder checkpoints on one immutable 512-game, seeded, seat-paired official-engine schedule.

**Architecture:** A project-local comparison runner selects audited checkpoints from the existing V6 metrics, exports each decoder over the unchanged 0806 actor package, and evaluates each candidate sequentially through the established N16E6/B128/1ms worker-local pipeline. Every arm uses the same candidate deck, doubled Frozen-0806 opponent allocation, engine/search/policy seeds, and physical deck slots; the aggregate report retains per-game paired outcomes and bootstrap-ready pair identifiers.

**Tech Stack:** Python 3.11, PyTorch model-only checkpoints, seeded official C++ runtime, evaluation engine pool, resident CUDA inference, HTML/JSON reporting, unittest.

## Global Constraints

- Never modify `engine/source/`; build and load only the hash-addressed seeded runtime under `engine/build/`.
- All strength results must come from completed official-engine games with zero worker errors.
- The focal deck is fixed to `dragapult_ex_07bedfffbfad` with exact-deck SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`.
- The comparison base seed is evaluation-only and must be distinct from 0034 rollout/training seeds.
- Each arm runs 512 games: the Frozen-0806 256-game allocation doubled, yielding 256 engine-seed pairs with opposite requested seats.
- Checkpoint selection uses existing metrics only; the new 512-game results may not be used to change the selected set after seeing outcomes.
- Preserve all unrelated dirty-worktree changes and do not overwrite existing checkpoint or training artifacts.
- Candidate inference uses N16E6, batch size 128, 1 ms wait, FP16, synchronous H2D, and the torch-free worker-local compiler.

---

### Task 1: Freeze the comparison contract and checkpoint selection

**Files:**
- Create: `train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`
- Test: `train/0034_dragapult_third_large_model_rl/tests/test_seeded_checkpoint_series.py`

**Interfaces:**
- Produces: `ComparisonArm(update: int, label: str, selection_reason: str, prior_metric: dict[str, float])`.
- Produces: `select_comparison_arms(metrics_path: Path, checkpoint_dir: Path) -> tuple[ComparisonArm, ...]` with updates `(0, 10, 20, 80, 100, 123)`.
- Produces: immutable evaluation base seed `341512806`, separate from V6 training seed `330031001`.

- [ ] Write a test fixture with update-0/every-10 frozen probes and a final update-123 rollout row; assert the six selected arms, labels, prior evidence type, and checkpoint existence checks.
- [ ] Run `python3 -m unittest -v train.0034_dragapult_third_large_model_rl.tests.test_seeded_checkpoint_series` and confirm the module is missing.
- [ ] Implement strict JSONL parsing, duplicate-update rejection, exact selection reasons, checkpoint SHA-256 capture, and fail-closed missing checkpoint handling.
- [ ] Re-run the focused test and require PASS.

### Task 2: Build the doubled seeded schedule

**Files:**
- Modify: `train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`
- Modify: `train/0034_dragapult_third_large_model_rl/tests/test_seeded_checkpoint_series.py`

**Interfaces:**
- Produces: `comparison_config(...) -> BatchConfig` using `games_by_opponent=tuple(2 * entry.games for entry in catalog.pool.schedule)`.
- Produces: a schedule commitment containing candidate/deck/pool/catalog/runtime hashes, 512 requests, 256 distinct engine seeds, two games per engine seed, and opposite seats inside every pair.

- [ ] Add a test that builds jobs without starting workers and asserts 512 games, exact doubled opponent counts, 256 engine seeds, two games per seed/opponent, opposite seats, and identical request seed tuples across two arms.
- [ ] Run the test and confirm it fails before schedule construction exists.
- [ ] Implement the BatchConfig builder with seed `341512806`, N16E6/B128/1ms, FP16, sync H2D, worker-local compiler, seeded runtime enabled, zero retries beyond the existing one-crash safety boundary, and `league_deck_quality` metrics.
- [ ] Hash a canonical JSON representation of `(game_id, opponent, seat, engine_seed, policy_seed, search_seed)` and store it as `request_schedule_sha256`.
- [ ] Re-run the focused tests and require PASS.

### Task 3: Export and evaluate all six decoder arms

**Files:**
- Modify: `train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`
- Create during execution: `.tmp/evaluation/0034_seeded_512_checkpoint_comparison/contract.json`
- Create during execution: `.tmp/evaluation/0034_seeded_512_checkpoint_comparison/arms/<label>/candidate/`
- Create during execution: `.tmp/evaluation/0034_seeded_512_checkpoint_comparison/arms/<label>/run-*/report.html`

**Interfaces:**
- Consumes: `export_candidate(source, checkpoint, output)` from the self-contained 0034 exporter.
- Produces: one immutable arm JSON containing checkpoint/export hashes, report path, W/L/D, first/second split, selections/s, wall time, runtime manifest, schedule hash, and 512 lightweight per-game outcomes.

- [ ] Add a dry-run test that mocks export/server/batch boundaries and asserts one shared opponent policy server plus six sequential candidate server lifetimes.
- [ ] Implement atomic contract/status writes before any official games; refuse to reuse an arm whose checkpoint or schedule commitment differs.
- [ ] Export each arm from the unchanged zero-shot candidate, validate its 60-card identity and decoder checkpoint update, and record model/package hashes.
- [ ] Start the frozen Policy-0806 opponent service once, start one candidate service per arm, run the 512 official games, and fail closed unless completion is 512/512 with zero errors and zero unfinished games.
- [ ] Preserve each generated `run_id/report.html` under the required `.tmp/evaluation/` tree and write resumable arm result JSON only after validation.

### Task 4: Aggregate paired checkpoint evidence

**Files:**
- Modify: `train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`
- Create during execution: `.tmp/evaluation/0034_seeded_512_checkpoint_comparison/comparison.json`
- Create during execution: `.tmp/evaluation/0034_seeded_512_checkpoint_comparison/index.html`
- Create: `experiments/0034_dragapult_third_large_model_rl/evaluation/V6_exact007_0023_selection_lambda095.html`
- Create: `rl_runs/0034_dragapult_third_large_model_rl/versions/V6_exact007_0023_selection_lambda095/artifact/evaluation.json`

**Interfaces:**
- Produces: arm rankings, first/second win rates, Wilson 95% intervals, paired deltas versus zero-shot, and exact discordant pair counts without claiming independent-game certainty for paired observations.

- [ ] Add deterministic unit tests for W/L/D aggregation, Wilson intervals, matched-game deltas, and the rule that rollout win rate remains selection context rather than final strength evidence.
- [ ] Implement an HTML report linking every source report and showing the frozen contract, selected-node rationale, per-arm outcomes, throughput, seat split, and paired zero-shot comparison.
- [ ] Copy the validated aggregate HTML to the unused V6 authoritative evaluation path and write its SHA-256 plus source `.tmp` paths into `artifact/evaluation.json`.
- [ ] Refresh `experiments/0034_dragapult_third_large_model_rl/evaluation/index.html` using the repository reporting contract without overwriting any existing V-report.

### Task 5: Verification and evidence boundary

**Files:**
- Modify: `evaluation/README.md`
- Modify: `experiments/0034_dragapult_third_large_model_rl/DECISIONS.md`

**Interfaces:**
- Consumes: all six official-engine arm results and the seeded runtime manifest.
- Produces: a documented default that checkpoint comparisons use at least 512 fixed seeded games, separate RL/evaluation seed namespaces, and matched-seat interpretation.

- [ ] Document the 512-game minimum as a checkpoint-comparison policy, not a universal statistical guarantee or a replacement for opponent-pool-specific formal contracts.
- [ ] Record selected updates and explain that general official rules and card/runtime semantics are unchanged; only the local evaluation experiment contract changed.
- [ ] Run `python3 -m unittest -v train.0034_dragapult_third_large_model_rl.tests.test_seeded_checkpoint_series tests.test_seeded_official_runtime tests.test_evaluation_batch tests.test_evaluation_engine_pool_worker tests.test_evaluation_inference_server`.
- [ ] Run `python3 -m py_compile train/0034_dragapult_third_large_model_rl/evaluate_seeded_checkpoint_series.py`, `git diff --check`, and `git diff --exit-code -- engine/source`.
- [ ] Inspect all six reports for 512/512 completion, zero errors, identical request schedule hashes, identical focal deck hashes, and the recorded seeded runtime ABI/library hashes.
