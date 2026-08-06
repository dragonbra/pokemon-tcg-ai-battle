# Frozen-0806 Full Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate all 55 Frozen-0806 exact decks as Policy-0806 candidates against the same fixed 256-game Policy-0019 distribution and publish one auditable official-engine strength report.

**Architecture:** Extend the existing evaluation batch contract with explicit per-opponent game counts while preserving uniform legacy runs. Materialize one shared Policy-0806 runtime and reuse one shared Policy-0019 runtime; two resident CUDA inference servers route observations by each game's exact deck while isolated worker processes run the official engine. Archive pre-0806 Combat Mat and Frozen assets before publishing the replacement report.

**Tech Stack:** Python 3.11, PyTorch CUDA inference, official `cg` engine runtime, evaluation worker processes, static HTML/JSON reports.

## Global Constraints

- The canonical batch size is 256 games per candidate, not 240.
- Every one of the 55 candidate exact decks must run the same committed Frozen-0806 schedule.
- Candidate inference uses Policy-0806 SHA `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.
- Opponent inference uses Policy-0019 SHA `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb` with neutral `source_id=0`.
- Policy models are loaded once per role into resident GPU services and route by exact 60-card deck.
- Every game runs in an isolated worker against the official engine runtime; `engine/source/` is read-only.
- A candidate report is publishable only when all 256 games finish with zero errors and the schedule/policy hashes match.
- Existing pre-0806 assets are moved to `archive/evaluation/` with an archive manifest; they are not deleted.

---

### Task 1: Variable-count batch contract

**Files:**
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Consumes: `BatchConfig.opponents` in deterministic schedule order.
- Produces: optional `BatchConfig.games_by_opponent: tuple[int, ...] | None`; when present, `_game_jobs()` creates exactly those counts and balances seats over the complete schedule.

- [x] Add failing tests for variable opponent counts, total manifest games, stable seeds, and 128/128 seats across a 256-game schedule.
- [x] Implement strict length/positive-integer validation and preserve legacy `games_per_opponent` behavior when the optional field is absent.
- [x] Record explicit per-opponent game counts and the fixed schedule identity in report manifests.
- [x] Run focused batch tests.

### Task 2: Frozen-0806 executable policy runtimes

**Files:**
- Create: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/policies/policy_0019/`
- Create: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/policies/policy_0806/`
- Modify: `evaluation/frozen_0806.py`
- Create: `tests/test_evaluation_frozen_0806_runtime.py`

**Interfaces:**
- Consumes: immutable pretrained archives and the existing audited candidate exporters.
- Produces: `Frozen0806RuntimeCatalog` containing one candidate runtime, one opponent runtime, and 55 deck-routed `SubmissionPackage` identities per role.

- [x] Add failing tests that both runtime packages validate, expose shared inference, match the fixed policy SHAs, and return every exact 60-card deck on initialization.
- [x] Export one Policy-0806 runtime from the archived epoch-11 checkpoint.
- [x] Materialize one Policy-0019 runtime with neutral source ID and preserve exact runtime hashes.
- [x] Implement deck-routed package identities without duplicating model weights per deck.
- [ ] Run package validation and a small official-engine smoke.

### Task 3: Archive obsolete evaluation assets

**Files:**
- Move: `evaluation/arena/combat_mat/` to `archive/evaluation/pre_0806_combat_mat/`
- Move: legacy Frozen-0019 deck/catalog assets to `archive/evaluation/0019_foundation_51_exact_decks_v4/`
- Create: `archive/evaluation/manifest.json`
- Modify: active evaluation docs/config references as required.

**Interfaces:**
- Consumes: current pre-0806 assets and their checksums.
- Produces: recoverable archive locations plus an active empty `evaluation/arena/combat_mat/` destination for Frozen-0806.

- [x] Inventory exact source paths, sizes, pool IDs, and key hashes before moving anything.
- [x] Move assets without overwriting any existing archive path.
- [x] Write an archive manifest containing old/new paths, reason, date, size, and identity hashes.
- [x] Verify archived reports and legacy Frozen assets remain readable and active defaults no longer point at missing paths.

### Task 4: GPU benchmark and full evaluator

**Files:**
- Create: `evaluation/frozen_0806_full_evaluation.py`
- Create: `tests/test_evaluation_frozen_0806_full_evaluation.py`
- Create: `.tmp/evaluation/frozen_0806_benchmark/` during calibration.

**Interfaces:**
- Consumes: `Frozen0806RuntimeCatalog`, fixed schedule counts, and resident candidate/opponent inference services.
- Produces: resumable per-deck runs and immutable published reports under `evaluation/arena/combat_mat/frozen/0806_kaggle_top100_plus_v1/`.

- [ ] Add tests for schedule expansion, candidate identity routing, report acceptance gates, resume behavior, and aggregate calculations.
- [ ] Implement one shared Policy-0806 CUDA service and one shared Policy-0019 CUDA service per 256-game candidate run.
- [ ] Benchmark stable worker/batch settings on the same official-engine workload and record throughput evidence.
- [ ] Select the fastest zero-error setting and run all 55 × 256 = 14,080 games.
- [ ] Reject and rerun any partial, errored, wrong-policy, wrong-deck, or wrong-schedule candidate report.

### Task 5: Publish the strength report

**Files:**
- Create: `evaluation/arena/combat_mat/frozen/0806_kaggle_top100_plus_v1/index.html`
- Create: `evaluation/arena/combat_mat/frozen/0806_kaggle_top100_plus_v1/manifest.json`
- Create: `evaluation/arena/combat_mat/frozen/0806_kaggle_top100_plus_v1/reports/<deck_id>.html`
- Modify: `evaluation/arena/combat_mat/index.html`
- Modify: `evaluation/README.md`

**Interfaces:**
- Consumes: 55 accepted 256-game reports.
- Produces: sortable 55-deck strength table with W-L-D, overall/first/second win rates, exact-deck identity, archetype, source rank, runtime duration, completion, and links to full per-game evidence.

- [ ] Aggregate only 55 complete zero-error reports totaling exactly 14,080 official-engine games.
- [ ] Render representative card images and sortable/searchable deck-strength results.
- [ ] Include immutable Policy-0806, Policy-0019, pool manifest, and schedule hashes in the report header and embedded manifest.
- [ ] Run report-contract tests, archive checks, Frozen regression tests, and `git diff --check`.
- [ ] Confirm no files under `engine/source/` changed and report the final ranked results.
