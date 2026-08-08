# SP09 And Frozen-002 Seed Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the exact SP09 deck and independently replicate Frozen deck 002 on 2,048 new seeded CUDA games, then publish an evidence-based comparison explaining the observed strength gap.

**Architecture:** Extend the existing SP-series archive and evaluator from eight to nine immutable exact decks. Add a replication namespace that changes only the evaluation-seed root while preserving the Frozen-0806 opponent schedule, seat balance, policies, FP32 inference, 50-full-round draw rule, and 20-repeat forfeit guard. Derive conclusions from paired deck-list differences, per-opponent counts, confidence intervals, and the independent Frozen-002 run rather than attributing the gap to a single Stadium by inspection.

**Tech Stack:** Python 3.11, resident CUDA official-state engine, Policy-0806 PyTorch inference, static HTML/JSON reports, unittest.

## Global Constraints

- Do not modify `engine/source/`.
- Every strength result must come from real simulated games through the project engine runtime.
- Each evaluation contains exactly 2,048 games, 1,024 in each seat, over the immutable 55-deck Frozen-0806 frequency schedule.
- SP09 uses the user-provided per-card quantities: 18 Pokémon, 35 Trainer, 7 Energy, exactly 60 cards.
- The Frozen-002 replication must use a seed namespace disjoint from the canonical evaluation seed while changing no other experimental variable.
- Reports use the exact `candidate-overview` deck-construction UI of the published Frozen-0806 report.

---

### Task 1: Archive and validate SP09

**Files:**
- Create: `docs/reports/sp-series/decks/SP09_MAGA/deck.csv`
- Create: `docs/reports/sp-series/decks/SP09_MAGA/manifest.json`
- Modify: `docs/reports/sp-series/manifest.json`
- Modify: `engine_cuda/tools/evaluate_sp_series_cuda.py`
- Test: `engine_cuda/tests/test_sp_series_cuda_evaluation.py`

**Interfaces:**
- Consumes: the official integer card IDs and `exact_deck_sha256(deck)`.
- Produces: `load_sp_candidates(catalog)` containing `SP09_MAGA` as an exact 60-card candidate.

- [x] Add a failing asset test asserting SP01–SP09 are present, SP09 has 60 rows, and its grouped counts are 18/35/7.
- [x] Run the focused test and confirm it fails because SP09 is absent.
- [x] Add the exact SP09 deck and immutable manifest, then extend the evaluator’s expected ID contract to nine decks.
- [x] Run the focused test and confirm it passes.
- [x] Run `python3 engine_cuda/tools/evaluate_sp_series_cuda.py --deck-id SP09_MAGA` and publish its Seeded-2048 report.

### Task 2: Independently replicate Frozen deck 002

**Files:**
- Modify: `engine_cuda/tools/evaluate_policy_0806_cuda.py`
- Create: `engine_cuda/tests/test_policy_0806_cuda_seed_replication.py`
- Generate: `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_2048_replication_002/`

**Interfaces:**
- Consumes: `build_cuda_schedule(..., evaluation_seed=<replication seed>)` and Frozen deck 002.
- Produces: a result manifest proving 2,048 unique games, 1,024 games per seat, and zero seed overlap with the canonical deck-002 schedule.

- [x] Add a failing test for a caller-supplied report/output namespace and independent evaluation seed.
- [x] Run the focused test and confirm the current fixed paths/metadata fail it.
- [x] Add the minimal seed-replication entry point without changing canonical report paths.
- [x] Assert schedule uniqueness, seat balance, and zero engine-seed overlap with the canonical deck-002 schedule.
- [x] Run the 2,048-game CUDA replication and render the deck-002 report.

### Task 3: Attribute the observed deck gap

**Files:**
- Create: `docs/reports/sp-series/seeded2048_cuda_v2/002_strength_analysis.md`
- Create: `docs/reports/sp-series/seeded2048_cuda_v2/002_strength_analysis.html`
- Modify: `docs/reports/sp-series/seeded2048_cuda_v2/index.html`

**Interfaces:**
- Consumes: canonical SP01–SP09 results, canonical Frozen-002 result, independent Frozen-002 replication, exact deck manifests, and per-opponent W/L/D rows.
- Produces: a table and narrative that separate measured facts, card-text/runtime facts, and causal hypotheses.

- [x] Compute Wilson 95% intervals for overall win rates and an interval for the two independent Frozen-002 proportions.
- [x] Build per-opponent deltas against the best compositionally comparable SP decks, including sample sizes and draw counts.
- [x] Audit exact deck-list differences so Nighttime Mine is not treated as the only changed variable.
- [x] Check whether the observed gain concentrates in Tera matchups or is distributed across non-Tera matchups.
- [x] Render Markdown and HTML analyses with direct links to every underlying report.
- [x] Run structural checks for all report deck sections, exact 60-card totals, 2,048-game totals, seat balance, result hashes, and Git diff hygiene.
