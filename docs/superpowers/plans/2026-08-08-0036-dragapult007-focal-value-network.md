# 0036 Dragapult 007 Focal Value Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare dedicated action-decision critics for Frozen-0806 exact deck 007 using 10,000 unique Dragapult-containing Episodes sampled across all available dates with a controlled recent-date bias.

**Architecture:** Preserve the frozen 0031 semantic Encoder and the existing MLP/query-head ablations, but replace the generic dual-perspective sampling frame with focal actors whose registered 60-card deck contains Dragapult ex card ID 121. Exact 007 is an audit/evaluation stratum, never a new actor-visible identity input; the existing exact-deck resource ledger is sufficient actor-visible conditioning.

**Tech Stack:** Python 3.11, PyTorch, official Kaggle Episode ZIPs, process-parallel archive scanning, gzip JSONL feature shards, TensorBoard, W&B online.

## Global Constraints

- Focal exact deck is `dragapult_ex_07bedfffbfad`, SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`.
- A focal actor qualifies only when its registered exact 60-card multiset contains card ID 121. Opponent-only perspectives are excluded.
- If both actors qualify, both trajectories may be retained but the whole Episode remains in one split.
- Selection scans all available official daily archives and takes at most 10,000 unique Episodes. Every eligible date is covered; 25% of allocation weight is uniform over dates and 75% uses a seven-day exponential recency half-life. It never duplicates Episodes to satisfy the target.
- Split isolation is whole-Episode and stratifies date bucket, focal outcome, opponent archetype and `is_exact_007`.
- Exact-deck hash, similarity, team/source and the `is_exact_007` flag are audit/metric fields only and never enter forward.
- Every example remains one real non-registration `agent(observation)` callback before its full action.
- Frozen source checkpoint and model/data/checkpoint/W&B contracts from the original 0036 plan remain unchanged.
- Existing generic catalog/dataset are retained as superseded pipeline evidence and are not formal training inputs.

---

### Task 1: Build a reusable all-perspective registration index

**Files:**
- Create: `train/0036_dedicated_action_value_network/data/dragapult_catalog.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_dragapult_catalog.py`

**Interfaces:**
- Consumes: official ZIPs dated 2026-07-10 through 2026-08-06, `registration_decks`, terminal outcome/prize helpers, exact 007 `deck.csv`.
- Produces: `scan_dragapult_archives(archives, target_trajectories) -> dict` with focal actor rows and payload locators.

- [ ] Write a synthetic replay test where only player 1 contains card 121 and assert that exactly player 1 is emitted.
- [x] Write a mirror test where both actors contain card 121 and assert two focal rows sharing one Episode identity.
- [ ] Write an exact-007 test that compares complete `Counter(card_id)` multisets rather than display name or prefix hash.
- [x] Implement all-date archive scanning with per-archive counts, payload SHA, exact decks, terminal labels, opponent class, signed final diff and focal outcome.
- [x] Allocate 10,000 unique Episodes with 25% uniform-date plus 75% seven-day-half-life recency weight, mandatory nonzero date coverage, capacity redistribution and deterministic within-date SHA ranking.
- [x] If all archives contain fewer than 10,000, emit every unique eligible Episode and record the shortfall without duplication.
- [ ] Run `python3 -m unittest -v train.0036_dedicated_action_value_network.tests.test_dragapult_catalog` and require PASS.

### Task 2: Create whole-Episode focal splits and audit metrics

**Files:**
- Modify: `train/0036_dedicated_action_value_network/data/dragapult_catalog.py`
- Create: `experiments/0036_dedicated_action_value_network/data_audit/dragapult_focal_recent_10k.json`
- Create: `experiments/0036_dedicated_action_value_network/data_audit/dragapult_focal_recent_10k_summary.json`
- Test: `train/0036_dedicated_action_value_network/tests/test_dragapult_catalog.py`

**Interfaces:**
- Consumes: Task 1 focal rows.
- Produces: immutable train/validation Episode assignments plus all-Dragapult and exact-007 strata.

- [x] Test that a mirror match never crosses splits and the validation count is exact at the Episode level.
- [ ] Test deterministic coverage for win/loss, exact/non-exact 007 and every observed opponent class.
- [x] Allocate approximately 10% of Episodes to validation; preserve at least 100 exact-007 focal trajectories in validation when the corpus supports it.
- [x] Record focal trajectories/Episodes, dates, outcomes, exact-007 count, deck variants, opponent classes and final-diff distribution.
- [x] Verify catalog commitment, unique `(episode_id, player_index)`, locator payload hashes and no opponent-only trajectory.

### Task 3: Materialize focal action decisions

**Files:**
- Modify: `train/0036_dedicated_action_value_network/data/materialize.py`
- Modify: `train/0036_dedicated_action_value_network/training/dataset.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_materialize.py`

**Interfaces:**
- Consumes: focal catalog whose Episodes may contain one or two selected actors.
- Produces: ignored local dataset `rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_recent_10k`.

- [ ] Make materialization compile only catalog-selected `player_index` entries, not both seats unconditionally.
- [x] Preserve one-over-focal-decisions trajectory weights and assert each selected focal trajectory sums to one.
- [ ] Carry `is_exact_007` and focal deck hash only in `audit`; do not add them to the 39 feature keys.
- [ ] Run a mixed win/loss/exact/non-exact smoke and validate hashes, labels and selected-actor counts.
- [ ] Materialize the full focal corpus, then independently decompress/hash every shard and persist the verification report.

### Task 4: Add deployment-aligned validation metrics

**Files:**
- Modify: `train/0036_dedicated_action_value_network/training/dataset.py`
- Modify: `train/0036_dedicated_action_value_network/training/metrics.py`
- Modify: `train/0036_dedicated_action_value_network/training/trainer.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_training.py`

**Interfaces:**
- Consumes: audit-only exact-007 masks alongside labels.
- Produces: `value/validation_exact007/*` and `value/validation_dragapult/*` metrics without feeding masks into the model.

- [ ] Test exact-007 BCE/Brier/ECE/AUROC against a hand-computed mixed batch.
- [ ] Keep optimization weights episode-equal across all focal trajectories; do not silently oversample exact 007 in the initial five-way comparison.
- [ ] Select checkpoints lexicographically by exact-007 validation BCE first and all-Dragapult validation BCE second.
- [ ] Persist per-opponent-class and exact/non-exact calibration diagnostics in canonical JSONL, TensorBoard and W&B.

### Task 5: Run the five focal ablations

**Files:**
- Modify: `train/0036_dedicated_action_value_network/run_ablation.py`
- Modify: `experiments/0036_dedicated_action_value_network/DESIGN.md`
- Modify: `experiments/0036_dedicated_action_value_network/DESIGN.html`
- Modify: `experiments/0036_dedicated_action_value_network/DECISIONS.md`

**Interfaces:**
- Consumes: verified focal dataset and frozen source checkpoint.
- Produces: immutable V1–V5 formal runs and a selected critic candidate.

- [ ] Lock all five versions to the same focal catalog SHA, split and optimization settings.
- [ ] Run raw-pool MLP, summary MLP, latent Value-only, latent+opponent-class and latent+class+final-diff.
- [ ] Require every epoch's complete exact-007 and all-Dragapult validation metrics before checkpoint selection.
- [ ] Run `V1_raw_pool_mlp`, then immediately run the full `V2_latent_value_archetype_diff`; defer summary, Value-only and single-auxiliary variants to V3–V5.
- [ ] Compare against constant-half BCE `0.693147` and Brier `0.25`; auxiliary gains never override worse exact-007 Value BCE.
- [ ] Update both authority designs with actual counts, shapes, run paths, metrics and current stage.

### Task 6: Optional exact-007 calibration after model selection

**Files:**
- Modify: `train/0036_dedicated_action_value_network/run_ablation.py`
- Modify: `train/0036_dedicated_action_value_network/export_value.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_export.py`

**Interfaces:**
- Consumes: selected V1–V5 head and exact-007 train subset.
- Produces: a new V6+ model-only critic only if held-out exact-007 BCE improves.

- [ ] Allocate a new version; never append calibration to the selected ablation run.
- [ ] Reinitialize the optimizer and calibrate on exact-007 train trajectories only.
- [ ] Reject calibration if exact-007 BCE or Brier worsens, even if generic Dragapult metrics improve.
- [ ] Export with source/value hashes and logit parity, then expose it only as PPO Value/GAE initialization.
- [ ] Continue on-policy calibration in PPO and make no actor-strength claim without official-engine evaluation.
