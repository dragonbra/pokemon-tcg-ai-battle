# 0044 Decks 068–069 Alakazam Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register two immutable Alakazam focal decks as IDs 068 and 069, preserve the frozen 001–067 training/opponent pool, and compare Champion-G3 with G4 U57 on both decks under Benchmark V2 CUDA-2048.

**Architecture:** The project deck registry grows to 69 contiguous identities, while roles keep 068/069 evaluation-only and the training pool remains exactly 001–067. Own-deck routing maps both new IDs to existing Alakazam class 03 without changing the model's 29-class embedding shape. Benchmark V2 pins its existing taxonomy/mapping identities and filters opponent members to training-role decks, so adding focal-only assets cannot change its established Core-16 jobs.

**Tech Stack:** Python 3.11, PyTorch, project CUDA Engine 2.0, JSON asset registries, pytest, static HTML reports.

## Global Constraints

- Deck IDs are append-only canonical numeric identities; 068 and 069 must never alias SP08 or an existing 001–067 deck.
- Deck 069 differs from 068 only by Night Stretcher `1097` −1 and Boss’s Orders `1182` +1.
- Both decks use own Meta Archetype 03 (Alakazam); the embedding class count remains 29.
- The Benchmark V2 opponent remains complete Policy-0809 over the frozen 001–067 training pool with the existing Core-16 fixed random schedule.
- Every evaluation arm uses official engine runtime, CUDA-2048, greedy inference, FP16 storage followed by FP32 runtime, and independent policy identities.
- No file under `engine/source/` may be modified.

---

### Task 1: Immutable deck assets and role boundary

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/assets/decks/definitions/068/deck.csv`
- Create: `train/0044_g2_dragapult_policy_option_lora/assets/decks/definitions/069/deck.csv`
- Modify: `train/0044_g2_dragapult_policy_option_lora/assets/decks/registry.json`
- Modify: `train/0044_g2_dragapult_policy_option_lora/assets.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_assets.py`

**Interfaces:**
- Consumes: user-provided exact card counts.
- Produces: registry entries with IDs `068` and `069`, role `evaluation`, and verified file/content SHA-256 identities.

- [ ] **Step 1: Write failing registry assertions**

```python
registry = AssetRegistry.load(PROJECT_ROOT)
assert tuple(deck.deck_id for deck in registry.decks) == tuple(f"{i:03d}" for i in range(1, 70))
assert tuple(deck.deck_id for deck in registry.decks if "training" in deck.roles) == tuple(f"{i:03d}" for i in range(1, 68))
assert tuple(deck.deck_id for deck in registry.decks if deck.roles == ("evaluation",)) == ("068", "069")
```

- [ ] **Step 2: Run the asset test and verify it fails because 068/069 are absent**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_assets.py`

- [ ] **Step 3: Add exact 60-card files and append registry entries**

Use 068 counts exactly as supplied. Build 069 from 068 by changing only `1097: 3→2` and `1182: 2→3`. Compute `file_sha256` from the physical file and `content_sha256` with `canonical_deck_sha256`; set `roles: ["evaluation"]`.

- [ ] **Step 4: Preserve the 67-deck training contract while admitting 69 registry identities**

Change `AssetRegistry.validate_all()` to require contiguous 001–069 registered identities, exact 001–067 training-role identities, and exact 068–069 evaluation-only identities. Keep duplicate content and canonical path checks unchanged.

- [ ] **Step 5: Run the asset tests**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_assets.py`

### Task 2: Own-deck routing without changing embedding shape

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/assets/taxonomy/own_archetypes_v2.json`
- Modify: `train/0044_g2_dragapult_policy_option_lora/assets/taxonomy/deck_own_archetype_mapping_v2.json`
- Modify: `train/0044_g2_dragapult_policy_option_lora/own_archetype.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_own_archetype_taxonomy_v2.py`

**Interfaces:**
- Consumes: immutable deck hashes from Task 1.
- Produces: exact mappings `068→3` and `069→3`; class count remains 29 and embedding width remains 16.

- [ ] **Step 1: Add failing exact-routing assertions**

```python
assert len(v2.mappings) == 69
assert mapping["068"] == mapping["069"] == names["alakazam"] == 3
assert v2.class_count == 29
```

- [ ] **Step 2: Run the taxonomy test and verify it fails at 67 mappings**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_own_archetype_taxonomy_v2.py`

- [ ] **Step 3: Append the two mappings and class memberships**

Use `old_archetype_id: 3`, `archetype_id: 3`, `decision: "retain_parent_alakazam"`, and record 068 as the user exact list and 069 as its one-card trainer allocation variant.

- [ ] **Step 4: Generalize the contiguous mapping validator**

Require mapping IDs to equal `001..len(mappings)` rather than hard-coded 001–067, while retaining exact registry hash verification and declared-class membership equality.

- [ ] **Step 5: Run taxonomy and policy-loader tests**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_own_archetype_taxonomy_v2.py train/0044_g2_dragapult_policy_option_lora/tests/test_external_focal_deck.py`

### Task 3: Freeze Benchmark V2 opponent semantics

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/evaluation/benchmark_v2_schedule.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/evaluation/render_g2_candidate_gate.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py`

**Interfaces:**
- Consumes: registry roles and expanded routing metadata.
- Produces: the same 2,048 Core-16 opponent jobs and the same frozen taxonomy/mapping identity fields used by existing Benchmark V2 reports.

- [ ] **Step 1: Add a failing test that asserts no 068/069 opponent job exists**

```python
schedule = materialize(PROJECT_ROOT, focal_deck_id="068", focal_deployment_identity="a" * 64)
assert {job["opponent_deck_id"] for job in schedule["jobs"]} <= {f"{i:03d}" for i in range(1, 68)}
assert "068" not in {job["opponent_deck_id"] for job in schedule["jobs"]}
assert "069" not in {job["opponent_deck_id"] for job in schedule["jobs"]}
```

- [ ] **Step 2: Run the schedule test and verify the expanded mapping leaks into opponent membership**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py`

- [ ] **Step 3: Filter schedule members by the `training` role and pin old identity hashes**

Use frozen values `taxonomy_sha256=9d9bc5cb95d253731270b064d27279448e1a099e0f548dfeb6bc8215bd891f60` and `mapping_sha256=6b7d97027d465031344d6851f8826fcf3955e2a9cc9fbd8ba49ff32fd6cb5120` in Benchmark V2 schedule payloads. This keeps old G3 reports pair-compatible after the focal-only append.

- [ ] **Step 4: Restrict reporting taxonomy/art to training-role opponent decks**

Filter `_meta_taxonomy()` and `_deck_representative_art()` to IDs with the `training` role so opponent detail tables continue to describe the frozen 001–067 pool.

- [ ] **Step 5: Run Benchmark V2 schedule/report regression tests**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_report.py train/0044_g2_dragapult_policy_option_lora/tests/test_g2_candidate_gate.py`

### Task 4: 068/069 G3 versus G4 U57 CUDA-2048 report

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/evaluation/benchmark_v2_068_069_g3_vs_g4_u57.py`
- Generate: `docs/evaluation/combat_mat/benchmark_v2/0044_g3_vs_g4_u57_068_069_alakazam_core16_policy0809_cuda2048_v2/index.html`
- Generate: `docs/evaluation/combat_mat/benchmark_v2/0044_g3_vs_g4_u57_068_069_alakazam_core16_policy0809_cuda2048_v2/manifest.json`

**Interfaces:**
- Consumes: registered exact decks, Champion-G3 U110, G4 U57, and frozen Benchmark V2 schedule.
- Produces: two paired deck comparisons with aggregate, per-Meta, per-opponent-deck, game-count, seed, card-art, and deployment-identity evidence.

- [ ] **Step 1: Write controller preflight and identity tests**

Assert exact IDs `(068, 069)`, parent relationship, one-card count delta, own archetype 03, G3/U57 checkpoint hashes, Policy-0809 opponent identity, and unused output paths.

- [ ] **Step 2: Run preflight without GPU**

Run: `python3 -m train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_068_069_g3_vs_g4_u57`

- [ ] **Step 3: Run four official CUDA-2048 arms sequentially**

Run: `python3 -m train.0044_g2_dragapult_policy_option_lora.evaluation.benchmark_v2_068_069_g3_vs_g4_u57 --launch`

Expected: 8,192 terminal games, zero error/unfinished, no focal/opponent storage sharing, and matched common-random job identities within each deck pair.

- [ ] **Step 4: Publish incremental HTML after each completed deck pair**

The index and per-deck pages must show Champion-G3 WR, G4 U57 WR, delta, 95% interval, first/second-player splits, per-Meta and exact opponent rows with game counts and card thumbnails.

- [ ] **Step 5: Run the complete 0044 targeted regression suite and inspect git diff**

Run: `PYTHONPATH=. python3 -m pytest -q train/0044_g2_dragapult_policy_option_lora/tests/test_assets.py train/0044_g2_dragapult_policy_option_lora/tests/test_own_archetype_taxonomy_v2.py train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_schedule.py train/0044_g2_dragapult_policy_option_lora/tests/test_external_focal_deck.py train/0044_g2_dragapult_policy_option_lora/tests/test_benchmark_v2_report.py`

Run: `git diff --check && git status --short`
