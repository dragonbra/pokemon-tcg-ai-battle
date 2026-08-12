# 0043 Own Archetype Taxonomy V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit all 65 exact decks, assetize an append-only Own Archetype Taxonomy V2, and create a self-contained Champion-G2-Seed whose expanded own-strategy embeddings preserve Champion-G1 at update zero.

**Architecture:** `train/0043_champion_league_rl/assets/` owns the exact 65-deck training registry, V1/V2 taxonomy assets, exact deck mapping, and immutable G2 seed artifacts. Project-local taxonomy, migration, runtime, and identity modules resolve vocabulary from checkpoint metadata; the opponent Meta vocabulary/head remains independently fixed at 15 classes. `experiments/0043_champion_league_rl/` owns the synchronized design and full audit report.

**Tech Stack:** Python 3.11, PyTorch, JSON, CSV, SHA-256, pytest.

## Global Constraints

- Champion-G1 artifacts, taxonomy metadata, and effective identity remain byte-for-byte unchanged.
- Existing Own Archetype IDs `0..14` retain their V1 meaning; V2 additions are append-only.
- Every known deck `001..065` resolves through exact content hash mapping; trigger rules are fallback-only.
- Both policy and value own embeddings retain width `16`; new rows copy declared migration-parent rows.
- Opponent Meta classifier vocabulary, 15-way logits, weights, and supervision are unchanged.
- PPO rollout size, optimizer settings, losses, reward, LoRA, action schema, and FrozenMeta256-V1 remain unchanged.
- G2 Seed uses a new optimizer; no optimizer state is migrated or stored.
- No formal RL update may start unless the taxonomy, G1 preservation, G2 loading, and zero-step parity gates pass.

---

### Task 1: Freeze Sources and Audit All 65 Decks

**Files:**
- Create: `train/0043_champion_league_rl/assets/decks/definitions/056/deck.csv` through `065/deck.csv`
- Modify: `train/0043_champion_league_rl/assets/decks/registry.json`
- Create: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.md`

**Interfaces:**
- Consumes: root `FrozenPool_65_decks_2026-08-12.zip`, official card CSV, immutable 001–055 registry.
- Produces: exact 65-row training deck registry and one explicit strategic decision per deck.

- [ ] Verify ZIP manifests, exact 60-card content, deterministic hashes, and immutable equality for 001–055.
- [ ] Copy only 056–065 into project-local numeric definitions and append their registry records.
- [ ] Record evolution, resource, Prize/tempo, board-allocation, sequencing, and decision for every deck.
- [ ] Prove FrozenMeta256-V1 still references exactly the historical 55 decks.

### Task 2: Append-Only Taxonomy and Exact Mapping Assets

**Files:**
- Create: `train/0043_champion_league_rl/assets/taxonomy/own_archetypes_v1.json`
- Create: `train/0043_champion_league_rl/assets/taxonomy/own_archetypes_v2.json`
- Create: `train/0043_champion_league_rl/assets/taxonomy/deck_own_archetype_mapping_v2.json`
- Create: `train/0043_champion_league_rl/own_archetype.py`
- Test: `train/0043_champion_league_rl/tests/test_own_archetype_taxonomy_v2.py`

**Interfaces:**
- Produces: `OwnArchetypeVocabulary.load_version(version)`, exact hash resolution, dynamic `class_count`, deterministic taxonomy/mapping hashes, and fallback-only heuristic classification.

- [ ] Write failing tests for 65 unique mappings, old-ID identity, append-only IDs, deterministic hashes, exact hash matching, and unknown-deck fallback.
- [ ] Implement registry validation without any own-vocabulary magic number.
- [ ] Assert own and opponent taxonomy types/assets cannot be substituted for each other.
- [ ] Run the focused taxonomy tests to PASS.

### Task 3: G1 to G2 Seed Tensor Migration

**Files:**
- Create: `train/0043_champion_league_rl/migrate_g2_seed.py`
- Create: `train/0043_champion_league_rl/assets/taxonomy/migrations/v1_to_v2.json`
- Create: `train/0043_champion_league_rl/assets/policies/definitions/champion_g002_seed/source_update_000000.pt`
- Create: `train/0043_champion_league_rl/assets/policies/definitions/champion_g002_seed/model.bin`
- Create: `train/0043_champion_league_rl/assets/policies/manifests/champion_g002_seed.json`
- Modify: `train/0043_champion_league_rl/assets/policies/registry.json`
- Test: `train/0043_champion_league_rl/tests/test_g2_seed_migration.py`

**Interfaces:**
- Consumes: immutable G1 FP32 model-only delta and portable FP16 artifact.
- Produces: expanded FP32 training seed and FP16-storage/FP32-runtime deployment seed with effective hashes and no optimizer state.

- [ ] Snapshot all G1 artifact hashes before migration.
- [ ] Expand every tensor named by the migration manifest; preserve rows `0..14` exactly and parent-copy every appended row.
- [ ] Reject random initialization, embedding-width changes, missing embedding tables, unrelated tensor changes, optimizer fields, and metadata/hash disagreement.
- [ ] Register G2 Seed as a new immutable candidate identity, never as G1.

### Task 4: Version-Aware Runtime, Checkpoint, and Export Loading

**Files:**
- Modify: `train/0043_champion_league_rl/semantic_runtime/deployment/compound_inference.py`
- Modify: `train/0043_champion_league_rl/runtime.py`
- Modify: `train/0043_champion_league_rl/policy_identity.py`
- Test: `train/0043_champion_league_rl/tests/test_g2_seed_runtime.py`

**Interfaces:**
- Produces: strict V1/G1 and V2/G2 construction from checkpoint-declared taxonomy class count/hash; opponent Meta remains 15-wide.

- [ ] Make own embedding row count constructor-driven while keeping width `16` and the policy MLP input width unchanged.
- [ ] Validate checkpoint taxonomy identity before strict-load and exact-deck lookup.
- [ ] Keep G1 loader behavior and hashes unchanged; add G2 Seed loader/materializer.
- [ ] Prove no active training/CPU/CUDA/export path assumes 15 own rows.

### Task 5: Zero-Step Parity, New Optimizer, and Readiness Gate

**Files:**
- Create: `train/0043_champion_league_rl/parity_g2_seed.py`
- Create: `train/0043_champion_league_rl/tests/test_g2_seed_parity.py`
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/OWN_ARCHETYPE_TAXONOMY_V2_AUDIT.md`

**Interfaces:**
- Produces: tensor and synthetic-inference parity audit for all old mappings, parent-row parity for every new class, fresh-optimizer initialization proof, and explicit RL readiness result.

- [ ] Compare G1/G2 adapter outputs, values, step logits/probabilities, and greedy actions under identical old-deck inputs.
- [ ] Assert exact FP32 equality and deployment-effective equality under the repository conversion contract.
- [ ] Initialize the approved optimizer groups from G2 model parameters without loading optimizer state.
- [ ] Run focused and full 0043 tests; record every command/result and any unrelated pre-existing failure.
- [ ] Mark readiness YES only when every hard gate passes; do not launch training.
