# 0042 V17–V21 U20 Archetype Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the active V11–V16 serialized curriculum with five missing focal archetypes, each trained for exactly 20 updates and handed off only after its U20 CUDA-2048 identity/evaluation gate passes.

**Architecture:** Keep the already-running V11–V16 controller untouched. Add a second fail-closed systemd-managed controller that waits for the immutable V16 U50 terminal checkpoint, then runs V17–V21 serially with fresh optimizer/on-policy state per version. Exact decks are copied into project-local focal-deck assets with immutable source provenance and derived own-archetype validation.

**Tech Stack:** Python 3, PyTorch PPO runtime, systemd user services, CUDA resident rollout/evaluation, JSON manifests, unittest.

## Global Constraints

- Do not modify `engine/source/`.
- V17 starts from V16 U50; V18–V21 each start from the immediately preceding version's U20 model-only checkpoint.
- Each new version uses `--updates 20`, `--eval-every 10`, and must finish its U20 CUDA-2048 evaluation before handoff.
- Preserve every V11 training parameter except focal deck, version, source checkpoint, and finite update count.
- Candidate deployment remains FP16 storage then strict FP32 runtime; Frozen opponent remains independent complete Policy-0809.
- Any missing checkpoint, identity mismatch, evaluation error, unfinished game, semantic fallback, or process failure blocks the queue.

---

### Task 1: Freeze five focal deck identities

**Files:**
- Create: `train/0042_full_model_design/focal_decks/barbaracle_cornerstone_mask_ogerpon_ex_019/{deck.csv,manifest.json}`
- Create: `train/0042_full_model_design/focal_decks/cynthia_s_garchomp_ex_roserade_013/{deck.csv,manifest.json}`
- Create: `train/0042_full_model_design/focal_decks/team_rocket_s_mewtwo_ex_spidops_021/{deck.csv,manifest.json}`
- Create: `train/0042_full_model_design/focal_decks/mega_starmie_ex_mega_froslass_ex_044/{deck.csv,manifest.json}`
- Create: `train/0042_full_model_design/focal_decks/archaludon_ex_cinderace_048/{deck.csv,manifest.json}`

**Interfaces:**
- Consumes: immutable Frozen pool deck and manifest for numbered decks 019, 013, 021, 044, and 048.
- Produces: exact-60 project-local focal decks with class IDs 7, 11, 9, 12, and 13.

- [x] **Step 1: Copy exact deck bytes and record source file SHA-256.**
- [x] **Step 2: Record exact-deck identity, display name, source, class ID, and class name.**
- [x] **Step 3: Validate all five manifests against `OwnArchetypeVocabulary` and immutable source bytes.**

### Task 2: Implement the U20 continuation controller

**Files:**
- Create: `train/0042_full_model_design/training/run_remaining_archetype_curriculum.py`
- Create: `train/0042_full_model_design/tests/test_remaining_archetype_curriculum.py`

**Interfaces:**
- Consumes: V16 U50 checkpoint validated by the existing controller contract.
- Produces: V17–V21, each terminating at update 20 with complete U20 evidence.

- [x] **Step 1: Define exact version/deck/class sequence 019→013→021→044→048.**
- [x] **Step 2: Implement `validate_u20(version: str) -> Path` checking terminal state, checkpoint sidecar, 2048 unique valid games, candidate PASS, and exact Policy-0809 opponent identity.**
- [x] **Step 3: Implement the preserved training command with `--updates 20`.**
- [x] **Step 4: Implement atomic waiting/running/blocked/complete state transitions.**
- [x] **Step 5: Test sequence, exact deck provenance, finite update count, and inherited hyperparameters.**

### Task 3: Synchronize project design authority

**Files:**
- Modify: `experiments/0042_full_model_design/DESIGN.md`
- Modify: `experiments/0042_full_model_design/DESIGN.html`

**Interfaces:**
- Consumes: final V17–V21 sequence and U20 handoff contract.
- Produces: authoritative human-readable training-stage record.

- [x] **Step 1: Document the five new focal identities and classes.**
- [x] **Step 2: Document V16 U50→V17 and subsequent U20→U20 inheritance semantics.**
- [x] **Step 3: State that optimizer and on-policy data restart for every version.**

### Task 4: Launch and verify the waiting queue

**Files:**
- Runtime state: `.tmp/training_monitor/0042_full_model_design/deck_curriculum_v17_v21/state.json`

**Interfaces:**
- Consumes: tested controller module.
- Produces: active systemd user service waiting on V16 U50 without disturbing V13–V16.

- [x] **Step 1: Run focused and project identity tests plus `git diff --check`.**
- [x] **Step 2: Start `pokemon-0042-v17-v21-u20-archetype-curriculum.service`.**
- [x] **Step 3: Verify the original V11–V16 service remains active and the new controller reports `waiting_for_u50` for V16.**
