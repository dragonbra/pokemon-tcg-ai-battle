# Rmy Ogerpon Frozen Default Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote Rmy's replay-audited Teal Mask Ogerpon ex exact deck into the Frozen pool and give it an isolated 0023 default decoder branch for future RL.

**Architecture:** Frozen Arena grows from 50 to 51 immutable exact-deck identities while retaining the shared frozen 0019 inference policy. The self-contained 0023 catalog receives the same deck as a Live/frozen-anchor identity; a new V8 initialization inherits all matching V7 update-71 decoders and materializes a Foundation-initialized update-0 model-only decoder/value checkpoint only for the new Ogerpon identity.

**Tech Stack:** Python 3.11, PyTorch model-only checkpoints, official evaluation catalog and engine runtime, unittest.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve the exact 60-card hash `e40278fd83d971c280b0fb9cd14d5e45cfe63a45c647741eb937440b4b19be34`.
- Do not mutate V7 or append new metrics to it.
- The new decoder must be deck-local, model-only, identity-checked, and optimizer-free.
- Frozen evaluation and RL initialization must record distinct, auditable policy semantics.

---

### Task 1: Frozen Pool Promotion

**Files:**
- Modify: `evaluation/configs/frozen.json`
- Modify: `evaluation/arena/frozen/manifest.json`
- Modify: `evaluation/frozen.py`
- Modify: `tests/test_evaluation_frozen_assets.py`
- Modify: `tests/test_evaluation_frozen_catalog.py`

**Interfaces:**
- Consumes: existing lightweight `evaluation/arena/frozen/rmy_teal_mask_ogerpon_001/`.
- Produces: a 51-entry immutable Frozen catalog with pool ID `0019_foundation_51_exact_decks_v4`.

- [ ] Add the Ogerpon catalog entry with representative Card ID `96`.
- [ ] Update strict loader constants and tests from 50/v3 to 51/v4.
- [ ] Run Frozen asset and catalog tests.

### Task 2: 0023 Deck Snapshot and Restart Compatibility

**Files:**
- Create: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/deck/rmy_teal_mask_ogerpon_001/deck.csv`
- Create: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/deck/rmy_teal_mask_ogerpon_001/manifest.json`
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/training/league_run.py`
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: V7 update-71 checkpoints for matching deck IDs and exact hashes.
- Produces: `resolve_project_version_checkpoint_map()` results where new current decks resolve to `None` with an explicit `foundation_default` audit.

- [ ] Add the exact deck and replay provenance to the self-contained 0023 catalog.
- [ ] Write a failing test showing a new current deck may be Foundation-initialized while prior decks remain mandatory.
- [ ] Permit only current-minus-prior additions; continue rejecting disappeared or hash-changed prior decks.
- [ ] Run deck and restart-resolution tests.

### Task 3: Materialize V8 Default Decoder

**Files:**
- Create: `rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V8_add_rmy_ogerpon_default/`

**Interfaces:**
- Consumes: V7 update-71 decoders for 50 prior decks and Foundation state for Rmy Ogerpon.
- Produces: 51 model-only `checkpoint/decks/*.pt` assets and an immutable catalog/config/status audit.

- [ ] Allocate V8 with all four required version directories unused.
- [ ] Resolve V7 inheritance and initialize V8.
- [ ] Audit the Ogerpon checkpoint identity, update `0`, value head, SHA sidecar, and absence of optimizer state.

### Task 4: Documentation and Verification

**Files:**
- Modify: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/DESIGN.md`
- Modify: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/DESIGN.html`

**Interfaces:**
- Consumes: final pool IDs, counts, V8 initialization audit, and checkpoint identity.
- Produces: synchronized authoritative design documentation.

- [ ] Record the 51-deck pool, V8 inheritance boundary, and Ogerpon Foundation-default decoder.
- [ ] Run focused 0023, Frozen catalog, model-only checkpoint, and official-engine smoke tests.
- [ ] Verify no long-running training process was started.
