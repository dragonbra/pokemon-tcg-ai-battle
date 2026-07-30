# 0022 League Training Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained 0022 framework that binds the immutable 0019 Epoch 13 Foundation, discovers pluggable exact-deck packages, and creates versioned decoder-only League state without duplicating Frozen baseline weights.

**Architecture:** `train/0022_league_training/` owns a frozen copy of the 0019 model/feature contract and loads only the archived model weights as immutable data provenance. Tracked deck plugins contain identity and exact 60-card lists; mutable Live decoder/value checkpoints live under the strict `rl_runs/0022_league_training/versions/<version>/` boundary. A fail-closed CLI validates plugins, verifies the Foundation, initializes a League snapshot, and audits decoder-only checkpoints before any official-engine PPO run is authorized.

**Tech Stack:** Python 3.11, PyTorch, standard-library `argparse`/`dataclasses`/`unittest`, existing `rl_environment.runs` version layout.

## Global Constraints

- Never modify `engine/source/`; official-engine runtime remains the only source of gameplay evidence.
- `train/0022_league_training/` must not import executable code from another numbered training project.
- Bind exactly archive asset `0019-0730-epoch13`, Epoch 13, SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- Source identity is provenance only in 0022 deployment; actor-visible `source_id` is fixed to neutral `0`.
- A deck plugin is exact-deck conditioned and must contain exactly 60 positive integer card IDs.
- Mutable weights belong in a new strict `rl_runs/0022_league_training/versions/V<n>_<tag>/`; never overwrite a used version.
- Checkpoints are model-only and must reject optimizer, scheduler, scaler, RNG, replay, or rollout state.
- Formal BC/value/PPO training must use W&B online unless a version status records why it is disabled.

---

### Task 1: Freeze The 0019 Foundation Contract

**Files:**
- Create: `train/0022_league_training/foundation/contract.py`
- Create: `train/0022_league_training/foundation/model_source/*.py`
- Create: `train/0022_league_training/assets/card_ontology.json`
- Create: `train/0022_league_training/foundation.py`
- Test: `train/0022_league_training/tests/test_foundation.py`

**Interfaces:**
- Produces: `load_foundation(device) -> tuple[SourceConditionedR15Policy, FoundationIdentity]`
- Produces: `neutral_source_id(batch_size, device) -> Tensor`

- [x] Copy the immutable 0019 model source bundle and ontology into 0022, excluding caches.
- [x] Write a failing test that rejects a changed archive SHA and verifies 17,756,162 parameters.
- [x] Implement strict archive identity, Epoch 13, schema, and state-dict loading.
- [x] Verify `source_id` is always zero in the 0022 deployment helper.
- [x] Run `python3 -m unittest -v train.0022_league_training.tests.test_foundation`.

### Task 2: Define Pluggable Deck Packages

**Files:**
- Create: `train/0022_league_training/deck/README.md`
- Create: `train/0022_league_training/deck/.gitkeep`
- Create: `train/0022_league_training/decks.py`
- Test: `train/0022_league_training/tests/test_decks.py`

**Interfaces:**
- Produces: `DeckPlugin`, `DeckRole`, `load_deck_plugins(root)`, `write_catalog_snapshot(path, plugins)`.
- Consumes: directories `deck/<deck_id>/manifest.json` and `deck/<deck_id>/deck.csv`.

- [x] Write tests for exact 60-card validation, ASCII snake-case IDs, unique IDs/hashes, explicit Frozen/Live role, provenance fields, and fail-closed unknown files.
- [x] Implement deterministic plugin discovery and SHA-256 identity.
- [x] Define `foundation` as the only decoder reference allowed to omit a checkpoint.
- [x] Document the exact package layout and a complete manifest example.
- [x] Run `python3 -m unittest -v train.0022_league_training.tests.test_decks`.

### Task 3: Implement Decoder-Only Checkpoints

**Files:**
- Create: `train/0022_league_training/decoder.py`
- Test: `train/0022_league_training/tests/test_decoder.py`

**Interfaces:**
- Produces: `DECODER_COMPONENTS`, `extract_decoder_state(model)`, `save_decoder_checkpoint(...)`, `load_decoder_checkpoint(...)`.
- Checkpoint payload: schema, Foundation SHA, deck ID/hash, policy role/version, update, decoder/value tensors only.

- [x] Write tests that decoder extraction contains exactly pointer key/query, option bias, decoder init, GRU decoder, stop head and optional value head.
- [x] Test that two initialized deck decoders do not share storage.
- [x] Test atomic model-only save/load and rejection of optimizer/RNG/unknown tensor keys.
- [x] Implement strict Foundation/deck hash checks and return a load audit.
- [x] Run `python3 -m unittest -v train.0022_league_training.tests.test_decoder`.

### Task 4: Initialize Immutable League Versions

**Files:**
- Create: `train/0022_league_training/league.py`
- Create: `train/0022_league_training/cli.py`
- Create: `train/0022_league_training/__main__.py`
- Test: `train/0022_league_training/tests/test_league.py`

**Interfaces:**
- Produces CLI commands `verify-foundation`, `validate-decks`, `initialize`, and `audit-version`.
- `initialize --version V<n>_<tag>` creates config/status/catalog plus Live decoder checkpoints under the standard run layout.

- [x] Test empty deck staging validation, fail-closed initialization without a focal Live deck, and refusal to reuse a version.
- [x] Test that Frozen foundation entries create catalog references but no duplicated decoder file.
- [x] Test that Live entries create independent model-only decoder/value files with immutable SHA sidecars.
- [x] Implement run allocation through `rl_environment.runs.initialize_version` and atomic status/config writes.
- [x] Run `python3 -m unittest -v train.0022_league_training.tests.test_league`.

### Task 5: Synchronize Design And Operator Documentation

**Files:**
- Modify: `experiments/0022_league_training/DESIGN.md`
- Modify: `experiments/0022_league_training/DESIGN.html`
- Create: `train/0022_league_training/README.md`

**Interfaces:**
- Documents the boundary between tracked deck plugins, versioned Live weights, baseline Frozen sentinels, and promoted Frozen snapshots.

- [x] Add the exact tensor/component checkpoint contract and filesystem topology to both authority documents.
- [x] Add copy-paste commands for foundation verification, deck validation, version initialization and audit.
- [x] State that initialization is not official-engine strength evidence and that PPO launch remains gated on a non-empty validated catalog and Frozen evaluation contract.
- [x] Run focused 0022 tests, `compileall`, and `git diff --check`.
- [x] Commit only 0022 framework files; preserve unrelated working-tree changes.
