# 0024 Marnie's Grimmsnarl League Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and launch a self-contained 0024 continuous PPO league whose focal policy is the most-used exact Marnie's Grimmsnarl ex / Froslass deck.

**Architecture:** Copy the stable 0023 training implementation and exact-deck catalog into a new numbered project, changing only project identity and focal ownership. Initialize all 51 Live decoder/value branches from the immutable 0023 V8 catalog, except the three Mega Lopunny identities which deliberately receive physical model-only copies of 0023's final focal Mega Lopunny decoder. Frozen opponents remain the shared 0019 Foundation; the Rmy Teal Mask Ogerpon ex Live opponent is sampled at 10 times the ordinary Live-deck weight so it is trained as the secondary policy alongside focal Marnie.

**Tech Stack:** Python 3.11, PyTorch CUDA, official Pokemon TCG engine runtime, W&B, TensorBoard, model-only checkpoints.

## Global Constraints

- Never modify `engine/source/`.
- All formal rollout and evaluation games use the official engine runtime.
- `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/` is self-contained and must not import numbered-project executable code.
- The formal run writes a strictly new `rl_runs/0024_marnies_grimmsnarl_ex_froslass_001_league_training/versions/V1_from_0023_v8_marnie_focal_24h/` version.
- Record Frozen pool `0019_foundation_51_exact_decks_v4`, W&B metadata, source checkpoint SHA-256 values, and all initialization exceptions.
- On-policy rollout sampling assigns `rmy_teal_mask_ogerpon_001` weight 10 only for the Live-opponent view; every other non-focal Live opponent and every Frozen opponent has weight 1.
- Keep the training process and local monitor in the same foreground terminal session; use no competing GPU work while it runs.

---

### Task 1: Scaffold the Self-Contained 0024 Project

**Files:**
- Create: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/`
- Create: `experiments/0024_marnies_grimmsnarl_ex_froslass_001_league_training/manifest.json`
- Create: `experiments/0024_marnies_grimmsnarl_ex_froslass_001_league_training/DESIGN.md`
- Create: `experiments/0024_marnies_grimmsnarl_ex_froslass_001_league_training/DESIGN.html`

**Interfaces:**
- Consumes: stable 0023 implementation and 51 exact deck assets.
- Produces: importable `train.0024_marnies_grimmsnarl_ex_froslass_001_league_training` with `PROJECT_ID` and `FOCAL_DECK_ID`.

- [ ] **Step 1: Copy only source, Foundation, policy, rollout, tests, and exact deck assets from 0023**

Run: `cp -a train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training train/0024_marnies_grimmsnarl_ex_froslass_001_league_training`

- [ ] **Step 2: Set the focal policy identity**

```python
PROJECT_ID = "0024_marnies_grimmsnarl_ex_froslass_001_league_training"
FOCAL_DECK_ID = "marnies_grimmsnarl_ex_froslass_001"
```

- [ ] **Step 3: Write 0024 manifest and design documents**

Document the exact deck SHA, 48-member environment evidence, Frozen/Live contract, independent Live decoders, and the Mega Lopunny initialization exception.

- [ ] **Step 4: Run import and deck-catalog checks**

Run: `python3 -m train.0024_marnies_grimmsnarl_ex_froslass_001_league_training validate-decks`
Expected: 51 valid plugins and exactly one focal Marnie plugin.

### Task 2: Define Initialization Provenance

**Files:**
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/training/league_run.py`
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/tests/test_training.py`
- Create: `experiments/0024_marnies_grimmsnarl_ex_froslass_001_league_training/decisions/001_v8_inheritance_and_lopunny_override.md`

**Interfaces:**
- Consumes: 0023 V8 `artifact/league_catalog.json`, `artifact/status.json`, and model-only `checkpoint/decks/*.pt` files.
- Produces: `resolve_v8_initial_checkpoint_map(...) -> (dict[str, Path], dict[str, dict[str, object]], dict[str, object])`.

- [ ] **Step 1: Write a resolver regression test for the three Lopunny overrides**

```python
assert audit["mega_lopunny_ex_001"]["initialization"] == "cross_loaded_0023_focal_decoder"
assert audit["mega_lopunny_ex_mega_froslass_ex_001"]["source_deck_id"] == "mega_lopunny_ex_001"
assert audit["mega_lopunny_ex_mega_froslass_ex_002"]["source_deck_id"] == "mega_lopunny_ex_001"
```

- [ ] **Step 2: Implement a fail-closed V8 resolver**

Verify catalog membership, exact deck SHA for all ordinary branches, expected model-only checkpoint metadata, and the overridden decoder's source deck identity. Reject missing assets or nonzero unexpected source update.

- [ ] **Step 3: Run focused resolver tests**

Run: `python3 -m unittest -v train.0024_marnies_grimmsnarl_ex_froslass_001_league_training.tests.test_training`
Expected: PASS.

### Task 3: Add the Weighted Secondary Ogerpon Curriculum

**Files:**
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/training/run.py`
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/training/league_run.py`
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: `schedule_jobs(..., evaluation=False)` and the exact deck ID `rmy_teal_mask_ogerpon_001`.
- Produces: balanced training rollout jobs where the Ogerpon Live matchup appears 10 times for each one appearance of an ordinary Live matchup across a complete deterministic scheduling cycle.

- [ ] **Step 1: Write a sampling-contract test**

```python
assert live_counts["rmy_teal_mask_ogerpon_001"] == 10 * live_counts["dragapult_ex_001"]
assert frozen_counts["rmy_teal_mask_ogerpon_001"] == frozen_counts["dragapult_ex_001"]
```

- [ ] **Step 2: Implement a named weight map used only outside evaluation**

```python
LIVE_OPPONENT_WEIGHTS = {"rmy_teal_mask_ogerpon_001": 10}
```

Expand the non-focal Live matchup list by this integer weight before deterministic pairing. Keep `evaluation=True` unweighted and complete, so no strength metric changes denominator or silently favors Ogerpon.

- [ ] **Step 3: Record secondary-policy evidence per update**

Log Ogerpon scheduled episode count, actor decisions, PPO loss fields, source-policy update, and weight under `rollout/secondary/*` and `ppo/deck/rmy_teal_mask_ogerpon_001/*`.

- [ ] **Step 4: Run the focused scheduling tests**

Run: `python3 -m unittest -v train.0024_marnies_grimmsnarl_ex_froslass_001_league_training.tests.test_training`
Expected: PASS.

### Task 4: Initialize and Verify V1

**Files:**
- Create: `rl_runs/0024_marnies_grimmsnarl_ex_froslass_001_league_training/versions/V1_from_0023_v8_marnie_focal_24h/artifact/`
- Create: `rl_runs/0024_marnies_grimmsnarl_ex_froslass_001_league_training/versions/V1_from_0023_v8_marnie_focal_24h/checkpoint/decks/`

**Interfaces:**
- Consumes: the resolver mapping from Task 2.
- Produces: independent 0024 model-only decoder/value checkpoints and `initialization_audit.json`.

- [ ] **Step 1: Initialize the version without PPO updates**

Run: `python3 -m train.0024_marnies_grimmsnarl_ex_froslass_001_league_training initialize-v8 --version V1_from_0023_v8_marnie_focal_24h`

- [ ] **Step 2: Audit the version**

Run: `python3 -m train.0024_marnies_grimmsnarl_ex_froslass_001_league_training audit-version --version V1_from_0023_v8_marnie_focal_24h`
Expected: all 51 checkpoints model-only, isolated, and initialized with recorded source SHA-256 values.

### Task 5: Preflight and Long-Run Supervision

**Files:**
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/monitor_training.py`
- Modify: `train/0024_marnies_grimmsnarl_ex_froslass_001_league_training/README.md`

**Interfaces:**
- Consumes: V1 artifact paths and training child PID.
- Produces: heartbeats and fail-closed alerts at `.tmp/training_monitor/0024_marnies_grimmsnarl_ex_froslass_001_league_training/V1_from_0023_v8_marnie_focal_24h/`.

- [ ] **Step 1: Run official-engine smoke and PPO canary**

Run: `python3 -m train.0024_marnies_grimmsnarl_ex_froslass_001_league_training smoke-rollout --device cuda:0 --workers 4`
Expected: all games finish without engine errors.

- [ ] **Step 2: Launch the 24-hour continuous multi-decoder league**

Run the training child and monitor in one foreground terminal session with `--workers 128`, `--coalesce-ms 5`, `--games-per-update 512`, and `--min-hours 24`.

- [ ] **Step 3: Verify the first completed update**

Confirm `training_metrics.jsonl`, status `checkpoint_update >= 1`, all 51 `checkpoint/live/<deck>/update-000001.pt` files, W&B run metadata, heartbeat, and safe GPU/disk headroom.

- [ ] **Step 4: Continue foreground observation**

Watch the session with intervals below 60 seconds. On a monitor alert, stop further GPU tasks, inspect status/log/heartbeat, preserve the failed version, and report the exact failure condition before considering recovery.
