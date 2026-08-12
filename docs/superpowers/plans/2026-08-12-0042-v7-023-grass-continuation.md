# 0042 V7 023 Grass-Deck Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop formal V6 only after update 70 and its Frozen-0809 CUDA-2048 evaluation complete, then start an unbounded formal V7 PPO branch from V6 update 70 using exact focal deck 023 (`Hydrapple ex / Meganium`).

**Architecture:** Use the existing per-run focal exact-deck interface; do not alter actor-visible schemas, network structure, optimizer hyperparameters, opponent pool, or Frozen policy identity. Strict-load the V6 update-70 model-only checkpoint into a fresh V7 optimizer and fresh on-policy rollout stream, while recomputing the focal own-archetype from exact deck 023 (class 6, `festival_lead`, because trigger card 93 precedes the grass-evolution triggers in the immutable priority taxonomy).

**Tech Stack:** Python 3, PyTorch PPO Protocol V2, resident official CUDA engine, W&B online, unittest.

## Global Constraints

- V6 stops only after `update-000070.pt` exists and `artifact/frozen_results/core-update-000070.json` plus the update-70 eval metrics are complete.
- V7 version is `V7_hydrapple_ex_meganium_023`; no V6 artifact is overwritten or appended.
- V7 source is V6 `checkpoint/update-000070.pt`; optimizer, rollout, and RNG stream start fresh and V7 update numbering starts at 0.
- Focal deck is the exact 60-card catalog-023 payload, deck ID `hydrapple_ex_meganium_415af0a5541c`, exact-deck SHA256 `415af0a5541c9c046b4a80abc365dbf91fc693b8546e49796fec6463a181c53d`, own-archetype class 6 (`festival_lead`) under the current priority-ordered taxonomy.
- All V6 training parameters remain unchanged: 256 games/update, 256 trajectory games/update, 256 CUDA lanes, PPO minibatch 2048, forward microbatch 1024, 3 epochs, decoder/allocation LR `2e-5`, policy-adapter LR `4e-5`, Value LR `1e-4`, and evaluation every 10 updates.
- Opponents remain independent, immutable full `Policy-0809` over the unchanged neutral 55-deck pool and exact-deck routing contract.
- Periodic evaluation remains `kaggle_fp16_storage_fp32_runtime_v1` Frozen-0809 CUDA-2048 with candidate and opponent identity audits required to PASS.
- V7 has no `--updates` limit and stops only through its own `artifact/STOP_REQUESTED` update-boundary sentinel.
- Official `engine/source/` remains read-only; no automatic Promote decision is made.

---

### Task 1: Close V6 exactly at the update-70 evaluated boundary

**Files:**
- Create at runtime: `rl_runs/0042_full_model_design/versions/V6_dragapult_ex_0042_v6/artifact/STOP_REQUESTED`
- Verify: `rl_runs/0042_full_model_design/versions/V6_dragapult_ex_0042_v6/checkpoint/update-000070.pt`
- Verify: `rl_runs/0042_full_model_design/versions/V6_dragapult_ex_0042_v6/artifact/frozen_results/core-update-000070.json`
- Verify: `rl_runs/0042_full_model_design/versions/V6_dragapult_ex_0042_v6/artifact/training_summary.json`

**Interfaces:**
- Consumes: V6 runner ordering: checkpoint save, cadence evaluation, metrics/status flush, then stop-sentinel check.
- Produces: immutable update-70 model-only source and a clean `stopped_by_request` V6 terminal record.

- [ ] Wait for `update-000070.pt` and only then atomically create `artifact/STOP_REQUESTED`.
- [ ] Wait for the trainer and watchdog to exit cleanly; require status `stopped_by_request`, final update 70, and no error.
- [ ] Require update-70 Frozen result to contain 2,048 terminal games, zero errors/unfinished/fallback, Policy-0809 opponent audit PASS, candidate deployment audit PASS, and `eval/checkpoint_update = 70` in canonical metrics.
- [ ] Verify the checkpoint sidecar and recomputed SHA-256 match and that updates 0..70 remain retained.

### Task 2: Validate deck 023 as the V7 focal identity

**Files:**
- Modify: `train/0042_full_model_design/tests/test_project_identity.py`
- Create: `train/0042_full_model_design/focal_decks/hydrapple_ex_meganium_023/deck.csv`
- Create: `train/0042_full_model_design/focal_decks/hydrapple_ex_meganium_023/manifest.json`
- Preserve as provenance: `train/0042_full_model_design/league/decks/023_hydrapple_ex_meganium/`

**Interfaces:**
- Consumes: existing `RunConfig` custom-focal interface and `OwnArchetypeVocabulary.classify_own_deck`.
- Produces: regression proof that exact deck 023 is routed as the focal deck and changes own-archetype conditioning to class 6 without changing the model schema.

- [ ] Copy the immutable catalog-023 60-card payload into a focal package whose manifest adds the required display/source fields and immutable back-reference; require the same exact-deck hash.
- [ ] Add a V7 configuration test with the focal-package path, exact ID, hash, display name, and frozen-catalog provenance.
- [ ] Assert 60 cards, recomputed exact-deck hash equality, focal job routing equality, and priority-taxonomy own-archetype class 6 (`festival_lead`).
- [ ] Run `python3 -m unittest train.0042_full_model_design.tests.test_project_identity` and require PASS.
- [ ] Run focused candidate-deployment and Policy-0809 contract tests and require PASS.

### Task 3: Synchronize the authoritative 0042 design

**Files:**
- Modify: `experiments/0042_full_model_design/DESIGN.md`
- Modify: `experiments/0042_full_model_design/DESIGN.html`

**Interfaces:**
- Consumes: verified V6 U70 checkpoint identity and verified V7 runtime configuration.
- Produces: authoritative documentation of the focal deck/archetype transition and unchanged model/training/Frozen semantics.

- [ ] Replace the current-run statement with V7 exact deck 023, exact hash, own-archetype class 6, and V6 U70 lineage.
- [ ] State explicitly that strict-loading the full checkpoint preserves learned actor/Value/adapters/allocation weights, while the focal exact-deck-derived own-archetype input changes from class 0 to class 6.
- [ ] Record fresh optimizer/on-policy data, unchanged hyperparameters/opponent pool/evaluation cadence, and unbounded manual-stop duration.
- [ ] Keep V6 U70 as immutable provenance rather than relabeling it as V7 or Policy-0809.

### Task 4: Launch and verify unbounded formal V7

**Files:**
- Create through the formal runner: `rl_runs/0042_full_model_design/versions/V7_hydrapple_ex_meganium_023/{artifact,checkpoint,tensorboard,wandb}`

**Interfaces:**
- Consumes: V6 `update-000070.pt`, exact focal deck 023, existing full Policy-0809 resolver, and unchanged V6 CLI hyperparameters.
- Produces: a healthy unbounded V7 W&B run with V7 update 0 baseline followed by PPO updates.

- [ ] Confirm the V7 version directory and evaluation destination are unused, no trainer remains active, W&B authentication works, CUDA assets exist, disk threshold passes, and the V6 U70 sidecar matches.
- [ ] Launch through `monitor_training` with the same watchdog thresholds, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, explicit unchanged hyperparameters, V6 U70 initial checkpoint, exact deck-023 focal flags, `--launch-formal`, and no `--updates` argument.
- [ ] Require V7 training config to record `max_updates = null`, deck 023 exact identity, own-archetype class 6 in candidate materialization, V6 U70 source checkpoint identity, full Policy-0809 opponent PASS, and unchanged 55-deck schedule.
- [ ] Observe update-0 Frozen CUDA-2048 and update 1; require 2,048 terminal eval games, 0 error/unfinished/fallback, candidate deployment PASS, lane routing PASS, finite PPO metrics, 100% first-epoch coverage, checkpoint save, and W&B online status.
- [ ] Leave both watchdog and trainer running; report their PIDs, log path, W&B URL, current update, update-0 Frozen result, and next evaluation milestone.
