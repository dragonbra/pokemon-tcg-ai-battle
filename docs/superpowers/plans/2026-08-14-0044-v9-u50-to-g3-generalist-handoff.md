# 0044 V9 U50 to G3 Generalist Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the fixed-Deck066 V9 run only after its durable U50 checkpoint and Benchmark V2 evidence exist, then automatically launch a new indefinite G3 run whose focal deck is seeded-random and frequency-balanced across exact decks 001–067.

**Architecture:** Extend the self-contained 0044 trainer with an explicit focal scheduling mode while retaining one resident focal model and per-lane exact-deck/own-embedding identity. A fail-closed handoff supervisor freezes the V9 U50 checkpoint hash, gracefully terminates V9, and launches a fresh V10 repository/W&B version with a new optimizer and local U0. Benchmark V2 remains a fixed Deck066 sentinel so U0 is directly comparable to V9 U50; rollout telemetry carries all 67 focal identities.

**Tech Stack:** Python 3, PyTorch PPO, CUDA Engine 2.0, official engine runtime, pytest, W&B online.

## Global Constraints

- Do not modify `engine/source/`.
- V9 may stop only after `update-000050.pt`, a PASS U50 Benchmark V2 report, and matching canonical metrics are durable.
- V10 is a new formal version and W&B run; it strict-loads V9 U50 model-only weights and initializes a fresh optimizer.
- Focal schedules cover every deck 001–067 in every 256-game update, with counts differing by at most one and a seeded random lane permutation.
- Focal deck ID and own-deck embedding ID remain per-lane/per-transition inputs; focal IDs never become model-cache or minibatch partition keys.
- Opponent policy remains immutable Champion-G2 and opponent exact decks remain uniform 001–067; PFSP remains disabled.
- Reference KL remains anchored to immutable Champion-G2, not the V9 parent checkpoint.
- All model-only checkpoints are retained; Benchmark V2 uses CUDA-2048 and Policy-0809 with common fixed seeds.
- Synchronize both `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Generalist focal scheduling contract

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v10_g3_generalist.py`

**Interfaces:**
- Consumes: `balanced_focal_schedule(seed, deck_ids=..., lanes=256)`.
- Produces: `run(..., focal_schedule_mode, evaluation_focal_deck_id)` and `_jobs(..., focal_schedule_mode)`.

- [ ] Write tests proving every 001–067 deck appears 3–4 times, lane bindings use the matching exact deck and own-archetype ID, and unsupported/mismatched scheduling modes fail closed.
- [ ] Run the focused tests and verify the new interface is absent or fixed-focal validation rejects the generalist case.
- [ ] Add explicit `fixed` and `seeded_frequency_balanced_random_v1` modes, using seed `430044711 + update` for the latter.
- [ ] Record schedule mode, all focal IDs, initialization deck, and fixed evaluation sentinel in config/status/checkpoint metadata.
- [ ] Run the focused tests and the existing V9 tests.

### Task 2: V10 G3 formal entrypoint and immutable U50 parent manifest

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v10.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/tests/test_v10_g3_generalist.py`

**Interfaces:**
- Consumes: V9 `checkpoint/update-000050.pt` plus `artifact/g3_handoff_u50.json`.
- Produces: formal version `V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2`.

- [ ] Write readiness tests for exact parent version/update/hash, all 67 focal IDs, fresh optimizer, uniform Champion-G2 opponents, no PFSP, and indefinite launch defaults.
- [ ] Implement fail-closed manifest verification and V10 CLI.
- [ ] Bind V10 local U0 and periodic Benchmark V2 to Deck066 as a named longitudinal sentinel while training all 67 focal decks.
- [ ] Run entrypoint/readiness tests with a synthetic parent manifest and checkpoint.

### Task 3: Exact-U50 stop and automatic handoff supervisor

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/training/handoff_v9_u50_to_v10.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_v9_u50_handoff.py`

**Interfaces:**
- Consumes: V9 PID, checkpoint, metrics, status, and U50 Benchmark report.
- Produces: immutable handoff manifest, graceful V9 stop, detached V10 launch, and auditable supervisor status/log.

- [ ] Test that a checkpoint without matching PASS evaluation cannot trigger a stop.
- [ ] Test exact process command validation, SHA-256 freezing, atomic manifest/status writes, and V10 launch command construction.
- [ ] Implement polling at a short interval; send SIGINT only to the validated V9 process after the complete U50 gate.
- [ ] Refuse V10 launch if U51 is already durable, V9 exits abnormally, the manifest changes, or V10 paths are non-pristine.
- [ ] Launch V10 in a new session with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and W&B online; record PID and command.

### Task 4: Design and operator documentation

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Modify: `train/0044_g2_dragapult_policy_option_lora/README.md`

**Interfaces:**
- Consumes: finalized scheduling, identity, benchmark, and handoff semantics.
- Produces: human-auditable V9→V10 phase record.

- [ ] Document V9’s exact U50 terminal gate and V10’s fresh-optimizer local-U0 boundary.
- [ ] Document one-resident-policy/per-lane-focal routing and why it does not fragment CUDA batching.
- [ ] Document Deck066 Benchmark V2 as a sentinel rather than an all-deck G3 strength aggregate.
- [ ] Document W&B run identity and relevant rollout coverage/routing metrics.

### Task 5: Verification and live arming

**Files:**
- Runtime outputs only under `rl_runs/0044_g2_dragapult_policy_option_lora/versions/`.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: an armed supervisor while V9 continues uninterrupted.

- [ ] Run all 0044 unit/contract tests without a GPU-heavy evaluation.
- [ ] Run V10 readiness in pre-parent mode and confirm it fails only because U50 does not exist yet.
- [ ] Start the handoff supervisor detached and verify its status says `waiting_for_v9_u50` with the exact V9 PID.
- [ ] Verify V9 continues to advance and has not been interrupted by implementation/tests.
- [ ] At the gate, verify V9 final U50 evidence, V10 U0 checkpoint/evaluation, V10 first completed rollout/update, W&B online URL, and no U51 V9 checkpoint.
