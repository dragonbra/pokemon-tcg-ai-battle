# 0023 Mega Lopunny ex / Mega Froslass ex League Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a continuously running 0023 League PPO experiment focused on `mega_lopunny_ex_mega_froslass_ex_002`, inheriting every existing deck's latest complete 0022 Live decoder and initializing only new decks from the immutable 0019 Foundation.

**Architecture:** Freeze one self-contained copy of the 0019 Foundation encoder and route it to 50 isolated decoder/value pairs. The 48 decks present in 0022 V11 load their exact update-81 model-only assets; `mega_lopunny_ex_001` and the focal `_002` deck start from Foundation because they were added after V11. Each PPO trainer anchors reference KL to its own 0023 initialization state, uses only that deck's actor-relative trajectory, and publishes bounded model-only checkpoints while Frozen and Live opponents remain separate views.

**Tech Stack:** Python 3.11, PyTorch PPO, official engine runtime workers, CUDA resident batching, TensorBoard, W&B online, JSON model-only checkpoints.

## Global Constraints

- Project ID is `0023_mega_lopunny_ex_mega_froslass_ex_002_league_training`; train, experiment, and run roots use the same ID.
- `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/` must not import executable code from another numbered training project.
- Official `engine/source/` is read-only; every rollout and Frozen/Live evaluation uses the official engine runtime.
- Focal deck is exactly `mega_lopunny_ex_mega_froslass_ex_002`, deck SHA-256 `7fb1536b191c0e0ff5f8819c674ba784e9e0b77927b58b3faea423db66b90e8b`.
- Existing 0022 V11 decks inherit their latest complete update-81 decoder/value; decks absent from V11 initialize from 0019 Foundation.
- Reference KL is deck-local and fixed to each deck's 0023 initialization state, not silently reset at every PPO update.
- Training has no duration-based automatic stop. It stops only on explicit SIGINT/SIGTERM, a fail-closed error, CUDA failure, or the configured SSD low-water/cap guard.
- Checkpoints remain model-only and bounded; optimizer, scheduler, RNG, rollout buffers, traces, and replays are never serialized.
- `training_metrics.jsonl` is canonical, followed by TensorBoard and W&B online mirroring in `dragon_bra/pokemon-tcg-policy-learning`.
- Sampled rollout metrics remain diagnostics; checkpoint strength requires fixed-seed, balanced-seat Frozen official-engine evaluation.

---

### Task 1: Freeze The 0023 Project Boundary

**Files:**
- Create: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/`
- Create: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/DESIGN.md`
- Create: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/DESIGN.html`
- Create: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/manifest.json`

**Interfaces:**
- Consumes: tracked 0022 source files, 0019 immutable Foundation assets, and the current 50 exact deck identities.
- Produces: an independently importable 0023 Python package and authoritative design contract.

- [ ] Copy only tracked source, tests, ontology, and exact deck identity files from 0022; omit caches and runtime output.
- [ ] Change `PROJECT_ID`, schema strings, prose, focal constants, and W&B tags to 0023-specific values.
- [ ] Make curated deck focal selection a named `FOCAL_DECK_ID` constant equal to `mega_lopunny_ex_mega_froslass_ex_002`.
- [ ] Run `rg 'train\.0022|from .*0022|import .*0022' train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training` and require no executable cross-project imports.

### Task 2: Implement Auditable Live Inheritance

**Files:**
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/training/league_run.py`
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/league.py`
- Test: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: `DeckPlugin`, V11 `league_catalog.json`, V11 `checkpoint/live/<deck>/update-000081.pt`, and Foundation SHA.
- Produces: `resolve_initial_checkpoint_map(plugins) -> (dict[str, str | None], dict[str, dict[str, object]])` and an immutable initialization audit.

- [ ] Add a failing test with one inherited deck and one new deck; assert inherited resolves to update 81 and new resolves to `None`/Foundation.
- [ ] Add identity validation for foundation SHA, deck ID, exact deck SHA, policy role, update number, SHA sidecar, and V11 catalog membership.
- [ ] Refuse partial or ambiguous inheritance: a V11 catalog member without an update-81 asset is a launch error, not a Foundation fallback.
- [ ] Record each deck's source project/version/update/path/SHA or explicit Foundation sentinel in `artifact/initialization_audit.json` and `training_config.json`.
- [ ] Load the 50 routed policies from that exact map and create each fixed reference after loading its inherited state.

### Task 3: Implement Continuous, Graceful League Training

**Files:**
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/cli.py`
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/training/run.py`
- Modify: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/training/league_run.py`
- Test: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/tests/test_training.py`

**Interfaces:**
- Consumes: `LeagueTrainingConfig`, 50 policies, official-engine rollout episodes, and `StopRequest`.
- Produces: a run loop without a time deadline and update-boundary graceful stopping.

- [ ] Replace `duration_hours` as a termination condition with elapsed-hours telemetry only; expose no misleading `20h` default stop.
- [ ] Install SIGINT/SIGTERM handling and complete the active rollout/update/checkpoint before recording `completed_stop_requested`.
- [ ] Keep SSD launch minimum, runtime low-water, and per-version cap as fail-closed stop conditions.
- [ ] Preserve 512-game balanced focal schedule, Frozen/Live views, per-deck actor trajectory filtering, isolated optimizers, and bounded checkpoint retention.
- [ ] Add focal-specific rollout/Frozen/Live/probe metrics under stable namespaces and record source-policy versus checkpoint update.

### Task 4: Verify Initialization And Runtime Contracts

**Files:**
- Test: `train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/tests/`
- Create at runtime: `.tmp/evaluation/0023_mega_lopunny_froslass_smoke/`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: unit-test, decoder-parity, CUDA-memory, and official-engine smoke evidence.

- [ ] Run the full 0023 unit suite and the shared evaluation asset tests.
- [ ] Materialize a disposable smoke version and audit all 50 decoder/value identities.
- [ ] Compare greedy logits/actions before and after save/load for inherited and Foundation-initialized examples; require exact equality.
- [ ] Run a small CUDA official-engine League canary with both Frozen and Live views, balanced seats, 100% completion, and zero errors.
- [ ] Record peak allocated/reserved VRAM, GPU utilization, episodes/s, engine selections/s, free disk, and estimated checkpoint growth.

### Task 5: Publish The Formal V1 Run

**Files:**
- Create: `rl_runs/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/versions/V1_mega_lopunny_ex_mega_froslass_ex_002_continuous_league/`
- Create: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/decisions/001_initialization_and_launch.md`

**Interfaces:**
- Consumes: all smoke gates and clean tracked code.
- Produces: one immutable repository version mapped to one stable W&B run.

- [ ] Commit code, tests, plan, DESIGN.md/DESIGN.html, and launch decision before starting the formal run.
- [ ] Confirm all four V1 output roots are unused and W&B is importable from the training interpreter.
- [ ] Launch V1 detached with an explicit PID/log/control record and no duration stop.
- [ ] Verify update 1 publishes 50 model-only decoder/value checkpoints, local metrics, TensorBoard events, W&B URL, and running status.
- [ ] Verify the focal deck receives the intended main-view trajectory and every Live deck only updates from its own real decisions.

### Task 6: Continuous Operations And Checkpoint Selection

**Files:**
- Update at runtime: `artifact/status.json`, `artifact/training_metrics.jsonl`, `artifact/training_summary.json`
- Create when evaluated: `experiments/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/evaluation/V<n>_<tag>.html`

**Interfaces:**
- Consumes: the running V1 process and model-only checkpoints.
- Produces: continuous health evidence and auditable candidate checkpoints.

- [ ] Monitor process liveness, CUDA allocation/errors, rollout completion, PPO finiteness/KL, per-deck decision counts, W&B sync, and SSD guards.
- [ ] Treat 20 GPU hours as a durability milestone only; do not stop the process there.
- [ ] Use rolling diagnostics only to nominate checkpoints, then run fixed-seed balanced-seat Frozen official-engine evaluation before strength claims.
- [ ] Never promote `latest` over `champion` without the Frozen gate; preserve bounded snapshots needed for comparison.
- [ ] Continue until the user explicitly requests a stop or a documented fail-closed safety condition occurs.
