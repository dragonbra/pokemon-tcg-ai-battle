# 0043 Champion League RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, auditable 0043 opponent league with immutable deck/policy/evaluation assets, exact 128/64/64 sampling, ten-update PFSP curricula, rollout telemetry, FrozenMeta evaluation, and a manual Promote Champion V2 workflow.

**Architecture:** `train/0043_champion_league_rl/` owns executable code and imported runtime assets; `experiments/0043_champion_league_rl/` owns audit, design, decisions, and formal reports; `rl_runs/0043_champion_league_rl/versions/` owns versioned mutable run state. The project freezes the approved 0042 PPO Protocol V2 optimizer contract and changes only opponent asset routing, curriculum sampling, telemetry, and promotion governance until a separately authorized experiment changes training semantics.

**Tech Stack:** Python 3.11, PyTorch, pytest, JSON manifests, official CPU/CUDA engine runtimes, TensorBoard, W&B.

## Global Constraints

- Project ID is exactly `0043_champion_league_rl`; numbered-project executable code may not import another numbered `train/<project_id>/` package.
- Formal rollout is exactly 256 games per update; V2 mixture is exactly 128 PFSP, 64 uniform, and 64 latest frozen champion lanes.
- PFSP deck and policy choices are independent; no unapproved deck-policy compatibility gate is allowed.
- Curriculum weights refresh every 10 updates and remain immutable inside each window.
- PPO rollout and optimizer/master weights remain FP32; Kaggle-strength candidate evidence uses `kaggle_fp16_storage_fp32_runtime_v1`.
- Policy identity is resolved and audited before lane routing, batching, caching, or sharing; hybrid opponent policies hard-fail.
- Latest champion must be an immutable registered policy and Promotion remains a human decision.
- FrozenMeta256-V1 preserves the existing 55-deck composition, integer frequencies, and seed/seat semantics; new training decks do not enter it automatically.
- Every 0043 deck uses its zero-padded numeric `deck_id` as the sole canonical identity: existing decks are immutable `001`–`055`, and future additions append monotonically from `056`; names, archetypes, hashes, and historical IDs are metadata rather than routing identities.
- Official engine runtime is the only strength-evaluation source; `engine/source/` remains read-only.
- Existing user changes and historical assets are not modified or deleted.

---

### Task 1: Phase 0 Audit and Frozen Baseline Contract

**Files:**
- Create: `experiments/0043_champion_league_rl/PHASE0_AUDIT.md`
- Create: `experiments/0043_champion_league_rl/active_training_config.json`
- Create: `experiments/0043_champion_league_rl/manifest.json`
- Create: `experiments/0043_champion_league_rl/DESIGN.md`
- Create: `experiments/0043_champion_league_rl/DESIGN.html`
- Create: `experiments/0043_champion_league_rl/DECISIONS.md`

**Interfaces:**
- Consumes: the actual V22 `training_config.json`, 0042 active entrypoint, approved Policy-0809 audit, Champion-G1 archive, and 55-deck schedule.
- Produces: immutable `active_training_config_sha256`, exact source inventory, explicit unresolved items, and the project-wide architecture contract used by every later task.

- [ ] **Step 1: Write a failing audit-schema test**

Create `train/0043_champion_league_rl/tests/test_phase0_contract.py` asserting the snapshot has `games_per_update == 256`, the approved PPO group learning rates, exact trainable set, and a canonical SHA-256 recorded in the project manifest.

- [ ] **Step 2: Run the test and verify missing project files fail**

Run: `pytest -q train/0043_champion_league_rl/tests/test_phase0_contract.py`

Expected: FAIL because the 0043 files do not yet exist.

- [ ] **Step 3: Freeze the audited config and source map**

Copy the full active V22 config without retyping parameters, canonicalize it for hashing, document the active training/evaluation/opponent paths, and record that G1 is the user-designated generation anchor from `archive/pretrained/0042_champion_g1/manifest.json` rather than a new automatic promotion.

- [ ] **Step 4: Add synchronized DESIGN documents**

Document asset boundaries, tensor/model inheritance, unchanged PPO objectives, independent focal/opponent identities, 256-game curriculum flow, current Phase 1 status, and future hard gates in both Markdown and HTML.

- [ ] **Step 5: Run the focused test**

Run: `pytest -q train/0043_champion_league_rl/tests/test_phase0_contract.py`

Expected: PASS.

### Task 2: Self-Contained Immutable Asset Registries

**Files:**
- Create: `train/0043_champion_league_rl/assets.py`
- Create: `train/0043_champion_league_rl/import_assets.py`
- Create: `train/0043_champion_league_rl/assets/decks/registry.json`
- Create: `train/0043_champion_league_rl/assets/decks/definitions/<deck_id>/deck.csv`
- Create: `train/0043_champion_league_rl/assets/policies/registry.json`
- Create: `train/0043_champion_league_rl/assets/policies/manifests/policy_0809.json`
- Create: `train/0043_champion_league_rl/assets/policies/manifests/champion_g001.json`
- Create: `train/0043_champion_league_rl/assets/evaluation/registry.json`
- Create: `train/0043_champion_league_rl/assets/evaluation/manifests/frozen_meta_256_v1.json`
- Create: `train/0043_champion_league_rl/assets/evaluation/seeds/frozen_meta_256_v1.json`
- Test: `train/0043_champion_league_rl/tests/test_assets.py`

**Interfaces:**
- Consumes: approved one-time import sources named in Task 1.
- Produces: `AssetRegistry.load(project_root)`, `validate_all() -> AssetAudit`, immutable project-relative assets under semantic directories (`definitions/001`, `definitions/policy_0809`, `definitions/champion_g001`), and separate training/evaluation pool manifests. SHA-256 values remain manifest fields and never serve as semantic directory names.

- [ ] **Step 1: Write failing registry integrity tests**

Cover stable canonical deck hashes, exactly 60 positive card IDs, immutable IDs, missing blob rejection, external/absolute path rejection, 55-deck/256-game FrozenMeta composition, and full manifest reconstruction.

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest -q train/0043_champion_league_rl/tests/test_assets.py`

Expected: FAIL because registries and loader are absent.

- [ ] **Step 3: Implement the fail-closed registry loader**

Use resolved-path containment checks, file and canonical-content SHA-256 validation, unique IDs/hashes, exact role membership, and no fallback to historical paths.

- [ ] **Step 4: Implement the one-time import command**

The command copies approved deck/policy/evaluation inputs into content-addressed 0043 paths, writes atomically, refuses overwrite on content disagreement, and records provenance only as non-runtime metadata.

- [ ] **Step 5: Import and validate assets**

Run: `python3 -m train.0043_champion_league_rl.import_assets --all`

Expected: `status=PASS`, 55 immutable decks, 256 FrozenMeta slots, Policy-0809, and Champion-G1 all load only through 0043-relative paths.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q train/0043_champion_league_rl/tests/test_assets.py`

Expected: PASS.

### Task 3: Policy Resolver and No-Hybrid Gate

**Files:**
- Create: `train/0043_champion_league_rl/policy_identity.py`
- Create: `train/0043_champion_league_rl/policy_runtime.py`
- Test: `train/0043_champion_league_rl/tests/test_policy_identity.py`

**Interfaces:**
- Consumes: Task 2 policy manifests and project-local blobs.
- Produces: `resolve_policy(policy_id, purpose, device) -> ResolvedPolicy` and `audit_role_isolation(focal, opponent) -> PolicyIsolationAudit`.

- [ ] **Step 1: Write failing identity/isolation tests**

Test complete effective component hashes for 0809/G1, reconstruction, no shared `Parameter` object or tensor storage, focal-step immutability, latest pointer frozen-only, and deliberate hybrid hard failure.

- [ ] **Step 2: Implement project-local materializers**

Port and freeze only the required 0042 model/runtime implementation into 0043, replacing all historical runtime paths with Task 2 registry resolution.

- [ ] **Step 3: Implement identity-before-routing gates**

Every resolver returns requested/materialized IDs, component hashes, complete effective hash, schema metadata, and `PASS`; missing or mismatched fields raise a fatal identity exception before engine creation.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q train/0043_champion_league_rl/tests/test_policy_identity.py`

Expected: PASS.

### Task 4: Exact 128/64/64 Opponent Sampler

**Files:**
- Create: `train/0043_champion_league_rl/league/sampler.py`
- Create: `train/0043_champion_league_rl/league/config.json`
- Test: `train/0043_champion_league_rl/tests/test_sampler.py`

**Interfaces:**
- Consumes: immutable training deck pool, active policy pool, latest champion pointer, and frozen curriculum weights.
- Produces: `build_schedule(update, seed, curriculum) -> tuple[OpponentLane, ...]` with branch, policy/deck IDs and hashes, seat slot, engine/search/policy seeds, and routing audit identity.

- [ ] **Step 1: Write exact-quota and reproducibility tests**

Assert 128/64/64/256 counts, independent deck/policy draws, reproducible seed output, shuffled branch positions, one-item behavior, empty-pool failure, and preserved 128 first/128 second seat slots.

- [ ] **Step 2: Implement quota construction before shuffle**

Build each branch independently, bind exact immutable identities, then apply one seeded permutation without modifying seat/seed contracts.

- [ ] **Step 3: Run focused tests**

Run: `pytest -q train/0043_champion_league_rl/tests/test_sampler.py`

Expected: PASS.

### Task 5: PFSP State and Ten-Update Curricula

**Files:**
- Create: `train/0043_champion_league_rl/league/pfsp.py`
- Create: `train/0043_champion_league_rl/league/state.py`
- Test: `train/0043_champion_league_rl/tests/test_pfsp.py`

**Interfaces:**
- Consumes: terminal game rows keyed by focal context, opponent deck, and opponent policy.
- Produces: Beta(1,1) smoothed per-deck/per-policy/joint statistics and immutable `Cxxx` manifests with configurable floor/cap and pool/config hashes.

- [ ] **Step 1: Write smoothing, numerical, freeze, and resume tests**

Assert zero-game WR 0.5, no NaN/all-zero weights, configured floor/cap, no weight change inside updates 0–9, exact refresh at update 10, and byte-equivalent resumed state/schedule.

- [ ] **Step 2: Implement atomic state and manifest persistence**

Write temp-then-replace JSON under the current `rl_runs/0043.../versions/<Vn>/artifact/` and reject pool/config hash changes on resume.

- [ ] **Step 3: Run focused tests**

Run: `pytest -q train/0043_champion_league_rl/tests/test_pfsp.py`

Expected: PASS.

### Task 6: PPO Integration and Regression Guard

**Files:**
- Create: `train/0043_champion_league_rl/training/run.py`
- Create: `train/0043_champion_league_rl/training/config.py`
- Create: `train/0043_champion_league_rl/training/regression.py`
- Test: `train/0043_champion_league_rl/tests/test_training_regression.py`

**Interfaces:**
- Consumes: Task 1 approved config, Tasks 3–5 policy/schedule/curriculum APIs, and a self-contained copy of required model/rollout code.
- Produces: formal versioned PPO entrypoint with unchanged optimizer behavior and opponent-lane manifests embedded in each trajectory.

- [ ] **Step 1: Write legacy-compatible regression tests**

Compare parameter groups, learning rates, trainable tensor names, update-step count, 256 games, complete shuffled PPO traversal, loss/reward/GAE/KL settings, action semantics, and gradients against the frozen Phase 0 contract.

- [ ] **Step 2: Port the minimum executable training boundary**

Copy required implementation into 0043 and remove numbered-project imports; shared `rl_environment/`, `evaluation/`, official data, and engine runtimes remain allowed.

- [ ] **Step 3: Bind sampled identities to episodes and PPO batches**

Reject any trajectory missing source focal update, opponent effective hash, exact-deck hash, curriculum version, or routing `PASS`.

- [ ] **Step 4: Run tests and a two-deck/two-policy smoke**

Run: `pytest -q train/0043_champion_league_rl/tests/test_training_regression.py`

Run: `python3 -m train.0043_champion_league_rl.training.run --smoke --wandb-mode offline`

Expected: all tests pass and the official-engine smoke completes with zero identity/routing fallback.

### Task 7: Rollout Strength and Curriculum Telemetry

**Files:**
- Create: `train/0043_champion_league_rl/telemetry.py`
- Test: `train/0043_champion_league_rl/tests/test_telemetry.py`

**Interfaces:**
- Consumes: the same 256 completed terminal game rows used by PPO/PFSP.
- Produces: scalar/table payloads for `rollout/*`, `strength/*`, `pfsp/*`, and `coverage/*` without extra games.

- [ ] **Step 1: Write aggregation tests**

Use fixed win/loss/draw fixtures to verify terminal prize margin, loss-only prizes, win-only opponent prizes, turns-to-win/loss, p10 WR, red counts, entropy, distributions, and curriculum boundaries.

- [ ] **Step 2: Implement one post-rollout aggregation pass**

Keep per-game rows local, emit bounded scalar sets and top/bottom tables, and preserve all existing PPO health metrics.

- [ ] **Step 3: Verify logger ordering and namespaces**

Assert `training_metrics.jsonl` flush precedes TensorBoard and W&B mirror, with `trainer/update`, `rollout/source_policy_update`, and `checkpoint/update` correctly distinguished.

### Task 8: FrozenMeta Assetization and Parity

**Files:**
- Create: `train/0043_champion_league_rl/evaluation/frozen_meta.py`
- Create: `train/0043_champion_league_rl/evaluation/run.py`
- Test: `train/0043_champion_league_rl/tests/test_frozen_meta.py`

**Interfaces:**
- Consumes: Task 2 FrozenMeta256-V1 assets and Task 3 shared resolver/materializer.
- Produces: CPU-256 and CUDA-2048 manifests with separate candidate/opponent audits, exact schedules, seeded toss/Agent choice/actual seat, and zero-error acceptance gate.

- [ ] **Step 1: Write composition/seed/deployment tests**

Pin 55 entries, 256 frequencies, schedule hashes, replica semantics, FP16-storage/FP32-runtime candidate gate, and raw-FP32 formal rejection.

- [ ] **Step 2: Implement evaluation through the shared identity boundary**

Use official engine games and project-local policies; no high-frequency automatic full evaluation is added.

- [ ] **Step 3: Run old/new parity diagnostics**

For identical immutable candidate/opponent/deck/seed inputs, compare schedule and first-divergence traces; store diagnostic reports under `.tmp/evaluation/0043_frozen_meta_parity/`.

### Task 9: Manual Promote Champion V2 Workflow

**Files:**
- Create: `train/0043_champion_league_rl/promote/freeze.py`
- Create: `train/0043_champion_league_rl/promote/audit.py`
- Create: `train/0043_champion_league_rl/promote/decision.py`
- Test: `train/0043_champion_league_rl/tests/test_promote.py`

**Interfaces:**
- Consumes: immutable candidate checkpoint, shared deployment materializer, passing audits, formal Frozen evidence, and explicit human decision file.
- Produces: frozen candidate manifest/report; only explicit `PROMOTE` atomically appends a generation and changes the latest champion pointer.

- [ ] **Step 1: Write freeze/audit/manual-decision tests**

Cover candidate immutability, reconstruction, deployment hash equality, evaluation completeness, HOLD/REJECT no-op behavior, missing human decision failure, generation increment, and preservation of old champions.

- [ ] **Step 2: Implement freeze and hard audit commands**

Materialize source FP32 → complete effective → FP16 artifact → strict FP32 runtime and record all V1/V2 identity fields.

- [ ] **Step 3: Implement atomic manual registry update**

Require a signed/explicit local decision record naming the exact candidate ID and evidence hashes; never infer promotion from metrics.

- [ ] **Step 4: Run full project gate**

Run: `pytest -q train/0043_champion_league_rl/tests`

Expected: PASS before any formal 0043 League training launch.
