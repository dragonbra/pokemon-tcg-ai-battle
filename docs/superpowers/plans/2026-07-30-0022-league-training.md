# 0022 League Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and implement a resident Frozen/Live opponent League that improves real RL rollout throughput without hiding policy-quality regressions.

**Architecture:** A frozen source-free BC Encoder is shared by 16 complete Frozen anchor policies and 16 independent Live Decoder/Value branches. Official-engine workers send batched observations to one resident inference service; each rollout records actor/opponent policy versions and an immutable pool snapshot. PPO updates only the actor branch that owns each trajectory, while Frozen anchor evaluations remain separate from training data.

**Tech Stack:** Python 3.11+, PyTorch CUDA, multiprocessing, official engine runtime, standard-library `unittest`, TensorBoard, W&B online, HTML/Markdown experiment reports.

## Global Constraints

- `engine/source/` is read-only and must never be modified.
- Official engine runtime remains the only source for formal training strength and evaluation until CUDA parity promotion passes.
- 0022 uses source-free deployment `pi(a|s,D)`; source/team identity is provenance only.
- Every exact deck keeps an independent package, deck hash, manifest and evaluation boundary.
- Every formal version uses `rl_runs/0022_league_training/versions/V<n>_<tag>/` with artifact, checkpoint, tensorboard and wandb subdirectories.
- A PPO rollout must record `rollout/source_policy_update`, `checkpoint/update`, actor policy version, opponent policy version and opponent pool snapshot.
- W&B formal runs use private project `dragon_bra/pokemon-tcg-policy-learning`; `training_metrics.jsonl` is canonical.
- Throughput claims require identical official-engine completion/error contracts and cannot use micro-opcode or policy-only speed as end-to-end evidence.

## Task 1: Freeze the design and catalog contracts

**Files:**
- Create: `experiments/0022_league_training/DESIGN.md`
- Create: `experiments/0022_league_training/DESIGN.html`
- Create: `experiments/0022_league_training/decisions/001_scope_and_gates.md`
- Create: `docs/superpowers/plans/2026-07-30-0022-league-training.md`

**Interfaces:**
- Consumes: source-free BC checkpoint metadata, current official-engine benchmark contract, and the future 16-deck catalog manifest.
- Produces: authoritative Frozen/Live roles, throughput gate, package gate, rollout fields and version layout.

- [x] **Step 1: Record the two-pool architecture and the non-goal.**

  Define complete Frozen packages as quality anchors and independent Live Decoder/Value branches as curriculum. State that 0022 measures local Decoder-only improvement and opponent throughput, not global optimality.

- [x] **Step 2: Record the pre-training throughput gate.**

  Require tuned CPU, resident CPU and resident GPU arms, with completed episodes, selections/s, opponent decisions/s, package load, encoding, IPC, H2D/D2H, latency and memory metrics. Require `2.0x` opponent decisions/s and `1.5x` end-to-end episodes/s before League self-evolution.

- [x] **Step 3: Record catalog selection criteria.**

  Require exact observed online decks, frequency/evidence, BC card coverage, diversity and package validation. Leave the final 16 IDs unselected until a separate catalog decision.

- [x] **Step 4: Render the HTML and cross-check Markdown facts.**

  Keep `DESIGN.html` synchronized with `DESIGN.md`, including the current stage, Frozen/Live roles, 320-game focal schedule, throughput gates and version paths.

- [x] **Step 5: Commit the design-only artifact.**

  ```bash
  git add experiments/0022_league_training docs/superpowers/plans/2026-07-30-0022-league-training.md
  git commit -m "docs: define 0022 league training contract"
  ```

## Task 2: Build the 16-deck catalog audit

**Files:**
- Create: `experiments/0022_league_training/catalog/README.md`
- Create: `experiments/0022_league_training/catalog/manifest.schema.json`
- Create: `train/0022_league_training/catalog.py`
- Create: `train/0022_league_training/tests/test_catalog.py`

**Interfaces:**
- Consumes: exact deck packages, official card IDs, BC dataset provenance and online environment evidence.
- Produces: `load_catalog(path) -> LeagueCatalog`, `validate_catalog(catalog) -> CatalogAudit`, and an immutable catalog manifest with deck/package/source hashes.

- [ ] **Step 1: Write failing catalog tests.**

  Test exact 60-card validation, required evidence fields, duplicate deck hash rejection, diversity labels, and rejection of a deck without online/BC coverage evidence.

- [ ] **Step 2: Implement typed manifest loading and fail-closed validation.**

  Reject missing hashes, duplicate package IDs, non-60-card decks, unvalidated packages, and catalog entries that lack frequency and BC coverage evidence. Preserve source identity only as provenance.

- [ ] **Step 3: Run the focused catalog suite.**

  ```bash
  python3 -m unittest -v train/0022_league_training/tests/test_catalog.py
  ```

- [ ] **Step 4: Commit the catalog contract.**

  ```bash
  git add experiments/0022_league_training/catalog train/0022_league_training
  git commit -m "feat: add 0022 league catalog audit contract"
  ```

## Task 3: Implement resident Frozen/Live policy routing

**Files:**
- Create: `train/0022_league_training/policy_pool.py`
- Create: `train/0022_league_training/inference_service.py`
- Create: `train/0022_league_training/tests/test_policy_pool.py`
- Modify: `train/0020_pluggable_deck_rl/rl/opponent_inference/resident_pool.py`

**Interfaces:**
- Consumes: validated catalog, source-free BC checkpoint, Frozen/Live model packages and batched observation requests.
- Produces: `ResidentLeaguePool.act(batch) -> PolicyActionBatch`, `snapshot() -> PoolSnapshot`, and explicit `policy_id` routing without per-decision package load.

- [ ] **Step 1: Write failing routing and snapshot tests.**

  Verify stable policy IDs, Frozen immutability, Live version increments, deck-to-policy routing, batch grouping by schema/dtype, and snapshot hash reproducibility.

- [ ] **Step 2: Implement one resident model service.**

  Load the shared Encoder once, keep Frozen and Live heads resident, group requests by compatible tensor contract, and return legal ordered actions with actor/opponent role metadata. Do not call `.cpu()`, `.numpy()`, or `.item()` in the steady-state policy path.

- [ ] **Step 3: Run focused pool tests and a bounded inference smoke.**

  ```bash
  python3 -m unittest -v train/0022_league_training/tests/test_policy_pool.py
  ```

- [ ] **Step 4: Commit the resident pool.**

  ```bash
  git add train/0022_league_training train/0020_pluggable_deck_rl/rl/opponent_inference/resident_pool.py
  git commit -m "feat: add resident frozen live league pool"
  ```

## Task 4: Add the three-arm throughput benchmark

**Files:**
- Create: `train/0022_league_training/benchmark.py`
- Create: `train/0022_league_training/tests/test_benchmark_contract.py`
- Create: `experiments/0022_league_training/benchmarks/README.md`

**Interfaces:**
- Consumes: one exact deck catalog snapshot, identical engine/runtime hashes, worker/thread settings and policy checkpoint.
- Produces: machine-readable benchmark JSON with baseline/resident CPU/resident GPU arms and a `throughput_gate` boolean.

- [ ] **Step 1: Write failing benchmark contract tests.**

  Assert that missing arm data, changed deck hashes, incomplete games, errors, or absent H2D/IPC timing fail closed. Assert that a policy-only speedup cannot set `throughput_gate=true`.

- [ ] **Step 2: Implement benchmark orchestration and comparison.**

  Run tuned CPU, resident CPU and resident GPU arms with identical completed-game and seat schedules. Calculate opponent decisions/s and end-to-end episodes/s; require `2.0x` and `1.5x` thresholds plus no completion/error regression.

- [ ] **Step 3: Run a bounded official-engine benchmark into `.tmp/evaluation/0022_throughput/`.**

  Preserve the run ID and report path; do not call it a formal strength evaluation.

- [ ] **Step 4: Commit the benchmark implementation and contract.**

  ```bash
  git add train/0022_league_training experiments/0022_league_training/benchmarks
  git commit -m "feat: benchmark resident league throughput"
  ```

## Task 5: Implement focal Frozen/Live rollout and PPO ownership

**Files:**
- Create: `train/0022_league_training/rollout.py`
- Create: `train/0022_league_training/ppo.py`
- Create: `train/0022_league_training/tests/test_rollout_contract.py`
- Create: `train/0022_league_training/tests/test_update_ownership.py`

**Interfaces:**
- Consumes: `PoolSnapshot`, official-engine workers, focal schedule and resident inference service.
- Produces: role-separated trajectories, `training_metrics.jsonl`, model-only Live checkpoints and explicit `rollout/source_policy_update` metadata.

- [ ] **Step 1: Write failing ownership tests.**

  Verify Frozen opponent actions never enter an actor loss, Live opponent actions enter only their own policy loss, reward signs are actor-relative, and one PPO batch contains one behavior/reference version per policy.

- [ ] **Step 2: Implement the 16 Frozen + 16 Live focal schedule.**

  Generate 160 Frozen and 160 Live games per focal iteration by default, with balanced seats, configurable weights, immutable pool snapshot and explicit focal deck ID. Keep Live-only win rates diagnostic.

- [ ] **Step 3: Implement independent Decoder/Value updates.**

  Update only the policy branches that own trajectories. Freeze the shared Encoder; create a new `V<n>_<tag>` for every semantic change; save model-only checkpoints with limited retention.

- [ ] **Step 4: Run focused rollout/PPO tests.**

  ```bash
  python3 -m unittest -v train/0022_league_training/tests/test_rollout_contract.py train/0022_league_training/tests/test_update_ownership.py
  ```

- [ ] **Step 5: Commit the league update path.**

  ```bash
  git add train/0022_league_training
  git commit -m "feat: add frozen live league rollout updates"
  ```

## Task 6: Add Frozen-anchor evaluation and promotion gates

**Files:**
- Create: `train/0022_league_training/promotion.py`
- Create: `train/0022_league_training/tests/test_promotion.py`
- Modify: `evaluation/runner/batch.py`
- Create: `experiments/0022_league_training/evaluation/index.html`

**Interfaces:**
- Consumes: Live candidate checkpoint, Frozen pool snapshot, historical snapshots, exact catalog and official engine reports.
- Produces: `PromotionDecision`, formal `evaluation/V<n>_<tag>.html`, index entry and `artifact/evaluation.json` reverse link.

- [ ] **Step 1: Write failing promotion tests.**

  Reject Live-only improvement without Frozen-anchor evidence, incomplete games, missing pool snapshots, stale engine hashes, overwritten report paths and non-monotonic version directories.

- [ ] **Step 2: Implement quality comparison.**

  Compare candidate vs zero-shot Frozen and historical best by deck, opponent, seat and metric profile. Keep Wilson intervals and report non-inferable differences as inconclusive.

- [ ] **Step 3: Run promotion parser tests and refresh the project index.**

  ```bash
  python3 -m unittest -v train/0022_league_training/tests/test_promotion.py
  ```

- [ ] **Step 4: Commit the promotion contract.**

  ```bash
  git add train/0022_league_training evaluation/runner/batch.py experiments/0022_league_training/evaluation
  git commit -m "feat: add league frozen anchor promotion gates"
  ```

## Task 7: Run the first end-to-end feasibility trial

**Files:**
- Create: `experiments/0022_league_training/decisions/002_feasibility_result.md`
- Create: `rl_runs/0022_league_training/versions/V1_throughput_feasibility/`

**Interfaces:**
- Consumes: completed BC checkpoint, audited catalog, throughput benchmark and resident service.
- Produces: an auditable decision: `throughput_gate_passed`, `decoder_only_gate_passed`, or a recorded failure with the next experiment.

- [ ] **Step 1: Verify BC completion and catalog evidence before rollout.**

  Refuse the run if source-free BC validation, exact deck evidence, package hashes or model contract are missing.

- [ ] **Step 2: Run the three-arm throughput benchmark.**

  Do not start League self-evolution unless both throughput thresholds and completion/error contracts pass.

- [ ] **Step 3: Run one focal Decoder-only Frozen trial.**

  Compare the Live focal policy to zero-shot and historical Frozen anchors under the same official-engine schedule.

- [ ] **Step 4: Record the decision and preserve failed artifacts.**

  Keep all metrics, pool snapshots, reports, status and W&B sync state; never overwrite a failed version.

- [ ] **Step 5: Commit the feasibility result.**

  ```bash
  git add experiments/0022_league_training rl_runs/0022_league_training
  git commit -m "exp: record 0022 league feasibility result"
  ```
