# 0015 T4 Marnie Unicode Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline because this repository does not
> expose `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Track each
> checkbox directly and do not delegate or commit.

**Goal:** Add a deterministic natural-distribution cap from the large Marnie/Munkidori pool to a
new auditable T4 dataset generation and train the next source-conditioned R15 policy without
changing the frozen V1–V14 records.

**Architecture:** Preserve all existing source IDs 1–84 and assign stable `unicode_<sha256[:12]>`
keys at IDs 85+ only when an NFKD ASCII slug is empty or collides with another normalized full team
name. Audit all 4,672 qualified Marnie candidates (including 191 trajectories in the seven-name
Unicode collision subset), then select 535 by a fixed outcome-stratified hash cap matching the
existing train-trajectory count. Build a new index containing the unchanged target/pure-Dragapult
trajectories plus that cap, then compile a new raw dataset and R15 cache. Train fresh R15 T4 on all
build IDs present while retaining target-only validation and the existing source persona.

**Tech Stack:** Python 3.11, SHA-256/Unicode normalization, PyTorch 2.11 BF16/CUDA, unittest,
TensorBoard, W&B online, official-engine evaluation.

## Global Constraints

- Never modify `engine/source/`, V1 data, V1–V14 versions, existing candidates or reports.
- Keep all qualified target and pure-Dragapult trajectories. Select exactly 535 of 4,672 qualified
  Marnie trajectories by the declared outcome-stratified hash cap; no win/loss filtering.
- Every exact team name must map to one stable unique source key; different names may not share a
  persona ID.
- Preserve the complete old source vocabulary mapping byte-for-byte and append only new IDs.
- Validation remains the same 1,000 target decisions; auxiliary rows never select checkpoints.
- V15 uses fresh R15 initialization, seed 20260723, batch 256, LR 3e-4, weight decay 0.02 and W&B
  online under `dragon_bra/pokemon-tcg-policy-learning`.
- No Kaggle submission, commit/push or opponent-pool admission is authorized.

---

### Task 1: Collision-safe source identity and immutable T4 index

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/data/source_index.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_catalog.py`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/trajectory_source_index_v2_t4.json`

**Interfaces:**
- Consumes: exact Unicode `team_name`, the frozen V1 source vocabulary and 0726 catalog/archive.
- Produces: `_source_key(team_name: str) -> str` and a V2 index with source IDs 1–84 unchanged,
  unique Unicode IDs appended, resolved `first_player`, and T4 trajectories in `core_trajectories`.

- [ ] Add tests proving ASCII compatibility, deterministic Unicode hashing, uniqueness for all seven
  observed Unicode names and unchanged old source IDs.
- [ ] Add `--include-t4` and `--base-index` contracts; reject inclusion unless the base vocabulary is
  exactly preserved and every team-name/key mapping is one-to-one.
- [ ] Audit all 4,672 Marnie candidates, select 535 with `0015_t4_outcome_cap_v1`, then read and
  hash-check only selected replay bytes, derive real first-player evidence and assign split `train`.
- [ ] Generate the V2 index and audit the 191-trajectory Unicode collision subset, seven distinct
  Unicode team identities, selected deck hashes and no duplicate `(episode_id, player_index)`.

### Task 2: New raw dataset, feature cache and CUDA smoke

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/V2_t4_raw/`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/V2_t4_r15_features/`
- Create: `.tmp/0015_dragapult_conditioned_bc/smoke_v15_t4/`

**Interfaces:**
- Consumes: V2 T4 index through existing `raw_dataset` and `materialize` module entrypoints.
- Produces: a content-addressed feature cache with build 4 rows and unchanged target validation.

- [ ] Build raw shards and verify every selected trajectory produces at least one causal decision.
- [ ] Compile the shared R15 cache and verify all rows are eligible, build/source/outcome counts are
  nonzero and target validation remains exactly 1,000 decisions.
- [ ] Run the 24 focused tests plus a five-batch CUDA BF16 T4 smoke with full validation, finite loss,
  legal rate 1.0, checkpoint reload and W&B disabled.
- [ ] Record cache hashes, exact natural data counts and smoke results in the 0015 data audit.

### Task 3: Formal V15 T4 training

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V15_t4_marnie_munkidori/`

**Interfaces:**
- Consumes: V2 T4 feature cache and the unchanged `--arm t4` R15 training contract.
- Produces: immutable checkpoints, canonical metrics, TensorBoard events and one W&B online run.

- [ ] Preflight all four version directories plus the authoritative report path and refuse overwrite.
- [ ] Launch V15 immediately after smoke and monitor each epoch for finite optimization, complete
  target validation, legality, target exact and overfitting.
- [ ] While V15 trains, prepare candidate export/validation commands and compare V2 epoch selectors
  read-only so the next experiment does not wait on development.
- [ ] Preserve early-stop, best-loss, best-teacher and best-greedy checkpoints; do not use engine
  results to choose an epoch.

### Task 4: Official-engine evaluation and next-version gate

**Files:**
- Create: `evaluation/arena/candidates/0015_v15_t4_marnie_munkidori_exact_best/`
- Create: `experiments/0015_dragapult_conditioned_bc/evaluation/V15_t4_marnie_munkidori.html`
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/009_t4_marnie_results.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/manifest.json`

**Interfaces:**
- Consumes: frozen V15 exact-best checkpoint, target deck/persona and the unchanged 20-opponent
  official-engine catalog at 10 games each.
- Produces: 200-game strength evidence and a single predeclared next gate.

- [ ] Export and validate the target package, then run 200 official-engine games with workers 8 and
  one CPU thread per worker; require 200 completed games and zero errors.
- [ ] Compare V15 with V2's 39/200 selected baseline and the observed V12/V13 seed spread; do not
  claim transfer from offline exact alone.
- [ ] If V15 clearly improves, allocate one independent-seed confirmation next; if it regresses,
  stop T4. If it lands within one win of V2 while target exact regresses, permit exactly one
  predeclared half-cap arm (267 Marnie trajectories) that keeps every target/pure trajectory; do not
  sweep additional cap sizes.
- [ ] Sync DESIGN Markdown/HTML, manifest, report backlink, W&B state and Decision 009; run focused
  tests, evaluation asset tests, scoped compile, candidate validate and `git diff --check`.

### Task 5: Bounded V16 half-cap follow-up

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/data/source_index.py`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/trajectory_source_index_v3_t4_half.json`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/V3_t4_half_raw/`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/V3_t4_half_r15_features/`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V16_t4_marnie_half_cap/`

**Interfaces:**
- Consumes: the same complete 4,672-candidate pool and `0015_t4_outcome_cap_v1` ordering.
- Produces: a strict prefix-by-stratum 267-trajectory Marnie view and one otherwise identical R15
  training run.

- [ ] Add an explicit positive `--t4-cap` option valid only with `--include-t4`; keep V15's default
  cap at the complete core-train trajectory count.
- [ ] Generate the 267-trajectory index/cache and verify all selected identities are a subset of the
  V15 selection, old source IDs remain fixed and validation remains 1,000 decisions.
- [ ] Run CUDA smoke, then train V16 from seed 20260723 with every other model/optimizer/validation
  field identical to V15.
- [ ] Export exact-best and run the same 200-game official-engine evaluation; stop cap exploration
  after V16 regardless of outcome.

### Task 6: Prepare a low-source-scale transfer branch

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`
- Conditionally create: `rl_runs/0015_dragapult_conditioned_bc/versions/V17_t4_low_source_scale/`

**Interfaces:**
- Consumes: CLI `--source-initial-scale`, the unchanged T4 data membership and source IDs.
- Produces: a model/checkpoint contract that records the exact source residual scale while retaining
  identical portable inference.

- [ ] Thread a validated `source_initial_scale` through plain R15 and rule-contract model builders;
  default 0.10 must reproduce every V1–V16 topology and parameter count.
- [ ] Add focused tests for scale 0.03, invalid endpoints and model-contract serialization; run a
  five-batch CUDA smoke on the V16 dataset.
- [ ] Do not allocate V17 before V16 official-engine evidence. If V16 is not a supported gain, one
  0.03 run is permitted as a distinct transfer hypothesis; do not sweep more scale values.
