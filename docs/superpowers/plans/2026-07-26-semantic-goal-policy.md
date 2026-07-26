# 0013 Semantic Goal Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Every behavior change uses TDD; every phase uses verification-before-completion. Do not commit or push unless the user separately requests it.

**Goal:** Build and empirically evaluate `0013_semantic_goal_policy`: a causal player-perspective, structured-semantic, Goal-QKV policy research system with M0–M5 ablations, W&B auditability, at least ten hours of healthy GPU exploration, and one official-engine formal evaluation of the best eligible checkpoint.

**Architecture:** Shared lifecycle/logging/evaluation mechanics remain in `rl_environment/` and `evaluation/`. All policy semantics live in `train/0013_semantic_goal_policy/`; tracked facts and reports live in `experiments/0013_semantic_goal_policy/`; generated datasets and versioned runtime assets live in `rl_runs/0013_semantic_goal_policy/`. Offline construction and online inference use the same causal knowledge state, typed feature schema, card semantics, and full-action probability implementation.

**Tech Stack:** Python 3.11+, standard-library `unittest`, PyTorch, gzip JSONL shards, TensorBoard, W&B SDK, existing official-engine evaluation runtime.

## Global Constraints

- Never modify, format, patch, or generate files under `engine/source/`.
- Treat `data/raw/` and `data/official/` as read-only.
- Preserve all pre-existing modified/untracked files; never reset, clean, checkout, or broadly format the tree.
- No commit or push unless separately requested.
- Preserve replay `selected` order; never use the old `sorted(action)` path.
- Source contract: Yushin Ito winner trajectories from 2026-07-18 through 2026-07-25, including manifested patched data; no other expert/team.
- Train/validation only, split by complete episode-player group; official-engine evaluation is the final test.
- Never expose omniscient state, opponent deck, future frames/logs, outcome, or reward as policy input.
- Never silently truncate long actions or silently repair schema/ledger/action errors.
- W&B target is private `dragon_bra/pokemon-tcg-policy-learning`, group `0013_semantic_goal_policy`, online for formal runs.
- Canonical metric order is JSONL flush → TensorBoard → W&B. Remote failure cannot roll back local facts.
- Never upload checkpoint, optimizer, full metrics JSONL, dataset, observation, replay, trace, or source patch.
- Candidate remains under `evaluation/arena/candidates/`; do not alter opponents/catalog and do not submit to Kaggle.
- Downstream phases stop when an upstream gate fails. Ten GPU hours cannot be satisfied with unsafe, idle, or failed loops.

## Frozen Initial Protocol

Before formal data build, write and hash `pre_run_protocol.json` with these defaults:

- Split: 90% train / 10% validation, seed `20260726`, complete episode-player groups, deterministic stratification by `(date, deck_manifest_hash)`.
- Digest input: `semantic_goal_policy_split_v1|20260726|<date>|<deck_hash>|<episode_id>|<player>`.
- Patched episode replaces archive original by canonical episode ID; ambiguity fails closed.
- Event window: newest 64 actor-visible events.
- Ontology: `card_effect_ontology_v1`; 100% registered-deck/observed-option card identity coverage, ≥99% effect-instance coverage; residuals use `UNKNOWN_EFFECT`.
- View kinds: `none`, `eligible_subset`, `full_membership`, `ordered_view`; exact Prize inference only after verified `full_membership`.
- Formal evaluation: profile `auto_iteration_v8_setup_relay`, revision 7; 20 games per enabled opponent; seed `20260726 + game_index`; alternating seats, ten per seat.
- Invariance: aligned agreement ≥0.999 and mean KL ≤1e-4.
- Free-running legality/completion: exactly 1.0 on validation audit cases.
- Counterfactual direction and unknown-vs-zero rates ≥0.80; irrelevant-goal mean absolute logit change ≤0.05.
- Numerical health: zero NaN/Inf, zero corrupt checkpoints, zero unacknowledged AMP overflow.
- Minimum formal allocation per smoke-passing M0–M5: one seed, three complete epochs, at least one complete pass over train.
- Runtime floor: lower of 10 decisions/s and 80% of measured M0 smoke throughput, frozen before formal runs.
- GPU telemetry every 10 seconds; count non-overlapping healthy GPU-work intervals, merge gaps ≤60 seconds, report idle/failure/overlap separately; target ≥36,000 active seconds.
- W&B snapshot limit: 1 MiB/file and 4 MiB/generation.

---

## Phase 1 — Infrastructure

### Task 1: Protect the working tree and baseline

**Files:** Read-only repository status, `engine/source/`, current tests.

**Produces:** protected-path inventory, baseline test failures, engine/source hash/diff evidence.

- [ ] Run `git status --short`, `git diff --name-only`, `git diff -- engine/source`, and record all pre-existing changes.
- [ ] Run `python3 -m unittest discover -s tests -p 'test_*.py'` and record baseline failures.
- [ ] Stop immediately if engine/source already differs; do not alter it.
- [ ] At every later gate compare changed files to this inventory.

**Gate:** Baseline is explicit and protected; no file changed.

### Task 2: Complete nested project/version lifecycle

**Files:** Modify `rl_environment/runs.py`, `tests/test_experiment_projects.py`.

**Interfaces:** Extend `VersionPaths` with `checkpoint_selection`, `model_contract`, `dataset_reference`, `metrics_snapshot`, and `wandb_snapshot_manifest`; add atomic `write_version_status()` and full four-directory occupancy validation.

- [ ] Write failing tests replacing obsolete 0013 example identity, rejecting any occupied artifact/checkpoint/tensorboard/wandb/report path, enforcing strict next version, atomic status merge, and immutable evaluation backlink.
- [ ] Run `python3 -m unittest -v tests.test_experiment_projects`; verify expected failures.
- [ ] Implement the minimum nested-path/status changes while preserving legacy 0001–0012 read behavior.
- [ ] Re-run focused tests and `git diff --check`.

**Gate:** Lifecycle tests pass; only declared files changed.

### Task 3: Repair W&B canonical path, axis, status, and exit semantics

**Files:** Modify `rl_environment/logging.py`, `rl_environment/wandb_logging.py`; tests in `tests/test_rl_wandb_logging.py`, `tests/test_rl_wandb_identity.py`, `tests/test_rl_framework.py`.

**Interfaces:** Explicit metric axis/namespace; nested canonical path recognition; sibling version `wandb/`; sync state including project/entity/run ID/URL/failure; exception-aware `close(exit_code)`.

- [ ] Add failing fake-SDK tests for nested canonical paths, version-local staging, explicit axes, local-before-remote ordering, remote-failure persistence, and nonzero exception exit.
- [ ] Run focused tests and confirm failures.
- [ ] Implement structural path validation under `rl_runs/<project>/versions/<version>/artifact/training_metrics.jsonl`.
- [ ] Add explicit BC/value/PPO/rollout/eval axes and representation/counterfactual/invariance namespaces.
- [ ] Persist sync state without raising remote failures through local logging.
- [ ] Re-run all focused tests.

**Gate:** JSONL survives simulated network failure; axes and status are deterministic.

### Task 4: Add immutable `policy="now"` W&B audit snapshots

**Files:** Create `rl_environment/wandb_snapshot.py`, `tests/test_rl_wandb_snapshot.py`; modify `rl_environment/wandb_logging.py`.

**Interfaces:** `SnapshotPolicy`, `validate_snapshot_files()`, `upload_snapshot_generation()`.

- [ ] Test rejection of outside-root paths, symlinks, wrong names, oversized files, secret/raw-data keys, and full JSONL content.
- [ ] Test exact whitelist: `training_summary.json`, `status.json`, `checkpoint_selection.json`, `model_contract.json`, `dataset_reference.json`, `metrics_snapshot.json`.
- [ ] Test `wandb.save(path, policy="now", base_path=artifact_root)` and payload-before-manifest ordering.
- [ ] Implement immutable generation manifests that list payload hashes but not their own hash; upload errors return failed state without deleting local files.
- [ ] Run focused tests with fake W&B only.

**Gate:** Snapshot policy is fail-closed; no real upload occurs in tests.

### Task 5: Migrate formal evaluation paths

**Files:** Modify `evaluation/runner/batch.py`, `evaluation/reporting/index.py`, `evaluation/cli.py`; focused evaluation tests.

**Interfaces:** New formal path `experiments/<project>/evaluation/Vn_tag.html`; temporary `.tmp/evaluation/...` unchanged; legacy reports read-only.

- [ ] Add failing tests for path rejection, overwrite refusal before workers, atomic report/index/backlink order, malformed legacy read-only handling, and no false backlink on failure.
- [ ] Implement formal/temporary path classification and atomic finalization.
- [ ] Run focused evaluation tests.

**Phase gate:** Infrastructure tests pass, `python3 -m compileall -q rl_environment evaluation`, `git diff --check`, and engine/source remains unchanged.

---

## Phase 2 — Project and Protocol

### Task 6: Create the 0013 three-root scaffold

**Files:** Create `train/0013_semantic_goal_policy/`, `experiments/0013_semantic_goal_policy/` metadata/design roots, and runtime roots through lifecycle APIs.

**Interfaces:** `PROJECT_ID`, thin CLI with `validate-protocol`, `audit-visibility`, `build-dataset`, `smoke-model`, `train`, `schedule`, `select-project`, `export`, `evaluate`; typed `ProjectConfig`.

- [ ] Write `test_scaffold.py` asserting imports, CLI commands, exact project identity, source contract, links, and absence of obsolete 0013 identity.
- [ ] Run test and verify failure.
- [ ] Implement the thin package/CLI and create roots through `initialize_project` rather than ad hoc directory logic.
- [ ] Write synchronized initial `DESIGN.md` and interactive `DESIGN.html` containing only approved current contracts, not empirical claims.
- [ ] Run scaffold tests and CLI help.

**Gate:** Three roots resolve correctly and no generated dataset/checkpoint is tracked.

### Task 7: Freeze `pre_run_protocol.json`

**Files:** Create `protocol.py`, `configs/pre_run_protocol.json`, `tests/test_protocol.py`, decision record.

**Interfaces:** `PreRunProtocol.from_json()`, `.validate()`, `.canonical_bytes()`, `.sha256()`, `freeze_protocol()`.

- [ ] Test every frozen field, canonical hash stability, missing threshold rejection, overwrite refusal, and formal-command hash mismatch refusal.
- [ ] Implement sorted-key canonical JSON and immutable freeze.
- [ ] Separate preflight-resolved M0 runtime value from immutable formula; require resolution before training.
- [ ] Validate through CLI.

**Phase gate:** Project/protocol tests pass and protocol contains no result-dependent placeholders.

---

## Phase 3 — Data Contracts

### Task 8: Read canonical sources with patched precedence

**Files:** Create `data/source.py`, `data/records.py`, `tests/test_source_reader.py`, tracked source manifest.

**Interfaces:** normalized team identity, `EpisodeSource`, `iter_canonical_episodes()`, `DecisionIdentity`, `SourceIdentity`.

- [ ] Build synthetic ZIP fixtures testing exact normalized Yushin match, winner filtering, patched replacement, duplicate/ambiguous/hash failure, and `submission_id_unavailable=true`.
- [ ] Implement precedence before trajectory parsing; keep payloads read-only.
- [ ] Produce daily source counts and hashes without storing episode bodies in tracked files.

**Gate:** Synthetic tests pass and real-source dry audit identifies the expected eight-day source set.

### Task 9: Preserve ordered full actions and duplicate-option identity

**Files:** Create `features/action_contract.py`, `tests/test_action_contract.py`.

**Interfaces:** `OptionSemanticIdentity`, non-feature `OptionOccurrenceIdentity`, `OrderedAction`, `validate_ordered_action()`, `remap_action()`.

- [ ] Test `[2,0]` remains ordered, duplicate/out-of-range/min/max failures, semantic duplicate occurrence numbering, bijective permutation round trip, and long-action rejection.
- [ ] Implement identities using actor-visible fields only.
- [ ] Represent optional STOP separately from max-count forced termination.

**Gate:** No action sorter is reachable; all remaps are bijective.

### Task 10: Implement deterministic stratified split

**Files:** Create `data/split.py`, `tests/test_split.py`.

**Interfaces:** `assign_groups()` returning a manifest with assignments, hashes, date/deck strata, and rare-stratum reports.

- [ ] Test input-order independence, no group overlap, no test split, both splits for strata ≥10, rare-stratum behavior, and byte-identical reproduction.
- [ ] Implement deterministic digest rank assignment; no `random.shuffle`.

**Gate:** Split is reproducible and every group appears exactly once.

### Task 11: Define BC/RL-compatible records and shards

**Files:** Create `data/dataset.py`, `data/shards.py`, `tests/test_dataset_contract.py`.

**Interfaces:** `DecisionRecord`, raw player-relative terminal outcome, `build_dataset()`, `iter_records()`.

- [ ] Test required identity/deck/observation/options/ordered-action/event-cursor/outcome/version fields.
- [ ] Test rejection of reward propensity/Q/discounted-return/omniscient/opponent-deck/future fields.
- [ ] Test error/truncated episodes have no value target, existing/partial output refusal, atomic shard finalization, and hash verification.
- [ ] Implement train/validation gzip JSONL shards and small hash-only `dataset_reference.json`.

**Phase gate:** Source/action/split/record suites pass. Build the real dataset only after audits; any source, long-action, split, or cross-date/deck conflict gate failure stops training.

---

## Phase 4 — Semantics and Causal Knowledge

### Task 12: Build card effect ontology v1

**Files:** Create `features/card_semantics.py`, `features/ontology.py`, `configs/card_effect_ontology_v1.json`, `tests/test_card_semantics.py`.

**Interfaces:** shared `CardSemanticRegistry`, `CardSemantics`, `EffectPrimitive`, `FunctionalCapability`, dedicated `UNK_CARD`/`UNKNOWN_EFFECT`.

- [ ] Test structural type/stage/HP/cost fields, ordered multi-effects, distinct unknown/missing/N/A/padding, shared encoder identity, deterministic capabilities, and coverage thresholds.
- [ ] Implement deterministic parsing/curation from read-only official data; no learned text encoder.
- [ ] Emit coverage and unknown-effect audits; stop below frozen thresholds.

**Gate:** Required deck/option card coverage and effect coverage pass.

### Task 13: Define typed feature and relation schemas

**Files:** Create `features/schema.py`, `features/observation.py`, `tests/test_schema.py`; update both DESIGN files.

**Interfaces:** typed state/entity/deck/ledger/event/option inputs; explicit epistemic state and independent padding/unknown/missing/N/A/overflow masks; named relation types.

- [ ] Test every shape/range/mask, overflow bit, opponent deck rejection, outcome/future rejection, serial exclusion from numeric features, and valid relation endpoints.
- [ ] Implement schema-versioned dataclasses and observation encoding from only actor observation plus causal state.
- [ ] Synchronize exact field and shape tables in DESIGN.md/HTML.

**Gate:** Tests and both DESIGN forms agree exactly.

### Task 14: Implement visibility audit and causal ledgers

**Files:** Create `knowledge/state.py`, `knowledge/ledger.py`, `knowledge/visibility.py`, tests, tracked visibility audit; inspect engine source read-only.

**Interfaces:** `CausalKnowledgeState.new_game()`, `.consume()`, `.encode_current()`, `.record_pending()`, `.reconcile()`; four view kinds; conservation validator.

- [ ] Test Prize unknown before full membership, causal unlock after valid view, no current-action consequence in `s_t`, shuffle/order behavior, Prize/Looking/recovery conservation, serial uniqueness, opponent unknown→known→moved/new-unknown lifecycle, and no future backfill.
- [ ] Read-only audit deck/hand-view contexts and redacted logs; unsupported contexts fail closed.
- [ ] Implement self exact ledger, opponent asymmetric knowledge, function aggregation, source/age/validity state.

**Phase gate:** No unclassified context, all causal/conservation tests pass, and `git diff -- engine/source` is empty.

---

## Phase 5 — M0–M3 and Probability

### Task 15: Implement model registry and M0–M3

**Files:** Create `model/registry.py`, `model/adapters.py`, `model/semantics.py`, `model/goal_qkv.py`, `model/variants.py`, tests.

**Interfaces:** explicit M0–M5 registry; shared `encode(TypedPolicyInput)`; four goal roles.

- [ ] Test distinct variant contracts: M0 ignores post-M0 features; M1 uses semantics/masks; M2 uses deck mean only; M3 returns four Goal-QKV contexts.
- [ ] Test shared semantic module identity, state independence from option order/count, and 18M–28M reference parameter budget.
- [ ] Implement `d_model=384`, six state layers, eight heads, FFN 1536, two option cross-attention layers; port only required 0012 behavior into project-local M0.

**Gate:** Variant isolation and parameter tests pass.

### Task 16: Implement one full-action probability engine

**Files:** Create `model/decoder.py`, `runtime/policy_api.py`, `tests/test_probability_contract.py`.

**Interfaces:** `sample_action()` and `evaluate_action()` return sequence, summed log-prob, summed nonforced entropy, mean-step entropy, value, decision count, forced-terminal marker.

- [ ] Test legality/uniqueness, min STOP mask, optional STOP probability, max forced terminal log-prob=0/entropy=0, teacher/sample/reevaluation parity, deterministic argmax, permutation alignment, and no ordinal feature.
- [ ] Implement one centralized decoder loop shared by all modes.

**Gate:** Fixed-logit and randomized numerical parity tests pass.

### Task 17: Add BC, probe, counterfactual, and invariance metrics

**Files:** Create `objective/bc.py`, `objective/probes.py`, `objective/counterfactual.py`, `objective/invariance.py`, tests.

- [ ] Test complete train/validation metric definitions, detachable nonleaking probes, all approved causal pairs, direction/margin/stability metrics, and permutation agreement/KL.
- [ ] Generate counterfactuals through the causal builder so both members are valid and conserve resources.
- [ ] Emit exact W&B namespaces.

**Phase gate:** M0–M3 CPU forward/backward smoke passes; freeze measured runtime floor before formal runs.

---

## Phase 6 — M4–M5 and Runtime

### Task 18: Implement M4 live ledger and M5 relation/event/value-ready paths

**Files:** Create `model/relational.py`, `model/heads.py`; modify variants; tests; update DESIGN files.

- [ ] Test separate theoretical-deck and live-ledger K/V, unknown-vs-zero, causal view changes, opponent known-hand effects only in M4/M5, relation-bias isolation, 64-event truncation, and finite explicitly uncalibrated `V(s)`.
- [ ] Implement independent capability/availability routing and auditable relation bias.
- [ ] Keep outcome/reward outside encoder.

**Gate:** M4/M5 and controlled counterfactual tests pass.

### Task 19: Prove offline/online causal parity

**Files:** Create `runtime/session.py`, `runtime/inference.py`, `tests/test_runtime_parity.py`.

- [ ] Test identical typed-input hashes at every decision for offline replay and online session, pending-event timing, new-game reset, schema mismatch refusal, legal deterministic output, and unknown-log failure.
- [ ] Implement both paths around the exact same causal state and codecs.

**Phase gate:** Runtime parity is exact, probability tests still pass, engine source unchanged.

---

## Phase 7 — Training and Selection

### Task 20: Implement formal epoch training and immutable checkpoints

**Files:** Create `training/trainer.py`, `training/checkpoints.py`, `training/status.py`, tests; narrowly extend shared checkpoint helper if necessary.

**Interfaces:** `train_version()`; immutable epoch checkpoints plus criterion manifests for latest/loss/exact/representation/decision quality.

- [ ] Test complete train+validation every epoch, identical local/remote scalar names, canonical logging order, resumable latest state, immutable criterion aliases, failure status, W&B-failure survival, and version occupancy refusal.
- [ ] Implement content-addressed checkpoints and full metadata hashes.
- [ ] Generate only approved small snapshot payloads.

**Gate:** Synthetic two-epoch CPU training and failure-recovery tests pass.

### Task 21: Implement two-level checkpoint selection

**Files:** Create `training/selection.py`, `tests/test_checkpoint_selection.py`.

**Interfaces:** version-local preliminary selection and project-level selection; no candidate/report fields until final provenance.

- [ ] Test hard gates, frozen composite/ties, fallback-to-exact reason, complete ranking/rejection reasons, last JSONL record, cross-version ranking, and valid `no_eligible_checkpoint`.
- [ ] Implement immutable selected checkpoint references and one project winner.

**Gate:** Selection can be replayed deterministically from local files.

### Task 22: Implement adaptive GPU scheduler and telemetry

**Files:** Create `training/scheduler.py`, `training/telemetry.py`, `tests/test_scheduler.py`.

- [ ] Test every smoke-passing M0–M5 receives minimum allocation before pruning, hard failures may stop, overlap counts once, idle/failure excluded, allowed gaps categorized, protocol cannot mutate, decisions are persisted, and completion requires ≥36,000 healthy seconds.
- [ ] Implement scheduler as deterministic state transitions; persist every decision before launch.

**Phase gate:** Training/selection/scheduler/W&B suites pass. Verify W&B login/import, GPU, disk, and protocol. Any failure becomes an audited blocker; do not launch formal runs.

### Task 23: Execute formal M0–M5 training campaign

**Files:** Generated immutable versions only under `rl_runs/0013_semantic_goal_policy/versions/`; tracked decisions/status summaries under `experiments/...`.

- [ ] Allocate each formal version through the strict allocator.
- [ ] Run minimum allocation for every healthy M0–M5 with online W&B.
- [ ] Verify each epoch’s JSONL/TensorBoard/W&B parity and checkpoint hashes before scheduling next work.
- [ ] Continue adaptive healthy GPU exploration until ≥36,000 active non-overlapping seconds, unless a documented stop-gate blocker occurs.
- [ ] Produce version-local selections, project ranking, snapshot files, W&B summaries, and complete failure/interruption records.

**Gate:** Either one project winner is eligible or `no_eligible_checkpoint` is recorded without relaxing gates.

---

## Phase 8 — Candidate and Official Evaluation

### Task 24: Export and validate ranked candidate fallbacks

**Files:** Create `export/candidate.py`, tests; generated candidate directories only.

- [ ] Test immutable checkpoint hash use, self-contained package, exactly 60 deck rows, no symlink/absolute dependency, runtime contract match, overwrite refusal, ranked fallback after validation failure, no-eligible outcome, and no catalog/opponent change.
- [ ] Export project winner, validate; if it fails, preserve evidence and try next frozen-ranked eligible version without changing scores.

**Gate:** Exactly one candidate validates or project ends no-eligible.

### Task 25: Run one formal official-engine evaluation and finalize provenance

**Files:** Create project evaluation orchestration/tests; generate formal HTML/index/backlink/final provenance.

- [ ] Test every enabled opponent has exactly 20 games, ten per seat, frozen seeds/profile/hashes/workers contract, overwrite refusal, one candidate only, idempotent same-ID W&B resume, and no invented eval metrics on failure.
- [ ] Rehash catalog, every opponent package, candidate, engine source/build, evaluation code, metric profile, git commit/status immediately before launch; abort on drift.
- [ ] Run candidate validation again, then official evaluation with workers=8 and worker CPU threads=1 unless frozen benchmark overrides.
- [ ] Atomically create report/index/backlink and `selection_provenance_final.json`.
- [ ] Resume the same W&B run with `resume="allow"`, mirror limited `eval/*`, backlink, and a new approved snapshot generation; finish cleanly.

**Gate:** Report game count/seats/hashes agree with provenance, or failure is explicit and no false report exists.

### Task 26: Final audit and design synchronization

**Files:** Update `experiments/0013_semantic_goal_policy/{README.md,DESIGN.md,DESIGN.html,manifest.json,decisions/,versions/}`.

- [ ] Cross-check docs against schema hashes, data audit, tensor shapes, parameters, checkpoint metadata, W&B state, selection, candidate hash, and report.
- [ ] Run `python3 -m unittest discover -s tests -p 'test_*.py'`.
- [ ] Run project tests under `train/0013_semantic_goal_policy/tests`.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_assets`.
- [ ] Run `python3 -m compileall -q evaluation visualization rl_environment train` and `git diff --check`.
- [ ] Run `git diff -- engine/source` and confirm empty.
- [ ] Confirm opponents/catalog unchanged, no Kaggle command, no prohibited W&B upload, and all pre-existing changes preserved.

**Final gate:** Deliver exact tests, W&B run URLs/status, GPU active-time evidence, checkpoint decision logic, candidate/report links, failures, and conclusions. Do not commit, push, promote, or submit.

## Review Checkpoint Rule

At every task: run focused tests, `git status --short`, `git diff --check`, review only declared file diffs, confirm protected changes and engine source remain untouched, and record the gate result. Use subagent implementation plus independent spec-compliance and code-quality reviews before advancing.
