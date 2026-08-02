# 0025 Canonical Semantic Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a new 0025 policy whose actor input is a complete typed canonical game/action contract compiled directly from raw observations and official prototypes, with no legacy tensor dependency.

**Architecture:** Preserve V3 as immutable historical evidence. Add a new `features/canonical/` compiler and `model/canonical/` policy package organized by responsibility: schema, deterministic facts, typed embeddings, state memories, option-to-prototype relations, and autoregressive decoding. Materialize a V2 dataset from the existing audited raw James Cox corpus, then train one canonical semantic arm in repository version `V4_canonical_semantic_foundation`.

**Tech Stack:** Python 3.11, PyTorch, gzip JSONL shards, official read-only engine prototype exports, TensorBoard, W&B online, repository model-only checkpoint and monitoring contracts.

## Global Constraints

- Never modify `engine/source/`.
- Do not import executable code from another numbered training project.
- The new actor payload must not contain `legacy`, source/team identity, or target actions.
- Every categorical field uses an explicit embedding; numeric fields use explicit numeric projection; padding, unknown, and not-applicable remain distinct.
- Every legal option explicitly binds its source/target card instances and applicable Card, Attack, Skill, and Effect prototypes.
- Offline and online compilers must share the same canonical implementation.
- Preserve the existing 981/92 whole-Episode split and exact James Cox deck identity.
- Formal training uses W&B online, per-epoch full validation, model-only checkpoints, and a foreground watchdog.

---

### Task 1: Canonical Feature Contract

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/features/canonical/schema.py`
- Create: `train/0025_semantic_foundation_pretraining/features/canonical/resolver.py`
- Create: `train/0025_semantic_foundation_pretraining/features/canonical/compiler.py`
- Test: `train/0025_semantic_foundation_pretraining/tests/test_canonical_features.py`

**Interfaces:**
- Consumes: raw `actor_observation`, `CausalSnapshot`, exact deck manifest, and `PrototypeIndex`.
- Produces: `compile_canonical_row(row, snapshot, prototypes) -> dict` with named global, card-instance, resource, event, option, option-skill, and option-effect records.

- [x] Write tests requiring no `legacy` key, exact option/action bounds, typed Grass-versus-Lightning energy facts, actual skill IDs, explicit effect parent-option links, and distinct field states.
- [x] Run `python3 -m unittest train.0025_semantic_foundation_pretraining.tests.test_canonical_features -v` and confirm the tests fail before implementation.
- [x] Implement named field specifications and canonical compilation directly from observation locations and engine prototypes.
- [x] Run the focused tests and confirm all canonical feature tests pass.

### Task 2: Structured Canonical Model

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/config.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/typed.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/prototypes.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/state.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/options.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/decoder.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/policy.py`
- Create: `train/0025_semantic_foundation_pretraining/model/canonical/README.md`
- Test: `train/0025_semantic_foundation_pretraining/tests/test_canonical_model.py`

**Interfaces:**
- Consumes: the exact collated tensor keys declared by `canonical/schema.py`.
- Produces: `CanonicalSemanticPolicy.teacher_logits`, `forward`, and `greedy_action` under the existing ordered-option-plus-STOP objective.

- [x] Write tests for finite forward, padding invariance, legal autoregressive decode, option-effect relation sensitivity, typed-energy sensitivity, and fail-closed missing/extra actor tensors.
- [x] Run the focused model tests and confirm they fail before implementation.
- [x] Implement separately readable typed encoders, prototype encoders, state memories, option relation fusion, and decoder.
- [x] Run the focused tests and confirm all model tests pass.

### Task 3: Dataset And Training Integration

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/data/materialize_canonical.py`
- Create: `train/0025_semantic_foundation_pretraining/training/canonical_dataset.py`
- Create: `train/0025_semantic_foundation_pretraining/run_canonical_bc.py`
- Modify: `train/0025_semantic_foundation_pretraining/monitor_training.py`
- Test: `train/0025_semantic_foundation_pretraining/tests/test_canonical_dataset.py`

**Interfaces:**
- Consumes: `V1_james_cox_raging_bolt_raw` and canonical compiler/model interfaces.
- Produces: immutable `V2_james_cox_raging_bolt_canonical` shards and V4 training entrypoint.

- [x] Write tests for shard commitments, whole-split counts, actor/target separation, deterministic collation, and model-only checkpoint reload.
- [x] Implement bounded-memory materialization and canonical batch collation without importing V3 batching or legacy codec.
- [x] Add a one-arm formal training entrypoint with W&B online and the existing per-epoch metrics contract.
- [x] Run all 0025 tests and `git diff --check`.

### Task 4: Materialization, Smoke, And Formal BC

**Files:**
- Create runtime assets under `rl_runs/0025_semantic_foundation_pretraining/dataset/V2_james_cox_raging_bolt_canonical/`.
- Create version assets under `rl_runs/0025_semantic_foundation_pretraining/versions/V4_canonical_semantic_foundation/`.

**Interfaces:**
- Consumes: Tasks 1-3 and the immutable raw corpus.
- Produces: verified dataset, smoke evidence, W&B-synced metrics, and finite model-only checkpoints.

- [x] Materialize a bounded smoke dataset and verify compiler/model forward and backward.
- [x] Materialize all 77,174 decisions and verify hashes, split counts, field coverage, and storage/time statistics.
- [x] Run a GPU smoke with finite loss, gradients, validation, checkpoint reload, and greedy decode.
- [x] Start V4 formal training inside the foreground monitor and continuously inspect process, status, metrics, W&B, checkpoints, GPU/CPU/RAM/swap/SSD, and training alerts until completion.

### Task 5: Authority Sync

**Files:**
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0025_semantic_foundation_pretraining/manifest.json`

**Interfaces:**
- Consumes: current code hashes, dataset manifest, V4 config/status/summary/checkpoints, and measured training results.
- Produces: authoritative model/data-flow documentation and an explicit next evaluation gate.

- [x] Document every canonical field family, tensor shape, relation path, model module, parameter count, loss, and project phase.
- [x] Record what is official rule/runtime fact, deterministic project derivation, learned representation, and still-unknown semantics.
- [ ] Cross-check both design documents against live code and V4 artifacts, then run final tests and HTML/JSON parsing checks.

### Task 6: Canonical Online Runtime And Export

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/deployment/canonical_online_runtime.py`
- Create: `train/0025_semantic_foundation_pretraining/deployment/canonical_inference.py`
- Modify: `train/0025_semantic_foundation_pretraining/export_candidate.py`
- Modify: `train/0025_semantic_foundation_pretraining/model/canonical/config.py`
- Test: `train/0025_semantic_foundation_pretraining/tests/test_canonical_deployment.py`

**Interfaces:**
- Consumes: one live official-engine observation stream, exact registered deck, canonical prototype tables, and a V4 model-only checkpoint.
- Produces: `OnlineCausalEncoder.encode(observation) -> dict[str, Tensor]`, `PortableCanonicalPolicy.select(observation) -> list[int]`, and a self-contained standard candidate package.

- [x] Write failing tests proving online actor contract, strict checkpoint loading, state reset, unique legal decoding, package self-containment, and zero legacy/source fields.
- [x] Add the explicit 128-option inference bound without changing learned parameter shapes.
- [x] Implement the stateful online adapter by calling the same `CausalKnowledge`, `compile_canonical_row`, and `collate_canonical_records` used for materialization.
- [x] Export both V4 checkpoint slots with copied canonical code, prototype commitments, exact deck, official `cg/`, and no symlinks or cross-project imports.

### Task 7: Generic Shared Inference Contract

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Modify: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Consumes: a package model and its online encoder output.
- Produces: a ragged shared batch containing exactly the fields required by that policy; legacy source-conditioned packages retain their existing zero `source_id`, canonical packages receive none.

- [x] Write a failing test showing that `source_id` is only injected for a policy that explicitly declares it.
- [x] Implement a package-level `requires_source_id` capability check and preserve the frozen 0019 behavior.
- [x] Run focused inference-server and 0025 deployment tests.

### Task 8: Same-Contract Official-Engine Evaluation

**Files:**
- Create: `evaluation/arena/candidates/0025_v4_best_validation_loss/`
- Create: `evaluation/arena/candidates/0025_v4_best_greedy_exact/`
- Create: `.tmp/evaluation/0025_v4_best_loss_frozen_all_fail_closed/<run_id>/report.html`
- Create: `.tmp/evaluation/0025_v4_best_exact_frozen_all_fail_closed/<run_id>/report.html`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0025_semantic_foundation_pretraining/manifest.json`

**Interfaces:**
- Consumes: the two validated candidate packages and the immutable 51-deck Frozen 0019 catalog.
- Produces: strict fail-closed local 510-game official-engine reports using 10 games per opponent, balanced seat assignment, identical catalog, worker, device, and metric-profile settings. The official engine internal RNG is not exposed, so checkpoint runs are same-contract independent stochastic samples rather than paired trials. Formal publication is deferred until the user accepts the checkpoint selection.

- [x] Validate both packages and run one-opponent smoke games to prove multi-turn online inference.
- [x] Run best-loss and best-exact against all 51 frozen opponents with `--workers 8 --worker-cpu-threads 1` and the same CUDA shared-inference settings.
- [x] Compare win rate, completion, errors, turn order, matchup deltas, runtime, and checkpoint provenance without treating offline exact accuracy as gameplay strength.
- [x] Synchronize authority docs, rerun all tests and document parsers, and leave formal publication as the next explicit gate.
