# 0028 Universal Semantic Foundation Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, readable universal BC project that preserves the complete 0025 V4 canonical actor semantics while training from every audited official winner perspective dated 2026-07-10 through the latest locally available official Episode archive.

**Architecture:** Raw official Episodes are projected into an actor-visible causal decision record, enriched only with official Card/Attack/Skill/Effect prototype facts and a causal exact-deck ledger, then compiled into a strict named tensor contract. The policy is organized as four explicit stages: prototype lookup, state-memory encoding, legal-option cross-attention, and ordered option-pointer decoding with STOP. Team/source identity remains provenance only and exact deck identity is represented through the registered-deck multiset, never through source/persona embeddings.

**Tech Stack:** Python 3.11, PyTorch, official engine-derived JSON prototype sidecars, gzip JSONL datasets, unittest, TensorBoard, W&B online, official engine runtime evaluation.

## Global Constraints

- Project ID is exactly `0028_universal_semantic_foundation_pretraining`; no `0027` project paths may be created.
- `train/0028_universal_semantic_foundation_pretraining/` must not import executable code from any numbered project.
- `engine/source/` is read-only; derived prototype artifacts may be built only under `engine/build/` and copied into 0028 as immutable assets.
- Actor forward accepts no source/team/persona identity, copied target action, raw replay payload, or legacy tensor compatibility fields.
- Exact registered deck, visible board, causal ledger, recent events, turn budgets, and known/unknown information boundaries remain actor-visible.
- Categorical identities use independent embeddings; continuous/count values use normalized numeric projections plus explicit known/unknown/not-applicable state.
- Every official Episode contributes only its unique positive winner perspective; whole Episodes are assigned deterministically to train or validation.
- The initial date interval is `2026-07-10` through `2026-08-01`, the latest locally available official archive at project creation.
- Formal training uses one train pass and one complete teacher-forced plus greedy validation pass per epoch, model-only checkpoints, TensorBoard, and W&B online.
- No full dataset materialization or formal 0028 training may contend with the active 0026 formal RL run; only low-resource smoke work is allowed until 0026 completes.

---

### Task 1: Project Contract And Authority Documents

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/__init__.py`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/manifest.json`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.md`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.html`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/decisions/2026-08-02-initial-contract.md`

**Interfaces:**
- Consumes: 0025 V4 code and artifacts as immutable design provenance only.
- Produces: authoritative project ID, schema names, actor visibility boundary, model data flow, and phase gate.

- [ ] **Step 1: Write the manifest validation test**

```python
def test_manifest_declares_persona_free_actor_and_0025_provenance():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["project_id"] == "0028_universal_semantic_foundation_pretraining"
    assert manifest["actor_source_identity_visible"] is False
    assert manifest["implementation_lineage"]["executable_dependency"] is False
```

- [ ] **Step 2: Run the test and verify it fails because 0028 does not exist**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_project_contract`

Expected: FAIL with an import or missing manifest error.

- [ ] **Step 3: Add the package and authority documents**

The documents must state that 0028 preserves 0025 V4 semantics, uses all audited winner perspectives, has no source-visible input, and is initially limited to smoke work while 0026 is active.

- [ ] **Step 4: Run the contract test**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_project_contract`

Expected: PASS.

### Task 2: Named Tensor And Prototype Contracts

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/contracts/fields.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/contracts/batch.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/domain/prototypes.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json`
- Create: `train/0028_universal_semantic_foundation_pretraining/assets/official_full_engine_prototypes_v1.json`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_contracts.py`

**Interfaces:**
- Consumes: immutable 0025 prototype sidecars after SHA-256 verification.
- Produces: `DecisionBatch`, `PrototypeIndex`, `ACTOR_KEYS`, field names, vocabulary sizes, and schema version `0028_canonical_semantic_decision_v1`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_actor_keys_exclude_provenance_and_legacy():
    assert "source_id" not in ACTOR_KEYS
    assert "team_name" not in ACTOR_KEYS
    assert "legacy" not in ACTOR_KEYS

def test_batch_rejects_missing_and_extra_tensors():
    with pytest.raises(ValueError, match="missing"):
        DecisionBatch.from_mapping({})
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_contracts`

Expected: FAIL because contracts are absent.

- [ ] **Step 3: Implement named contracts and prototype validation**

Use separate categorical vocabularies for every field. Preserve the 0025 V4 global/card/resource/event/option/skill/effect fields and attach human-readable names and shape assertions to each tensor family.

- [ ] **Step 4: Verify the sidecar commitments and tests**

Run: `sha256sum train/0025_semantic_foundation_pretraining/assets/official_*_prototypes_v1.json train/0028_universal_semantic_foundation_pretraining/assets/official_*_prototypes_v1.json`

Expected: each 0028 file matches its corresponding 0025 immutable source.

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_contracts`

Expected: PASS.

### Task 3: Readable Four-Stage Policy

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/model/config.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/typed_fields.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/prototype_encoder.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/state_encoder.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/option_encoder.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/action_decoder.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/policy.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/model/README.md`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_model.py`

**Interfaces:**
- Consumes: `DecisionBatch` and `PrototypeIndex`.
- Produces: `SemanticPolicy.encode_state`, `SemanticPolicy.encode_options`, `SemanticPolicy.forward`, `SemanticPolicy.teacher_logits`, and `SemanticPolicy.greedy_action`.

- [ ] **Step 1: Write tests for stage boundaries and exact batch usage**

```python
def test_forward_exposes_four_named_stages(model, batch):
    state = model.encode_state(batch)
    options = model.encode_options(batch, state)
    logits = model.decode_next(batch, state, options)
    assert state.tokens.ndim == 3
    assert options.ndim == 3
    assert logits.shape[-1] == batch.option_mask.shape[1] + 1

def test_every_actor_tensor_is_consumed_or_declared_mask(model):
    assert model.expected_batch_keys == ACTOR_KEYS | MASK_KEYS | {"targets"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_model`

Expected: FAIL because model modules are absent.

- [ ] **Step 3: Implement the four stages**

`StateEncoder` creates global, card-instance, exact-deck ledger, and event tokens and retains the full token memory. `OptionEncoder` binds option fields to source/target card instances and Card/Attack/Skill/Effect prototypes, then cross-attends to full state memory. `ActionDecoder` autoregressively selects ordered legal option indices and an explicit STOP action.

- [ ] **Step 4: Add residual and mask invariance tests**

```python
def test_padding_does_not_change_logits(model, one_row_batch, padded_batch):
    torch.testing.assert_close(model(one_row_batch), model(padded_batch))

def test_energy_type_change_reaches_logits(model, lightning_batch, grass_batch):
    assert not torch.allclose(model(lightning_batch), model(grass_batch))
```

- [ ] **Step 5: Run model tests**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_model`

Expected: PASS with finite logits and legal greedy actions.

### Task 4: Universal Winner Catalog

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/data/replay_contract.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/data/winner_catalog.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_winner_catalog.py`
- Create: `experiments/0028_universal_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json`

**Interfaces:**
- Consumes: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD.zip` and audited patches.
- Produces: a deduplicated Episode catalog with winner player index, exact 60-card deck hash/counts, seat, source provenance, payload hash, locator, and deterministic whole-Episode split.

- [ ] **Step 1: Write catalog fail-closed tests**

```python
def test_catalog_keeps_only_unique_positive_terminal_winner(): ...
def test_catalog_rejects_non_60_card_registration(): ...
def test_source_identity_is_provenance_only(): ...
def test_duplicate_episode_with_different_payload_fails_closed(): ...
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_winner_catalog`

Expected: FAIL because the catalog implementation is absent.

- [ ] **Step 3: Implement deterministic scanning and split assignment**

The split digest is `0028_episode_split_v1|20260802|<deck_sha256>|<seat>|<episode_id>`. Team/source fields are retained in audit records but omitted from actor records.

- [ ] **Step 4: Run a two-archive smoke audit while 0026 is active**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.data.winner_catalog --archives-dir data/raw/episodes/archives --start-date 2026-07-10 --end-date 2026-07-11 --workers 1 --output .tmp/0028/catalog_smoke.json`

Expected: nonzero unique winners, two audited archives, zero duplicate identity conflicts.

- [ ] **Step 5: Build the full catalog after 0026 releases resources**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.data.winner_catalog --archives-dir data/raw/episodes/archives --patches-dir data/raw/episodes/patches --start-date 2026-07-10 --end-date 2026-08-01 --workers 8 --output experiments/0028_universal_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json`

Expected: 23 archives audited, unique Episode IDs, exact 60-card winning decks, and deterministic train/validation counts.

### Task 5: Causal Decision Compiler

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/features/knowledge.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/features/relations.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/features/compile_decision.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/features/collate.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_features.py`

**Interfaces:**
- Consumes: actor-visible observation, ordered legal action, exact deck manifest, causal prior observations, and `PrototypeIndex`.
- Produces: canonical record mappings accepted exactly by `DecisionBatch`.

- [ ] **Step 1: Write semantic regression tests**

```python
def test_attack_option_binds_attack_skill_and_effect_ids(): ...
def test_grass_and_lightning_energy_have_different_typed_deficits(): ...
def test_exact_deck_multiset_survives_collation(): ...
def test_opponent_private_cards_never_enter_actor_tensors(): ...
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_features`

Expected: FAIL because feature modules are absent.

- [ ] **Step 3: Implement causal compilation with explicit relations**

Card parent, option source/target, option-skill parent, and option-effect parent remain one-based relational indices. Numeric missingness uses explicit state tensors rather than conflating unknown, not-applicable, and numeric zero.

- [ ] **Step 4: Run feature and model integration tests**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_features train.0028_universal_semantic_foundation_pretraining.tests.test_model`

Expected: PASS.

### Task 6: Audited Dataset Materialization

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/data/raw_decisions.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/data/materialize.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/training/dataset.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_dataset.py`

**Interfaces:**
- Consumes: full winner catalog and official Episode locators.
- Produces: immutable gzip JSONL train/validation shards, manifest hashes, row/byte counts, maxima, exclusion audit, and per-source/per-deck metrics metadata under `rl_runs/0028.../dataset/V1_universal_winners_canonical/`.

- [ ] **Step 1: Write interrupted-build and commitment tests**

```python
def test_partial_output_is_never_published_as_complete(): ...
def test_loader_rejects_shard_hash_mismatch(): ...
def test_episode_never_crosses_train_and_validation(): ...
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_dataset`

Expected: FAIL because materialization is absent.

- [ ] **Step 3: Implement atomic sharding and manifest commitments**

Write into a unique temporary sibling directory, fsync files, validate counts and SHA-256 commitments, then atomically rename to the unused V1 dataset path.

- [ ] **Step 4: Run a 32-decision smoke and size benchmark**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.data.materialize --catalog .tmp/0028/catalog_smoke.json --output .tmp/0028/dataset_smoke --max-decisions 32 --workers 1`

Expected: reload, collate, forward, and backward pass; report raw/canonical bytes per decision and encoding milliseconds per decision.

- [ ] **Step 5: Materialize the full dataset after 0026 completes**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.data.materialize --catalog experiments/0028_universal_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json --output rl_runs/0028_universal_semantic_foundation_pretraining/dataset/V1_universal_winners_canonical --workers 8`

Expected: complete committed dataset with no partial status and an auditable capacity estimate.

### Task 7: Formal Universal BC Trainer

**Files:**
- Create: `train/0028_universal_semantic_foundation_pretraining/training/objective.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/training/checkpoints.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/training/logger.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/training/trainer.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/run_bc.py`
- Create: `train/0028_universal_semantic_foundation_pretraining/monitor_training.py`
- Test: `train/0028_universal_semantic_foundation_pretraining/tests/test_training.py`

**Interfaces:**
- Consumes: `V1_universal_winners_canonical` and `SemanticPolicy`.
- Produces: versioned V1 training config, JSONL metrics, TensorBoard, W&B online run, model-only checkpoints, status, and monitor heartbeat/alerts.

- [ ] **Step 1: Write objective, checkpoint, and epoch-contract tests**

```python
def test_loss_masks_after_stop_and_padded_targets(): ...
def test_checkpoint_contains_no_optimizer_rng_or_loader_state(): ...
def test_epoch_has_one_train_pass_and_one_full_validation_pass(): ...
def test_source_identity_never_reaches_forward(): ...
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3 -m unittest -v train.0028_universal_semantic_foundation_pretraining.tests.test_training`

Expected: FAIL because training modules are absent.

- [ ] **Step 3: Implement formal training and monitoring**

Use `trainer/epoch`, `bc/optimization/*`, and `bc/validation/*`. Flush `training_metrics.jsonl` before TensorBoard and W&B. Store only latest, best validation loss, best teacher exact, and best greedy exact model-only checkpoints.

- [ ] **Step 4: Run a CPU/CUDA micro-training smoke without W&B**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.run_bc --dataset .tmp/0028/dataset_smoke --version V1_universal_semantic_foundation_smoke --epochs 2 --batch-size 4 --wandb-mode disabled`

Expected: two finite epochs, two complete validation records, four or fewer model-only checkpoint slots, and clean monitor completion.

- [ ] **Step 5: Update both authority documents with measured shapes and costs**

Cross-check `DESIGN.md` and `DESIGN.html` against schema constants, parameter count, dataset manifest, and smoke metrics. Do not publish estimates as measured facts.

### Task 8: Full Verification And Formal Training Gate

**Files:**
- Modify: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0028_universal_semantic_foundation_pretraining/DESIGN.html`
- Create: `rl_runs/0028_universal_semantic_foundation_pretraining/versions/V1_universal_semantic_foundation/artifact/training_config.json`

**Interfaces:**
- Consumes: all prior tasks and completed 0026 resource release.
- Produces: a ready-to-launch formal V1 universal BC package; launching the long run remains a separately observed action.

- [ ] **Step 1: Run the complete 0028 test suite**

Run: `python3 -m unittest discover -v -s train/0028_universal_semantic_foundation_pretraining/tests`

Expected: all tests PASS.

- [ ] **Step 2: Verify self-containment and official-engine immutability**

Run: `rg -n 'train\.(00[0-2][0-9])_' train/0028_universal_semantic_foundation_pretraining`

Expected: no executable cross-project import.

Run: `git diff -- engine/source`

Expected: empty.

- [ ] **Step 3: Verify docs and source consistency**

Run: `python3 -m train.0028_universal_semantic_foundation_pretraining.tools.audit_design --design experiments/0028_universal_semantic_foundation_pretraining/DESIGN.md --manifest experiments/0028_universal_semantic_foundation_pretraining/manifest.json`

Expected: schema, parameter count, data interval, actor visibility, and phase all match code and artifacts.

- [ ] **Step 4: Run formatting and diff checks**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Prepare, but do not silently detach, the formal training command**

```bash
python3 -m train.0028_universal_semantic_foundation_pretraining.monitor_training \
  --version V1_universal_semantic_foundation \
  -- \
  python3 -m train.0028_universal_semantic_foundation_pretraining.run_bc \
  --dataset rl_runs/0028_universal_semantic_foundation_pretraining/dataset/V1_universal_winners_canonical \
  --version V1_universal_semantic_foundation \
  --wandb-mode online
```

Expected: the formal run starts only in a long-lived foreground terminal and is continuously monitored at intervals no longer than 60 seconds.
