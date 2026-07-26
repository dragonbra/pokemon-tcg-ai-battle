# 0014 Faithful Board and Causal Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build experiment `0014_faithful_board_causal_features`, whose single model-ready dataset is a strict actor-visible feature superset of 0010, train faithful A0 on the frozen 07-25 split, and then train the all-feature AC model.

**Architecture:** Reuse `rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1` and its existing split as the sole provenance layer. Materialize one immutable superset containing faithful 0010 board/option tensors plus card capability, public-zone inventory, own-resource ledger, event-memory, and registered-deck channels. A0 selects only the bitwise-compatible 0010 view; AC reads every new family through typed lightweight adapters and board-conditioned retrieval. Fine-grained ablations are reserved for diagnosing AC only if needed.

**Tech Stack:** Python 3.11, PyTorch, standard-library `unittest`, existing official replay archives and raw JSONL shards, TensorBoard/W&B through `TrainingLogger`, official engine evaluation.

## Global Constraints

- Project ID is exactly `0014_faithful_board_causal_features`; use `train/`, `experiments/`, and `rl_runs/` roots with strictly increasing `V<n>_<tag>` versions.
- Never modify `engine/source/`; all strategy-strength claims require official-engine evaluation.
- Raw records and split stay immutable at `rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1`; never overwrite any 0013 dataset.
- The 0014 raw source, split, compiler digest, schema digest, card registry digest, switch set, checkpoint, and evaluation report must be recorded together.
- All feature values must be actor-visible at the decision time. Encode unknown, missing, not-applicable, padding, bounded, and inferred-exact states distinctly; do not use reward, outcome, future frames, or opponent hidden-card identity.
- The first formal control must use the original 0010 ID-only architecture and ordered-pointer action contract as faithfully as practical; capacity changes are a separate experimental variable.
- `training_metrics.jsonl` remains canonical, TensorBoard follows it, and W&B is a failure-isolated mirror only.
- Reuse 0013's optimized training contract: one train update pass per epoch, no static train evaluation, no train greedy decode, BF16 AMP, one shared validation encoding, pinned/non-blocking transfer, full validation every epoch, and host `python3` W&B online logging.

---

## Feature-switch contract

The 0014 materializer always emits the faithful 0010-compatible base and every additional family. Model views select channels without rebuilding data:

| Switch | Information added beyond 0010 | Required audit invariant |
|---|---|---|
| `card_capability` | deterministic structured card semantics: type/stage/HP/type/weakness/resistance/retreat, move cost/damage/effect primitives and capabilities, plus card-ID residual | field provenance is official card CSV; unknown/missing/not-applicable are not equal to zero |
| `public_zone_inventory` | observed own/opponent Active, Bench, hand, discard, deck and Prize counts plus visible attachment/damage aggregates | opponent hand/deck/Prize expose counts only; hidden identities never enter this family |
| `own_resource_ledger` | registered multiplicity and causal own deck/Prize identity counts with epistemic state | exact only after verified full deck membership plus conservation; shuffle never invents identity knowledge |
| `event_memory` | deduplicated actor-visible event stream, event age, event card semantics, and optionally opponent hand known/unknown composition | each source event is consumed once; a hidden draw changes only unknown-card count, never card identity |
| `registered_deck` | actual own 60-card deck tokens with multiplicity and card semantics | exactly 60 registered cards, independent of later visible zones |

`relation_graph` and redesigned board-conditioned Goal-QKV are AC model readers, not evidence that an information family exists.

## Task 1: Allocate 0014 and freeze the faithful-control contract

**Files:**
- Create: `train/0014_faithful_board_causal_features/__init__.py`
- Create: `train/0014_faithful_board_causal_features/config.py`
- Create: `experiments/0014_faithful_board_causal_features/manifest.json`
- Create: `experiments/0014_faithful_board_causal_features/DESIGN.md`
- Create: `experiments/0014_faithful_board_causal_features/DESIGN.html`
- Create: `experiments/0014_faithful_board_causal_features/decisions/001_faithful_superset_contract.md`
- Create: `train/0014_faithful_board_causal_features/configs/pre_run_protocol.json`
- Test: `train/0014_faithful_board_causal_features/tests/test_protocol.py`

**Interfaces:**
- Consumes: immutable `decision_record_v3` rows and their registered deck manifests.
- Produces: `ExperimentConfig`, `FeatureSwitches`, and an immutable protocol that names the exact raw source/split and the faithful `0010_id_only_control` model family.

- [ ] **Step 1: Write the failing allocation/contract test**

```python
def test_control_has_no_new_information_switches() -> None:
    config = ExperimentConfig.from_file(CONFIG)
    assert config.model_family == "0010_id_only_control"
    assert config.feature_switches == FeatureSwitches()
    assert config.feature_schema_version == "faithful_board_causal_input_v1"
```

- [ ] **Step 2: Run the test to verify the missing contract fails**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_protocol`

Expected: FAIL because the 0014 package/configuration does not exist.

- [ ] **Step 3: Add the 0014 roots and immutable contract**

Define `FeatureSwitches(card_capability=False, public_zone_inventory=False, own_resource_ledger=False, event_memory=False, registered_deck=False, relation_graph=False, goal_qkv=False)`. Bind the protocol to `rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1/dataset_reference.json`. State explicitly that A0 and AC use the same rows, split, labels, and superset cache.

- [ ] **Step 4: Run the allocation test**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_protocol`

Expected: PASS; an enabled data/model switch in `0010_id_only_control` is rejected.

## Task 2: Replay-level visibility and causal-transition audit gate

**Files:**
- Create: `train/0014_faithful_board_causal_features/audit/visibility.py`
- Create: `train/0014_faithful_board_causal_features/audit/replay_audit.py`
- Create: `train/0014_faithful_board_causal_features/tests/test_visibility_audit.py`
- Create: `experiments/0014_faithful_board_causal_features/data_audit/visibility_transition_audit.json`
- Modify: `experiments/0014_faithful_board_causal_features/DESIGN.md`

**Interfaces:**
- Consumes: chronological raw rows, `event_cursor`, actor observation, and canonical replay source hashes.
- Produces: `TransitionAudit` with counts and exemplars for deck view, Prize inference, draw, move, shuffle, reveal, hand view, and duplicate-log behavior. `assert_transition_audit_ready(audit)` raises on an unclassified or contradictory transition.

- [ ] **Step 1: Write failing transition tests using a two-decision fixture**

```python
def test_duplicate_cumulative_logs_are_consumed_once() -> None:
    audit = audit_episode_rows(rows_with_same_prior_log_twice())
    assert audit.event_consumptions == 1

def test_hidden_opponent_draw_only_increases_unknown_hand_slot() -> None:
    state = replay_knowledge(rows_with_opponent_hidden_draw())[-1]
    assert state.opponent_hand.unknown_slots == 1
    assert state.opponent_hand.known == ()

def test_exact_prize_requires_full_visible_deck_and_conservation() -> None:
    assert replay_knowledge(rows_with_partial_search())[-1].own_prize.is_unknown
    assert replay_knowledge(rows_with_full_deck_view())[-1].own_prize.is_inferred_exact
```

- [ ] **Step 2: Run the audit tests and record current failures**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_visibility_audit`

Expected: FAIL before implementation; specifically, no code may assume that `select.deck` is full membership without replay evidence, or that the same log array is incremental.

- [ ] **Step 3: Implement cursor-aware transition replay**

Use the row's `event_cursor.incoming_log_count` and canonical frame identity to select only newly visible logs. Build a typed transition mapper from actual engine numeric log payloads observed in the audit; unknown numeric types fail closed and are counted. Preserve a source-event ID per consumed event.

For own resources, permit exact deck/Prize identity only when an audited full-deck view contains exactly `deckCount` cards and registered-deck conservation holds. On a shuffle, retain membership but clear order; on an unidentified deck/Prize transition, degrade only the affected identity count to bounded/unknown.

For opponent hands, represent `known_cards` only after an actor-visible reveal with retained card IDs/serials. Represent a hidden draw as `unknown_slots += 1`; a later hand count reconciles the total without assigning an ID. If the raw projection lacks the engine field needed for a verified hand reveal, record that capability as unavailable rather than synthesizing it.

- [ ] **Step 4: Run fixture tests and an audit sample**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_visibility_audit`

Run: `python3 -m train.0014_faithful_board_causal_features.audit.replay_audit --source rl_runs/0013_semantic_goal_policy/dataset/V1_causal_semantic_v1 --output experiments/0014_faithful_board_causal_features/data_audit/visibility_transition_audit.json --sample-groups 256`

Expected: PASS; audit lists every observed log type, reports zero double-consumed events, and distinguishes verified capabilities from unavailable ones.

## Task 3: Implement the 0010-faithful board and option base schema

**Files:**
- Create: `train/0014_faithful_board_causal_features/features/schema.py`
- Create: `train/0014_faithful_board_causal_features/features/board.py`
- Create: `train/0014_faithful_board_causal_features/features/options.py`
- Create: `train/0014_faithful_board_causal_features/tests/test_faithful_0010_parity.py`
- Modify: `experiments/0014_faithful_board_causal_features/DESIGN.md`

**Interfaces:**
- Consumes: one actor-visible observation and legal options.
- Produces: `FaithfulBoardFeatures` with `global_cat[4]`, `global_num[12]`, `entity_cat[N,7]`, `entity_num[N,5]`, `entity_mask[N]`, `option_cat[O,12]`, `option_mask[O]`, targets, min/max count, and named entity/source/target indexes.

- [ ] **Step 1: Write a parity fixture with active Pokémon, attachments, evolution, statuses, stadium, looking, select.deck, and a source-target option**

```python
def test_new_base_matches_0010_codec_for_every_base_tensor() -> None:
    old = IDOnlyCodec(IDOnlyConfig()).encode(OBSERVATION, ACTION)
    new = compile_faithful_base(OBSERVATION, ACTION)
    assert new.global_cat == old["global_cat"]
    assert new.entity_cat == old["entity_cat"]
    assert new.entity_num == old["entity_num"]
    assert new.option_cat == old["option_cat"]
```

- [ ] **Step 2: Run the parity test to verify it fails**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_faithful_0010_parity`

Expected: FAIL because no faithful board compiler exists.

- [ ] **Step 3: Implement the base by porting behavior, not by approximating it**

Port the 0010 semantics for relative owner, zones, parent entity index, status bits, damage ratio, energy/tool/evolution counts, `appearThisTurn`, all global flags/counts, stadium/looking/select.deck, and source/target option resolution. Preserve the 0010 capacity/configuration for the control; if a row exceeds a control limit, audit and reject it consistently rather than silently truncating it.

- [ ] **Step 4: Run parity and dataset coverage checks**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_faithful_0010_parity`

Expected: PASS on synthetic fixtures and a committed raw-row sample; all differences require an explicit, documented source-contract reason.

## Task 4: Add all five data families to one superset compiler

**Files:**
- Create: `train/0014_faithful_board_causal_features/features/card_capability.py`
- Create: `train/0014_faithful_board_causal_features/features/public_zone_inventory.py`
- Create: `train/0014_faithful_board_causal_features/features/ledger.py`
- Create: `train/0014_faithful_board_causal_features/features/events.py`
- Create: `train/0014_faithful_board_causal_features/features/registered_deck.py`
- Create: `train/0014_faithful_board_causal_features/features/compiler.py`
- Create: `train/0014_faithful_board_causal_features/tests/test_feature_switches.py`
- Modify: `experiments/0014_faithful_board_causal_features/DESIGN.md`

**Interfaces:**
- Consumes: `FaithfulBoardFeatures`, `CausalTransitionState`, `FeatureSwitches`, and `CardSemanticRegistry`.
- Produces: `Compiled0014Features`; disabled switches produce no reader-visible token/channel and cannot alter the base tensor values.

- [ ] **Step 1: Write switch-isolation tests**

```python
def test_disabling_all_switches_is_exactly_the_faithful_base() -> None:
    assert compile_0014(ROW, FeatureSwitches()).base == compile_faithful_base(OBS, ACTION)

def test_each_switch_changes_only_its_declared_family() -> None:
    base = compile_0014(ROW, FeatureSwitches())
    ledger = compile_0014(ROW, FeatureSwitches(own_resource_ledger=True))
    assert ledger.base == base.base
    assert ledger.ledger_tokens != ()
    assert ledger.event_tokens == ()
```

- [ ] **Step 2: Run the switch tests and verify they fail**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_feature_switches`

Expected: FAIL because the five compiler families do not exist.

- [ ] **Step 3: Implement all five channels**

Use the audited 0013 card registry as the source for card capability. Emit public-zone counts for both players while keeping opponent hidden identities absent. Use only audited causal transition state for ledger/event values. Add registered-deck tokens from the exact row deck manifest. If opponent-hand knowledge passed Task 2, expose it as a separately named event-memory subfield with known IDs/serials and unknown-slot count; otherwise keep it absent and document the limitation. The materializer writes all families on every row; switches are model-view metadata, not separate dataset builds.

- [ ] **Step 4: Run isolation, invariance, and leakage tests**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_feature_switches`

Expected: PASS; changing a future frame, reward, terminal field, or opponent hidden-card ID cannot change compiled features.

## Task 5: Materialize a new immutable dataset and train the faithful 0010 control

**Files:**
- Create: `train/0014_faithful_board_causal_features/data/materialize.py`
- Create: `train/0014_faithful_board_causal_features/model/id_only_control.py`
- Create: `train/0014_faithful_board_causal_features/training/campaign.py`
- Create: `train/0014_faithful_board_causal_features/tests/test_materialized_parity.py`
- Create: `rl_runs/0014_faithful_board_causal_features/dataset/V1_full_feature_superset/` at execution time only
- Create: `rl_runs/0014_faithful_board_causal_features/versions/V1_0010_faithful_control/` at execution time only

**Interfaces:**
- Consumes: frozen raw records and the full enabled materialization schema.
- Produces: hash-committed tensor shards and an independently initialized 0010-compatible pointer model whose config, parameter count, action decoder, and BC objective are recorded.

- [ ] **Step 1: Write a raw-cache parity test**

```python
def test_materialized_control_matches_raw_compilation_for_1024_rows() -> None:
    for raw, cached in paired_rows(1024):
        assert materialize_all_features(raw) == load_cached(cached)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_materialized_parity`

Expected: FAIL because no 0014 materializer exists.

- [ ] **Step 3: Implement atomic materialization and control training**

Publish only into the unused `V1_full_feature_superset` dataset directory after all shard hashes, split counts, compiler/schema digests, and visibility audit SHA are written. The control loader selects only faithful 0010 base tensors; it must not pass zero-filled new tokens, change masks, change sequence length, or instantiate new readers. Record the source/split comparison against 0010 as a new-dataset control, not as a continuation or reproduction of old 0010 metrics.

- [ ] **Step 4: Validate, train, and evaluate the control**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_materialized_parity`

Run: `python3 -m unittest discover -s tests -p 'test_*.py'`

Run: `python3 -m compileall -q train/0014_faithful_board_causal_features`

Expected: raw/cache parity passes, complete per-epoch validation exists, local metrics/TensorBoard/W&B contract is intact, and any later official-engine evaluation uses a self-contained candidate only after user authorization.

## Task 6: Add the AC reader model and reserve diagnostic ablations

**Files:**
- Create: `train/0014_faithful_board_causal_features/model/superset_policy.py`
- Create: `experiments/0014_faithful_board_causal_features/ablations.json`
- Create: `train/0014_faithful_board_causal_features/tests/test_ablation_contract.py`
- Modify: `experiments/0014_faithful_board_causal_features/DESIGN.md`

**Interfaces:**
- Consumes: a `Compiled0014Features` schema and one immutable `AblationSpec`.
- Produces: one model whose enabled readers exactly match the ablation spec.

- [ ] **Step 1: Write the ablation-contract test**

```python
def test_primary_campaign_runs_a0_then_ac_on_one_dataset() -> None:
    specs = load_ablation_specs(PATH)
    assert specs[0].name == "A0_0010_faithful_control"
    assert specs[1].name == "AC_all_features_lightweight_readers"
    assert specs[0].dataset_sha256 == specs[1].dataset_sha256
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_ablation_contract`

Expected: FAIL because the ablation registry does not exist.

- [ ] **Step 3: Register the initial ladder**

Register primary runs `A0_0010_faithful_control` and `AC_all_features_lightweight_readers`. AC uses local card-semantic residual adapters, a global public-zone adapter, one token per registered card identity combining deck/ledger fields, a causal event encoder, and board-conditioned role retrieval with zero-initialized normalized gates. Reserve A1-A5, relation-only, Goal-off, and capacity-matched controls as diagnostic specs that are not allocated unless AC underperforms or becomes numerically unhealthy. Every spec names the exact dataset digest, model capacity, optimizer, seed, and expected input channels.

- [ ] **Step 4: Run the contract test**

Run: `python3 -m unittest -v train.0014_faithful_board_causal_features.tests.test_ablation_contract`

Expected: PASS; a spec that enables a model reader for absent data, changes capacity silently, or changes multiple unregistered variables is rejected.

## Self-review

- The plan covers: one full superset cache, 0010 faithful A0, direct all-feature AC, five information families, replay/causal audit before publication, 0013 training optimizations, model-reader separation, immutable versioned runs, and validation/evaluation boundaries.
- The visibility audit is deliberately a hard gate: current 0013 code has an internal opponent-hand representation but does not tensorize it, filters reveal-card payloads from raw logs, and does not consume `event_cursor`; none of those semantics may be claimed for 0014 until replay evidence and tests establish them.
- No step modifies official engine source or overwrites historical 0013 assets.
