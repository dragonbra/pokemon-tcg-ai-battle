# 0031 Rule-Faithful Semantic Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained next-generation BC pretraining project whose prototypes and observations preserve official-engine facts and relations without substituting compressed, guessed, or incorrectly derived values.

**Architecture:** A read-only C++ exporter inventories and serializes the complete rule-relevant `CardMaster`, `Attack`, `Skill`, `Effect`, `Target`, and condition structures. A causal compiler turns each public current frame into a typed relational graph of physical instances, prototype-program nodes, persistent known-information ledger entries, bounded semantic events, and legal options; it preserves direct facts and explicit unknown states but does not reconstruct a rules engine or precompute attack gaps, final damage, KO, discard priority, or other arithmetic the model can learn. A relation-aware policy encodes each field according to its true data type, preserves ordered rule programs, and points to ordered legal options with no source/persona input.

**Tech Stack:** Python 3.11, C++20 read-only official-engine header extraction, PyTorch, gzip JSONL, unittest, TensorBoard, W&B online, official engine runtime evaluation.

## Global Constraints

- Project ID is exactly `0031_rule_faithful_semantic_foundation_pretraining`.
- `train/0031_rule_faithful_semantic_foundation_pretraining/` must not import executable code from another numbered training project.
- `engine/source/` is read-only. The 0031 exporter source lives under `train/0031.../tools/`; binaries and runtime probes are built only under `engine/build/0031/`.
- Official full-engine fields take precedence over the compressed public API. Public values may never overwrite a more expressive full-engine value.
- Every official prototype field must appear in `prototype_field_inventory.json` with disposition `actor`, `relation`, or `audit_only`; `audit_only` requires a concrete non-gameplay rationale.
- Every observation, log, and legal-option field must appear in `observation_field_inventory.json` with the same explicit disposition contract.
- An unavailable value is encoded as `UNKNOWN`; it is never encoded as zero, Colorless, false, or not-applicable.
- `PAD`, `PRESENT`, `UNKNOWN`, and `NOT_APPLICABLE` are field-specific states and must remain bound to their field after model projection.
- Every semantic field declares exactly one encoding kind: categorical enum/boolean/symbol uses its own embedding; exact numeric constants and counts use their exact scalar value with a field-specific projection; continuous values use a continuous projection; bitmasks use lossless raw identity plus named bits; references use typed relation edges.
- Physical cards retain local instance identity, relative owner/controller, current zone/slot, and explicit attachment/evolution relations. Raw serial magnitude is audit metadata, not a learnable scalar.
- Energy uses the official bitmask and unit count. `SPECIAL_LOOKUP` is distinct from Colorless and padding.
- Direct runtime facts such as `benchMax`, physical attached Energy identities, resolved current `energies`, prototype attack cost requirements, base retreat cost, and engine-exposed remaining retreat payment are actor-visible. An effective dynamic attack or retreat cost is actor-visible only when the public observation provides an exact factual source; otherwise it remains `UNKNOWN` and the model receives the visible modifiers and rule semantics.
- Derived answers such as typed gap, extra Energy, newly enabled attack, remove-Energy counterfactual, final damage, KO, preferred discard, or “surplus Energy” are excluded from actor forward.
- Ordered attacks, skills, triggers, effects, target areas, and target conditions retain identity, parent, phase, and ordinal relations; they are not mean-pooled into unordered summaries.
- Current-frame facts are complete. History is retained only as bounded semantic events and rule-justified persistent knowledge; 0031 does not attempt to replay every state transition inside the feature compiler.
- Team/source identity remains provenance only and never enters actor forward.
- Historical 0025/0028 datasets and checkpoints remain immutable evidence. 0031 rematerializes from raw decisions and trains from random initialization.
- Formal training uses one train update pass and one complete teacher-forced plus greedy validation pass per epoch, model-only checkpoints, TensorBoard, W&B online, and a foreground watchdog.
- `experiments/0031.../DESIGN.md` and `DESIGN.html` must be synchronized whenever the schema, model, objective, or project phase changes.

---

### Task 1: Authority Documents And Fail-Closed Inventories

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/__init__.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/prototype_field_inventory.json`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/observation_field_inventory.json`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/manifest.json`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/decisions/2026-08-04-initial-contract.md`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_project_contract.py`

**Interfaces:**
- Consumes: official engine structures and the 0025/0028 audit as immutable provenance.
- Produces: project identity, field disposition vocabulary, authority boundary, and phase gates.

- [ ] **Step 1: Write the failing authority test**

```python
def test_manifest_declares_truth_precedence_and_persona_free_actor():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["project_id"] == "0031_rule_faithful_semantic_foundation_pretraining"
    assert manifest["prototype_truth_precedence"] == ["official_full_engine", "public_api"]
    assert manifest["actor_source_identity_visible"] is False
    assert manifest["derived_actor_answers"] == []
```

- [ ] **Step 2: Run the test and verify the missing project failure**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_project_contract`

Expected: FAIL because the 0031 package and manifest do not exist.

- [ ] **Step 3: Create the authority files and inventories**

The inventories use records with exact keys:

```json
{
  "owner_type": "CardMaster",
  "source_field": "energyType",
  "disposition": "actor",
  "actor_field": "energy_type_mask",
  "encoding": "raw_bitmask_plus_multihot",
  "rationale": "Official type alternatives are rule semantics."
}
```

- [ ] **Step 4: Add a no-unexplained-exclusion test**

```python
def test_audit_only_fields_have_non_gameplay_rationale():
    for row in load_inventory(PROTOTYPE_INVENTORY):
        if row["disposition"] == "audit_only":
            assert row["rationale"].startswith("Non-gameplay:")
```

- [ ] **Step 5: Run the authority test**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_project_contract`

Expected: PASS.

- [ ] **Step 6: Commit the authority boundary**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining experiments/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "docs: define 0031 rule-faithful semantic contract"
```

### Task 2: Complete Read-Only Prototype Export

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/tools/full_engine_prototype_export.cpp`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/tools/export_prototypes.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/domain/prototypes.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/assets/official_full_engine_prototypes_v2.json`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_prototype_export.py`

**Interfaces:**
- Consumes: read-only official `CardMaster`, `Attack`, `Skill`, `Effect`, `Target`, trigger, and condition tables.
- Produces: `PrototypeIndex.load(path)` and schema `0031_official_full_engine_prototypes_v2`.

- [ ] **Step 1: Write the Team Rocket Energy regression test**

```python
def test_team_rocket_energy_keeps_official_semantics():
    card = prototypes.cards[15]
    assert card["card_type"] == 6
    assert card["energy_type_mask"] == 80
    assert card["energy_count"] == 2
    assert card["only_team_rocket"] is True
    assert card["ability_skill_id"] == 8
```

- [ ] **Step 2: Write source-field coverage tests**

Parse field declarations from `Card.h`, `Skill.h`, and the structures containing `Effect`, `Target`, triggers, and target conditions. Normalize C++ names to the inventory `source_field` values and assert exact set equality.

```python
def test_every_official_struct_field_has_a_disposition():
    declared = parse_official_struct_fields(ENGINE_SOURCE)
    inventoried = inventory_source_fields(PROTOTYPE_INVENTORY)
    assert declared == inventoried
```

- [ ] **Step 3: Run the tests and verify that v1 omits fields**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_prototype_export`

Expected: FAIL and identify at least `CardMaster.onlyTeamRocket` as absent.

- [ ] **Step 4: Implement the exporter**

Export raw enums and bitmasks without `EnergyTypeIndex()` compression. Export every ordered list with an explicit ordinal. Export name-bearing rule symbols through a deterministic symbol table so `evolvesFrom` and `Target.name` remain relational rather than free-text scalars.

The build command is:

```bash
mkdir -p engine/build/0031
g++ -std=c++20 -O2 -I"engine/source/ptcgProgram 22" \
  train/0031_rule_faithful_semantic_foundation_pretraining/tools/full_engine_prototype_export.cpp \
  -o engine/build/0031/full_engine_prototype_export
```

- [ ] **Step 5: Export atomically and validate content commitment**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.tools.export_prototypes \
  --binary engine/build/0031/full_engine_prototype_export \
  --output train/0031_rule_faithful_semantic_foundation_pretraining/assets/official_full_engine_prototypes_v2.json
```

Expected: 1,267 cards, 1,556 attacks, 433 skills, a deterministic SHA-256 commitment, and no unclassified source fields.

- [ ] **Step 6: Run prototype tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_prototype_export`

Expected: PASS, including Team Rocket, Prism, Neo Upper, and Ignition Energy fixtures.

- [ ] **Step 7: Commit the prototype exporter**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "feat: export complete official prototype semantics"
```

### Task 3: Typed Schema Without False Defaults

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/fields.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/batch.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/domain/field_state.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_typed_contract.py`

**Interfaces:**
- Consumes: prototype v2 and observation field inventories.
- Produces: `DecisionRecord`, `DecisionBatch`, named field groups, relation groups, and `FieldState`.

- [ ] **Step 1: Write false-default rejection tests**

```python
def test_unknown_energy_is_not_colorless():
    value = EnergyValue.unknown()
    assert value.state is FieldState.UNKNOWN
    assert value.type_mask is None

def test_special_lookup_is_distinct_from_colorless_and_padding():
    assert ENERGY_SPECIAL_LOOKUP != ENERGY_COLORLESS
    assert ENERGY_SPECIAL_LOOKUP != ENERGY_PAD

def test_each_semantic_field_declares_a_supported_encoding_kind():
    supported = {"categorical", "exact_numeric", "continuous", "bitmask", "relation"}
    assert all(field.encoding_kind in supported for field in ALL_FIELD_SPECS)
```

- [ ] **Step 2: Write field-state alignment tests**

```python
def test_each_numeric_field_has_its_own_state():
    assert OPTION_NUMERIC.names == OPTION_NUMERIC_STATES.names
    assert PROTOTYPE_NUMERIC.names == PROTOTYPE_NUMERIC_STATES.names
```

- [ ] **Step 3: Implement exact typed contracts**

Use a separate categorical vocabulary per enum, boolean, and symbolic-identity field. Preserve exact numeric constants and counts as exact scalar values with field-specific projections. Use continuous projections only for genuinely continuous values. Preserve bitmasks as raw identity plus named bits, relations as indexed typed edges, and one state code per scalar. Do not include `typed_energy_gap`, `extra_energy`, `newly_enabled`, `remove_energy_counterfactual`, `final_damage`, `base_damage_is_ko`, `preferred_discard`, or `surplus_energy` in `ACTOR_KEYS`.

- [ ] **Step 4: Run typed contract tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_typed_contract`

Expected: PASS.

- [ ] **Step 5: Commit the typed schema**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining/contracts train/0031_rule_faithful_semantic_foundation_pretraining/domain
git commit -m "feat: add non-lossy typed semantic schema"
```

### Task 4: Physical Instances, Persistent Knowledge, And Semantic Events

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/instance_graph.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/events.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/knowledge/state.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_instance_graph.py`

**Interfaces:**
- Consumes: one actor observation plus causal knowledge state.
- Produces: `InstanceGraph` with local node IDs and typed edges `ATTACHED_TO`, `EVOLVED_FROM`, `OPTION_SOURCE`, `OPTION_TARGET`, `EVENT_SOURCE`, and `EVENT_TARGET`.

- [ ] **Step 1: Write duplicate-instance separation tests**

```python
def test_same_card_same_hp_with_different_energy_remains_distinguishable():
    graph = compile_fixture("duplicate_ogerpon_different_energy.json")
    left, right = graph.find_cards(card_id=1071)
    assert left.local_id != right.local_id
    assert graph.children(left.local_id) != graph.children(right.local_id)
```

- [ ] **Step 2: Write event relation tests**

```python
def test_attach_and_evolve_logs_keep_both_serial_relations():
    graph = compile_fixture("attach_evolve_events.json")
    attach = graph.events_of_type(LogType.ATTACH)[0]
    evolve = graph.events_of_type(LogType.EVOLVE)[0]
    assert attach.source_local_id and attach.target_local_id
    assert evolve.source_serial != evolve.target_serial

def test_known_opponent_hand_identity_persists_after_event_window():
    state = reveal_opponent_card_then_advance_events(event_count=80)
    assert state.known_opponent_hand[0].card_id == REVEALED_CARD_ID
```

- [ ] **Step 3: Implement local identity resolution**

Resolve `(playerIndex, area, slot)` and raw serial into stable row-local IDs. Preserve raw serial only in audit payloads. Give attached Energy, Tool, and pre-evolution cards their own nodes and parent edges. Encode relative owner/controller, zone/slot, `benchMax`, actual stadium owner, and known opponent hand card nodes. Each log field still receives an inventory disposition, but actor-visible history is limited to bounded semantic event tokens and facts needed to update persistent knowledge; non-persistent transition narration may be `audit_only` with an explicit rationale instead of being used to reconstruct the complete game state.

- [ ] **Step 4: Add permutation-equivariance tests**

Consistently permuting node storage and remapping all relation indices must preserve graph semantics and remap option targets exactly.

- [ ] **Step 5: Run instance graph tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_instance_graph`

Expected: PASS with no duplicate selected option lacking a distinguishing relation.

- [ ] **Step 6: Commit the instance graph**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining/features train/0031_rule_faithful_semantic_foundation_pretraining/knowledge
git commit -m "feat: preserve physical card and event relations"
```

### Task 5: Energy Facts And Runtime Oracle

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/domain/energy.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/tools/energy_oracle.cpp`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/energy.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_energy_semantics.py`

**Interfaces:**
- Consumes: physical Energy card prototype, target prototype/state, and direct observation `energies`.
- Produces: factual `EnergyContribution(type_mask, unit_count, state)` and physical attachment relations; no attack-satisfaction, removal, discard-priority, or surplus result.

- [ ] **Step 1: Write exact special-energy tests**

```python
def test_team_rocket_energy_is_two_psychic_or_darkness_units():
    value = resolve_energy(card_id=15, target=TEAM_ROCKET_BASIC)
    assert value.type_mask == 80
    assert value.unit_count == 2

def test_prism_energy_depends_on_target_stage():
    assert resolve_energy(16, BASIC_TARGET).type_mask == ENERGY_ALL
    assert resolve_energy(16, STAGE_ONE_TARGET).type_mask == ENERGY_COLORLESS
```

- [ ] **Step 2: Write the no-gap contract test**

```python
def test_energy_features_contain_facts_not_attack_answers():
    forbidden = {
        "typed_gap", "total_gap", "newly_enabled", "extra_energy",
        "remove_counterfactual", "preferred_discard", "surplus_energy",
    }
    assert forbidden.isdisjoint(ENERGY_ACTOR_FIELDS)
```

- [ ] **Step 3: Implement the read-only runtime oracle**

Build `energy_oracle.cpp` under `engine/build/0031/`. It initializes official cards, constructs audited target fixtures, calls official `getEnergyInfo()`, and emits card ID, target traits, raw type mask, unit count, and attach legality.

- [ ] **Step 4: Implement the Python factual resolver**

Mirror only the finite target-dependent rules required to reproduce `getEnergyInfo()`. Current attached state additionally preserves the direct observation `energies` list and checks that it agrees with the physical-card decomposition when decomposition is reconstructible.

- [ ] **Step 5: Compare every Energy card against the oracle**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.tests.energy_oracle_audit \
  --oracle engine/build/0031/energy_oracle
```

Expected: zero mismatches for all Energy cards across Basic, Stage 1, Stage 2, Team Rocket, and relevant dynamic target fixtures.

- [ ] **Step 6: Run Energy tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_energy_semantics`

Expected: PASS; `0` is never ambiguously interpreted as both Colorless and special lookup.

- [ ] **Step 7: Commit Energy facts**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "feat: preserve official energy type and unit facts"
```

### Task 6: Lossless Rule-Program And Decision Compiler

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/rule_program.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/compiler.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/collate.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_compiler.py`

**Interfaces:**
- Consumes: raw decision, `PrototypeIndex`, `InstanceGraph`, and causal snapshot.
- Produces: schema `0031_rule_faithful_decision_v1` and strict collated batches.

- [ ] **Step 1: Write ordered-program tests**

```python
def test_effect_order_target_areas_and_conditions_survive_compilation():
    program = compile_attack_program(attack_id=1)
    assert [node.ordinal for node in program.effects] == list(range(len(program.effects)))
    source = prototypes.attacks[1]["post_effects"][0]["target"]
    compiled = program.effects[0].target
    assert compiled.area_ids == tuple(source["areas"])
    assert compiled.condition_nodes == compile_conditions(source["conditions"])
```

- [ ] **Step 2: Write option binding tests**

```python
def test_energy_and_skill_options_bind_physical_instances():
    row = compile_fixture("energy_and_skill_options.json")
    assert all(option.source_local_id for option in row.energy_options)
    assert len({option.source_local_id for option in row.skill_options}) == len(row.skill_options)
```

- [ ] **Step 3: Implement exact skill selection**

Bind only the ability/play/delay skill relevant to the legal action and context. Attack options bind the selected `attackId` and its ordered effects; they do not inherit unrelated card skills.

Target areas are emitted as the original ordered discrete area IDs, each using the Area embedding rather than an area-count scalar. Every target condition is emitted as its own ordered child node: target/comparator enums use embeddings, numeric thresholds use exact numeric fields, bitmasks use the bitmask encoding, and name/card symbols use the deterministic symbol embedding.

- [ ] **Step 4: Implement compiler coverage assertions**

At compilation, compare all encountered raw keys against `observation_field_inventory.json`. Unknown keys fail with the exact JSON path. Missing known keys receive `UNKNOWN` only where the inventory declares optionality.

- [ ] **Step 5: Add collision audit tests**

Run the compiler on the 77,174-row 0025 raw corpus and assert:

```python
assert audit.selected_indistinguishable_skill_groups == 0
assert audit.selected_indistinguishable_attachment_targets == 0
assert audit.false_present_numeric_fields == 0
```

- [ ] **Step 6: Run compiler tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_compiler`

Expected: PASS.

- [ ] **Step 7: Commit the compiler**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining/features train/0031_rule_faithful_semantic_foundation_pretraining/tests
git commit -m "feat: compile lossless relational decisions"
```

### Task 7: Relation-Aware Policy Without Cross-Field Collapse

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/config.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/field_encoder.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/prototype_encoder.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/graph_encoder.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/option_encoder.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/action_decoder.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/model/policy.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_model.py`

**Interfaces:**
- Consumes: `DecisionBatch` and prototype v2.
- Produces: `RuleFaithfulPolicy.teacher_logits()` and `RuleFaithfulPolicy.greedy_action()`.

- [ ] **Step 1: Write independent-field encoding tests**

```python
def test_unknown_state_remains_bound_to_its_field():
    hp_unknown = batch_with_unknown("current_hp")
    cost_unknown = batch_with_unknown("retreat_cost")
    assert not torch.equal(encoder(hp_unknown), encoder(cost_unknown))
```

- [ ] **Step 2: Write relation sensitivity tests**

```python
def test_swapping_attachment_edges_changes_only_remapped_targets():
    original, swapped, permutation = attachment_edge_fixture()
    expected = model(original).index_select(-1, permutation)
    torch.testing.assert_close(expected, model(swapped), atol=1e-5, rtol=1e-5)
```

- [ ] **Step 3: Implement per-field encoders**

Give each exact numeric or continuous field its own scalar projection and its own field-state embedding. Give every enum, boolean, discrete area, comparator, condition type, and symbolic identity its own embedding table. Encode raw bitmasks through both exact-value identity and named bit projections. Do not coerce categorical values to floats and do not apply LayerNorm across heterogeneous raw fields before projection.

- [ ] **Step 4: Implement ordered prototype programs**

Use parent-kind, parent ID, phase, ordinal, target, area, and condition edges with ordinal embeddings. Preserve child token sequences; do not mean-pool effects, triggers, areas, or conditions.

- [ ] **Step 5: Implement bidirectional typed graph messages**

Each attachment edge contributes child-to-parent and parent-to-child messages with distinct relation embeddings. Option queries gather exact source/target local nodes and relevant prototype-program roots.

- [ ] **Step 6: Run model tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_model`

Expected: PASS with finite teacher logits, legal greedy actions, relation sensitivity, and padding invariance.

- [ ] **Step 7: Commit the model**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining/model train/0031_rule_faithful_semantic_foundation_pretraining/tests
git commit -m "feat: add relation-aware rule-faithful policy"
```

### Task 8: Raw Projection And Dataset Materialization

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/data/replay_contract.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/data/raw_decisions.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/data/materialize.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/training/dataset.py`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_dataset.py`

**Interfaces:**
- Consumes: official Episode archives and patches, plus a hash-verified immutable copy of the audited 0028 winner catalog as provenance.
- Produces: immutable dataset `rl_runs/0031.../dataset/V1_rule_faithful_winners_20260710_20260801`.

- [ ] **Step 1: Write projection completeness tests**

```python
def test_projection_does_not_drop_inventory_actor_fields():
    projected = project_raw_decision(FULL_FIXTURE)
    assert inventory_actor_paths(OBSERVATION_INVENTORY) <= json_paths(projected)
```

- [ ] **Step 2: Implement deterministic episode-grouped materialization**

Use whole-Episode splits, exact deck conditioning, source provenance outside actor forward, atomic staging, finite shard sizes, SHA-256 commitments, and fail-closed compiler coverage.

- [ ] **Step 3: Freeze and verify the winner catalog provenance**

Copy the immutable catalog into 0031 and verify both copies have SHA-256 `237477fc3f2deb34c893dd7da164651f156887151e06a5d2aa91d671e628881b` before materialization.

```bash
mkdir -p experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit
cp experiments/0028_universal_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json \
  experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json
sha256sum \
  experiments/0028_universal_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json \
  experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json
```

- [ ] **Step 4: Run a 1,024-decision smoke materialization**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.data.materialize \
  --archives-dir data/raw/episodes/archives \
  --patches-dir data/raw/episodes/patches \
  --winner-catalog experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json \
  --output .tmp/0031/materialize_smoke \
  --limit 1024 --workers 1
```

Expected: 1,024 records, zero unknown raw keys, zero prototype field omissions, and zero selected indistinguishable relations.

- [ ] **Step 5: Run dataset tests and a one-batch model smoke**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_dataset`

Expected: PASS.

- [ ] **Step 6: Materialize the full dataset only after all semantic gates pass**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.data.materialize \
  --archives-dir data/raw/episodes/archives \
  --patches-dir data/raw/episodes/patches \
  --winner-catalog experiments/0031_rule_faithful_semantic_foundation_pretraining/data_audit/winners_2026-07-10_2026-08-01.json \
  --output rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/V1_rule_faithful_winners_20260710_20260801 \
  --workers 8
```

Expected: manifest decisions and Episode counts match the audited raw source, with new schema and prototype commitments.

- [ ] **Step 7: Commit code and dataset manifest, not dataset shards**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining experiments/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "feat: materialize audited 0031 semantic decisions"
```

### Task 9: Training, Checkpoints, And Monitoring

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/training/objective.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/training/checkpoints.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/training/trainer.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/run_bc.py`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/monitor_training.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_training.py`

**Interfaces:**
- Consumes: the complete V1 0031 dataset and `RuleFaithfulPolicy`.
- Produces: versioned model-only checkpoints and canonical local/TensorBoard/W&B metrics.

- [ ] **Step 1: Write checkpoint rejection tests**

```python
def test_checkpoint_is_model_only():
    payload = build_checkpoint(model, metadata=FIXED_METADATA)
    forbidden = {"optimizer", "scheduler", "scaler", "rng", "dataloader", "replay"}
    assert forbidden.isdisjoint(payload)
```

- [ ] **Step 2: Implement ordered full-action BC**

Use autoregressive option-pointer cross-entropy with STOP, one train pass per epoch, online teacher diagnostics, and one complete teacher-forced plus greedy validation pass after every epoch.

- [ ] **Step 3: Run a CPU two-batch training smoke**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.run_bc \
  --dataset .tmp/0031/materialize_smoke \
  --version V1_contract_smoke --epochs 1 --max-train-batches 2 \
  --max-validation-batches 2 --wandb-mode disabled
```

Expected: finite loss, finite gradients, validation metrics, and model-only checkpoint reload.

- [ ] **Step 4: Run all 0031 tests before formal training**

Run: `python3 -m unittest discover -v train/0031_rule_faithful_semantic_foundation_pretraining/tests`

Expected: PASS with no skipped semantic coverage tests.

- [ ] **Step 5: Start formal training in a new immutable version**

Use version `V2_rule_faithful_full_bc`, W&B project `dragon_bra/pokemon-tcg-policy-learning`, foreground monitor session, model-only latest/best-loss/best-greedy checkpoints, and the repository hardware-calibrated batch size.

- [ ] **Step 6: Commit training implementation before launching the formal run**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining experiments/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "feat: add monitored 0031 BC training"
```

### Task 10: Official-Engine Gate And Authority Synchronization

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/export_candidate.py`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/evaluation/index.html`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/evaluation/V2_rule_faithful_full_bc.html`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`

**Interfaces:**
- Consumes: selected V2 checkpoint and immutable 0031 assets.
- Produces: self-contained candidate and formal official-engine evidence; no automatic opponent promotion.

- [ ] **Step 1: Write self-containment tests**

```python
def test_candidate_has_no_numbered_project_dependency_or_symlink():
    package = export_fixture_candidate()
    assert not any(path.is_symlink() for path in package.rglob("*"))
    assert "train.0028" not in read_all_python(package)
```

- [ ] **Step 2: Export and validate a candidate**

Run:

```bash
python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate \
  --checkpoint rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V2_rule_faithful_full_bc/checkpoint/best_greedy.pt \
  --output evaluation/arena/candidates/0031_rule_faithful_full_bc
python3 -m evaluation validate evaluation/arena/candidates/0031_rule_faithful_full_bc
```

Expected: exact 60-card deck response, strict checkpoint load, no symlink, and no adjacent project dependency.

- [ ] **Step 3: Run the formal official-engine evaluation**

Run with `--workers 8 --worker-cpu-threads 1`, the fixed opponent catalog, fixed seeds/seats, and the current setup/relay metric profile. Write the report to the exact V2 HTML path and refresh the project evaluation index.

- [ ] **Step 4: Synchronize DESIGN with code and artifacts**

Cross-check schema widths, parameter count, dataset manifest hash, checkpoint hash, W&B identity, current project phase, known unknowns, and formal report link in both `DESIGN.md` and `DESIGN.html`.

- [ ] **Step 5: Run the final contract suite**

Run:

```bash
python3 -m unittest discover -v train/0031_rule_faithful_semantic_foundation_pretraining/tests
python3 -m evaluation validate evaluation/arena/candidates/0031_rule_faithful_full_bc
```

Expected: all tests pass and candidate validation succeeds. Official-engine strength remains whatever the report measures; no automatic promotion occurs.

- [ ] **Step 6: Commit the evaluated candidate metadata and authority documents**

```bash
git add train/0031_rule_faithful_semantic_foundation_pretraining experiments/0031_rule_faithful_semantic_foundation_pretraining
git commit -m "docs: record 0031 official-engine evaluation"
```

## Plan Self-Review

- Spec coverage: prototype truth precedence, full field inventory, Team Rocket Energy, target-dependent Energy, owner/controller/zone/attachment relations, bounded semantic events, persistent known opponent hand identities, `benchMax`, original Target Area IDs, complete Target Conditions, ordered effects, per-type field encoding, non-derived actor facts, rematerialization, retraining, and official-engine evaluation each have an explicit task and test.
- Placeholder scan: the plan contains no unresolved placeholder marker, generic error-handling step, or undefined cross-task shorthand.
- Type consistency: `PrototypeIndex`, `FieldState`, `DecisionRecord`, `DecisionBatch`, `InstanceGraph`, `EnergyContribution`, and `RuleFaithfulPolicy` have one stable name across producer and consumer tasks.
- Historical safety: no task modifies `engine/source/`, 0025, 0028, or an existing run version.
- Training safety: formal materialization and training are gated behind semantic coverage and smoke tests; W&B and foreground monitoring are required only for the formal run.
