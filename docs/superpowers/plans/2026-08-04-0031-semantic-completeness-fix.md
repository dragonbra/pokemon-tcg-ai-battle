# 0031 Semantic Completeness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 0031 preserve every audited actor-visible discrete fact, multiplicity, physical-instance relation, and persistent knowledge fact without adding derived rules-engine answers.

**Architecture:** Bump the actor schema and represent resolved Energy units as variable-length typed child tokens, while keeping physical Energy cards as separate attachment children. Extend option and event contracts with exact discrete identities and participant relations, add conservative set-valued persistent knowledge, and make the static prototype encoder reject any collection it cannot encode losslessly.

**Tech Stack:** Python 3.11, PyTorch, official read-only C++ engine headers, unittest, JSON prototype assets.

## Global Constraints

- Never modify `engine/source/`; it is read-only evidence.
- Do not expose Energy deficits, surplus/removal preferences, counterfactual attach results, final damage, or KO labels.
- Preserve missing, unknown, not-applicable, and present zero as distinct states.
- Encode discrete identities/enums with embeddings or typed relations; encode exact counts and magnitudes numerically.
- Keep 0031 self-contained and update both `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md` and `DESIGN.html` for every schema/model change.
- Do not create a formal run or materialize a formal dataset in this task.

---

### Task 1: Lock the V2 actor contract with regression tests

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/fields.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_features.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_contracts.py`

**Interfaces:**
- Produces: `0031_rule_faithful_semantic_decision_v2` with resolved Energy-unit tokens, exact option ordinal/serial/index fields, typed event participants, and explicit categorical applicability.

- [ ] Add failing tests showing `[1,1,1,6]` preserves all four units, same-card Skill options bind different serials, Switch/MoveAttached identities survive, and `failSkip=2` differs from `failSkip=1`.
- [ ] Add contract fields and batch relation keys required by those tests.
- [ ] Run `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_contracts train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_features` and confirm only implementation-dependent tests remain failing.

### Task 2: Preserve dynamic observation facts and relations

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/features/compiler.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/batch.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/state_encoder.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/option_encoder.py`

**Interfaces:**
- Produces: one resolved-unit token per official `PokemonJson.energies` entry, source-parent relations for Energy units, serial-bound Skill options, non-fabricated Attack relations, and exact Switch/Change/MoveAttached event participant relations.

- [ ] Compile each resolved Energy unit as a categorical `EnergyTypeIndex` token parented to its Pokémon; retain physical Energy card children independently.
- [ ] Add option ordinal and exact discrete option parameters; resolve type-15 source by official serial and never attach all card skills as if they were selected.
- [ ] Remove fabricated Active source/target assertions when the public Attack option cannot identify them; preserve explicit candidate/unknown state and option ordinal.
- [ ] Read official event key names, encode `hasBasicPokemon`, `isRecover`, coin result, damage-counter mode, result and reason with correct discrete applicability, and add participant edges for both sides of Switch/Change/MoveAttached.
- [ ] Preserve `contextCard` and `effect` as physical nodes with serial/owner relations, and use each attached child's own official `playerIndex`.
- [ ] Run feature, batch, and model tests.

### Task 3: Preserve actor knowledge without false certainty

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/data/replay_contract.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/knowledge/state.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_dataset.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_features.py`

**Interfaces:**
- Produces: explicit looking visibility/count state, persistent ordered self-deck knowledge until invalidation, and opponent hidden-hand candidate sets after ambiguous departures.

- [ ] Preserve `null`, opaque list length, and visible looking cards as distinct observations.
- [ ] Store a full ordered deck view when `select.deck` exposes it, update it on exact draws/moves, and invalidate order only on shuffle or an order-unknown transition.
- [ ] Replace destructive clearing of known opponent-hand identities with conservative candidate-group knowledge that never claims a particular card certainly remains.
- [ ] Persist revealed opponent identities and last-known/candidate zones beyond the 64-event presentation window.
- [ ] Add chronological tests for reveal, ambiguous departure, exact departure, full-deck view, draw, and shuffle.

### Task 4: Make static prototypes lossless and fail closed

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/prototype_encoder.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/contracts/prototype_field_inventory.json`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_features.py`

**Interfaces:**
- Produces: exact numeric `failSkip`, no duplicate `condition_type`, and explicit capacity validation against official struct limits/current assets.

- [ ] Move `fail_skip` out of booleans and encode its exact integer magnitude.
- [ ] Remove the duplicate `condition_type` categorical column.
- [ ] Validate card attack, attack Energy/effect, skill trigger/effect, target area, and target condition lengths before table construction; raise a descriptive error instead of slicing silently.
- [ ] Use official capacities where finite (`Target.areas=4`, `Attack.energies=7`) and current audited maxima for vectors, guarded by validation.
- [ ] Run prototype and lineage tests.

### Task 5: Verify semantics, quantify cost, and synchronize design

**Files:**
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`
- Create: `.tmp/0031_feature_benchmark/<run-id>/feature_benchmark.json`

**Interfaces:**
- Produces: audited V2 field/tensor shapes, old-vs-new token/compile/forward measurements, and a truthful readiness status.

- [ ] Run all 0031 unit tests and chronological real-row compilation.
- [ ] Benchmark old-compatible counts versus V2 on a fixed real-row sample: compile wall time, collation time, mean/p50/p95 card/event/option/resolved-unit token counts, padded batch token counts, and model forward time.
- [ ] Explain why tensors are variable length, how padding and masks turn them into dense batch tensors, and which bucketing/prefetch/shared-prototype optimizations can improve throughput without semantic loss.
- [ ] Update DESIGN.md, DESIGN.html, inventory, and decisions with exact V2 widths, benchmark evidence, remaining public-API limitations, and readiness conclusion.
- [ ] Confirm `git diff --check`, no changes under `engine/source/`, and no formal version directories were created.
