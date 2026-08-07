# 0035 Versioned Observation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move change detection to a persistent observation-session boundary so unchanged canonical card/event segments are reused without per-entity recursive signatures, while preserving the exact 0031 record and 39-tensor contract.

**Architecture:** A session-owned `ObservationVersionTracker` compares only compiler-consumed top-level zones and causal tuple revisions, assigns monotonic card/event/resource/selection versions, and returns an immutable transition object. `IncrementalCanonicalCompiler` consumes these versions: unchanged card inputs reuse the complete immutable `CardLayer`; unchanged event window plus unchanged card layout reuses the complete `EventLayer`; action-owned option/global output remains rebuilt. Stateless compilation remains the authority for invalid types, ambiguous equality, chronology resets, and every incremental exception.

**Tech Stack:** Python 3.11, frozen/slotted dataclasses, PyTorch tensor parity tests, `unittest`, the existing 0035 chronological benchmark and official-engine evaluation infrastructure.

## Global Constraints

- Do not modify `engine/source/`, the official Engine ABI, or its JSON observation format.
- Keep actor schema `0031_rule_faithful_semantic_decision_v2`, all 39 tensor keys, model parameters, logits, decoder, and action contract unchanged.
- `train/0035_lifetime_aware_feature_compiler/` remains self-contained and must not import executable code from another numbered project.
- Treat turn as an epoch only; supporter/stadium/energy/retreat flags, status, `appearThisTurn`, selection, and board state remain action-mutable.
- Any unclassified value, same-object mutation ambiguity, duplicate serial, actor/deck/schema mismatch, or non-chronological decision must fail closed to the stateless compiler.
- V2 remains disabled by default until record/tensor parity, compiler median, compiler p95, and official-engine gates pass.

---

### Task 1: Persistent observation versions

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/features/observation_versions.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_observation_versions.py`

**Interfaces:**
- Consumes: canonical raw row, `CausalSnapshot`, actor index.
- Produces: `ObservationVersions` and `ObservationVersionTracker.observe(row, snapshot)`.

- [ ] **Step 1: Write failing lifecycle tests**

Test first observation versions, option-only change, HP/layout change, event-only change, turn-only change, primitive-type ambiguity, same-object mutation, and reset. Assert that option-only changes do not advance `card`, HP changes do, unchanged causal event identity does not advance `event`, and unsafe inputs raise `ObservationVersionError`.

- [ ] **Step 2: Run the focused test and verify the import fails**

Run: `python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_observation_versions`

Expected: FAIL because `features.observation_versions` does not exist.

- [ ] **Step 3: Implement immutable version tokens**

Define:

```python
@dataclass(frozen=True, slots=True)
class ObservationVersions:
    decision: int
    battle: int
    turn: int
    cards: int
    events: int
    resources: int
    selection: int
    globals: int

class ObservationVersionTracker:
    def observe(
        self, row: Mapping[str, Any], snapshot: CausalSnapshot
    ) -> ObservationVersions: ...
    def reset(self) -> None: ...
```

Store previous references and normalized top-level projections. Use equality only for fresh engine mappings after validating consumed primitive/container types; if an input subtree is the same object as the previous subtree, compute the strict projection rather than trusting self-equality. Card dependencies must cover both players' consumed zones/status, stadium, looking, select deck/context/effect, and the four causal synthetic-card tuples. Event dependencies are `(source_event, object identity)` plus the card version. Resource dependencies use the immutable ledger entries and deck-order flag. Selection and globals enumerate their existing compiler inputs separately.

- [ ] **Step 4: Run lifecycle tests**

Run the focused test above. Expected: PASS.

### Task 2: Whole immutable-layer reuse

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/features/incremental.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_incremental_features.py`

**Interfaces:**
- Consumes: optional `ObservationVersions` from Task 1.
- Produces: `compile(row, snapshot, versions=None)` with complete-card and complete-event layer hit counters.

- [ ] **Step 1: Write failing reuse tests**

Compile two chronological rows differing only in `turnActionCount` or option order. Assert exact full-record parity, `segment/card/hit == 1`, and card compiler call count does not advance. Add HP, context-card, known-hand, card-reorder and event-append cases proving the corresponding versions miss and relations remain exact.

- [ ] **Step 2: Verify the tests fail before integration**

Run: `python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_incremental_features`

Expected: FAIL because `compile` does not consume versions or reuse complete layers.

- [ ] **Step 3: Add version-aware layer storage**

Keep `_last_cards`, `_last_events`, `_last_card_version`, and `_last_event_version` inside the compiler. When the supplied version matches, reuse the frozen layer object directly. Otherwise call the existing exact fragment compiler. Event reuse additionally requires the same card version because its relation arrays use canonical card indexes. Reset and fallback clear every stored layer/version. Record `segment/card/{hit,miss}` and `segment/event/{hit,miss}`.

- [ ] **Step 4: Run parity and invalidation tests**

Run incremental, card-v2, semantic-lineage, and extended-fixture tests. Expected: all pass with zero audited fallback.

### Task 3: Move tracking into the online session

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/online_runtime.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_deployment.py`

**Interfaces:**
- `OnlineCausalEncoder` owns one `ObservationVersionTracker` per battle.
- `encode_record` calls `knowledge.consume`, then `tracker.observe`, then version-aware compile.

- [ ] **Step 1: Write a failing session-ownership test**

Assert two encoders never share trackers or versions, process prototypes remain shared, option-only changes reuse a card layer, and a new encoder starts versions from one without old card/event state.

- [ ] **Step 2: Integrate the tracker**

Construct it in `__init__`; pass its result only when `incremental=True`. Do not change the stateless path or public observation mapping. Expose a read-only diagnostic snapshot for benchmark/evaluation profiling.

- [ ] **Step 3: Run deployment tests**

Run: `python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_deployment`

Expected: PASS.

### Task 4: Chronological semantic and performance gates

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/benchmark_incremental_features.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_benchmark_protocol.py`

**Interfaces:**
- Benchmark includes observation tracking in the incremental compiler arm and reports version/segment counters.

- [ ] **Step 1: Extend benchmark protocol tests**

Require version counters, card/event segment hit/miss totals, alternating arm order, and exact parity before timing.

- [ ] **Step 2: Run the 560-decision semantic gate**

Compare complete Python records and all 39 tensors over the 35-decision golden plus 525-decision extended fixtures. Require zero mismatch and zero fallback.

- [ ] **Step 3: Run paired performance benchmark**

Run:

```bash
python3 -m train.0035_lifetime_aware_feature_compiler.benchmark_incremental_features \
  --mode compare --repetitions 7 --decisions 10000 --warmup-decisions 1000 \
  --output .tmp/0035_feature_compiler/versioned_observation_v2_7x10000.json
```

Admit only if compiler median is at most 0.25 ms, at least 50% below paired full, and compiler p95 does not regress. Otherwise retain opt-in status and record the exact result.

### Task 5: Documentation and final verification

**Files:**
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.html`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DECISIONS.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/manifest.json`

**Interfaces:**
- Records V2 implementation, exact semantic evidence, benchmark artifact, admission decision, and next stage.

- [ ] **Step 1: Synchronize Markdown, HTML, decisions and manifest**

Distinguish official rules, Engine JSON facts, and compiler strategy. State that observation versioning occurs after JSON decoding and therefore does not eliminate Engine serialization/parsing; it eliminates repeated feature-tree interpretation and layer materialization when versions are unchanged.

- [ ] **Step 2: Run all project gates**

Run:

```bash
python3 -m unittest discover -s train/0035_lifetime_aware_feature_compiler/tests -v
python3 -m compileall -q train/0035_lifetime_aware_feature_compiler
python3 -m json.tool experiments/0035_lifetime_aware_feature_compiler/manifest.json >/dev/null
git diff --check -- train/0035_lifetime_aware_feature_compiler experiments/0035_lifetime_aware_feature_compiler
test -z "$(git diff --name-only -- engine/source/)"
```

Expected: all tests pass, JSON parses, diff check is clean, and no official-engine source path is changed.

## Self-review

- Spec coverage: persistent early versions, battle/turn/action ownership, complete-layer reuse, fail-closed semantics, tensor parity, benchmark and documentation are each assigned to a task.
- Placeholder scan: no implementation placeholder is used; every task names exact files, interfaces, commands and assertions.
- Type consistency: `ObservationVersionTracker.observe` returns `ObservationVersions`; the compiler accepts that same type and the online runtime owns exactly one tracker per encoder session.
