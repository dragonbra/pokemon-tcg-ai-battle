# 0031 Incremental Feature Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Policy-0806 online feature-compilation cost by caching immutable/session state and rebuilding only dirty canonical feature layers while preserving tensor-exact parity with the existing 39-key actor contract.

**Architecture:** Keep `compile_canonical_row()` as the authoritative stateless full-rebuild path used by offline materialization. Factor its internals into pure layer compilers, then add a session-owned `IncrementalCanonicalCompiler` used only by `OnlineCausalEncoder`; it compares explicit dependency projections, reuses clean immutable layer values, rebases cross-layer card relations during assembly, and falls back to the full compiler whenever a dependency or ordering invariant cannot be proven. Benchmark a newly exported temporary runtime against the immutable Frozen-0806 package; never modify the frozen runtime or `engine/source/`.

**Tech Stack:** Python 3.11, dataclasses, PyTorch tensor collation, official CPU engine runtime, `unittest`, Frozen-0806 evaluation profiler.

## Global Constraints

- Do not modify `engine/source/` or any file under `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/policies/`.
- Preserve schema `0031_rule_faithful_semantic_decision_v2`, all 39 `ACTOR_KEYS`, masks, ordering, relation endpoints, action targets, logits, and deterministic actions.
- Keep `train/0031_rule_faithful_semantic_foundation_pretraining/` self-contained; no runtime imports from another numbered project.
- The stateless compiler remains the offline/training authority. Incremental state is owned by exactly one actor/game session and is cleared when a new encoder is constructed or the session closes.
- Dependency projections may contain only actor-visible observation fields, registered-deck facts, and `CausalSnapshot` facts already consumed by the current compiler.
- If card ordering, serial identity, visibility, causal-event chronology, or a layer dependency cannot be classified, rebuild all layers and record the fallback reason.
- Tensor-exact chronological parity is a hard gate. Throughput results without parity cannot justify enabling the incremental runtime.
- Test and benchmark artifacts go below `.tmp/evaluation/evaluation_inference_profile/` or `.tmp/0031_incremental_features/`; no formal run/version is created.
- Because the runtime pipeline changes while the actor schema does not, update both authoritative `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md` and `DESIGN.html` in the same change.

---

### Task 1: Freeze Chronological Parity And Cost Evidence

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/fixtures/incremental_feature_trajectory.json.gz`
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/benchmark_incremental_features.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`
- Create during execution: `.tmp/0031_incremental_features/full_rebuild_baseline.json`

**Interfaces:**
- Consumes: chronological raw rows with `actor_observation`, `event_cursor`, `deck_manifest`, Episode identity, and ordered action.
- Produces: `load_parity_trajectories(path: Path) -> tuple[ParityTrajectory, ...]` and a baseline JSON containing decision counts plus knowledge, compile, collate, and total milliseconds per decision.

- [ ] **Step 1: Select and freeze representative public trajectories**

  Select complete actor-local chronological sequences from the existing audited validation shard. Include setup, ordinary main actions, nested selections, search/deck views, evolution, ability, attack, switch/KO replacement, and terminal-near observations. Store only the exact deck, provenance identity, event cursor, and actor-visible observations required for compilation. The fixture loader must reject a sequence whose `actor_decision_index` is not exactly `0..N-1`.

- [ ] **Step 2: Write the full-rebuild reference helper and failing fixture test**

  Add a helper with the exact contract:

  ```python
  def full_rebuild_sequence(trajectory: ParityTrajectory) -> tuple[dict[str, torch.Tensor], ...]:
      knowledge = CausalKnowledge(trajectory.actor, trajectory.deck)
      batches = []
      for decision in trajectory.decisions:
          snapshot = knowledge.consume(decision.observation, decision.event_cursor)
          record = compile_canonical_row(decision.row(), snapshot, PROTOTYPES)
          batches.append(collate_canonical_records([record]))
      return tuple(batches)
  ```

  Assert all fixture sequences are chronological and collectively contain the required phase/log categories.

- [ ] **Step 3: Run the fixture test**

  Run:

  ```bash
  python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment
  ```

  Expected: PASS for the existing encoder and new fixture audit; no implementation change yet.

- [ ] **Step 4: Add a layer-neutral baseline benchmark**

  Time `CausalKnowledge.consume`, `compile_canonical_row`, and `collate_canonical_records` separately over at least 2,000 chronological decisions after warm-up. Emit medians, p95, decisions/s, Python version, CPU identity, dataset/fixture SHA-256, and schema version. Do not use repeated compilation of one unchanged row as the headline number.

- [ ] **Step 5: Record the baseline artifact**

  Run:

  ```bash
  python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.benchmark_incremental_features --mode full --decisions 2000 --output .tmp/0031_incremental_features/full_rebuild_baseline.json
  ```

  Expected: finite non-negative timings, exactly 2,000 compiled decisions, and schema `0031_rule_faithful_semantic_decision_v2`.

### Task 2: Factor The Stateless Compiler Into Exact Layers

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/layers.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/features/compiler.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/features/__init__.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_features.py`

**Interfaces:**
- Consumes: the same `row`, `CausalSnapshot`, and `PrototypeIndex` accepted by `compile_canonical_row()`.
- Produces: immutable `CardLayer`, `ResourceLayer`, `EventLayer`, `OptionLayer`, and `GlobalLayer`; `assemble_canonical_record(...) -> dict[str, Any]`.

- [ ] **Step 1: Write decomposition parity tests before moving code**

  For every fixture decision, compare the existing compiler with a not-yet-implemented `compile_canonical_layers()` result. Require equality of the complete Python record, not only tensor shapes:

  ```python
  self.assertEqual(
      compile_canonical_layers(row, snapshot, prototypes),
      compile_canonical_row(row, snapshot, prototypes),
  )
  ```

- [ ] **Step 2: Run the focused test and verify failure**

  Run:

  ```bash
  python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_features
  ```

  Expected: FAIL because `compile_canonical_layers` and the layer value types do not exist.

- [ ] **Step 3: Introduce immutable layer value types**

  Define frozen, slotted dataclasses. Card relations remain zero-based internally and are converted to the existing one-based-with-zero-sentinel representation only during assembly:

  ```python
  @dataclass(frozen=True, slots=True)
  class CardLayer:
      cat: tuple[tuple[int, ...], ...]
      num: tuple[tuple[float, ...], ...]
      state: tuple[tuple[int, ...], ...]
      parent: tuple[int, ...]
      locations: Mapping[tuple[int, int, int], int]
      serial_locations: Mapping[int, int]
      child_locations: Mapping[tuple[int, str, int], int]

  @dataclass(frozen=True, slots=True)
  class ResourceLayer:
      cat: tuple[tuple[int, ...], ...]
      num: tuple[tuple[float, ...], ...]
      state: tuple[tuple[int, ...], ...]
  ```

  Define corresponding event, option, and global types with every existing relation array represented explicitly.

- [ ] **Step 4: Extract pure compilers without changing ordering**

  Preserve the current exact card sequence: players/zones, stadium, looking, select deck, known self-deck order, known/possible/remembered opponent cards, then missing context/effect cards. `compile_event_layer` and `compile_option_layer` receive the final `CardLayer` so every relation endpoint is computed against the authoritative ordering.

- [ ] **Step 5: Make `compile_canonical_row` assemble the pure layers**

  Keep its public signature unchanged. It must call the pure layer functions and `assemble_canonical_record`, assert `set(actor_record) == ACTOR_KEYS`, and return the same mutable JSON-compatible record shape expected by dataset materialization and collation.

- [ ] **Step 6: Run exact record and tensor parity tests**

  Run:

  ```bash
  python3 -m unittest -v \
    train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_features \
    train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment
  ```

  Expected: every chronological record and all collated tensors are exactly equal to the pre-refactor reference.

### Task 3: Add Dependency-Proven Incremental Compilation

**Files:**
- Create: `train/0031_rule_faithful_semantic_foundation_pretraining/features/incremental.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/online_runtime.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`

**Interfaces:**
- Consumes: sequential `(row, CausalSnapshot)` pairs for one actor/deck session.
- Produces: `IncrementalCanonicalCompiler.compile(row, snapshot) -> dict[str, Any]`, `IncrementalCompileStats.snapshot() -> dict[str, int | float]`, and fail-closed `reset(reason: str) -> None`.

- [ ] **Step 1: Write dirty-layer and fallback tests**

  Cover these transitions independently: only legal options change; only selection context/deck changes; only global turn flags change; board HP changes; attachment/evolution changes card children; a new causal log changes event/resource memory; a card insertion shifts relation indexes; full deck order becomes known then invalidates; session chronology skips or rewinds. Assert the expected dirty layer set and exact equality with a fresh full rebuild after every transition.

- [ ] **Step 2: Run the tests and verify failure**

  Run:

  ```bash
  python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment
  ```

  Expected: FAIL because the incremental compiler does not exist.

- [ ] **Step 3: Define explicit dependency projections**

  Implement typed immutable projections rather than hashing the entire observation JSON:

  ```python
  @dataclass(frozen=True, slots=True)
  class LayerDependencies:
      cards: tuple[Any, ...]
      resources: tuple[Any, ...]
      events: tuple[Any, ...]
      options: tuple[Any, ...]
      global_facts: tuple[Any, ...]
      decision_index: int
  ```

  Each projection must enumerate every source read by its layer. Event and option dependencies include a compact card-layout token `(serial, card_id, owner, zone, slot, parent)` so relation caches invalidate when an endpoint moves even if the event/option JSON is unchanged.

- [ ] **Step 4: Implement cache ownership and fail-closed invalidation**

  `IncrementalCanonicalCompiler` owns exactly one previous dependency set and layer bundle. Reuse only layers whose dependency projection compares equal. Rebuild dependent layers according to this graph:

  ```text
  cards changed     -> cards + events + options
  resources changed -> resources
  events changed    -> events
  options changed   -> options
  globals changed   -> globals
  unknown/invalid   -> all layers
  ```

  A non-monotonic `snapshot.decision_index`, actor mismatch, schema mismatch, invalid relation endpoint, or projection error must call `reset()` and execute the stateless full path for that decision.

- [ ] **Step 5: Route online encoding through the incremental compiler**

  Construct the incremental compiler once in `OnlineCausalEncoder.__init__`. In `encode`, consume knowledge exactly once, compile incrementally, and keep the existing return contract. Add an explicit constructor flag `incremental: bool = True`; `False` must select the authoritative full path for parity and ablation tests.

- [ ] **Step 6: Prove chronological tensor equality**

  Run every frozen trajectory through two independent encoders, one full and one incremental. Assert `torch.equal` for every key in `EXPECTED_BATCH_KEYS` at every decision. Also compare model logits, sequence lengths, legality flags, and deterministic actions using the same checkpoint.

- [ ] **Step 7: Verify cache effectiveness without timing assertions**

  Require nonzero hits for card, resource, and event layers on the representative trajectories; require every miss/fallback reason to be counted; assert `sum(hit + miss) == decisions` for each layer. Do not make unit tests depend on wall-clock speed.

### Task 4: Remove Remaining Repeated Allocation At The Batch Boundary

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/features/collate.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/deployment/online_runtime.py`
- Modify: `evaluation/runner/inference_server.py`
- Modify: `tests/test_evaluation_inference_server.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`

**Interfaces:**
- Consumes: immutable canonical layer tuples from multiple resident sessions.
- Produces: one `collate_canonical_actors(records, structural_targets) -> dict[str, Tensor]` call per resident batch, with no per-request single-row tensor construction.

- [ ] **Step 1: Add batch-boundary parity tests**

  Compare legacy `collate_canonical_records(records)` with the actor-only online batch path for variable card/event/option/skill/effect/action lengths. Require equality for all actor tensors and masks; reconstruct only the online structural target sentinel and require its equality too.

- [ ] **Step 2: Make online runtime return canonical records explicitly**

  Add `OnlineCausalEncoder.encode_record(observation) -> dict[str, Any]`. Preserve `encode(observation) -> dict[str, Tensor]` as a compatibility wrapper that calls `collate_canonical_records([encode_record(...)])`.

- [ ] **Step 3: Use the explicit record API in resident inference**

  Replace the current module monkey-patch of `collate_canonical_records` with capability detection for `encode_record`. Collect records, collate once, transfer once, and leave legacy/source-conditioned policies on their existing path.

- [ ] **Step 4: Run resident-server regression tests**

  Run:

  ```bash
  python3 -m unittest -v \
    tests.test_evaluation_inference_server \
    train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment
  ```

  Expected: exact batch parity, one collation call per inference batch, unchanged fallback behavior, and no source identity entering the actor batch.

### Task 5: Benchmark, Gate, Export, And Document

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/benchmark_incremental_features.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/export_candidate.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`
- Modify: `evaluation/performance_profile.py`
- Modify: `tests/test_evaluation_performance_profile.py`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`
- Create during execution: `.tmp/0031_incremental_features/incremental_benchmark.json`
- Create during execution: `.tmp/evaluation/evaluation_inference_profile/policy0806_incremental_n16e1_g128.json`
- Create during execution: `.tmp/evaluation/evaluation_inference_profile/policy0806_incremental_n16e8_g128.json`

**Interfaces:**
- Consumes: the unchanged Policy-0806 checkpoint and exact Frozen-0806 workload.
- Produces: a self-contained temporary exported runtime, parity evidence, cache-hit/fallback statistics, microbenchmark comparison, and official-engine end-to-end profiles.

- [ ] **Step 1: Ensure exports contain the incremental runtime**

  Extend the export inventory to copy `features/layers.py` and `features/incremental.py`. The existing self-contained export test must dynamically load the extracted package without repository `PYTHONPATH` and perform chronological encoding.

- [ ] **Step 2: Add a profile-only runtime override**

  Let `evaluation.performance_profile` accept an explicit package root only for diagnostic output below `.tmp/`. Validate its checkpoint SHA, exact deck handling, `cg` compatibility, and schema, but label the result `non_frozen_runtime_profile`; do not route it into formal Frozen-0806 reports or overwrite the immutable package manifest.

- [ ] **Step 3: Run the microbenchmark comparison**

  Run:

  ```bash
  python3 -m train.0031_rule_faithful_semantic_foundation_pretraining.benchmark_incremental_features --mode compare --decisions 2000 --output .tmp/0031_incremental_features/incremental_benchmark.json
  ```

  Gate: zero tensor/logit/action mismatches, zero unexplained fallback reasons, and report per-layer hit rates plus full/incremental median and p95 compile time. Treat a speed result as descriptive unless the same host is idle and both arms are interleaved after warm-up.

- [ ] **Step 4: Run equal-contract official-engine profiles**

  Export the unchanged Policy-0806 checkpoint into a fresh `.tmp` package and run 128 completed games with zero errors for `N16E1` and `N16E8`, FP16, batch size 64, 2 ms coalescing, seed `8062026`, and the same balanced Frozen-0806 opponent schedule used by the existing baselines.

- [ ] **Step 5: Apply the admission gate**

  Enable incremental compilation by default only when all parity gates pass and both of these hold on repeated equal-contract trials:

  - feature compilation is at least 30% faster per engine selection than the matching full-rebuild arm;
  - end-to-end decisions/s does not regress by more than 3% at either `N16E1` or `N16E8`.

  If the gate fails, keep the factored compiler and profiler, set `incremental=False` by default, and record the measured layer responsible for the miss.

- [ ] **Step 6: Synchronize authoritative documentation**

  Update both DESIGN files with cache ownership, dependency projections, dirty graph, fallback semantics, unchanged shapes/schema, exact parity evidence, and measured throughput. Record in `DECISIONS.md` that official rules/runtime facts remain unchanged and that incremental invalidation is a project implementation strategy rather than a game-rule inference.

- [ ] **Step 7: Run the complete verification set**

  Run:

  ```bash
  python3 -m unittest -v \
    train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_features \
    train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_deployment \
    tests.test_evaluation_inference_server \
    tests.test_evaluation_performance_profile
  git diff --check
  ```

  Expected: PASS, no changes under `engine/source/` or the frozen runtime tree, and every benchmark/report link points to a repository-local `.tmp` artifact.
