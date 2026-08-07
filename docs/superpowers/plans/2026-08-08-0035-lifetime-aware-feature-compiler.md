# 0035 Lifetime-Aware Feature Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create self-contained project `0035_lifetime_aware_feature_compiler` from 0031 and reduce online canonical-feature materialization by updating only battle-, turn-, and action-owned fragments while preserving the exact 0031 actor semantics.

**Architecture:** Freeze the copied 0031 stateless compiler as the semantic oracle and add a session-owned compiler whose cache is partitioned by lifetime. Process/battle constants are pre-encoded once; turn transitions invalidate only turn-derived fragments; action transitions update serial-indexed cards, append-only events, resource rows, legal options, and affected relation endpoints. Engine logs provide dirty hints, cheap primitive signatures prove reuse, and ambiguity triggers a full rebuild. The official Engine and its observation JSON format remain unchanged.

**Tech Stack:** Python 3.11, frozen/slotted dataclasses, PyTorch tensor collation, `ctypes` official-engine runtime, `unittest`, 0031 schema `0031_rule_faithful_semantic_decision_v2`.

## Global Constraints

- Never modify `engine/source/` or the official observation format.
- `train/0035_lifetime_aware_feature_compiler/` must not import executable code from any other numbered `train/` project.
- Preserve all 39 actor fields, values, ordering, masks, relation endpoints, target sentinels, logits, legal flags, and deterministic actions from 0031.
- Source/team/persona identity remains actor-invisible. No new model input is permitted.
- The copied stateless full compiler remains the semantic oracle and fallback path.
- One incremental compiler instance belongs to exactly one `(battle session, actor, exact deck)` tuple and is reset on any ownership mismatch or chronology rewind.
- Whole-observation and whole-layer cryptographic hashing are forbidden in the hot path. Dirty checks use explicit primitive signatures only.
- Unknown log types, duplicate/unstable serials, invalid relation endpoints, observation-shape ambiguity, actor changes, or non-monotonic chronology must fail closed to a full rebuild.
- Performance evidence must use chronological changing observations. Repeating one unchanged row is not an admission benchmark.
- Update both `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md` and `DESIGN.html` in the same implementation.
- Temporary artifacts belong under `.tmp/0035_feature_compiler/` and `.tmp/evaluation/0035_feature_compiler/` and remain untracked.

---

### Task 1: Fork A Self-Contained 0035 Semantic Oracle

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/` from the complete executable 0031 package
- Create: `experiments/0035_lifetime_aware_feature_compiler/manifest.json`
- Create: `experiments/0035_lifetime_aware_feature_compiler/DECISIONS.md`
- Create: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md`
- Create: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.html`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_project_contract.py`

**Interfaces:**
- Consumes: the 0031 source tree, immutable prototype assets, schema, chronological fixture, and model/checkpoint contract.
- Produces: an independently importable 0035 package with no `train.0031_*` imports and a manifest containing the exact 0031 fork commit plus copied-asset SHA-256 values.

- [ ] **Step 1: Write the failing self-containment test**

  Add a test that scans every 0035 Python file and rejects imports matching `train.00[0-9][0-9]_` or relative filesystem access into another numbered project:

  ```python
  forbidden = re.compile(r"(?:from|import)\s+train\.00\d\d_|train/00\d\d_")
  for path in PROJECT_ROOT.rglob("*.py"):
      self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), path)
  ```

- [ ] **Step 2: Run the test and verify failure**

  Run:

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_project_contract
  ```

  Expected: import failure because the 0035 package does not exist.

- [ ] **Step 3: Copy the complete executable implementation mechanically**

  Copy the 0031 package, including `assets/`, `contracts/`, `domain/`, `features/`, `knowledge/`, `model/`, `deployment/`, `training/`, benchmark/export entrypoints, and tests. Exclude `__pycache__`, `.pyc`, generated outputs, checkpoints, W&B, TensorBoard, and dataset shards. Relative imports remain relative; replace project-ID literals only where they describe the owning project rather than the preserved actor schema.

- [ ] **Step 4: Freeze lineage commitments**

  Record the source Git commit and SHA-256 for both prototype JSON files, the trajectory fixture, `contracts/fields.py`, the stateless compiler, collator, causal knowledge implementation, and model config. State explicitly that schema `0031_rule_faithful_semantic_decision_v2` is intentionally preserved.

- [ ] **Step 5: Run the project contract test**

  Run the command from Step 2. Expected: PASS and no runtime import from 0031.

### Task 2: Freeze Cross-Project Semantic Reference Data

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/tests/fixtures/canonical_reference_hashes.json`
- Modify: `train/0035_lifetime_aware_feature_compiler/benchmark_incremental_features.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_semantic_lineage.py`
- Create during execution: `.tmp/0035_feature_compiler/full_rebuild_baseline.json`

**Interfaces:**
- Consumes: the copied 35-decision chronological fixture and 0035 stateless compiler.
- Produces: immutable per-decision canonical-record and 39-key tensor commitments generated at the fork boundary, plus knowledge/compile/collate timing baselines.

- [ ] **Step 1: Generate stable reference commitments before optimizing**

  For each decision, compute canonical JSON bytes with sorted keys and tensor commitments including key, dtype, shape, and contiguous CPU bytes:

  ```python
  record_hash = sha256(canonical_json(record)).hexdigest()
  tensor_hash = sha256(
      b"".join(name.encode() + tensor.dtype.__str__().encode()
               + repr(tuple(tensor.shape)).encode()
               + tensor.contiguous().numpy().tobytes()
               for name, tensor in sorted(batch.items()))
  ).hexdigest()
  ```

  Store fixture SHA-256, schema, Episode/player identity, decision index, record hash, and tensor hash. This file is immutable provenance data, not an executable dependency on 0031.

- [ ] **Step 2: Write semantic-lineage tests**

  Recompile all 35 decisions through fresh `CausalKnowledge` and the stateless compiler. Assert every record and tensor hash equals its stored commitment and all batch keys equal the 0031 contract.

- [ ] **Step 3: Run the semantic-lineage test**

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_semantic_lineage
  ```

  Expected: 35/35 record commitments and 35/35 tensor commitments PASS.

- [ ] **Step 4: Record an interleaved full-rebuild baseline**

  ```bash
  python3 -m train.0035_lifetime_aware_feature_compiler.benchmark_incremental_features \
    --mode full --decisions 5000 --warmup-decisions 500 \
    --output .tmp/0035_feature_compiler/full_rebuild_baseline.json
  ```

  Require separate knowledge, compile, assembly/copy, collate, and total median/p95 timings; reject non-finite or zero-decision output.

### Task 3: Define Lifetime Ownership And Dirty Evidence

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/features/lifetimes.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_feature_lifetimes.py`

**Interfaces:**
- Produces:
  - `BattleSignature(actor: int, deck: tuple[int, ...], first_player: int | None)`
  - `TurnSignature(turn: int, turn_actor: int, flags: tuple[int, ...])`
  - `CardSignature(serial: int, static: tuple[Any, ...], dynamic: tuple[Any, ...])`
  - `ZoneSignature(owner: int, zone: int, ordered_serials: tuple[int, ...])`
  - `SelectionSignature(header: tuple[Any, ...], options: tuple[tuple[Any, ...], ...])`
  - `DirtySet(full_rebuild: bool, turn: bool, zones: frozenset[tuple[int, int]], cards: frozenset[int], resources: frozenset[int], events_appended: int, selection: bool, globals: bool, reason: str | None)`
  - `classify_transition(previous, observation, snapshot) -> DirtySet`

- [ ] **Step 1: Write isolated lifetime tests**

  Cover: unchanged battle constants; turn increment; supporter/energy/retreat flags changing within a turn; HP-only card mutation; attachment/evolution child mutation; card zone move; append-only logs; full deck view appearance/disappearance; option-only nested selection; duplicate serial; actor mismatch; and chronology rewind. Assert exact `DirtySet` values.

- [ ] **Step 2: Verify tests fail before implementation**

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_feature_lifetimes
  ```

  Expected: FAIL because `features.lifetimes` is absent.

- [ ] **Step 3: Implement cheap explicit signatures**

  Normalize only primitive fields read by the compiler. Card static signatures contain identity/prototype selectors; card dynamic signatures contain owner, zone, position, HP/damage, conditions, action flags, and ordered child serials. Selection signatures contain type/context/bounds and normalized option primitives. Do not call `json.dumps`, `hashlib`, `repr(observation)`, or recursively freeze the full observation.

- [ ] **Step 4: Implement fail-closed transition classification**

  Use incoming typed logs as positive dirty hints, then verify relevant zone/card/selection signatures. If a serial is missing where identity is required, appears twice, changes card ID, or a log type is outside the audited enum, return `full_rebuild=True` with a stable reason code.

- [ ] **Step 5: Run lifetime tests**

  Run the command from Step 2. Expected: PASS for every isolated mutation and fallback case.

### Task 4: Cache Battle- And Turn-Owned Fragments

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/features/session.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/online_runtime.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_feature_session.py`

**Interfaces:**
- Produces `FeatureCompileSession(actor, registered_deck, prototypes)` with `compile(row, snapshot) -> dict[str, Any]`, `reset(reason)`, and `stats.snapshot()`.
- Owns immutable deck manifest/count rows, actor-relative mapping, prototype expansions, turn signature, and action-level caches.

- [ ] **Step 1: Write battle/session ownership tests**

  Assert deck counts are materialized once, prototype static expansions are reused by identity, a turn transition invalidates only turn-owned fragments, and actor/deck/chronology mismatch forces a full rebuild and cache reset.

- [ ] **Step 2: Run tests and verify failure**

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_feature_session
  ```

  Expected: FAIL because `FeatureCompileSession` is absent.

- [ ] **Step 3: Implement battle constants**

  Precompute the sorted deck manifest, initial resource order, actor-relative player indexes, and card/skill/attack/effect prototype fragments once in `__init__`. Store immutable tuples; never copy the observation into the cache.

- [ ] **Step 4: Implement turn epoch ownership**

  Update turn header and truly turn-owned derived values only when `(current.turn, current.yourIndex)` changes. Keep supporter/energy/stadium/retreat flags action-owned because they can change inside the turn.

- [ ] **Step 5: Route online encoding through one session**

  `OnlineCausalEncoder` constructs one `FeatureCompileSession`; `encode_record` consumes knowledge exactly once and calls the session. Keep `incremental=False` as the default until Task 7 passes all admission gates.

- [ ] **Step 6: Run session and deployment tests**

  ```bash
  python3 -m unittest -v \
    train.0035_lifetime_aware_feature_compiler.tests.test_feature_session \
    train.0035_lifetime_aware_feature_compiler.tests.test_deployment
  ```

  Expected: PASS with full-path output unchanged.

### Task 5: Incrementally Materialize Action-Owned Fragments

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/features/card_cache.py`
- Create: `train/0035_lifetime_aware_feature_compiler/features/event_cache.py`
- Create: `train/0035_lifetime_aware_feature_compiler/features/resource_cache.py`
- Create: `train/0035_lifetime_aware_feature_compiler/features/option_cache.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/features/session.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_incremental_materialization.py`

**Interfaces:**
- Card cache is keyed by serial and zone order; event cache is a bounded append-only deque keyed by source event; resource cache is keyed by card ID; option prototype fragments are keyed by normalized semantic option tuples.
- Every cache exposes `update(...)`, immutable output tuples, hit/miss/update counters, and `clear()`.

- [ ] **Step 1: Write chronological red tests**

  At each of the 35 fixture decisions, compare incremental and stateless Python records and all collated tensors. Add isolated tests asserting that an HP-only mutation rebuilds one card dynamic fragment; append-only logs encode only new immutable event payloads; unchanged resource IDs reuse static columns; and option-only transitions do not rebuild board fragments.

- [ ] **Step 2: Implement the serial-indexed card cache**

  Cache static card categorical/prototype fragments separately from dynamic state. Maintain `zone -> ordered serials`, rebuild only changed serials, and produce one authoritative `serial -> final card index` map after concatenating zone segments. A changed zone order invalidates relation endpoints but not unchanged card feature fragments.

- [ ] **Step 3: Implement the bounded event cache**

  Encode immutable event type/payload/value/count/number/state and participant serials only when an event is appended. Represent age using the current decision index at assembly time; evict only events outside the existing 64-event window. Resolve participant relations from serials against the current card index map.

- [ ] **Step 4: Implement resource-row updates**

  Preserve static resource ID, initial count, and row order. Update only visible counts, exact bounds, knowledge class, and source-event ages for dirty card IDs; if a hidden-zone transition cannot identify the card ID, rebuild the resource layer.

- [ ] **Step 5: Implement option semantic reuse**

  Cache prototype-only expansion by normalized option type/card ID/skill ID/attack ID/effect IDs. Recompute per-decision ordinals, numeric values, target/card relations, bounds, and availability against the current selection and card index map.

- [ ] **Step 6: Assemble without semantic copies**

  Reuse immutable tuples from clean fragments, concatenate only dirty zone/event/option segments, and convert to the legacy mutable record exactly once at the public boundary. Relation arrays are rebased from serials after final card ordering; no cached numeric position may survive a card-layout version change.

- [ ] **Step 7: Run exact parity and cache-work tests**

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_incremental_materialization
  ```

  Expected: 35/35 exact record parity, 35/35 39-key tensor parity, zero unexplained fallbacks, and counters showing fewer card/event/resource prototype encodes than full rebuild.

### Task 6: Add Shadow Verification And Export Safety

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/features/session.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/online_runtime.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/export_candidate.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_deployment.py`

**Interfaces:**
- `FeatureCompileSession(..., shadow_every: int = 0)` runs the stateless oracle every N decisions, compares full Python records, resets on mismatch, returns the full result for that decision, and increments `shadow_mismatches`.

- [ ] **Step 1: Write mismatch/fallback tests**

  Inject a corrupted cached fragment, trigger a shadow comparison, and assert the returned record equals the oracle, caches reset, the stable reason is `shadow_parity_mismatch`, and no corrupted tensor reaches the model.

- [ ] **Step 2: Implement deterministic shadow verification**

  Use `decision_index % shadow_every == 0`; compare records directly without tolerance because all compiler outputs are discrete integers or deterministically derived floats. Production default is zero after admission; debug/benchmark exports may select one for 100% shadowing.

- [ ] **Step 3: Make exports self-contained**

  Include every 0035 lifetime/cache module and both prototype assets. Extract to a fresh directory, remove repository `PYTHONPATH`, dynamically execute `main.py` without `__file__`, and verify exact 60-card initialization plus chronological encoding.

- [ ] **Step 4: Run deployment/export tests**

  ```bash
  python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_deployment
  ```

  Expected: full/incremental/shadow outputs match and extracted runtime has no 0031 dependency.

### Task 7: Benchmark, Admit, And Synchronize Documentation

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/benchmark_incremental_features.py`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.html`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DECISIONS.md`
- Create during execution: `.tmp/0035_feature_compiler/compare.json`
- Create during execution: `.tmp/evaluation/0035_feature_compiler/n16e1_full.json`
- Create during execution: `.tmp/evaluation/0035_feature_compiler/n16e1_incremental.json`
- Create during execution: `.tmp/evaluation/0035_feature_compiler/n16e8_full.json`
- Create during execution: `.tmp/evaluation/0035_feature_compiler/n16e8_incremental.json`

**Interfaces:**
- Produces parity evidence, per-cache work counters, compiler latency, and equal-contract official-engine throughput profiles.

- [ ] **Step 1: Run an interleaved 5,000-decision microbenchmark**

  ```bash
  python3 -m train.0035_lifetime_aware_feature_compiler.benchmark_incremental_features \
    --mode compare --decisions 5000 --warmup-decisions 500 \
    --output .tmp/0035_feature_compiler/compare.json
  ```

  Alternate full and incremental arms per trajectory cycle to reduce thermal/order bias. Report median/p95 for knowledge, transition classification, materialization, assembly, collate, and total, plus work avoided per cache rather than relying on coarse layer hit rates.

- [ ] **Step 2: Apply the compiler admission gate**

  Require all of:

  - 35/35 frozen chronological record and tensor parity;
  - zero shadow mismatches under `shadow_every=1`;
  - zero unexplained fallbacks;
  - incremental compiler median at most `0.25 ms/decision` and at least 50% below the paired full compiler median;
  - incremental feature pipeline p95 no worse than the full path;
  - repeated runs agree on the direction of the result.

- [ ] **Step 3: Run equal-contract official-engine profiles**

  Export the same 0031-compatible checkpoint and exact deck twice, differing only in `incremental=False/True`. Run 128 completed games, zero errors, FP16, batch size 64, 2 ms wait, seed 8062026, and the same opponent schedule at N16E1 and N16E8. Store reports only below `.tmp/evaluation/0035_feature_compiler/`.

- [ ] **Step 4: Apply the end-to-end admission gate**

  Enable incremental compilation by default only if both N16E1 and N16E8 show no more than 3% regression and N16E8 improves decisions/s by at least 10% in repeated equal-contract runs. Otherwise keep `incremental=False`, preserve the implementation and evidence, and record the dominant miss/update cause.

- [ ] **Step 5: Synchronize the authoritative designs**

  In both DESIGN files, record the unchanged 39-key shapes/schema and model architecture; the Engine JSON → Python dict boundary; battle/turn/action ownership table; dirty graph; fallback and shadow semantics; benchmark hardware/commands/results; current project stage; and the boundary between the formal 0031-compatible package and research-only compiler experiments. Record in DECISIONS that lifetime ownership is an implementation strategy, not an official game-rule claim.

- [ ] **Step 6: Run the complete verification set**

  ```bash
  python3 -m unittest discover -s train/0035_lifetime_aware_feature_compiler/tests -v
  python3 -m unittest -v tests.test_evaluation_inference_server tests.test_incremental_benchmark
  git diff --check
  git diff --name-only -- engine/source
  ```

  Expected: all tests PASS, diff check clean, and the final command prints nothing.

- [ ] **Step 7: Commit only after evidence is complete**

  Stage the 0035 project, both 0035 DESIGN files, manifest/decisions, plan, and directly required evaluation integration. Do not stage unrelated 0031 experiments or `.tmp` artifacts. Use a commit message describing measured lifetime-aware materialization rather than claiming an unmeasured speedup.
