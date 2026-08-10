# Official CPU Search Phase 2 Decision And Determinism Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use systematic-debugging to establish each source and runtime claim before classifying behavior. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the official CPU Search decision boundary, legal-selection semantics, actor ownership, and RNG consumption ground truth without implementing Value Search.

**Architecture:** Exhaustively map official engine selection primitives and RNG sites, then extend the Phase 1 direct-C-ABI fixture with an independent Phase 2 probe. Keep classification helpers research-only and cross-check every report statement against both unmodified source and runtime output.

**Tech Stack:** C++20 official engine headers and `Export.cpp`, Python 3.11 build driver, direct internal state inspection, source-call-site audit, Markdown evidence report.

## Global Constraints

- Use only `engine/source/ptcgProgram 22/` as behavior ground truth.
- Do not modify `engine/source/`, CUDA, production inference, RL wrappers, or compatibility wrappers.
- Do not connect a Value Network or implement production forced-step advancement.
- Do not produce card/action allowlists or design stochastic sampling/CRN.
- Concrete cards may be fixtures only; conclusions must remain engine-primitive level.

---

### Task 1: Source-Level Decision And Stop-Rule Audit

**Files:**
- Read: `engine/source/ptcgProgram 22/Search.h`
- Read: `engine/source/ptcgProgram 22/State.h`
- Read: `engine/source/ptcgProgram 22/ApiType.h`
- Read: every official caller of `State::setSelect`

**Interfaces:**
- Consumes: `SelectType`, `SelectContext`, `SelectOption`, `selectMin`, `selectMax`, `options`, `selected`, `selectPlayer`.
- Produces: complete engine-level decision taxonomy and exact `Search::step` return predicate.

- [x] Enumerate all `SelectType` values and map each to its option payload and representative official callers.
- [x] Trace `setSelect`, `clearSelect`, `addOption`, `checkPlayerSelect`, `State::step`, and terminal-state behavior in full.
- [x] Identify optional, singleton, acknowledgement, empty-option, and multi-player/ordered selection patterns.
- [x] Trace every way `selectPlayer` is assigned directly or through `setSelect`.

### Task 2: Legal Selection Enumerator And Research Classifier

**Files:**
- Create: `tests/official_cpu_search_decision_boundary_probe.cpp`
- Create: `tests/official_cpu_search_decision_boundary_probe.py`

**Interfaces:**
- Consumes: official `State` decision metadata and Phase 1 deterministic fixture pattern.
- Produces: `enumerate_legal_selections(const State&)`, legal selection counts, and research-only decision classifications.

- [x] Implement ordered-permutation enumeration exactly matching `State::checkPlayerSelect`: unique in-range option indices and every cardinality in `[selectMin, selectMax]`.
- [x] Assert representative enumerated vectors pass the official checker and duplicate vectors fail.
- [x] Implement research labels `TERMINAL`, `NO_SELECTION`, `FORCED_SINGLE_SELECTION`, `FOCAL_BRANCHING_DECISION`, `OPPONENT_BRANCHING_DECISION`, and `UNKNOWN` using internal state.
- [x] Determine and report which classification inputs are available from public observation fields.

### Task 3: Actor Boundary And RNG Runtime Evidence

**Files:**
- Modify: `tests/official_cpu_search_decision_boundary_probe.cpp`

**Interfaces:**
- Consumes: official public `SearchBegin/SearchStep`, internal `SearchState`, and actual official card/effect handlers.
- Produces: six required runtime traces with actor, type, context, min/max, option count, legal-selection count, and RNG hashes.

- [x] Reproduce ordinary automatic resolution and the Retreat forced-singleton chain.
- [x] Construct a focal-owned decision with at least two legal selections.
- [x] Construct a real effect path whose next selection is opponent-owned.
- [x] Verify a complete deterministic chain leaves search-side RNG unchanged.
- [x] Verify a public `SearchStep` path consumes search-side RNG, or explicitly separate an internal official RNG probe if no stable public path is available.

### Task 4: Exhaustive RNG Call-Site Audit

**Files:**
- Read: all official source files containing `rng`, `random_device`, `shuffle`, or coin-processing calls.

**Interfaces:**
- Consumes: concrete RNG call sites and their callers.
- Produces: operation-level RNG table distinguishing shuffle, draw, prize, coin, random selection, damage, KO, turn transitions, and automatic effects.

- [x] Enumerate every direct `Game::rng` and `std::random_device` call site in official source.
- [x] Trace trigger conditions and whether each operation consumes RNG or only observes fixed State order.
- [x] Separate transition determinism, RNG freedom, and observation determinism.

### Task 5: Report And Verification

**Files:**
- Create: `docs/reports/OFFICIAL_CPU_SEARCH_PHASE2_DECISION_BOUNDARY_DETERMINISM.md`

**Interfaces:**
- Consumes: Tasks 1-4 source anchors and passing probe output.
- Produces: the requested ten-section Phase 2 report.

- [x] Run the Phase 2 probe from a clean rebuild and retain all six PASS results.
- [x] Write the stop rule, taxonomy, forced-selection semantics, ownership, RNG table, determinism definitions, probe results, and recognizer limits.
- [x] Cross-check the report against current source line numbers and probe assertions.
- [x] Run formatting/diff checks and confirm `engine/source/` remains unchanged.
