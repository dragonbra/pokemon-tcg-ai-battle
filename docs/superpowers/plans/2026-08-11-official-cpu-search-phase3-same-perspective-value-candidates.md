# Official CPU Search Phase 3 Same-Perspective Value Candidate Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use systematic-debugging to establish each source and runtime claim before classifying behavior. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a conservative, evidence-backed V0 eligibility contract for one-`SearchStep`, same-perspective, same-turn deterministic Value candidates and prove whether official Search observations can be encoded by the existing 0031/0036 Value pipeline without feature drift.

**Architecture:** Keep official CPU semantics in an unmodified C++ probe that directly compiles `Export.cpp`; emit paired live/Search observations and decision metadata as research artifacts. Drive the existing 0031 `OnlineCausalEncoder` from Python with cloned branch-local causal contexts, compare every tensor exactly, perturb Search hidden determinizations, and summarize only PASS-ALL decision-node families as V0 candidates.

**Tech Stack:** C++20 official CPU engine, public `SearchBegin/SearchStep`, Python 3.11, 0031 canonical observation compiler and causal ledger, PyTorch tensor parity, Markdown evidence report.

## Global Constraints

- `engine/source/ptcgProgram 22/` is the only rules and Search-semantics ground truth and must remain unmodified.
- CUDA, RL wrappers, inference wrappers, and compatibility wrappers cannot establish engine semantics.
- Do not modify production inference, enable Value override, invoke a Value argmax, or modify CUDA.
- Exclude Attack, End, opponent-perspective Value, fixed-focal rendering, forced-step traversal, stochastic search, minimax/expectimax, Monte Carlo, and multi-layer search from V0.
- Classify official decision nodes, not human-level complete actions; unknown evidence is V0 reject.
- Concrete cards are runtime fixtures only and do not create card allowlists.

---

### Task 1: Source And Pipeline Contract Audit

**Files:**
- Read: `engine/source/ptcgProgram 22/Search.h`
- Read: `engine/source/ptcgProgram 22/ToJson.h`
- Read: official decision handlers in `GameProc.h`, `SelectProc.h`, `EffectProc.h`, and `EffectInstant.h`
- Read: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/`
- Read: `train/0040_dragapult_0809_action_boundary_rl/semantic_policy/deployment/online_runtime.py`
- Read: `train/0040_dragapult_0809_action_boundary_rl/semantic_policy/knowledge/state.py`
- Read: `train/0040_dragapult_0809_action_boundary_rl/semantic_policy/features/compiler.py`

**Interfaces:**
- Consumes: official `State`, Search JSON, `OnlineCausalEncoder`, `CausalKnowledge`, canonical `DecisionBatch`.
- Produces: strict `SAME_PERSPECTIVE_STEP` and V0 eligibility predicates plus a complete inventory of persistent feature dependencies.

- [x] Trace perspective selection from `State::selectPlayer` through `ToJsonApi` and prove what same perspective means at a nonterminal return boundary.
- [x] Map decision type/context/handler families to same-turn, RNG, hidden-zone, and ownership risks without admitting an entire `SelectType` wholesale.
- [x] Trace all Value input tensors back to official observation, exact deck, causal ledger, known-card memory, logs, prior observations, and actor/first-player metadata.
- [x] Identify which branch-local context operations are possible with existing APIs and which require a research-only clone strategy.

### Task 2: Official Decision-Node Runtime Probe

**Files:**
- Create: `tests/official_cpu_search_same_perspective_probe.cpp`
- Create: `tests/official_cpu_search_same_perspective_probe.py`

**Interfaces:**
- Consumes: public official `SearchBegin/SearchStep`, legal live CPU fixtures, Phase 2 hash/RNG helpers.
- Produces: machine-readable root/live/Search observations and metadata for Bench, Attach, Switch, payment, opponent-target, deterministic effect-target, and rejected hidden-sensitive cases.

- [x] Build every fixture through official Battle setup and legal selections; do not mutate official State fields to fabricate game semantics.
- [x] For Bench and same-source Energy attach targets, branch at least two candidates and assert nonterminal, focal ownership, unchanged turn/active player, and unchanged RNG.
- [x] Build a two-Energy Retreat payment decision and a multi-target Switch decision; branch each candidate independently from its source search id.
- [x] Build a focal-owned opponent-Pokémon target decision and a focal-owned deterministic Trainer/Ability effect-target decision.
- [x] Build a same-perspective or RNG-free-looking hidden-sensitive rejection case and record the exact violated predicate.
- [x] Emit enough observation/log/state metadata for Python feature and hidden-determinization comparisons.

### Task 3: Feature Equivalence And Hidden Sensitivity Probe

**Files:**
- Modify: `tests/official_cpu_search_same_perspective_probe.py`

**Interfaces:**
- Consumes: paired live/Search JSON emitted by the C++ probe and the exact 0031 causal feature implementation paired with Value V9.
- Produces: exact tensor diff reports for `global/card/resource/event/option` fields and hidden-determinization invariance verdicts.

- [x] Clone the current `OnlineCausalEncoder` before each hypothetical branch so Search encoding cannot mutate the live mainline context.
- [x] Feed the real successor observation to one clone and the Search successor observation to another clone, then compare all `DecisionBatch` keys, shapes, dtypes, and values exactly.
- [x] Verify that feeding Search observation into the original live encoder would mutate decision index, ledger, event memory, and hidden-zone knowledge; document this as forbidden production usage.
- [x] Perturb opponent hand/deck/prize and unknown own-deck order inputs while holding focal-visible root observation fixed; compare returned observations and Value-visible tensors.
- [x] Report every nonmatching tensor and trace it to JSON/log or persistent-context provenance.

### Task 4: Research-Only V0 Eligibility Checker

**Files:**
- Create: `tests/research/official_cpu_search_v0_eligibility.py`
- Test: `tests/test_official_cpu_search_v0_eligibility.py`

**Interfaces:**
- Consumes: root/branch evidence containing ownership, terminal, turn, active player, RNG hashes, hidden-invariance result, and feature-parity result.
- Produces: `EligibilityResult` values `ELIGIBLE`, `REJECT_NOT_ACTUAL_CHOICE`, `REJECT_PERSPECTIVE_FLIP`, `REJECT_TURN_CHANGE`, `REJECT_TERMINAL`, `REJECT_RNG`, `REJECT_HIDDEN_SENSITIVE`, `REJECT_FEATURE_INCOMPATIBLE`, and `UNKNOWN`.

- [x] Write fail-closed tests proving every missing or failed predicate maps to its specific rejection and only PASS-ALL maps to `ELIGIBLE`.
- [x] Implement a pure research classifier that never calls Search, Value, production inference, or engine mutation.
- [x] Run the focused Python tests and preserve complete rejection-reason coverage.

### Task 5: Candidate Matrix And Report

**Files:**
- Create: `docs/reports/OFFICIAL_CPU_SEARCH_PHASE3_SAME_PERSPECTIVE_VALUE_CANDIDATES.md`

**Interfaces:**
- Consumes: Tasks 1-4 source anchors, runtime traces, tensor parity, and hidden-sensitivity results.
- Produces: the Phase 3 ground-truth report and practical PASS-ALL/UNKNOWN/reject matrix.

- [x] Define `SAME_PERSPECTIVE_STEP` and the stricter V0 contract, including actual-choice, no forced traversal, information-set determinism, and feature-context isolation.
- [x] Report Tests A-G with source handlers, actor/turn/RNG metadata, observation perspective, tensor parity, and hidden perturbations.
- [x] Build a decision-family matrix keyed by type/context/handler rather than card names; all UNKNOWN rows default reject.
- [x] State whether existing logs permit exact branch-local causal-context updates and whether clone-before-consume is sufficient.
- [x] Run the C++/Python probe, eligibility tests, `git diff --check`, and confirm official engine source has zero diff.
