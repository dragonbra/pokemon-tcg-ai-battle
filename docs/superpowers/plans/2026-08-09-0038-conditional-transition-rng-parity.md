# 0038 Conditional Transition and RNG Semantic Parity Plan

> **Execution contract:** implement inline in this session. Do not start RL, alter checkpoint weights, modify `engine/source/`, submit Kaggle, or use aggregate win rate to override an upstream parity failure.

**Goal:** prove or falsify both (1) conditional rule parity for an identical canonical state, primitive action, and explicit random outcome, and (2) statistical equivalence of the independent CPU/CUDA random transition distributions.

**Architecture:** keep the deployment-visible public observation contract separate from a new authoritative, backend-independent rule-state record. Add test-only random-outcome adapters in standalone parity executables so production rollout code and kernels have no additional branch or provider lookup. Use exact field comparisons for categorical/rule state, declared tolerances for floating model tensors, and predeclared equivalence intervals for independent stochastic samples.

**Evidence roots:** tracked contracts/tests live under `train/0038_action_boundary_rl/semantic_parity/` and `engine_cuda/tests/`; generated traces and large samples live only under `.tmp/evaluation/0038_semantic_parity_audit/`; the release verdict lives in root `SEMANTIC_PARITY_AUDIT.md`.

## Frozen Acceptance Contract

- Conditional rule parity: `canonical_next_cpu == canonical_next_cuda` for identical canonical input, primitive action, and explicit random outcome.
- Snapshot feature parity: discrete tensors/relations/masks/orderings exact; floating features `atol=1e-6, rtol=1e-6`; FP32 model outputs `atol=1e-5, rtol=1e-5`, plus identical greedy intent unless a documented exact tie exists.
- RNG scalar categorical total-variation distance: upper 95% confidence bound `<= 0.01` for binary/small-cardinality tests and `<= 0.02` for 60-position tests.
- Batch invariance: pairwise metric difference upper 95% bound `<= 0.01` across batch sizes `1/8/256/512` for binary probabilities and `<= 0.02` TV for position distributions.
- Cross-game duplication/correlation: observed collision/autocorrelation must lie within a precomputed 99% reference envelope; absence of a significance rejection is never itself a PASS.
- End-to-end equivalence margins: win rate/seat/matchup absolute difference CI contained in `[-0.03,+0.03]`; mean game length within `±0.5` full turns; mean Prize differential within `±0.25`; strategic/macro/forced/action-type/error rates use an absolute `±0.03` or relative `±10%` margin, whichever is wider. Any nonzero semantic error/continuation/fallback caused by the audited runtime is a FAIL even when aggregate rates are close.
- `submission-ready=YES` requires Gates A–G PASS. U215/U230 remain ineligible as release candidates because their weights were learned under the pre-fix CUDA feature distribution; they may only serve as diagnostic checkpoints.

---

### Task 1: Authoritative Canonical State and Exclusion Ledger

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/canonical_state.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_canonical_state.py`
- Update: `train/0038_action_boundary_rl/semantic_parity/__init__.py`

**Interfaces:**
- `CanonicalAuthorityState.from_official(snapshot) -> CanonicalAuthorityState`
- `CanonicalAuthorityState.from_cuda(snapshot) -> CanonicalAuthorityState`
- `canonical_state_diff(left, right) -> list[FieldDifference]`
- `CANONICAL_STATE_SCHEMA_VERSION` and `EXCLUDED_FIELDS` with a non-rule justification for every exclusion.

- [ ] Add failing tests requiring zones, ownership, stable serials, board slots, HP/damage/status, attachments/tools, complete authoritative hand/deck/discard/prize state, turn/phase/seat flags, once-per-turn flags, continuation/effect stack, pending choice, legal options, event history, terminal/winner/error and reward-hook fields.
- [ ] Add tests proving missing required rule fields fail closed and excluded fields are limited to allocation addresses, timestamps, debug counters and backend layout/padding.
- [ ] Implement explicit CPU and CUDA adapters; do not use permissive recursive dumping as the schema.
- [ ] Preserve `canonical_public_observation` separately for hidden-information-safe policy comparison.
- [ ] Run exact round-trip/hash/diff tests.

### Task 2: Gate B — Fixed Snapshot Feature and Model Parity

**Files:**
- Update: `train/0038_action_boundary_rl/semantic_parity/gate_c_snapshot.py`
- Create: `train/0038_action_boundary_rl/semantic_parity/fixed_snapshot_parity.py`
- Update: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_c.py`
- Create: `train/0038_action_boundary_rl/tests/test_fixed_snapshot_feature_parity.py`

**Interfaces:**
- `compare_fixed_snapshot(cpu_record, cuda_record, tolerances) -> SnapshotParityResult`
- Ordered stages: observation fields → option skill/effect relations → event/resource/deck-membership → option order/mask → DecisionGate → model features → root/allocation logits → Value.

- [ ] Turn the frozen decision-0 feature diff, decision-60 greedy divergence and maximum Value sign-flip snapshots into immutable fixtures.
- [ ] Assert exact equality for schema/category/relations/order/mask/gate and declared tolerances for floating tensors.
- [ ] Report first divergence only; neural differences are not evaluated when observation/legal/mask already differs.
- [ ] Run deterministic greedy, `model.eval()`, dropout off, TF32 off, and record dtype/device/backend/tie margin.

### Task 3: Gate C — Deterministic Primitive Transition Parity

**Files:**
- Update: `train/0038_action_boundary_rl/semantic_parity/gate_a_rules.py`
- Create: `train/0038_action_boundary_rl/semantic_parity/deterministic_transition.py`
- Update: `train/0038_action_boundary_rl/tests/test_semantic_parity_gate_a.py`
- Create: `train/0038_action_boundary_rl/tests/test_deterministic_transition_parity.py`
- Reuse: `engine_cuda/tools/run_official_battle_ordered_matrix_cuda.py`

**Interfaces:**
- `compare_primitive_trace(cpu_steps, cuda_steps) -> TransitionParityResult`
- Exact step record: pre-state hash, primitive action, explicit/non-random marker, post-state, continuation, legal set, events and reward delta.

- [ ] Keep `gate_a_zoroark_decision129.json` and the fixed 283-action trace immutable.
- [ ] Execute the old pre-fix binary/commit artifact and require the known decision-129 FAIL.
- [ ] Rebuild current source and require decision-129 plus all Dragapult/Dusknoir, Area Zero and Zoroark/Munkidori fixtures to PASS.
- [ ] Compare continuation stack/lifecycle and reward fields instead of suppressing engine errors.

### Task 4: Gate D — Test-Only Explicit Random Outcome Injection

**Files:**
- Create: `engine_cuda/include/ptcg_cuda/testing/random_outcome_fixture.cuh`
- Create: `engine_cuda/extractor/official_random_outcome_fixture.cpp`
- Create: `engine_cuda/benchmarks/official_random_outcome_injection_paired.cu`
- Create: `engine_cuda/tools/run_official_random_outcome_injection_paired.py`
- Create: `engine_cuda/tests/test_random_outcome_injection.py`
- Create: `train/0038_action_boundary_rl/semantic_parity/random_outcomes.py`
- Create: `train/0038_action_boundary_rl/tests/test_random_outcome_schema.py`

**Interfaces:**
- Versioned `RandomOutcomeFixture` fields: shuffle permutation, Prize indices, coin results, random target indices and named effect results.
- Standalone test executable consumes the same fixture on official CPU and CUDA paths and emits exact canonical post-state records.

- [ ] Write schema validation tests for permutation completeness, bounds, consumption order, underflow and unused outcomes.
- [ ] Implement injection only inside standalone test translation units; no production header/source path imports the provider.
- [ ] Exercise real rule continuation after the injected random primitive, not a mocked final state.
- [ ] Cover shuffle/setup/Prize, coin, random target and random effect outcomes.
- [ ] Prove production binary symbols/config and benchmark throughput are unchanged within measurement noise.

### Task 5: Gate E — Independent Statistical RNG Equivalence

**Files:**
- Create: `engine_cuda/benchmarks/official_rng_distribution.cu`
- Create: `engine_cuda/tools/run_official_rng_distribution.py`
- Create: `engine_cuda/tests/test_official_rng_distribution.py`
- Create: `train/0038_action_boundary_rl/semantic_parity/statistical_equivalence.py`
- Create: `train/0038_action_boundary_rl/tests/test_statistical_equivalence.py`

**Interfaces:**
- `equivalence_ci_binary`, `multinomial_tv_bound`, `batch_stability`, `correlation_envelope`.
- Output one immutable JSON summary for CPU/CUDA independent samples at batch sizes `1/8/256/512`.

- [ ] Freeze margins and sample-size calculations before sampling.
- [ ] Sample coin, shuffled card position, opening hand inclusion, Prize inclusion and available random effects using independent backend seeds.
- [ ] Measure cross-game collisions and lag autocorrelation; test batch-size stability without reusing the same draw stream.
- [ ] Require confidence bounds to fall wholly inside the declared equivalence margins.
- [ ] Store counts/sufficient statistics, not millions of raw random draws.

### Task 6: Gate F — Action Boundary Regression

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_f_action_boundary.py`
- Create: `train/0038_action_boundary_rl/tests/test_gate_f_action_boundary.py`
- Update only if required by a proven bug: existing Agent/macro/trajectory modules under `train/0038_action_boundary_rl/`.

**Interfaces:**
- Gate summary includes strategic/policy/value/PPO counters, observe-only history hashes, macro lifecycle, primitive sequence and official/CUDA post-state hashes.

- [ ] Prove forced choices consume observation/history/events but add zero Policy, Value and PPO calls.
- [ ] Exhaust Phantom Dive `n=1..8`: one macro decision, stable-serial relocation, canonical allocation alias collapse and one transaction.
- [ ] Prove drift invalidates/fails closed without a second Policy call.
- [ ] Compare CUDA fused execution with official primitive unfolding under identical explicit random fixtures.
- [ ] Verify root-plus-allocation joint log-prob and one-transition trajectory semantics.

### Task 7: Gate G — End-to-End Independent-Seed Distribution Equivalence

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/gate_g_distribution.py`
- Create: `train/0038_action_boundary_rl/tests/test_gate_g_distribution.py`
- Reuse: official CPU evaluation package and CUDA frozen evaluator entrypoints from runtime inventory.

**Interfaces:**
- `compare_runtime_distributions(cpu_games, cuda_games, contract) -> DistributionParityResult`
- Metrics: overall/seat/matchup outcome, full turns, Prize differential, strategic/macro/forced counts, action family, error/continuation/fallback.

- [ ] Build independent, versioned CPU/CUDA seed manifests with identical deck/opponent/seat strata but no requirement for per-game trajectory identity.
- [ ] Start with a CPU 256 / CUDA 2,048 diagnostic to catch gross regressions, then run enough official CPU games for every predeclared interval to be statistically decidable; otherwise mark the affected metric INCOMPLETE.
- [ ] Save per-game compact metric rows and aggregate confidence/equivalence intervals.
- [ ] Never treat a similar overall win rate as recovery from an upstream Gate failure.

### Task 8: Gate H — Sensitivity and Negative Controls

**Files:**
- Create: `train/0038_action_boundary_rl/semantic_parity/sensitivity.py`
- Create: `train/0038_action_boundary_rl/tests/test_semantic_parity_sensitivity.py`

**Interfaces:**
- Executes fixtures against old/current build identities and isolated in-memory perturbations; source files are never left modified.

- [ ] Show old decision-129 fixture FAIL and current fixture PASS.
- [ ] Perturb one `option_effect` relation in memory and require fixed-snapshot parity FAIL.
- [ ] Perturb one continuation frame in memory and require transition parity FAIL.
- [ ] Remove perturbations and require both suites to PASS again.

### Task 9: Performance, Manifest and Release Report

**Files:**
- Update: `train/0038_action_boundary_rl/semantic_parity/submission_manifest.py`
- Update: `train/0038_action_boundary_rl/tests/test_semantic_parity_submission_manifest.py`
- Update: `SEMANTIC_PARITY_AUDIT.md`

**Interfaces:**
- Immutable report records pre-fix/post-fix commits, component/schema/rules hashes, exact commands, A–H matrix, first differences, statistical intervals, test-only/provider isolation, performance and remaining card/rule coverage.

- [ ] Require zero critical missing/unexpected state-dict keys and no silent fallback.
- [ ] Benchmark existing production CUDA path before/after test additions; the test-only provider must not appear in production symbols or change rollout throughput materially.
- [ ] Record all code changes and root-cause evidence, not merely test counts.
- [ ] Set `submission-ready=YES` only if A–G PASS and the candidate itself is eligible; otherwise enumerate blockers precisely.
- [ ] Run focused unit/property tests, CUDA standalone parity binaries, fixed-snapshot inference, distribution tests and compact official-engine smoke. Do not train or submit.

## Self-Review

- The authoritative canonical state and deployment-visible observation are deliberately separate, preventing hidden information from entering Actor parity while retaining complete rule-state evidence.
- Explicit random injection proves conditional transitions; independent statistical sampling proves distributions. Neither same-seed identity nor aggregate win rate is an acceptance premise.
- Random injection exists only in test executables, preserving production CUDA throughput and ABI.
- Negative controls demonstrate that the suite is capable of failing on known semantic defects.
- No task modifies official engine source or checkpoint tensors.
