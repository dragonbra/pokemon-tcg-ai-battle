# Seeded Official Engine Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic local battle/Search runtime from the untouched official C++ source, then make evaluation and the next RL rollout contract use recorded engine seeds and paired seats.

**Architecture:** A new external C++ translation unit includes the official export implementation while replacing only the `BattleStart` and `AgentStart` entrypoints with seed-configured equivalents. The official source tree remains byte-identical; packages select the local build through the existing `PTCG_CG_LIBRARY` hook. Evaluation keeps candidate and opponent in fixed physical slots, forces `IS_FIRST` in the harness, and assigns one engine seed to each two-seat pair.

**Tech Stack:** C++20, official header-only PTCG engine, Python 3.11 `ctypes`, `unittest`, evaluation worker processes.

## Global Constraints

- Never edit, format, patch, or generate files under `engine/source/`.
- Source-built native outputs go only under ignored `engine/build/`.
- Kaggle packages continue to carry the original official precompiled `cg` runtime.
- Real policy-strength evidence must still execute the official engine rules; the manifest must identify the source-built seeded ABI and its hashes.
- Engine, policy, and Search seeds are separate recorded semantics.
- A paired seat block keeps `deck0=candidate` and `deck1=opponent`; only the harness-owned `IS_FIRST` answer changes.
- Any numbered RL training-semantic change requires a new immutable version and synchronized `DESIGN.md` plus `DESIGN.html`.
- Existing unrelated dirty-worktree changes are preserved.

---

### Task 1: Native seeded ABI and reproducible builder

**Files:**
- Create: `evaluation/runtime/seeded_engine.cpp`
- Create: `evaluation/runtime/seeded.py`
- Modify: `evaluation/runtime/__init__.py`
- Test: `tests/test_seeded_official_runtime.py`

**Interfaces:**
- Produces: `build_seeded_runtime() -> SeededRuntimeManifest` and `configure_seeded_library(library, *, engine_seed: int, search_seed: int) -> None`.
- Produces native exports `ConfigureSeeds`, `BattleStartSeeded`, `AgentStartSeeded`, plus the standard official ABI used by `cg/game.py` and `cg/api.py`.

- [ ] Write a failing test that verifies the official source tree hash is unchanged, the build lands under `engine/build/seeded_official/`, and the new library exposes standard and seeded symbols.
- [ ] Run `python3 -m unittest -v tests.test_seeded_official_runtime` and confirm the missing module/native source failure.
- [ ] Implement the external C++ ABI with `recordLog=true`, `deviceRand=false`, nonzero `uint32_t` seeds, and official-equivalent deck validation.
- [ ] Implement an atomic, hash-addressed C++20 build in `evaluation/runtime/seeded.py`; write a JSON manifest containing compiler, source hashes, ABI schema, and library SHA-256.
- [ ] Re-run the native-runtime tests and confirm they pass.

### Task 2: Deterministic battle and Search acceptance

**Files:**
- Modify: `tests/test_seeded_official_runtime.py`

**Interfaces:**
- Consumes: the library and seed configuration API from Task 1.
- Produces: regression evidence for exact trace replay and Search/Battle RNG isolation.

- [ ] Add a failing test that runs the same two decks, engine seed, forced seat, and deterministic legal actions twice and compares canonical observation/action trace SHA-256.
- [ ] Add a failing test that changes only the engine seed and proves the post-setup state can differ.
- [ ] Add a failing test that starts `AgentStart` with a fixed Search seed and verifies repeated identical Search call sequences are identical.
- [ ] Add a failing isolation test showing that Search calls on the agent pointer do not change a battle pointer replaying the same actions.
- [ ] Run the tests, fix only native ABI defects exposed by them, and re-run until all acceptance checks pass.

### Task 3: Evaluation engine seed and fixed physical slots

**Files:**
- Modify: `evaluation/runner/models.py`
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/runner/worker.py`
- Modify: `evaluation/runner/engine_pool_worker.py`
- Modify: `evaluation/cli.py`
- Test: `tests/test_evaluation_batch.py`
- Test: `tests/test_evaluation_worker.py`
- Test: `tests/test_evaluation_engine_pool_worker.py`

**Interfaces:**
- Produces: requests carrying `engine_seed`, `policy_seed`, `search_seed`, `requested_candidate_first`, and the seeded runtime path/hash.
- Produces: harness-owned `IS_FIRST` selection and observation-backed seat validation.

- [ ] Change the job-builder tests so adjacent first/second games against one opponent share an engine seed while retaining distinct policy seeds.
- [ ] Change worker tests so candidate is always physical player 0, opponent is always physical player 1, and context 41 is answered by the harness rather than either policy.
- [ ] Add a failure test for an observed `firstPlayer` that disagrees with the requested seat.
- [ ] Implement request serialization and worker environment setup through `PTCG_CG_LIBRARY`, followed by `ConfigureSeeds` before battle/Search initialization.
- [ ] Update the engine-pool path to call the explicit seeded start API without process-global environment races.
- [ ] Run the three evaluation test modules and fix regressions without touching package assets.

### Task 4: Paired evaluation schedule and manifest contract

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/reporting/models.py`
- Modify: `evaluation/README.md`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Produces: `seed_policy=sha256(base_seed:candidate:opponent:pair_number)` and `swap_policy=fixed_physical_slots_force_first_player_paired`.

- [ ] Add tests for even per-opponent paired schedules, deterministic job order, and worker-count-independent request payloads.
- [ ] Fail closed for a formal evaluation count that cannot form complete seat pairs; retain an explicitly labeled unpaired diagnostic mode only for non-formal tests if needed.
- [ ] Replace the old manifest claim that engine RNG is unavailable with the native ABI schema, build hash, library hash, and separate seed policies.
- [ ] Document paired-game interpretation and seed-pair bootstrap/statistical grouping.
- [ ] Run evaluation batch/reporting tests.

### Task 5: Next-version RL paired rollout contract

**Files:**
- Modify: `train/0034_dragapult_third_large_model_rl/rollout/protocol.py`
- Modify: `train/0034_dragapult_third_large_model_rl/rollout/worker.py`
- Modify: `train/0034_dragapult_third_large_model_rl/training/run_full_semantic.py`
- Test: `train/0034_dragapult_third_large_model_rl/tests/test_project_identity.py`
- Test: `train/0034_dragapult_third_large_model_rl/tests/test_rollout_batch.py`

**Interfaces:**
- Produces: adjacent two-game blocks with identical opponent/deck slots/engine seed and opposite forced first player.
- Produces: disjoint update seed windows and separate stochastic policy seeds.

- [ ] Add failing tests for fixed physical deck slots, shared pair engine seed, opposite seats, separate policy/Search seeds, and disjoint windows across source-policy updates.
- [ ] Make the rollout worker load and configure the seeded official runtime and force context 41 before policy inference.
- [ ] Implement a paired schedule for the next immutable 0034 version without rewriting existing V1-V6 artifacts or metrics.
- [ ] Record engine/policy/Search seeds in each `RolloutJob` and episode diagnostics.
- [ ] Run the focused 0034 tests and an official-engine paired smoke under `.tmp/evaluation/seeded_official_rl_smoke/`.

### Task 6: Design synchronization and final verification

**Files:**
- Modify: `experiments/0034_dragapult_third_large_model_rl/DESIGN.md`
- Modify: `experiments/0034_dragapult_third_large_model_rl/DESIGN.html`
- Modify: `engine/README.md`

**Interfaces:**
- Consumes: final ABI, evaluation, and rollout contracts from Tasks 1-5.
- Produces: authoritative documentation with accurate seed and runtime boundaries.

- [x] Correct the historical statement that existing probes controlled official engine seeds; state that prior runs controlled only scheduling/Python-side randomness.
- [x] Document the next immutable version's paired rollout/evaluation seed contract in both DESIGN formats.
- [x] Document that 0025/0031 binaries are read-only prototype exporters and that the seeded runtime is the first alternate battle ABI built from untouched official source.
- [x] Verify `git diff -- engine/source` is empty and hash the official source tree.
- [x] Run all focused tests, then execute a 10-game extracted-package official-engine smoke into `.tmp/evaluation/seeded-runtime-0001-smoke/` with 10/10 finished and zero errors.
- [x] Inspect the generated report and manifests for runtime hashes, seed pairs, actual first player, and zero presentation errors.
