# 0043 CUDA Engine 2.0 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CUDA Engine 2.0 the audited GPU engine backend for 0043 rollout and Frozen evaluation without weakening policy identity, exact-deck, official-engine, or promotion gates.

**Architecture:** Publish the isolated `engine_cuda_2_0/` subtree as shared infrastructure, then place every project-specific choice behind a self-contained `train/0043_champion_league_rl/cuda_engine_2/` adapter. The adapter resolves 0043 numeric deck IDs and full policies before lane creation, records source/ABI/rule-pack/binary identities, and treats the official CPU engine as the transition oracle. No executable import from another numbered training project is allowed.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, CMake/Ninja, CUDA C++ SM120, pytest, official CPU engine runtime, CUDA Engine 2.0 PyTorch extension.

## Global Constraints

- Do not modify `engine/source/`; it remains the read-only official oracle.
- 0043 deck identity is numeric `001`–`055`; future identities append from `056`.
- Hashes are integrity metadata and never semantic directory names.
- Requested policy identity must be fully materialized and verified before CUDA routing or sharing.
- Policy-0809 and Champion-G1 remain independent complete effective policies with no mutable storage or cache sharing.
- Formal candidate inference uses FP16 storage followed by strict FP32 runtime materialization.
- CPU-256 and CUDA-2048 remain separate benchmarks; promotion requires an explicit human decision.
- CUDA parity or smoke results do not by themselves prove policy strength.

---

### Task 1: Publish CUDA Engine 2.0 shared infrastructure

**Files:**
- Add: `engine_cuda_2_0/**`
- Add: `engine_cuda_2_0/docs/rtx5080_wsl2_quickstart_zh.md`
- Add: `engine_cuda_2_0/docs/server_native_usage_zh.md`

**Interfaces:**
- Consumes: audited source commits `46b5824` and `5859da4`.
- Produces: importable package `ptcg_cuda_engine`, CMake targets, and SM120 build instructions.

- [x] Cherry-pick only the isolated engine and native usage commits; exclude unrelated Frozen65/ranked assets.
- [x] Verify `engine_cuda_2_0/` never imports a numbered training project at runtime.
- [ ] Run CPU-only engine contract tests and rule-pack compiler checks.
- [x] Configure and finish native SM120 build under ignored `engine_cuda_2_0/build/native/`.
- [x] Run the native CUDA smoke and record the binary identity.

### Task 2: Add the 0043 engine identity adapter

**Files:**
- Create: `train/0043_champion_league_rl/cuda_engine_2/__init__.py`
- Create: `train/0043_champion_league_rl/cuda_engine_2/identity.py`
- Create: `train/0043_champion_league_rl/cuda_engine_2/config.json`
- Test: `train/0043_champion_league_rl/tests/test_cuda_engine_2_identity.py`

**Interfaces:**
- Consumes: `AssetRegistry`, policy materializer, CUDA 2.0 source tree, private rule-pack manifest, built binary/extension.
- Produces: `CudaEngineIdentity.resolve(repository_root: Path) -> CudaEngineIdentity` and fail-closed audit payload.

- [ ] Write tests rejecting missing source identity, wrong ABI, stale binary, missing private rule pack, and unsupported compute capability.
- [x] Pin tracked source bytes, state ABI, expected rule-pack hash, runtime dtype, and SM120 in `config.json`.
- [x] Implement read-only source-tree and artifact hashing with project-relative semantic labels.
- [x] Verify the tests pass on the current RTX 5080 environment.

### Task 3: Resolve 0043 decks and policies before CUDA routing

**Files:**
- Create: `train/0043_champion_league_rl/cuda_engine_2/routing.py`
- Test: `train/0043_champion_league_rl/tests/test_cuda_engine_2_routing.py`

**Interfaces:**
- Consumes: `LeagueLane`, `AssetRegistry`, `materialize_policy_bundle()`.
- Produces: immutable `CudaLaneRequest` values containing numeric deck ID, exact-deck hash, requested/materialized policy ID, effective hash, seeds, seat slot, and curriculum identity.

- [x] Test exact 256-lane conversion, independent deck/policy resolution, latest-champion routing, and fail-closed identity mismatch.
- [x] Implement routing without importing 0042 or any other numbered project.
- [ ] Assert opponent tensors/caches cannot alias focal mutable storage.
- [x] Add a stable request-schedule hash.

### Task 4: Build and validate the PyTorch extension

**Files:**
- Modify only if required: `engine_cuda_2_0/CMakeLists.txt`
- Create: `train/0043_champion_league_rl/cuda_engine_2/build.py`
- Test: `train/0043_champion_league_rl/tests/test_cuda_engine_2_build.py`

**Interfaces:**
- Produces: reproducible build command/manifest and importable `ptcg_cuda_ext` from an ignored build directory.

- [x] Reproduce the existing Torch include/compiler failure and capture the first failing command.
- [x] Fix build configuration without hard-coded user/repository paths.
- [x] Build with CUDA architecture 120 and the active Torch CMake directory.
- [x] Run the project-local extension smoke and verify device-resident reset/status tensors.
- [x] Record extension SHA-256, architecture, source, ABI and rule-pack identity.

### Task 5: Add official observation and action first-divergence parity

**Files:**
- Create: `train/0043_champion_league_rl/cuda_engine_2/parity.py`
- Test: `train/0043_champion_league_rl/tests/test_cuda_engine_2_parity.py`

**Interfaces:**
- Consumes: project-local semantic runtime, Policy-0809/Champion-G1 loaders, official CPU observations, CUDA 2.0 semantic batches.
- Produces: first-divergence reports over tensor fields, logits/value, ordered greedy action, engine state/status, and terminal outcome.

- [ ] Pin deterministic test cases covering decks `001`, `048`, both seats, context 41, and at least one long causal-history trajectory.
- [ ] Compare every actor-visible tensor and relation before model forward.
- [ ] Compare FP32 outputs within declared tolerances and require exact greedy action equality.
- [ ] Continue the same action sequence through CPU and CUDA transitions and require zero state/status/outcome mismatch.
- [ ] Save temporary reports under `.tmp/evaluation/0043_cuda_engine_2/`.

### Task 6: Wire 0043 rollout and Frozen schedules

**Files:**
- Create: `train/0043_champion_league_rl/cuda_engine_2/rollout.py`
- Modify: `train/0043_champion_league_rl/preflight.py`
- Modify: `train/0043_champion_league_rl/evaluation/preflight.py`
- Test: `train/0043_champion_league_rl/tests/test_cuda_engine_2_rollout.py`

**Interfaces:**
- Consumes: resolved `CudaLaneRequest`, exact 128/64/64 league schedule, FrozenMeta CPU/CUDA schedules.
- Produces: one-pass terminal episode records accepted by `telemetry.py`, with exact identity and seeded-toss metadata.

- [x] Run a bounded official-engine CUDA transition across Policy-0809 and Champion-G1 lanes.
- [ ] Require 0 fallback, error, unfinished, identity mismatch, and illegal action.
- [x] Verify 256-game league schedules retain 128/64/64 branches and 128/128 seat slots.
- [x] Verify CUDA-2048 is eight immutable replicas and is stored separately from CPU-256.
- [x] Keep `formal_training_authorized=false` until all formal version/W&B gates pass.

### Task 7: Synchronize authoritative 0043 documentation

**Files:**
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/manifest.json`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`
- Modify: `train/0043_champion_league_rl/README.md`

**Interfaces:**
- Produces: current source/ABI/build/parity identities, precise evidence boundary, commands, and remaining hard gates.

- [x] Document CUDA 2.0 as the required 0043 GPU backend and official CPU as oracle.
- [x] Record `001`–`065` training scope while retaining the separate `001`–`055` FrozenMeta contract.
- [x] Record smoke/parity evidence without presenting it as Frozen strength evidence.
- [ ] Run all 0043 tests, CUDA smoke/parity, `git diff --check`, and preflight.
- [ ] Commit only 0043, CUDA 2.0 infrastructure, plan, and synchronized docs.
