# Shared CUDA Resident Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one audited CUDA-resident rollout stack that accelerates both Frozen-0806 evaluation and 0037 PPO while preserving exact-deck, seed, action, reward, value, and report contracts.

**Architecture:** A shared semantic0031 router owns one frozen state/prototype trunk, one focal Option/Decoder/Value branch, and one frozen opponent Option/Decoder branch. A resident lane scheduler uses masked seeded resets to refill terminal lanes without waiting for the longest game; evaluation and PPO provide different sinks over the same routed decisions and terminal events.

**Tech Stack:** Python 3.11, PyTorch 2.11 CUDA 12.8, `_ptcg_cuda.OfficialCudaEngine`, Semantic0031-v2, PPO/GAE, unittest, NVTX, Nsight Compute.

## Global Constraints

- Never modify `engine/source/`; CUDA runtime changes remain below `engine_cuda/`.
- Keep FP32 + TF32 as the initial numerical contract; BF16/FP16 requires a separate action-parity experiment.
- Frozen evaluation remains seed `341512806`, exactly 512 games per deck, balanced seats, 0 error, 0 unfinished.
- A repeated leading Ability option by the same actor forfeits on occurrence 20, including across turn-clock increments.
- Evaluation strength claims remain labelled CUDA-engine evidence until official-CPU parity is established.
- 0037 keeps turn-clock GAE λ=0.95, terminal win/loss reward, episode-equal weighting, Value Network fine-tuning, Last Option Q/V LoRA, and model-only checkpoints.
- Canonical metrics remain local-first; no formal training run is launched by this infrastructure change.

---

### Task 1: Lock Correctness and Baseline Evidence

**Files:**
- Modify: `engine_cuda/tests/test_policy_0806_cuda_evaluation.py`
- Modify: `engine_cuda/tests/test_progress_guard.py`
- Modify: `engine_cuda/tools/evaluate_policy_0806_cuda.py`
- Modify: `engine_cuda/python/ptcg_cuda_engine/progress_guard.py`
- Create: `.tmp/engine_cuda_optimization/shared_resident_20260808/baseline.json`

**Interfaces:**
- Consumes: numbered Frozen-0806 catalog, exact-deck hashes, existing 4×128 CUDA results.
- Produces: `resolve_candidate_deck_path(candidate, deck_root) -> Path`, repeat-guard reset semantics, baseline throughput and component timing record.

- [ ] Add a regression test proving a catalog candidate whose `root` is the shared policy package resolves its numbered exact deck instead.
- [ ] Add tests for multi-token and cross-turn repeated Ability counting plus selective lane reset.
- [ ] Validate each focal deck tuple with `exact_deck_sha256` before engine reset.
- [ ] Record corrected 0806 aggregate baseline (17.50 games/s) and Update32 152-lane baseline (15.25 games/s) with hashes and component percentages.
- [ ] Run `python3 -m unittest -q engine_cuda.tests.test_policy_0806_cuda_evaluation engine_cuda.tests.test_progress_guard engine_cuda.tests.test_benchmark_0037_update32_cuda_rollout`.

### Task 2: Shared Semantic0031 Focal/Opponent Router

**Files:**
- Create: `engine_cuda/python/ptcg_cuda_engine/semantic0031_router.py`
- Create: `engine_cuda/tests/test_semantic0031_router.py`
- Modify: `engine_cuda/python/ptcg_cuda_engine/semantic0031_bridge.py`

**Interfaces:**
- Consumes: focal `SemanticActorCritic` or frozen semantic actor, immutable 0806 base weights, Semantic0031-v2 device batch, focal route mask, opponent route mask.
- Produces: `Semantic0031ResidentRouter.route(batch, *, focal_route, opponent_route, focal_mode) -> RoutedDecisionBatch` containing actions, lengths, stopped, focal log-probability/entropy/value, validated batch, state, and focal options.

- [ ] Write a failing test showing router construction creates exactly one PrototypeMemory and one relation cache.
- [ ] Write a failing parity test comparing routed greedy actions with the existing two-model benchmark on a captured Semantic0031-v2 batch.
- [ ] Factor `Semantic0031DeviceAdapter.encode_option_inputs` into a public cached shared-input method.
- [ ] Compute frozen state tokens and pre-transform Option inputs once; run focal Last Option LoRA and frozen opponent Option branches only for their routed rows.
- [ ] Hold only one full actor; keep an immutable opponent Option Transformer and Decoder snapshot instead of a second full model.
- [ ] Accumulate diagnostics as CUDA tensors and transfer once at the end; remove per-decision `ready.sum().item()` synchronization.
- [ ] Run router parity tests in greedy and stochastic focal modes with behavior log-probability error ≤1e-4.

### Task 3: Resident Lane Scheduler with Masked Refill

**Files:**
- Create: `engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py`
- Create: `engine_cuda/tests/test_semantic0031_resident.py`
- Modify: `engine_cuda/python/ptcg_cuda_engine/progress_guard.py`

**Interfaces:**
- Consumes: ordered `ResidentJob` records (schedule index, two decks, engine seed, search/policy seed, focal player), router, lane count.
- Produces: ordered `TerminalRecord` events and optional `FocalDecisionRecord` batches through a sink protocol; all terminal lanes are refilled with `reset_seeded_interactive_masked` until the finite job queue is exhausted.

- [ ] Write a fake-engine test where one long lane cannot stall completion of short queued games.
- [ ] Write a scheduler test proving terminal results return in schedule-index order despite out-of-order lane completion.
- [ ] Write a paired-seat seed test proving both jobs retain the same engine seed and opposite focal players after multiple refills.
- [ ] Add `DeviceRepeatForfeitGuard.reset(mask)` so a reused lane cannot inherit counts from its previous Episode.
- [ ] Implement device-resident masked resets, per-lane job metadata, progress-guard adjudication, and fail-closed engine error capture.
- [ ] Ensure no observation/action tensor moves to CPU inside the decision loop; only terminal metadata and sink flushes may cross the boundary.
- [ ] Benchmark lane counts 64, 128, 152, 256, and 512 using identical jobs; select by median completed games/s subject to memory <90% VRAM.

### Task 4: Frozen-0806 Evaluation Integration

**Files:**
- Modify: `engine_cuda/tools/evaluate_policy_0806_cuda.py`
- Modify: `engine_cuda/tools/benchmark_0037_update32_cuda_rollout.py`
- Modify: `engine_cuda/tests/test_policy_0806_cuda_evaluation.py`
- Modify: `evaluation/arena/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cuda_seeded_512_v1/index.html` (generated)

**Interfaces:**
- Consumes: resident scheduler with greedy focal/opponent routing and the fixed 512-job schedule.
- Produces: the existing per-deck result schema, deterministic game-result hash, terminal chunk/state hash, 001–055 HTML reports, and resumable manifest.

- [ ] Replace four subprocess/model reloads with one resident process and one loaded router.
- [ ] Preserve schedule-index result order and exact report aggregation while lanes refill dynamically.
- [ ] Store lane topology, router hash, extension hash, repeat forfeits, games/s, decisions/s, and peak VRAM in the manifest.
- [ ] Regenerate the UI using the legacy Policy-0806 visual language: header, summary cards, representative card images, search, matchup bars, exact deck, and evidence boundary.
- [ ] Run deck 001 twice and require exact game results, terminal hashes, and forfeit indices.
- [ ] Resume the remaining 55-deck evaluation only after 001 is 512/512 with 0 error and 0 unfinished.

### Task 5: 0037 PPO CUDA Collector Integration

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/rollout/cuda_collector.py`
- Create: `train/0037_dragapult_value_initialized_rl/tests/test_cuda_collector.py`
- Modify: `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py`
- Modify: `train/0037_dragapult_value_initialized_rl/training/ppo_full_semantic.py`
- Modify: `train/0037_dragapult_value_initialized_rl/rollout/protocol.py`

**Interfaces:**
- Consumes: the same 512 paired jobs, current `SemanticActorCritic`, frozen 0806 snapshot, resident scheduler.
- Produces: the existing `EpisodeTrajectory`/`PreparedBatch` semantic contract with focal action sequence, stopped flag, rollout log-probability, entropy, Value Network output, turn clock, terminal reward, and source policy update.

- [ ] Write a collector contract test comparing CPU and CUDA field names/shapes on the same tiny seeded schedule.
- [ ] Write a turn-clock GAE test with lane reuse to prove Episode boundaries do not leak rewards, values, or repeat counts.
- [ ] Keep focal stochastic actions and opponent greedy actions on device; flush focal decision tensors to host once per rollout batch for the existing audited PPO preparation path.
- [ ] Add `--rollout-engine cuda|cpu` with CUDA as the new explicit experiment variable; keep CPU fallback for parity diagnosis.
- [ ] Log `rollout/engine=cuda`, lane count, completed games/s, decisions/s, refill count, peak VRAM, engine errors, unfinished games, and behavior log-probability parity.
- [ ] Run a no-update CUDA collection smoke and a one-update PPO smoke; require behavior log-probability MAE ≤1e-4 and unchanged trainable-parameter inventory.

### Task 6: Profile, Optimize, and Select the Champion

**Files:**
- Create: `.tmp/engine_cuda_optimization/shared_resident_20260808/profile/`
- Modify only if evidence requires: `engine_cuda/python/ptcg_cuda_engine/semantic0031_router.py`
- Modify only if evidence requires: `engine_cuda/include/ptcg_cuda/*` or `engine_cuda/src/*`

**Interfaces:**
- Consumes: resident Evaluation and PPO benchmark entry points.
- Produces: median-of-three benchmark matrix, NVTX/NCU reports, ablation attribution, and selected champion configuration.

- [ ] Add NVTX ranges for observation encoding, state encoder, focal/opponent Option branches, decoders, engine advance, masked reset, and sink flush.
- [ ] Measure corrected 0806 greedy evaluation and Update32 stochastic-vs-greedy rollout at 64/128/152/256/512 lanes, three repetitions each.
- [ ] Run NCU on the 152-lane Update32 hot range; record top compute, memory, and latency metrics under `.tmp/engine_cuda_optimization/shared_resident_20260808/profile/`.
- [ ] Ablate resident refill, shared Prototype/input cache, routed Option compaction, and host-sync removal one at a time.
- [ ] Select a champion only if correctness hashes/parity pass and median throughput improves beyond noise; target ≥20 games/s on the RTX 5080 Update32 pure-rollout contract.
- [ ] If the remaining bottleneck is a specific custom CUDA kernel, start the full roofline/branch/SASS kernel workflow; otherwise do not manufacture kernel changes.

### Task 7: Documentation and Final Verification

**Files:**
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`
- Modify: `docs/superpowers/plans/2026-08-08-shared-cuda-resident-rollout.md`

**Interfaces:**
- Consumes: champion benchmark, final collector contract, code/tests.
- Produces: synchronized authoritative design, reproducible commands, evidence boundary, and completion record.

- [ ] Document CUDA-resident observation/action flow, shared/focal/opponent tensor paths, lane refill, trajectory materialization, and CPU fallback boundary.
- [ ] Document model inputs/shapes, Value Network path, PPO losses, seed/seat contract, progress guard, and selected lane topology.
- [ ] Run all engine CUDA, 0037 rollout/PPO, Frozen-0806 asset, and evaluation tests.
- [ ] Run `git diff --check`, verify no `.tmp`, dataset, model, optimizer, or trace asset is staged, and create a focused implementation commit.
- [ ] Report Evaluation and PPO throughput separately; do not compare pure rollout games/s with end-to-end PPO update wall time.
