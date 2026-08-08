# 0037 RL Engine Pool And Prototype Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 0037 rollout use the admitted frozen-prototype GPU cache and a real N-process/E-engine official-engine pool, select the fastest valid RL topology on this machine, and relaunch the value-initialized seeded-512 PPO baseline as a new immutable version.

**Architecture:** Keep all PyTorch collation, actor/value inference, stochastic decoding, and trajectory capture in the parent GPU process. Each of N spawned CPU workers loads one official seeded runtime, owns E independent battle pointers, maintains one causal stateless compiler per battle side/session, and multiplexes at most I synchronous inference channels back to the parent. Prototype embeddings are a derived non-checkpoint cache owned by each frozen semantic policy and are reused in eval mode and frozen-encoder PPO while remaining invalidated for trainable prototype parameters.

**Tech Stack:** Python 3.11, PyTorch/CUDA, `multiprocessing` spawn, `ctypes` official seeded `libcg`, worker threads around independent battle pointers, W&B online, unittest.

## Global Constraints

- Never modify `engine/source/`; only the existing seeded official runtime under shared evaluation infrastructure may execute games.
- `train/0037_dragapult_value_initialized_rl/` remains self-contained and must not import executable code from another numbered training project.
- Preserve the exact 39-tensor actor schema, action decoder contract, 0036 V2 epoch-5 value initialization, terminal-only reward, turn-clock GAE lambda `0.95`, and 256 paired seed scenarios / 512 Episodes per update.
- Preserve V1 artifacts as an immutable failed/incomplete-topology record; use `V2_engine_pool_prototype_cache_seeded512` for the corrected formal run.
- Keep model-only checkpoint retention `all`; do not serialize prototype caches, optimizers, RNG state, rollout buffers, or engine state.
- Formal training uses W&B online project `dragon_bra/pokemon-tcg-policy-learning` and a run name beginning with `0037`.
- A topology is admissible only with exact compiler/action/replay parity, every requested official-engine game completed, zero errors, and no Encoder representation change.
- Benchmark topology, not policy strength: all arms consume the same checkpoint, exact paired schedule, inference mode, dtype, and coalescing contract.

---

### Task 1: Preserve the incomplete V1 and encode the corrected runtime contract

**Files:**
- Modify: `rl_runs/0037_dragapult_value_initialized_rl/versions/V1_value_initialized_turn_clock_seeded512/artifact/status.json`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DECISIONS.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`
- Modify: `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_project_identity.py`

**Interfaces:**
- Consumes: V1 status, metrics, schedules, checkpoint 0/1, and the confirmed runtime implementation.
- Produces: immutable failure reason plus `RunConfig.worker_processes`, `RunConfig.engines_per_worker`, and `RunConfig.inference_channels_per_role` recorded in every V2 config/status/metric payload.

- [ ] **Step 1: Add a failing configuration test**

Require invalid N/E/I values to fail; require `I <= E`; require V2 defaults to name the corrected topology and include a structured `rollout_topology` block in serialized config.

- [ ] **Step 2: Run the focused test and confirm failure**

Run `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_project_identity` and require the new assertions to fail before implementation.

- [ ] **Step 3: Add explicit topology fields and V1 failure evidence**

Replace the ambiguous `workers` runtime field with explicit N/E/I fields at the collector boundary, retain a compatibility-only `workers` CLI alias only if tests require it, and record V1's reason as `missing_engine_pool_and_prototype_cache_contract` without deleting its artifacts.

- [ ] **Step 4: Synchronize DESIGN and DECISIONS**

Document that V1 used N32E1 per-game spawning and uncached `encode_all()`, while V2 requires worker-local stateless compilation, frozen prototype cache, and an empirically selected N/E/I topology.

- [ ] **Step 5: Re-run the focused test**

Require the project identity/config tests to pass and parse both design documents for the new version and topology terms.

### Task 2: Port the admitted frozen prototype GPU cache into 0037

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/semantic_policy/model/policy.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_model.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_actor_critic.py`

**Interfaces:**
- Consumes: `OfficialPrototypeEncoder.encode_all()` and frozen Encoder/prototype parameters.
- Produces: `prototype_memory()`, `prepare_prototype_cache()`, `clear_prototype_cache(reason)`, and `prototype_cache_stats()` with no checkpoint keys.

- [ ] **Step 1: Write failing cache lifecycle tests**

Wrap `encode_all()` and require two eval forwards to build once, reuse four same-device tensors, preserve exact logits/actions, and leave `state_dict()` keys unchanged. Require `_apply` and `load_state_dict` to invalidate. Require trainable prototype parameters in train mode to bypass the cache, while fully frozen prototypes reuse detached cache and still permit gradients into decoder/value parameters.

- [ ] **Step 2: Confirm the tests fail against direct `encode_all()`**

Run the two focused test modules and verify cache API/call-count assertions fail.

- [ ] **Step 3: Implement the exact admitted lifecycle**

Port the 0035 cache semantics into the self-contained 0037 `SemanticPolicy`: a plain derived attribute, `torch.no_grad()` build, invalidation on `_apply`/`load_state_dict`, and top-level encode paths routed through `prototype_memory()`.

- [ ] **Step 4: Verify focal actor and frozen opponent behavior**

Load the real Policy-0806 checkpoint twice, require both policies to report one build and subsequent hits during repeated inference, and require exact deterministic action parity with the pre-cache commitment.

- [ ] **Step 5: Run cache/model regressions**

Run `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_model train.0037_dragapult_value_initialized_rl.tests.test_actor_critic train.0037_dragapult_value_initialized_rl.tests.test_checkpoint`.

### Task 3: Add a session-aware RL official-engine pool

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/rollout/pool_worker.py`
- Modify: `train/0037_dragapult_value_initialized_rl/rollout/collector.py`
- Modify: `train/0037_dragapult_value_initialized_rl/rollout/protocol.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_rollout_pool.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_worker_feature_parity.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_rollout_termination.py`

**Interfaces:**
- Consumes: `list[RolloutJob]`, N worker processes, E battle sessions per process, I inference channels per role, and central actor/opponent models.
- Produces: the same ordered `list[EpisodeTrajectory]`, plus topology and timing counters; each request carries `session_id`, `game_id`, role, turn, selection index, canonical record, and compiler time.

- [ ] **Step 1: Write failing pool protocol tests**

Use fake pointer battles and fake channel endpoints to require independent per-session compiler history, correct response routing under interleaving, deterministic output ordering, exactly-once battle finish/session close, isolated single-game failure, and no torch import inside the worker compiler path.

- [ ] **Step 2: Confirm the new tests fail because no pool exists**

Run `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_rollout_pool`.

- [ ] **Step 3: Implement one seeded pointer battle per session**

Wrap the existing official ABI with `BattleStartSeeded`, `GetBattleData`, `Select`, and `BattleFinish`. Guard library entry with one per-process lock, but allow other battle threads to compile or wait for GPU inference concurrently. Keep first-player forcing, terminal reward, full-round draw, repeated-action forfeit, step limit, and diagnostic payload identical to the E1 worker.

- [ ] **Step 4: Implement N/E/I worker scheduling**

Spawn N processes once per `collect()`. Give each process a stable job shard, execute up to E games with threads, and lease at most I duplex inference channels per role. The parent polls all channels, batches ready focal/opponent records separately, samples focal actions with the job-specific `torch.Generator`, and appends `TrajectoryDecision` to the matching session before responding on the same leased channel.

- [ ] **Step 5: Preserve canonical ordering and failure cleanup**

Return trajectories in input job order regardless of completion order. On timeout/error, close channels, join/terminate only the affected pool process as needed, and synthesize auditable errors containing game, opponent, seat, seed, last decision/action, and worker exit code.

- [ ] **Step 6: Run protocol, compiler, and termination tests**

Run the three focused test modules and require every new/existing assertion to pass.

### Task 4: Prove semantic parity with the real official engine

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py`
- Create: `.tmp/evaluation/0037_rl_pool_gate/` runtime reports
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_ppo_full_semantic.py`

**Interfaces:**
- Consumes: legacy N32E1 collector as a temporary reference and pooled collector using identical jobs/checkpoint.
- Produces: exact job/schedule parity, compiler 39-tensor parity, greedy action parity, stochastic behavior-logprob replay error <= `1e-4`, finite PPO gradients, and unchanged Encoder hash.

- [ ] **Step 1: Add a collector gate report schema**

Record N/E/I, requested/completed/error games, schedule hash, decisions/selections, compiler/model/IPC/wall timings, batches, max/mean batch, games/s, selections/s, RSS, prototype cache build/hit counts, behavior replay MAE, and representation hash.

- [ ] **Step 2: Run an 8-game E1-vs-pool greedy parity gate**

Use the same official paired jobs and require identical completion, reward, turn, ordered action traces, and 39 input tensors for every focal decision.

- [ ] **Step 3: Run a 16-game stochastic PPO canary**

Require 16/16 valid, zero errors, finite value/GAE/PPO metrics, behavior logprob MAE <= `1e-4`, a changed decoder/value checkpoint, unchanged Encoder hash, and model-only checkpoint schema.

- [ ] **Step 4: Run the complete 0037 test suite**

Run `python3 -m unittest discover -v train/0037_dragapult_value_initialized_rl/tests` and `python3 -m compileall -q train/0037_dragapult_value_initialized_rl`.

### Task 5: Select the fastest RL topology on this machine

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/benchmark_cpu_workers.py`
- Create: `.tmp/evaluation/0037_rl_pool_benchmark/quick_*.json`
- Create: `.tmp/evaluation/0037_rl_pool_benchmark/final_*.json`
- Create: `.tmp/evaluation/0037_rl_pool_benchmark/summary.json`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DECISIONS.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`

**Interfaces:**
- Consumes: fixed Policy-0806/value checkpoint, fixed 64-pair quick schedule and fixed 256-pair final schedule.
- Produces: selected N/E/I/coalescing configuration for V2.

- [ ] **Step 1: Implement isolated one-arm benchmark execution**

Each arm runs in a fresh Python process and writes atomic JSON. Reject mismatched schedule/checkpoint hashes, incomplete games, errors, behavior replay mismatch, or representation changes before comparing speed.

- [ ] **Step 2: Run the 128-game quick screen**

Measure at least `N32E1/I1` legacy, `N16E1/I1`, `N16E4/I4`, `N16E8/I8`, `N16E16/I8`, `N8E8/I8`, and `N8E16/I8` with the same paired schedule. Use 2 ms coalescing initially and capture games/s, selections/s, mean/max GPU batch, model utilization, CPU time, peak RSS, compiler time, and IPC wait.

- [ ] **Step 3: Tune coalescing for the best two N/E/I arms**

Compare 0.5, 2, and 5 ms with the same quick schedule. Reject settings that improve batch size but lower games/s or introduce tail stalls/errors.

- [ ] **Step 4: Re-run the best two arms on 512 games**

Use the exact paired seeded-512 schedule intended for training and run each finalist at least twice if their games/s differ by less than 5%. Select the smallest-memory arm within 3% of the best repeated games/s; otherwise select the outright fastest valid arm.

- [ ] **Step 5: Record the decision and evidence**

Write exact artifact paths, hardware, N/E/I meaning, all rejected arms/reasons, final throughput relative to V1's 512/417.389 s baseline, and the chosen defaults into both DESIGN formats and DECISIONS.

### Task 6: Launch the corrected formal V2 baseline

**Files:**
- Create: `rl_runs/0037_dragapult_value_initialized_rl/versions/V2_engine_pool_prototype_cache_seeded512/artifact/`
- Create: `rl_runs/0037_dragapult_value_initialized_rl/versions/V2_engine_pool_prototype_cache_seeded512/checkpoint/`
- Create: `rl_runs/0037_dragapult_value_initialized_rl/versions/V2_engine_pool_prototype_cache_seeded512/tensorboard/`
- Create: `rl_runs/0037_dragapult_value_initialized_rl/versions/V2_engine_pool_prototype_cache_seeded512/wandb/`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DECISIONS.md`

**Interfaces:**
- Consumes: selected topology, passed semantic gates, source actor/value hashes, and unchanged seeded-512 training contract.
- Produces: W&B-online V2 run, fixed update-0 evaluation, per-update schedules/metrics, and model-only checkpoints retained for every update.

- [ ] **Step 1: Assert the V2 version is fresh**

Require artifact/checkpoint/tensorboard/wandb paths and any matching formal evaluation HTML to be unused before creating them.

- [ ] **Step 2: Launch W&B online with selected topology**

Use run ID/name beginning `0037`, include N/E/I/cache tags/config, and write local `training_metrics.jsonl` before TensorBoard/W&B mirrors.

- [ ] **Step 3: Verify fixed baseline and first PPO update**

Require seeded512 evaluation 512/512 with zero errors, then require update 1 rollout 512/512, finite explained variance/value/GAE/KL metrics, replay MAE <= `1e-4`, checkpoint 1 present, prototype cache build counts bounded by loaded model count, and Encoder hash unchanged.

- [ ] **Step 4: Commit the implementation and immutable run evidence**

Run `git diff --check`, verify `git diff --name-only -- engine/source/` is empty, stage only 0037/plans/tests/docs plus V1/V2 small tracked evidence, and exclude dataset/checkpoint/W&B staging via `.gitignore`.

## Self-Review

- Spec coverage: both missing admitted optimizations, explicit N/E/I semantics, RL-specific benchmarking, paired-seed/value/PPO invariants, W&B, checkpoint retention, docs, and formal relaunch each have an implementation and verification task.
- Placeholder scan: the plan contains no TBD/TODO/fill-later step; every benchmark arm, threshold, path, and required metric is explicit.
- Type consistency: N=`worker_processes`, E=`engines_per_worker`, I=`inference_channels_per_role` are used consistently from config through collector, report, benchmark, docs, and launch.
