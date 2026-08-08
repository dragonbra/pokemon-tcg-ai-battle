# 0038 CUDA-Resident Rollout Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA-resident rollout backend to 0038 that preserves its DecisionGate, Phantom Dive macro, joint-logprob, semi-MDP trajectory, and official primitive-action contracts while reusing the shared `engine_cuda` resident scheduler.

**Architecture:** Extend the shared Python resident loop with an optional, backward-compatible action-adapter hook. A self-contained 0038 adapter classifies device semantic rows before policy routing, bypasses true forced callbacks, plans Phantom Dive allocation after the root decision using the already encoded state/options, and expands the cached plan across six CUDA-engine callbacks. A 0038 CUDA collector materializes the same `EpisodeTrajectory`/`PolicyTransition` types consumed by the existing GAE/PPO path; CPU remains a diagnostic backend.

**Tech Stack:** Python 3.11, PyTorch CUDA, `_ptcg_cuda` official-rules runtime, Semantic0031 device compiler/router, 0038 PPO and Action Boundary modules, `unittest`.

## Global Constraints

- Do not modify `engine/source/`, official ABI, official observation schema, or primitive `select` shape.
- Do not import executable code from `train/0037_*`; 0038 remains self-contained.
- Reuse only shared `engine_cuda` Python/runtime components.
- Strict FP32: `torch.set_float32_matmul_precision("highest")`, TF32 disabled.
- Preserve per-game policy-seed sampling independent of lane/refill order.
- Forced and macro parameter callbacks do not create Policy/Value samples or gamma/lambda steps.
- One Phantom Dive macro stores one root logprob, one allocation logprob, one joint old logprob, and one PolicyTransition.
- Any chance/information/priority/identity/mask/terminal drift fails closed and invalidates the macro PPO sample.
- Do not start formal PPO; only tests, parity gates, a short rollout/PPO smoke, and a lane-count benchmark.
- Keep the existing V4 formal training guard until the user explicitly approves launch.

---

### Task 1: Add a backward-compatible resident action-adapter interface

**Files:**
- Modify: `engine_cuda/python/ptcg_cuda_engine/semantic0031_resident.py`
- Create: `engine_cuda/tests/test_semantic0031_resident_action_adapter.py`

**Interfaces:**
- Consumes: ready-lane Semantic0031 tensors, `lane_job`, focal/opponent route masks, turn and selection counters.
- Produces: `ResidentActionBypass(actions, lengths, bypass_mask, metadata)` and optional `post_route(...)` metadata; default `None` preserves 0037/evaluation behavior exactly.

- [ ] **Step 1: Write a failing unit test for route bypass**

```python
def test_bypass_rows_are_not_sent_to_router_and_keep_primitive_shape():
    adapter = FakeAdapter(bypass_lane=1, action=3)
    result = run_resident_greedy_jobs(..., action_adapter=adapter)
    assert adapter.pre_route_calls > 0
    assert router.seen_route_masks[0][1].item() is False
    assert result.action_adapter_bypasses == 1
```

- [ ] **Step 2: Run the focused test and verify the missing argument/interface failure**

Run: `python3 -m unittest -v engine_cuda.tests.test_semantic0031_resident_action_adapter`

- [ ] **Step 3: Implement the minimal hook with no behavior change when disabled**

```python
@dataclass(frozen=True)
class ResidentActionBypass:
    actions: torch.Tensor
    lengths: torch.Tensor
    bypass_mask: torch.Tensor
    metadata: dict[str, Any]

def run_resident_greedy_jobs(..., action_adapter=None):
    bypass = action_adapter.pre_route(...) if action_adapter is not None else None
    policy_ready = ready & ~bypass.bypass_mask if bypass is not None else ready
    routed = router.route(..., focal_route=focal_route & policy_ready,
                          opponent_route=opponent_route & policy_ready)
    if bypass is not None:
        routed.actions = torch.where(bypass.bypass_mask[:, None], bypass.actions, routed.actions)
        routed.lengths = torch.where(bypass.bypass_mask, bypass.lengths, routed.lengths)
```

- [ ] **Step 4: Add counters without per-request disk logging**

Extend `ResidentRunResult` with `action_adapter_bypasses`, `forced_bypasses`, `macro_bypasses`, and `invalid_macro_count`; aggregate in memory only.

- [ ] **Step 5: Run resident/router regression tests**

Run: `python3 -m unittest -v engine_cuda.tests.test_semantic0031_resident engine_cuda.tests.test_semantic0031_router engine_cuda.tests.test_semantic0031_resident_action_adapter`

### Task 2: Implement the 0038 device DecisionGate and Phantom transaction

**Files:**
- Create: `train/0038_action_boundary_rl/rollout/cuda_action_adapter.py`
- Create: `train/0038_action_boundary_rl/tests/test_cuda_action_adapter.py`
- Reuse: `train/0038_action_boundary_rl/action_boundary/dragapult.py`
- Reuse: `train/0038_action_boundary_rl/policy/allocation_head.py`

**Interfaces:**
- Consumes: device `semantic` tensors plus routed `validated`, `state`, `focal_options`, root actions/logprob/entropy/value.
- Produces: per-job pending macro state, forced/macro primitive actions, parameter logprob/entropy, macro metadata, invalid/fallback reasons.

- [ ] **Step 1: Write tensor-fixture tests for gate classifications**

```python
def test_one_option_without_stop_is_forced():
    semantic = semantic_row(option_types=[6], min_count=1, max_count=1)
    assert CudaDecisionGate.classify(semantic).name == "FORCED"

def test_one_option_with_stop_is_strategic():
    semantic = semantic_row(option_types=[6, STOP_TYPE], min_count=0, max_count=1)
    assert CudaDecisionGate.classify(semantic).name == "STRATEGIC"
```

Cover empty terminal/pass/mask error, aliases, priority actor change, chance status, and 2+ strategic options.

- [ ] **Step 2: Implement field extraction from the existing Semantic0031 schema**

Use `global_cat[:, 1]` for context, `global_num[:, 10:14]` for option/min/max/remaining counters, `option_mask`, `option_cat`, `option_target`, and target `card_cat/card_num`. Do not read hidden opponent zones or true deck ID.

- [ ] **Step 3: Write exhaustive device allocation tests**

```python
for n, expected in {1: 1, 2: 7, 3: 28, 4: 84, 5: 210}.items():
    targets = adapter.visible_bench_targets(semantic_fixture(n))
    assert len(enumerate_allocations(targets)) == expected
```

Assert identities use `(relative player, serial, card id, initial bench slot)` from visible card rows.

- [ ] **Step 4: Implement root-time allocation scoring**

After router output identifies attack ID 154, gather opponent Bench state embeddings and build the same 12 visible allocation features from device tensors. Call `DragapultAllocationHead` once and save:

```python
PendingCudaMacro(
    schedule_index=job,
    actor=actor,
    target_serials=...,
    counters=...,
    allocation_logprob=...,
    allocation_entropy=...,
    remaining=6,
)
```

For `n=1`, parameter logprob/entropy are exactly zero.

- [ ] **Step 5: Implement six callback bypasses with fail-closed validation**

On context 14, verify actor, remaining count, legal target serial/card/slot set, no chance/information/priority change, and one matching option. Return a length-1 primitive action. Clear on completion; invalidate and expose reason on drift.

- [ ] **Step 6: Test confused Phantom Dive and target drift**

Confused root must not precommit a macro. Missing target, changed serial, empty mask, unexpected terminal, actor transfer, and remaining-counter mismatch must increment invalid/fallback and never emit a valid macro PPO sample.

### Task 3: Build the self-contained 0038 CUDA collector

**Files:**
- Create: `train/0038_action_boundary_rl/rollout/cuda_collector.py`
- Modify: `train/0038_action_boundary_rl/rollout/__init__.py`
- Create: `train/0038_action_boundary_rl/tests/test_cuda_collector.py`

**Interfaces:**
- Consumes: 0038 `RolloutJob`, common update-0 actor/value/allocation head, Frozen-0806 opponent, shared resident scheduler.
- Produces: existing 0038 `EpisodeTrajectory` with `TrajectoryDecision`, `PolicyTransition`, compact diagnostics, and collector scalar metrics.

- [ ] **Step 1: Write construction and job-conversion tests**

Assert exact focal deck, unique game IDs, one source policy update, policy seeds, focal seats, engine seeds, and opponent IDs survive conversion to `ResidentJob`.

- [ ] **Step 2: Implement `CudaActionBoundaryRolloutCollector`**

Mirror 0037's resident assembly but import only shared `engine_cuda`. Use `Semantic0031DeviceAdapter`, `Semantic0031ResidentRouter`, and the Task 2 action adapter. Preserve 0038 auxiliary values and macro metadata in the decision sink.

- [ ] **Step 3: Materialize only strategic decisions**

```python
TrajectoryDecision(
    log_prob=root_logprob + parameter_logprob,
    parameter_log_prob=parameter_logprob,
    value=pre_action_value,
    macro_action=macro_metadata,
    engine_event_index=official_selection_count,
)
```

Forced and macro callback rows never enter `decisions`.

- [ ] **Step 4: Build semi-MDP transitions**

Use the next strategic decision's value, terminal reward, and `accumulated_discount=1.0`. Set terminal `next_value=0`. Record prize counts/turn from already produced semantic tensors. Keep `post_commit_observation=None` because no private JSON observation is available; record an explicit CUDA metadata flag rather than synthesizing observation fields.

- [ ] **Step 5: Add trace retention policy**

Normal successful games retain compact selection/turn/prize/macro metadata. Invalid/fallback/error and deterministic sampled games retain device state hashes and semantic boundary summaries. Do not write per-step files.

- [ ] **Step 6: Test trajectory semantics**

Assert one Phantom macro produces one decision/transition, `joint=root+allocation`, six callback bypasses, no extra Value sample, no extra discount, and invalid fallback transitions have `valid=False`.

### Task 4: Wire backend selection into the guarded 0038 training entry

**Files:**
- Modify: `train/0038_action_boundary_rl/training/run_full_semantic.py`
- Modify: `train/0038_action_boundary_rl/training/scaling.py`
- Modify: `train/0038_action_boundary_rl/configs/BASE.json`
- Modify: `train/0038_action_boundary_rl/configs/INTEGRATED.json`
- Modify: `train/0038_action_boundary_rl/configs/PRIZE.json`
- Modify: `train/0038_action_boundary_rl/configs/META.json`
- Modify: `train/0038_action_boundary_rl/configs/PRIZE_META.json`
- Modify: `train/0038_action_boundary_rl/tests/test_project_identity.py`
- Modify: `train/0038_action_boundary_rl/tests/test_scaling_and_frozen_panel.py`

**Interfaces:**
- Consumes: `engine_backend in {"official_cpu", "cuda_resident"}` and CUDA topology/path configuration.
- Produces: one `build_collector(config, model, opponent, mode)` factory used identically by rollout and Frozen evaluation.

- [ ] **Step 1: Add failing config tests**

Assert `cuda_lane_count`, `cuda_check_interval`, `cuda_rules_path`, `cuda_extension_dir`, strict FP32, and backend version are validated and serialized.

- [ ] **Step 2: Implement the collector factory**

```python
def build_collector(config, model, opponent, *, mode):
    if config.engine_backend == "cuda_resident":
        return CudaActionBoundaryRolloutCollector(...)
    return FullSemanticRolloutCollector(...)
```

- [ ] **Step 3: Keep optimization intensity independent of rollout size**

Do not change PPO epochs, learning rates, effective minibatch, or optimizer steps when selecting CUDA or increasing `rollout_games_per_update`. Preserve `fixed_optimizer_budget` as default.

- [ ] **Step 4: Record immutable backend metadata**

Checkpoint/training config must include engine backend/version, extension ABI, rules hash, lane count, check interval, strict-FP32 settings, rollout games, valid transitions, optimizer steps, effective minibatch, and PPO epochs.

- [ ] **Step 5: Verify the formal launch guard remains active**

Run: `python3 -m unittest -v train.0038_action_boundary_rl.tests.test_project_identity`

### Task 5: Run parity, replay, PPO, and throughput gates

**Files:**
- Create: `train/0038_action_boundary_rl/evaluation/cuda_rollout_gate.py`
- Create: `train/0038_action_boundary_rl/benchmark_cuda_rollout.py`
- Create: `train/0038_action_boundary_rl/tests/test_cuda_rollout_gate.py`
- Generate locally: `.tmp/evaluation/0038_cuda_rollout_gate/<run-id>/`

**Interfaces:**
- Consumes: identical small fixed job manifest for CPU and CUDA backends.
- Produces: JSON/HTML-or-Markdown gate report with action, state, trajectory, logprob, determinism, throughput, and resource results.

- [ ] **Step 1: Add deterministic CPU/CUDA action parity gate**

Compare strategic root action, legal mask, forced classification, Phantom allocation, primitive expansion, terminal result, winner, turn, repeat-forfeit, and terminal-state hash on fixed seeds.

- [ ] **Step 2: Add stochastic behavior-logprob replay**

Re-evaluate saved root/allocation actions with the frozen behavior checkpoint and require mean absolute joint-logprob error `<=1e-4`.

- [ ] **Step 3: Run one real PPO minibatch**

Prepare CUDA-collected episodes through the unchanged 0038 GAE/batch pipeline. Require finite losses/gradients, frozen old logprob, and no forced samples in policy/entropy/KL/clip denominators.

- [ ] **Step 4: Run same-seed CUDA determinism twice**

Require identical per-game outcome, repeat-forfeit, terminal turn, terminal-state hash, strategic actions, allocations, and trajectory length independent of refill order.

- [ ] **Step 5: Benchmark lane counts without formal training**

Run fixed 64/128/256-lane short batches and report hot-loop/total games per second, strategic decisions per second, PolicyTransitions per second, staged bytes, CUDA allocated/reserved, CPU RSS, invalid/fallback, and peak lane utilization. Select a default from measured results.

- [ ] **Step 6: Run the complete 0038 unit suite**

Run: `python3 -m unittest discover -s train/0038_action_boundary_rl/tests -p 'test_*.py' -q`

### Task 6: Synchronize authoritative design and status documents

**Files:**
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Modify: `experiments/0038_action_boundary_rl/DECISIONS.md`
- Modify: `ACTION_BOUNDARY_0038_STATUS.md`
- Create after gates: `experiments/0038_action_boundary_rl/CUDA_ROLLOUT_GATE.md`

**Interfaces:**
- Consumes: actual code/config, parity results, benchmark results, extension/rules hashes.
- Produces: auditable current/next-stage documentation; no unsupported strength claim.

- [ ] **Step 1: Document the device action-boundary flow**

Include tensor fields, gate/bypass masks, macro transaction lifecycle, hierarchical probability, trajectory schema, strict FP32, seed contract, and CPU fallback semantics.

- [ ] **Step 2: Document evidence boundaries**

State separately: official TCG rules, CPU official-engine evidence, CUDA official-rules runtime parity evidence, and project policy assumptions. CUDA throughput alone is not official-engine equivalence evidence.

- [ ] **Step 3: Record the launch decision**

Keep formal V4 fail closed until CPU/CUDA gates pass and the user explicitly approves training. Do not run 50 updates as part of this plan.

## Self-review

- Spec coverage: shared resident reuse, 0038 self-containment, action-boundary semantics, macro hierarchy, PPO logprob, seed determinism, backend configuration, parity, smoke, throughput, and docs are each assigned to a task.
- Placeholder scan: no TBD/TODO or unspecified test step remains.
- Type consistency: Tasks 1–3 share `ResidentActionBypass`; Tasks 3–5 use existing `EpisodeTrajectory` and `PolicyTransition`; Task 4 exposes one backend-neutral collector factory.
