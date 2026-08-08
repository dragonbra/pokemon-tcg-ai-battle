# 0037 Action Boundary V1 Implementation Plan

> **For agentic workers:** Implement task-by-task with a review gate after every task. Do not start PPO or any long training run from this plan.

**Goal:** Correct PPO decision boundaries for structurally forced selections and make Dragapult ex Phantom Dive one hierarchical macro action while preserving the official engine ABI, observation, primitive action format, and callback count.

**Architecture:** A worker-side `DecisionGate` consumes every official observation before tensor compilation, while an `OfficialProtocolExecutor` either forwards a strategic request to the existing batched inference service or returns a verified forced/pending-macro primitive action. The actor keeps its root decoder unchanged and adds a conditional permutation-invariant allocation head that reuses the same state/card/option encodings. Engine events remain lossless and policy transitions become pending records finalized only at the next focal-visible policy boundary or terminal.

**Tech Stack:** Python 3.11, PyTorch, official seeded C++ runtime through the existing ctypes wrapper, `unittest`, JSONL/GZip telemetry, model-only checkpoints.

## Global Constraints

- Never modify `engine/source/`, the official ABI, official observation fields, or the required `list[int]` primitive action.
- Preserve the current request-local GRU contract; no hidden state crosses an official callback.
- Shadow mode observes only and must preserve action bytes, model outputs, sampling RNG consumption, and the existing PPO trajectory.
- V1 only shortcuts structurally proven forced choices and only compounds Phantom Dive; search, chance, opponent-controlled, setup, and information-reveal chains keep their current decision semantics.
- `MASK_ERROR` fails closed. Macro drift may finish online with the legacy per-callback policy, but the whole affected Episode is excluded from PPO in V1.
- V5 rollout data is never used for a V6 PPO ratio. V6 initializes a new optimizer, scheduler, and on-policy buffer.
- Model/action/trajectory changes require synchronized updates to both `experiments/0037_dragapult_value_initialized_rl/DESIGN.md` and `DESIGN.html`.
- Existing uncommitted progress-guard edits in `rollout/worker.py`, `rollout/pool_worker.py`, `rollout/protocol.py`, `training/run_full_semantic.py`, tests, and DESIGN files are user-owned and must be preserved.

---

## File Map

### New production modules

- `train/0037_dragapult_value_initialized_rl/action_boundary/__init__.py` — public V1 boundary interfaces and version constants.
- `train/0037_dragapult_value_initialized_rl/action_boundary/types.py` — enums and immutable telemetry, boundary, macro, trace, and policy-transition records.
- `train/0037_dragapult_value_initialized_rl/action_boundary/decision_gate.py` — completion counting, exit-option detection, boundary flags, and fail-closed classification.
- `train/0037_dragapult_value_initialized_rl/action_boundary/telemetry.py` — shadow aggregators and JSON-serializable per-context histograms.
- `train/0037_dragapult_value_initialized_rl/action_boundary/event_trace.py` — lossless per-session engine event trace and optional atomic `.jsonl.gz` writer.
- `train/0037_dragapult_value_initialized_rl/action_boundary/dragapult.py` — Phantom identification, allocation enumeration, stable-target extraction, visible allocation features, and alias counts.
- `train/0037_dragapult_value_initialized_rl/action_boundary/macro_protocol.py` — `MacroPlanner`, `PendingMacroTransaction`, and `OfficialProtocolExecutor`.
- `train/0037_dragapult_value_initialized_rl/policy/allocation_head.py` — conditional permutation-invariant allocation scorer.
- `train/0037_dragapult_value_initialized_rl/data/extract_dragapult_allocations.py` — audited replay/trace extractor and coverage manifest generator.
- `train/0037_dragapult_value_initialized_rl/training/allocation_bc.py` — bounded allocation-head-only BC warm-start entry point.
- `train/0037_dragapult_value_initialized_rl/render_action_boundary_smoke.py` — short smoke/benchmark HTML/JSON report.

### Existing production files to modify

- `train/0037_dragapult_value_initialized_rl/rollout/worker_compiler.py` — add observation-only consumption without tensor compilation.
- `train/0037_dragapult_value_initialized_rl/semantic_policy/deployment/online_runtime.py` — mirror `observe_only` for deployment parity.
- `train/0037_dragapult_value_initialized_rl/rollout/worker.py` — insert gate/executor before compilation in the single-engine path.
- `train/0037_dragapult_value_initialized_rl/rollout/pool_worker.py` — insert per-job gate/executor, trace, timing, pending transaction, and cleanup in the production pooled path.
- `train/0037_dragapult_value_initialized_rl/rollout/protocol.py` — versioned internal IPC and `PolicyTransition`/Episode schema.
- `train/0037_dragapult_value_initialized_rl/rollout/collector.py` — shadow aggregation, hierarchical inference, pending transition finalization, invalid-macro handling, and new metrics.
- `train/0037_dragapult_value_initialized_rl/policy/action_distribution.py` — one-encode root/allocation sampling and joint evaluation.
- `train/0037_dragapult_value_initialized_rl/policy/actor_critic.py` — attach allocation head, expose encoded components, and update trainable inventory.
- `train/0037_dragapult_value_initialized_rl/training/batch_full_semantic.py` — accumulated reward/discount GAE and macro batching.
- `train/0037_dragapult_value_initialized_rl/training/ppo_full_semantic.py` — joint root/allocation PPO, masks, and strategic-only metrics.
- `train/0037_dragapult_value_initialized_rl/training/storage_full_semantic.py` — V6 model-only schema, metadata guards, and explicit V5 migration.
- `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py` — V6 config/version gate, shadow/smoke modes, metric names, and no implicit long run.
- `train/0037_dragapult_value_initialized_rl/export_full_semantic_candidate.py` — support the V6 allocation head and compound inference runtime without changing Kaggle action output.
- `experiments/0037_dragapult_value_initialized_rl/DESIGN.md` — authoritative action, tensor, loss, checkpoint, and stage contract.
- `experiments/0037_dragapult_value_initialized_rl/DESIGN.html` — same facts as DESIGN.md.
- `experiments/0037_dragapult_value_initialized_rl/DECISIONS.md` — record the V6 action-contract decision and V5 incompatibility.

### New or expanded tests

- `train/0037_dragapult_value_initialized_rl/tests/test_decision_gate.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_shadow_telemetry.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_observe_only.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_dragapult_allocations.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_allocation_head.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_macro_protocol.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_policy_trajectory_v2.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_action_boundary_checkpoint.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_dragapult_trace_extractor.py`
- `train/0037_dragapult_value_initialized_rl/tests/test_dragapult_official_parity.py`
- Modify existing `test_actor_critic.py`, `test_full_terminal_credit.py`, `test_rollout_batch.py`, `test_rollout_pool.py`, `test_rollout_termination.py`, `test_storage.py`, and `test_worker_feature_parity.py` where their old trajectory/checkpoint assumptions change.

---

## Public Interfaces

```python
class DecisionClassification(str, Enum):
    TERMINAL = "TERMINAL"
    LEGAL_EMPTY_PASS = "LEGAL_EMPTY_PASS"
    MASK_ERROR = "MASK_ERROR"
    FORCED = "FORCED"
    STRATEGIC = "STRATEGIC"

class BoundaryFlag(str, Enum):
    CHANCE_BOUNDARY = "CHANCE_BOUNDARY"
    INFORMATION_BOUNDARY = "INFORMATION_BOUNDARY"
    PRIORITY_TRANSFER = "PRIORITY_TRANSFER"

@dataclass(frozen=True, slots=True)
class DecisionGateResult:
    classification: DecisionClassification
    boundary_flags: tuple[BoundaryFlag, ...]
    raw_option_count: int
    legal_completion_count: int
    canonical_completion_count: int
    stop_legal: bool
    cancel_legal: bool
    decline_legal: bool
    pass_legal: bool
    forced_action: tuple[int, ...] | None
    action_family: str
    same_outcome_alias_count: int

def classify_observation(
    observation: Mapping[str, object],
    previous: BoundarySnapshot | None,
) -> DecisionGateResult: ...
```

`CHANCE_BOUNDARY`, `INFORMATION_BOUNDARY`, and `PRIORITY_TRANSFER` are orthogonal flags rather than mutually exclusive choice counts. A strategic request after a reveal remains `classification=STRATEGIC` and carries the corresponding flag; this keeps policy-loss masking unambiguous.

```python
@dataclass(frozen=True, slots=True)
class StableTargetIdentity:
    player_index: int
    serial: int
    card_id: int
    initial_bench_slot: int

@dataclass(frozen=True, slots=True)
class DragapultDamageAllocation:
    target_ids: tuple[StableTargetIdentity, ...]
    counters: tuple[int, ...]
    total: int = 6

def enumerate_dragapult_allocations(
    targets: Sequence[StableTargetIdentity],
) -> tuple[DragapultDamageAllocation, ...]: ...
```

Target order is canonicalized by `(initial_bench_slot, serial, card_id)`. Reordering target rows and the matching counter columns leaves the allocation score unchanged.

```python
@dataclass(slots=True)
class PendingMacroTransaction:
    session_id: str
    actor: int
    root_selection_index: int
    root_event_start: int
    root_action: tuple[int, ...]
    allocation: DragapultDamageAllocation
    expected_effect_card_id: int
    expected_effect_serial: int
    next_counter_ordinal: int
    created_monotonic: float
    expires_monotonic: float
    ppo_valid: bool = True

class MacroPlanner:
    def plan_dragapult(... ) -> MacroPlan | None: ...

class OfficialProtocolExecutor:
    def consume(
        self,
        observation: Mapping[str, object],
        selection_index: int,
    ) -> ProtocolDirective: ...
    def accept_model_response(self, response: Mapping[str, object]) -> list[int]: ...
    def clear(self, reason: str) -> None: ...
```

`ProtocolDirective` is one of `REQUEST_POLICY`, `RETURN_FORCED`, `RETURN_PENDING_MACRO`, `FAIL_CLOSED`. Only the resulting `list[int]` is passed into the official `Select`/`battle_select` call.

```python
@dataclass(frozen=True, slots=True)
class PolicyTransition:
    pre_action_features: dict[str, Tensor]
    canonical_action: PrimitivePolicyAction | PhantomDiveMacroAction
    root_indices: tuple[int, ...]
    root_stopped: bool
    root_old_logprob: float
    allocation_old_logprob: float
    joint_old_logprob: float
    root_entropy: float
    allocation_entropy: float
    pre_action_value: float
    accumulated_reward: float
    accumulated_discount: float
    next_value: float
    done: bool
    engine_event_span: tuple[int, int]
    boundary: BoundaryMetadata
    policy_valid: bool
    policy_update: int
    turn: int
```

The engine trace contains the raw actor-view observation. `PolicyTransition` stores actor-visible tensors and trace indices, never an opponent-private observation. `post_commit_observation` is represented by the end of `engine_event_span`; bootstrap features/value are attached only at the next focal-visible true decision. Terminal before that boundary yields `next_value=0`.

---

### Task 1: Freeze the dirty-worktree baseline and add Shadow Telemetry

**Files:** Create `action_boundary/types.py`, `decision_gate.py`, `telemetry.py`, and their tests. Modify `worker.py`, `pool_worker.py`, `collector.py`, and `protocol.py` only additively.

- [ ] Record hashes/diffs of the existing progress-guard changes and run the current 0037 unit baseline before edits.
- [ ] Write DecisionGate truth-table tests for terminal, legal empty, mask error, structural singleton, singleton plus implicit STOP, explicit No/End, and multi-option requests.
- [ ] Implement ordered completion counts as `sum(P(n,k), k=min..max)` with overflow-safe capping for telemetry. Keep `canonical_completion_count=legal_completion_count` unless a registered action-family canonicalizer proves otherwise.
- [ ] Detect STOP from `minCount < maxCount`, decline from explicit `No`, pass from explicit `End`, and treat unknown cancellation semantics conservatively as strategic.
- [ ] Add shadow records keyed by `session_id/selection_index`; classify before compiler tensor construction but continue through the exact old compiler/model/action/trajectory path.
- [ ] Time compile, state+option Transformer, Value, GRU decode, and service roundtrip. Use CUDA events plus one batch synchronization on CUDA; attribute per-request time with batch size recorded instead of claiming isolated latency.
- [ ] Record focal callbacks, estimated strategic/forced counts, Phantom root selections and completed callback chains, trajectory/value inclusion, alias counts, and context/action-family histograms.
- [ ] Prove shadow mode returns byte-identical action lists and consumes the same policy RNG sequence as telemetry-disabled mode.
- [ ] Run a short fixed-seed shadow smoke only; write its report below `.tmp/evaluation/0037_action_boundary_shadow/<run-id>/`.

**Gate:** No shortcut code is enabled until shadow action parity is zero-diff and all current baseline tests pass.

### Task 2: Lossless Observation Consumption and Engine Event Trace

**Files:** Create `event_trace.py`; modify both online encoders and both worker paths; add `test_observe_only.py`.

- [ ] Add `WorkerLocalCompiler.observe_only(observation) -> CausalSnapshot` and `OnlineCausalEncoder.observe_only(observation) -> CausalSnapshot` that call `knowledge.consume` exactly once and do not compile/collate tensors.
- [ ] Refactor `compile`/`encode` to share a private consume function and reject duplicate observation/event serial consumption in tests.
- [ ] Record every official observation, selected primitive action, incremental logs, gate classification, reward/result, and event-span index in `EngineEventTrace`.
- [ ] Support optional atomic `.jsonl.gz` persistence to a unique session path; keep checkpoint and policy features free of raw traces.
- [ ] Compare legacy full inference against `observe_only` shortcut for knowledge snapshots, next compiled tensors, next logits, official action, and event serials.

**Gate:** The next strategic observation/history/features/logits must be identical under request-local GRU semantics.

### Task 3: Enable Structural Forced Shortcut

**Files:** Create/finish `macro_protocol.py`; modify workers, collector, protocol, batch, and PPO masking tests.

- [ ] Place `OfficialProtocolExecutor.consume` after observation/event consumption and before `WorkerLocalCompiler.compile`.
- [ ] For `FORCED`, return the sole completion, skip compiler/IPC/Transformer/GRU/Value, and write no `PolicyTransition`.
- [ ] For `LEGAL_EMPTY_PASS`, return `[]` only when the select contract explicitly permits it.
- [ ] For `MASK_ERROR`, terminate the Episode invalid with trace preserved; never call legacy fallback.
- [ ] Auto-advance consecutive forced callbacks until strategic, chance/info/priority, action commit, or terminal.
- [ ] Accumulate real reward/terminal into the previous pending policy transition. Record discount from actual turn-boundary changes, never callback count.
- [ ] Restrict policy loss, entropy, approximate KL, and clip fraction to `policy_valid` strategic transitions; Value samples are the same transition set.

**Gate:** Forced shortcut metrics show fewer inference requests while official selections and fixed-seed final states remain identical.

### Task 4: Phantom Allocation Canonicalizer and Visible Features

**Files:** Create `dragapult.py`, `test_dragapult_allocations.py`, and trace extractor tests.

- [ ] Identify root Phantom strictly by selected option `attackId == 154`; identify callbacks by zero-based `context == 14`, effect card `id == 121`, `min=max=1`, and `remainDamageCounter` descending 6 to 1.
- [ ] Extract stable target identities from actor-visible opponent Bench `serial/card id/slot`; never use hidden deck/hand/prize identities.
- [ ] Enumerate weak compositions in deterministic lexicographic counter order and assert counts `1, 7, 28, 84, 210` for `n=1..5`.
- [ ] Build per-target allocation scalars: counters, nominal/effective post-HP, immediate-KO bit, visible prize value, damage-to-KO/headroom, card identity, Bench slot, current/max HP, energy counts, status bits, public evolution headroom/risk, and damage-counter prevention.
- [ ] Build global structure scalars: targets hit, maximum count, concentration `sum(x_i^2)/36`, spread, KO count, and immediate visible prizes.
- [ ] Obtain prize/evolution/prevention facts only from current visible cards/stadium plus audited public prototypes; do not add official observation fields.
- [ ] Report same-outcome aliases as `n^6 - C(n+5,6)` for Phantom shadow telemetry, while retaining all 210 semantic allocations even when visible prevention makes outcomes coincide.

**Gate:** Pure exhaustive tests cover all 330 allocations and 20,515 ordered aliases; official parity is a separate Task 9 gate.

### Task 5: Hierarchical Root/Allocation Policy

**Files:** Create `allocation_head.py`; modify `actor_critic.py` and `action_distribution.py`; add allocation/head tests.

- [ ] Preserve the existing root decoder and its logits exactly.
- [ ] Split `SemanticActorCritic.encode` timing without changing tensors: actor state/options first, Value from the same encoded memory second.
- [ ] After root sampling, invoke allocation scoring only for rows whose selected root option has `attack_id=154` and at least two allocations. `n=0` has no allocation choice; `n=1` is deterministic with allocation log-probability/entropy zero.
- [ ] Gather opponent-Bench `state.cards` tokens through existing `card_cat` relative-owner/zone/slot fields; do not rerun State Encoder.
- [ ] Condition each target representation on state summary, selected Phantom root option token, target card token, counter embedding, and visible scalar features.
- [ ] Aggregate targets with masked sum/mean/max DeepSets pooling, concatenate global structure features, and emit one logit per canonical allocation.
- [ ] Sample/greedy-select from the conditional `Categorical`; return root, allocation, separate log-probabilities/entropies, and their joint sum.
- [ ] During PPO evaluation, recompute the selected root and allocation log-probabilities from the same stored pre-action features/candidate definition. Assert pre-update joint old-logprob MAE within the existing tolerance.
- [ ] Use sampled hierarchical entropy `H(root) + I[root=Phantom] H(allocation|root)`; document that V6 entropy/KL/clip denominators differ from V5.

**Gate:** Adding 210 allocation candidates cannot change root logits/probabilities, and permuting target rows with matching counters cannot change allocation logits after inverse permutation.

### Task 6: Pending Phantom Macro Transaction

**Files:** Finish `macro_protocol.py`; modify both workers, collector, protocol, and deployment/export runtime.

- [ ] Return an internal IPC response containing the unchanged root `action: list[int]` plus a side-band macro plan. Pass only `action` to official `Select`.
- [ ] Cache one pending transaction inside each `_run_job`/battle session; key diagnostics and trace by `game_id`, and use a deployment-local battle generation when no external session ID exists.
- [ ] On each of six callbacks, call `observe_only`, validate actor/context/effect serial/remain count/target set, relocate the planned target by stable serial, and return `[current_option_index]`.
- [ ] Close the transaction after the sixth official callback; keep all six callbacks in EngineEventTrace and create no model request or policy sample for them.
- [ ] Clear transactions on terminal, worker exception, game finish/reset, selection-index expiry, or monotonic timeout.
- [ ] On drift, mark the Episode and macro transition PPO-invalid, preserve trace, clear the transaction, and optionally complete the game through the old per-callback model path. Do not salvage that Episode into the normal V6 PPO batch.

**Gate:** One Phantom root causes one focal model/Value request and the official runtime still receives the root action plus six primitive target calls.

### Task 7: PolicyTrajectory V2, GAE, and PPO

**Files:** Modify `protocol.py`, `collector.py`, `batch_full_semantic.py`, and `ppo_full_semantic.py`; add fixed-trace tests.

- [ ] Create a pending transition at the focal strategic pre-action boundary with one Value and joint old-logprob.
- [ ] Span forced/macro callbacks without adding samples. Seal the macro commit event index after the sixth target action.
- [ ] Finalize `next_value` at the next focal-visible true policy boundary using that boundary's already-computed Value; never encode an opponent-private observation for the focal critic.
- [ ] If terminal occurs first, set `done=True`, `next_value=0`, and attach terminal reward/winner to the pending transition.
- [ ] Store explicit `accumulated_discount`; for the current turn clock this changes only across actual engine turns, not internal callbacks.
- [ ] Compute `delta = accumulated_reward + accumulated_discount * next_value - pre_action_value`; apply lambda only across true policy transitions/turn boundaries.
- [ ] Batch primitive and Phantom macro actions together with a macro mask and padded allocation data. Exclude invalid/fallback Episodes.
- [ ] Freeze joint old-logprob throughout PPO epochs and compute ratio/KL/clip from the joint new logprob.

**Gate:** Fixed traces prove one Phantom transition, correct terminal accumulation, strategic-only loss metrics, and compressed GAE equality to a hand-calculated reference.

### Task 8: Trace Dataset, Coverage Audit, and Allocation BC Warm Start

**Files:** Create extractor and `allocation_bc.py`; add manifests/tests. Do not start PPO.

- [ ] Extract only chains with root `attackId=154`, effect card 121, contexts/remain values 6..1, exact six legal primitive actions, stable serial mapping, and a reconstructable actor-visible pre-action history.
- [ ] Emit provenance, Episode identity, source/team, exact deck hash, pre-action record, canonical targets/counters, and split group; source identity is never actor-visible.
- [ ] Deduplicate by Episode/root-step/actor and split by whole Episode.
- [ ] Fail closed on source/provenance that is not approved for BC.
- [ ] If approved samples are insufficient, collect a bounded V5 shadow trace corpus first; do not use random allocation-head PPO exploration as a substitute.
- [ ] Warm-start only `allocation_head` with cross-entropy, keep root/encoder/Value frozen, and save a model-only V6 checkpoint with dataset hash and coverage metrics.
- [ ] Report allocation entropy, top-1/top-2 margin, exact accuracy, target-count strata, and allocation-shape coverage. Treat 78 mechanically extractable examples as evidence, not automatically eligible training data.

**Gate:** Formal PPO remains blocked until allocation BC is non-random under a frozen validation split and provenance is accepted.

### Task 9: Official-Engine Exhaustive Parity and Hidden-Information Tests

**Files:** Create `test_dragapult_official_parity.py` and extend pool/runtime parity tests.

- [ ] Build/reuse the official engine only into `engine/build/` and never patch source.
- [ ] For each `n=1..5`, execute every canonical allocation from the same deterministic root-state fixture and compare the expanded primitive sequence with the legacy path.
- [ ] Execute all ordered aliases in the slow parity suite, or use official search-state cloning only if it is proven ABI-equivalent; compare normalized final authoritative state hashes.
- [ ] Compare official primitive return format/count, final state, actor/turn/phase/priority, legal successors, reward, terminal, winner, both player observations, RNG/search state, deck order, and future continuation under a fixed suffix.
- [ ] Cover KO, visible prize values, prevention, multiple simultaneous KO thresholds, target serial drift, empty mask, actor/context drift, and terminal during the chain.
- [ ] Assert PolicyTransition contains only focal-visible tensors and trace references, never the opponent's private hand/deck/prize identities.

**Gate:** Zero parity failures and zero hidden-information violations are required before the new action contract can be enabled by default.

### Task 10: V6 Checkpoint, Documentation, and Short Smoke

**Files:** Modify checkpoint/run/export files and all 0037 authoritative docs.

- [ ] Reserve `V6_action_boundary_phantom_macro` only after confirming its artifact/checkpoint/tensorboard/wandb/evaluation paths are unused.
- [ ] Add checkpoint metadata keys exactly: `action_schema_version`, `decision_gate_version`, `canonicalizer_version`, `trajectory_schema_version`, and `official_protocol_adapter_version`.
- [ ] Implement explicit V5 migration that loads State Encoder, Option Encoder, root decoder, Value trunk, and current LoRA, initializes allocation head, and rejects optimizer/scheduler/rollout state.
- [ ] Save all V6 model-only checkpoints atomically and reject forbidden recovery-state keys.
- [ ] Synchronize DESIGN.md/HTML shapes, data flow, heads, losses, stages, and checkpoint boundaries; record the action-contract decision in DECISIONS.md.
- [ ] Run only unit/integration tests and a bounded fixed-seed smoke/benchmark. Do not run 50/200 PPO updates.
- [ ] Produce `.tmp/evaluation/0037_action_boundary_v1/<run-id>/report.html` with forward count, official primitive count, focal inference/transition counts, forced shortcuts, wall times, CUDA allocated/reserved, process-tree RSS, allocation entropy/margins, fallback/invalid counts, parity failures, and BC trace coverage.

**Final acceptance:** Phantom Dive uses one focal State Encoder/Value/root+allocation decision, six official target callbacks remain unchanged, all policy/trajectory/checkpoint contracts are versioned, and no long training has started.

---

## Current Read-Only Trace Coverage

- Formal 0037 V5 rollout EngineEventTrace persisted on disk: **0**. The collector retained only in-memory tensorized `TrajectoryDecision` records, so completed V5 runs cannot reconstruct macro labels.
- Repository scan found **29 unique official Episode replay files** containing Phantom Dive chains: 7 tracked under `data/replays/0016_alakazam_multideck_bc/...` and 22 disposable representative replays under `.tmp/environment_daily/...`.
- Mechanically complete/reconstructable root-to-allocation chains: **78/78**. Every chain has root `attackId=154`, exact `remainDamageCounter=6..1`, stable Bench serials, six actions, and exact 60-card decks for both players.
- Target-count coverage: `n=1:5`, `n=2:3`, `n=3:14`, `n=4:18`, `n=5:38`.
- Allocation-shape coverage: `(6):42`, `(5,1):15`, `(4,2):10`, `(4,1,1):7`, `(3,3):1`, `(3,2,1):1`, `(2,2,1,1):1`, `(2,1,1,1,1):1`.
- This corpus is highly concentrated and is not automatically BC-eligible. The 7 tracked replays belong to another project's audited source boundary, and the 22 `.tmp` files are not durable training assets. Source approval, deduplication, immutable manifesting, and Episode-level splitting are required; otherwise V1 must first collect a bounded shadow corpus.

## Planned Verification Commands

```bash
python3 -m unittest -v \
  train.0037_dragapult_value_initialized_rl.tests.test_decision_gate \
  train.0037_dragapult_value_initialized_rl.tests.test_shadow_telemetry \
  train.0037_dragapult_value_initialized_rl.tests.test_observe_only

python3 -m unittest -v \
  train.0037_dragapult_value_initialized_rl.tests.test_dragapult_allocations \
  train.0037_dragapult_value_initialized_rl.tests.test_allocation_head \
  train.0037_dragapult_value_initialized_rl.tests.test_macro_protocol

python3 -m unittest -v \
  train.0037_dragapult_value_initialized_rl.tests.test_policy_trajectory_v2 \
  train.0037_dragapult_value_initialized_rl.tests.test_action_boundary_checkpoint \
  train.0037_dragapult_value_initialized_rl.tests.test_dragapult_trace_extractor

python3 -m unittest -v \
  train.0037_dragapult_value_initialized_rl.tests.test_dragapult_official_parity
```

The final smoke command will be added only as an explicit bounded `--episodes`/`--max-updates 0` interface; the implementation must reject accidental long-run defaults in smoke mode.
