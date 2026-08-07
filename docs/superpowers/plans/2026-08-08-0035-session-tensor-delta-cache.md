# 0035 Session Tensor Delta Cache Implementation Plan

> **For agentic workers:** Execute this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep each battle session's unchanged canonical tensors resident on the inference GPU and transfer/materialize only tensor rows changed by the current official-engine observation.

**Architecture:** Preserve the official Engine JSON and the existing 39-key `DecisionBatch` model contract. A session-owned CPU `PersistentTensorBank` becomes the change authority and emits exact row patches; a GPU session store applies those patches, then assembles the current multi-session batch from resident tensors using device-to-device copies. Model-static prototype embeddings remain checkpoint/device/dtype scoped, exact-deck constants naturally become first-decision-only rows, turn changes are invalidation hints rather than blanket resets, and action-dynamic rows are the only steady-state H2D payload.

**Tech Stack:** Python 3.11, PyTorch CPU/CUDA tensors, CUDA events, official-engine resident inference, unittest, 0035 audited 35+525 chronological fixtures.

## Global Constraints

- Never modify `engine/source/`, official observation JSON, ctypes ABI, the 39-key tensor values/shapes seen by the model, checkpoint weights, logits, decoder, or action contract.
- The stateless collator remains the semantic authority. Every emitted/reconstructed tensor must match it in key, dtype, shape, and value before action parity is considered.
- Cache ownership is `checkpoint × device × dtype` for prototype memory, `exact deck × actor × session` for canonical tensor state, and one chronological update for decision-dynamic patches.
- Session ID, exact deck, actor, schema, non-chronological input, capacity overflow, malformed patch, or relation inconsistency fails closed and reseeds from a full canonical record.
- `supporterPlayed`, `stadiumPlayed`, `energyAttached`, `retreated`, status, `appearThisTurn`, options, counters, event age, and relation indices remain action-mutable even inside one turn.
- No whole-observation JSON/SHA fingerprint and no Python recursive freeze may be introduced on the hot path.
- The existing evaluation/0031 working-tree changes are preserved and excluded from 0035 commits unless a deliberately isolated integration hunk is required.
- Commits use a Conventional Commit prefix followed by a Chinese message.

---

### Task 1: Exact CPU tensor patch contract

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/features/tensor_bank.py`
- Test: `train/0035_lifetime_aware_feature_compiler/tests/test_tensor_delta.py`

**Interfaces:**
- Produces `TensorPatch(name: str, logical_shape: tuple[int, ...], rows: Tensor, values: Tensor, replace: bool)`.
- Produces `TensorDelta(tensors: Mapping[str, Tensor], patches: tuple[TensorPatch, ...], full_reseed: bool)`.
- Produces `PersistentTensorBank.collate_delta(record) -> TensorDelta`; `collate(record)` remains unchanged and delegates to the new method.

- [x] **Step 1: Add red tests for first-decision and unchanged-decision behavior**

  Compile an audited record, require the first delta to mark every key as `replace`, then collate an exact copy and require zero patch values while `delta.tensors` still exactly equals `collate_canonical_records([record])`.

- [x] **Step 2: Add red tests for sparse row changes, growth, shrink, and empty ragged fields**

  Mutate one global scalar, append one event, shrink an option family, and transition option skills from nonempty to empty. Apply patches to a reference tensor map and require exact equality with the stateless collator after every transition.

- [x] **Step 3: Implement patch metadata at the point where slots already discover changes**

  `_FixedSlot.update`, `_RaggedSlot.update`, and `_MaskSlot.update` return their tensor plus changed logical row indices and whether storage replacement is required. `collate_delta` records patches without rescanning tensor values. Empty ragged transitions emit an explicit zero placeholder patch.

- [x] **Step 4: Run chronological tensor-delta parity**

  Run all 560 audited decisions and require exact reconstruction for all 39 keys, zero unreported changes, zero unexpected reseeds, and nonzero avoided bytes for resource/event/card families.

### Task 2: GPU-resident per-session tensor store

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/deployment/session_tensor_cache.py`
- Test: `train/0035_lifetime_aware_feature_compiler/tests/test_session_tensor_cache.py`

**Interfaces:**
- Produces `SessionTensorKey(session_id: str, actor: int, deck_sha256: str)`.
- Produces `GpuSessionTensorStore(device, floating_dtype)`.
- Produces `apply(key, delta: TensorDelta) -> None`, `batch(keys: Sequence[SessionTensorKey]) -> DecisionBatch`, `close_session(key) -> None`, `clear(reason) -> None`, and `stats() -> dict[str, int | float]`.

- [x] **Step 1: Add CPU-device reconstruction tests**

  Use `device='cpu'` as a deterministic test backend. Interleave at least four sessions from different audited trajectories, apply chronological deltas, batch arbitrary session orders, and compare every output tensor to `collate_canonical_records` on the same records.

- [x] **Step 2: Add lifecycle and fail-closed tests**

  Reject an update before full reseed, key/actor/deck mismatch, malformed row indices, wrong dtype/width, and missing 39-key state. Verify `close_session` removes one session and `clear` removes all sessions.

- [x] **Step 3: Implement resident capacity and device-side batch assembly**

  Store one unbatched logical tensor per key/session on the target device. Apply `replace` patches with one transfer and sparse patches with `index_copy_`; preserve logical lengths separately. For `batch(keys)`, compute each field's current maximum logical length, allocate the canonical batched shape on the target device, copy each resident view device-to-device, and preserve zero padding/masks/targets exactly.

- [x] **Step 4: Add CUDA transfer accounting**

  On CUDA, synchronize only benchmark/profile events. Record `initial_h2d_bytes`, `delta_h2d_bytes`, `resident_bytes`, `d2d_batch_bytes`, patch counts, reseeds, hits, and session count. A same-record update must transfer zero bytes.

### Task 3: 0035 runtime capability without changing the actor contract

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/online_runtime.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/inference.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/export_candidate.py`
- Test: `train/0035_lifetime_aware_feature_compiler/tests/test_deployment.py`

**Interfaces:**
- Produces `OnlineCausalEncoder.encode_delta(observation) -> TensorDelta` when `persistent_tensors=True`.
- Produces `PortableSemanticPolicy.new_session_tensor_bank(actor, deck) -> OnlineCausalEncoder`.
- Existing `encode`, `encode_record`, `select`, `agent(observation)`, and exported package behavior remain backward compatible.

- [ ] **Step 1: Add portable capability and export tests**

  Require an exported package to include `features/tensor_bank.py` and `deployment/session_tensor_cache.py`, expose the delta capability, retain the same checkpoint keys, and return the same action through the ordinary single-session API.

- [ ] **Step 2: Implement the opt-in delta encoder**

  Reuse the same chronological `CausalKnowledge` and canonical compiler. `encode_delta` compiles one record and immediately calls the session `PersistentTensorBank.collate_delta`; it never changes observation parsing or canonical semantics.

- [ ] **Step 3: Run record/tensor/logit/action parity**

  On the real checkpoint, compare stateless and delta-reconstructed inputs, logits, legal lengths, and deterministic actions. Require zero mismatch across the audited 560 decisions.

### Task 4: Resident inference integration and profiling

**Files:**
- Modify: `evaluation/runner/inference_server.py` only through isolated capability-detection paths
- Modify: `tests/test_evaluation_inference_server.py` only for the new capability
- Create: `train/0035_lifetime_aware_feature_compiler/benchmark_session_tensor_cache.py`
- Test: `train/0035_lifetime_aware_feature_compiler/tests/test_session_tensor_benchmark.py`

**Interfaces:**
- `PolicyServer` detects `OnlineCausalEncoder.encode_delta` and `strategy.deployment.session_tensor_cache.GpuSessionTensorStore` only when a new opt-in runtime flag is enabled.
- The profiler records cache/H2D/D2D counters alongside existing compile, collate, H2D, model, ingress, and handoff stages.

- [ ] **Step 1: Add an equal-order server parity test**

  Feed interleaved fake sessions through ordinary raw-record collation and resident-delta assembly, then require identical model input tensors, deterministic action tensors, request ordering, close-session behavior, and errors.

- [ ] **Step 2: Integrate the opt-in resident-delta path**

  Encode each active session to `TensorDelta`, apply it to the GPU store, and call the unchanged model with `store.batch(active_keys)`. Keep all legacy, source-conditioned, worker-compiled-record, and unsupported packages on the existing path.

- [ ] **Step 3: Add a paired component benchmark**

  Alternate ordinary `collate + H2D` and `delta apply + resident batch` over chronological 64-way batches. Report CPU materialization, H2D bytes/time, D2D assembly, total preparation median/p95, resident memory, and exact action commitment.

- [ ] **Step 4: Run strict official-engine profiles**

  Export cached and control packages from the same 0035 source/checkpoint. Run three repetitions per arm with 128 games, N16E16, batch 64, wait 2 ms, FP16, seed 8062026. Require every game complete, zero errors, equal schedule/deck/checkpoint identities, and report three-run medians.

### Task 5: Admission, documentation, and scoped commit

**Files:**
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.html`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DECISIONS.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/manifest.json`

**Interfaces:**
- Produces a V5 result that distinguishes initial full seed, deck/battle-static reuse, action-delta transfer, D2D batch assembly, and model time.

- [ ] **Step 1: Apply admission gates**

  Admit only with 560/560 tensor parity, zero checkpoint logit/action mismatch, 768/768 official games, zero errors, lower median H2D bytes and CPU collate time, no preparation p95 regression above 5%, and at least 5% end-to-end throughput gain over the adjacent strict control. Otherwise preserve it as an opt-in negative result with exact bottleneck evidence.

- [ ] **Step 2: Synchronize authoritative documentation**

  Record the lifecycle table, delta protocol, memory ownership, fail-closed conditions, exact artifacts, component and official metrics, training/RL invalidation semantics, and the next remaining bottleneck in both DESIGN formats, DECISIONS, and manifest.

- [ ] **Step 3: Run all gates and commit only scoped changes**

  Run the complete 0035 suite, focused evaluation tests, compileall, JSON parse, `git diff --check`, and assert `git diff --name-only -- engine/source/` is empty. Stage only V5 files and isolated integration hunks, then commit with `perf: 增量传输并常驻对局特征张量` and push `dev/cyd_main`.

## Self-review

- Spec coverage: model-static cache remains intact; deck/battle-static and decision-dynamic data now have explicit CPU, H2D, GPU, batching, reset, RL, and official-engine contracts.
- Placeholder scan: every task names exact files, APIs, failure cases, parity scope, benchmark contract, and admission thresholds; no deferred implementation placeholder remains.
- Type consistency: `PersistentTensorBank.collate_delta` produces `TensorDelta`; `GpuSessionTensorStore.apply` consumes the same type; the resident server passes its reconstructed ordinary `DecisionBatch` to the unchanged policy.
