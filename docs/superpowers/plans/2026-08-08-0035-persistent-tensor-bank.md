# 0035 Persistent Tensor Bank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve unchanged actor features as session-owned Tensor storage and patch only changed fields, eliminating repeated Python-to-Tensor conversion while retaining the exact 39-key actor contract.

**Architecture:** Each battle owns a `PersistentTensorBank`. The canonical compiler remains the semantic authority, but the bank keeps capacity-backed CPU tensors for every fixed and ragged field, reuses storage when values are unchanged, and patches changed tensors in place. Returned tensors expose only the exact current logical lengths, so dtype, shape, mask, relation and STOP-target semantics match `collate_canonical_records([record])` exactly.

**Tech Stack:** Python 3.11, PyTorch CPU tensors, frozen 0035 canonical records, `unittest`, paired chronological benchmarks.

## Global Constraints

- Do not modify `engine/source/`, `GetBattleData`, the JSON observation format, actor schema, model, logits, decoder or action contract.
- Keep all runtime code self-contained under `train/0035_lifetime_aware_feature_compiler/`.
- One bank belongs to one actor-local battle session; no storage or previous values cross sessions.
- Never return padded garbage as valid input: logical shape and mask must match the existing unbucketed collator exactly.
- Unsupported/malformed canonical records fail closed with the same validation boundaries as the current collator.
- Do not enable the persistent path by default until 560-decision tensor parity and paired median/p95 gates pass.

---

### Task 1: Capacity-backed tensor slots

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/features/tensor_bank.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_tensor_bank.py`

**Interfaces:**
- Produces `PersistentTensorBank.collate(record) -> dict[str, Tensor]`, `reset()`, and `stats.snapshot()`.

- [x] **Step 1: Write failing storage tests**

Create two exact canonical records. Assert first output equals the reference collator; unchanged second output uses the same underlying storage pointers; a global-only change patches only the global tensor; a card numeric change preserves card categorical/state storage and values; length growth reallocates only the affected family; reset prevents cross-session reuse.

- [x] **Step 2: Verify the new module is absent**

Run `python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_tensor_bank` and expect an import failure.

- [x] **Step 3: Implement slots and exact validation**

Use `_FixedSlot` for global/min/max and `_RaggedSlot` for card/resource/event/option/skill/effect fields. Ragged storage grows geometrically, returns `storage[..., :logical_length]`, and maintains a bool mask with one false position for empty families. Store the previous canonical Python sequence reference; equality means no Tensor write. On change, copy a correctly typed Tensor into the existing capacity. Count allocations, slot hits/misses and written elements.

- [x] **Step 4: Run focused tests**

Expected: exact key/dtype/shape/value parity and storage-pointer assertions pass.

### Task 2: Session runtime integration

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/deployment/online_runtime.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_deployment.py`

**Interfaces:**
- `OnlineCausalEncoder(..., persistent_tensors: bool = False)` owns one bank.
- `encode()` uses the bank only when enabled; `encode_record()` remains unchanged.

- [x] **Step 1: Add failing isolation/parity tests**

Assert persistent and reference `encode()` dictionaries are exactly equal for chronological observations, two encoders have independent storage, and initialization does not share battle tensors even though process prototypes remain shared.

- [x] **Step 2: Integrate behind an explicit flag**

Construct the bank per encoder and route `encode_record()` output through it. Preserve the current stateless default and export behavior.

- [x] **Step 3: Run deployment and export tests**

Expected: checkpoint inference, self-contained export and full/persistent parity pass.

### Task 3: 560-decision semantic gate and paired benchmark

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/benchmark_incremental_features.py`
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_benchmark_protocol.py`

**Interfaces:**
- Compares `collate_canonical_records` with one bank per chronological trajectory.
- Reports compiler, collation, total, allocation/hit/write counters and seven paired repetitions.

- [x] **Step 1: Add benchmark protocol tests**

Require alternating arm order, warmup, median/MAD/p95, exact pre-timing parity and bank counters.

- [x] **Step 2: Run record/tensor parity**

Use the 35-decision golden plus 525-decision extended fixtures. Require all 39 keys, dtype, shape and values to match at every decision.

- [x] **Step 3: Run paired benchmark**

Run seven repetitions of 10,000 chronological decisions with 1,000 warmup decisions. Compare full compiler + normal collator against the best incremental compiler + persistent bank, and separately report collation-only savings.

### Task 4: Admission, documentation and verification

**Files:**
- Modify: `experiments/0035_lifetime_aware_feature_compiler/{DESIGN.md,DESIGN.html,DECISIONS.md,manifest.json}`

- [x] **Step 1: Record exact results and decision**

Document that Tensor persistence begins after canonical semantics, does not remove Engine JSON parsing, and directly eliminates unchanged Tensor conversion/allocation. Record any aliasing/lifetime boundary and whether default enablement is admitted.

- [x] **Step 2: Run all gates**

Run the full 0035 unittest suite, compileall, manifest JSON parsing, diff check, and verify no `engine/source/` path changed.

## Self-review

- Spec coverage: static Tensor storage, dynamic patching, session isolation, exact 39-key semantics, benchmark and admission are explicit.
- Placeholder scan: every task names concrete files, APIs, assertions and commands.
- Type consistency: the only production API is `PersistentTensorBank.collate(record)` and `OnlineCausalEncoder(..., persistent_tensors=...)`.
