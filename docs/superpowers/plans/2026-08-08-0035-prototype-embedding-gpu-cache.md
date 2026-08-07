# 0035 Prototype Embedding GPU Cache Implementation Plan

Steps use checkbox syntax to retain the implementation record.

**Goal:** Compute the frozen card/attack/skill/effect prototype embeddings once per model device/dtype/weight state and reuse them from GPU memory on every inference forward without changing logits or the 39-key actor contract.

**Architecture:** `SemanticPolicy` owns a derived, non-checkpoint `PrototypeEmbeddings` cache. Eval mode lazily builds it on the model's current device and dtype; `_apply`, `load_state_dict`, trainable prototype parameters, and training-mode transitions invalidate or bypass it. A benchmark compares identical forwards with the cache disabled and enabled, then an exported 0035 candidate is profiled under the same official-engine resident-inference contract.

**Tech Stack:** Python 3.11, PyTorch inference mode, CUDA RTX 5080, unittest, official-engine evaluation profiler.

## Global Constraints

- Do not modify `engine/source/`, the official observation JSON, the 39-key actor schema, model weights, logits, decoder, or action contract.
- Cache tensors are derived runtime state, never `nn.Parameter`, checkpoint state, or a persistent buffer.
- Training with any trainable prototype-encoder parameter must execute a fresh autograd-connected `encode_all()` on every forward.
- Training with every prototype-encoder parameter frozen may reuse the detached cache so later Transformer/LoRA parameters still receive gradients.
- Any device/dtype change or state-dict load invalidates the cache; the next eligible forward rebuilds it exactly once.
- Formal throughput claims require exact tensor/logit/action parity and equal-contract paired measurements.

---

### Task 1: Model-owned prototype cache lifecycle

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/model/policy.py`
- Test: `train/0035_lifetime_aware_feature_compiler/tests/test_model.py`

**Interfaces:**
- Produces: `SemanticPolicy.prototype_memory() -> PrototypeEmbeddings`
- Produces: `SemanticPolicy.prepare_prototype_cache() -> PrototypeEmbeddings`
- Produces: `SemanticPolicy.clear_prototype_cache(reason: str) -> None`
- Produces: `SemanticPolicy.prototype_cache_stats() -> dict[str, int]`

- [x] **Step 1: Write failing inference-cache tests**

Add tests that wrap `prototype_encoder.encode_all`, execute two eval forwards, and require one encode call, bit-exact logits, four cached GPU/CPU tensors on the model device, and cache hit/build counters.

- [x] **Step 2: Write failing invalidation and training tests**

Require `.to(dtype=...)` and `load_state_dict` to rebuild. Require ordinary train mode to call `encode_all` every forward with gradients. Freeze all prototype-encoder parameters, enable train mode, and require one detached cache build while a downstream Transformer parameter still receives a gradient.

- [x] **Step 3: Implement the minimal lifecycle**

Store the dataclass in a plain private attribute. `prototype_memory` returns live `encode_all()` when training and any prototype parameter requires gradients; otherwise it lazily calls `prepare_prototype_cache` under `torch.no_grad()`. `no_grad` is required here because frozen-prototype training must still be able to save the cached tensors for downstream trainable layers. Override `_apply` and `load_state_dict` only to clear derived state before mutation. Replace all top-level direct `encode_all()` calls in `SemanticPolicy` with `prototype_memory()`.

- [x] **Step 4: Run focused model tests**

Run `python3 -m unittest -v train.0035_lifetime_aware_feature_compiler.tests.test_model` and require all tests to pass, including CUDA compile parity.

### Task 2: Deployment and checkpoint semantic gates

**Files:**
- Modify: `train/0035_lifetime_aware_feature_compiler/tests/test_deployment.py`
- Verify: `train/0035_lifetime_aware_feature_compiler/tests/test_semantic_lineage.py`

**Interfaces:**
- Consumes: the automatic eval-mode cache from Task 1.
- Produces: checkpoint action parity and cache lifecycle evidence through the existing portable policy API.

- [x] **Step 1: Add portable-policy cache assertions**

Load the model-only checkpoint, run the same chronological observation twice through independent policies, and require identical legal action plus exactly one prototype cache build per loaded model.

- [x] **Step 2: Re-run canonical semantic gates**

Run deployment, semantic lineage, model, and export tests. Require no new checkpoint keys and verify candidate export physically includes the updated self-contained model implementation.

### Task 3: Paired CUDA and official-engine benchmark

**Files:**
- Create: `train/0035_lifetime_aware_feature_compiler/benchmark_prototype_cache.py`
- Create: `train/0035_lifetime_aware_feature_compiler/tests/test_prototype_cache_benchmark.py`

**Interfaces:**
- Produces: paired JSON with uncached/cached GPU forward median, p95, batches/s, cache bytes, build count, and exact output commitments.

- [x] **Step 1: Implement an alternating paired CUDA benchmark**

Load the real 0031 checkpoint and audited chronological batches, warm up both arms, alternate leading order over seven repetitions, synchronize CUDA around timed regions, and clear the cache before every uncached forward. Compare deterministic action tensors exactly before timing.

- [x] **Step 2: Export isolated cached and uncached 0035 candidates**

Use the existing 0035 exporter with the frozen Policy-0806 model-only checkpoint and exact deck. Write the cached package under `.tmp/0035_lifetime_aware_feature_compiler/prototype_cache_candidate` and an otherwise identical uncached package under `prototype_uncached_candidate`; do not alter the existing arena candidate.

- [x] **Step 3: Run equal-contract official-engine profiles**

Profile the uncached source candidate and cached 0035 package with identical games, workers, engine pool, batch, wait, dtype, seed, and GPU device. Require every game finished with zero errors; compare selections/s and GPU model time per batch/selection. Treat results as provisional unless each arm has at least three paired repetitions.

### Task 4: Documentation, full verification, and commit

**Files:**
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DESIGN.html`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/DECISIONS.md`
- Modify: `experiments/0035_lifetime_aware_feature_compiler/manifest.json`

**Interfaces:**
- Consumes: exact artifacts and measurements from Tasks 1-3.

- [x] **Step 1: Record cache semantics and measurements**

Document the model-static/deck-static distinction, cache size, inference and frozen-prototype RL lifecycle, exact artifact paths, paired medians/p95, official-engine result, and whether default enablement is admitted.

- [x] **Step 2: Run all gates**

Run the complete 0035 unittest suite, compileall, JSON parsing, `git diff --check`, and assert `git diff --name-only -- engine/source/` is empty.

- [x] **Step 3: Commit and push only scoped files**

Stage only this plan, 0035 implementation/tests/benchmark, and 0035 experiment documents. Commit with `perf: cache prototype embeddings for inference` and push `dev/cyd_main` without staging the pre-existing evaluation/0031 worktree changes.

## Self-review

- Spec coverage: GPU-resident model-static cache, RL frozen-parameter safety, exact parity, component throughput, official-engine throughput, docs, and isolated commit are all assigned.
- Placeholder scan: every task names exact files, APIs, commands, and pass conditions; no implementation placeholder remains.
- Type consistency: every consumer uses `PrototypeEmbeddings`; cache lifecycle methods are defined once in Task 1 and reused unchanged.
