# 0045 PinHaoLong V3 Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public-observation-only eight-checkpoint PinHaoLong V3 router that reconstructs and routes each V14/V19 checkpoint's complete stored effective Actor.

**Architecture:** Preserve the V2 monotonic public-card `if/elif` memory contract, replace its route table with the eight Eval-512 winners, and make unknown/conflict default selection an injected immutable configuration chosen after Benchmark V2. Materialize each checkpoint from the immutable Policy-0814 base under FP16-storage/FP32-runtime, group CUDA rows by route, and execute each route's complete effective Actor; share tensor storage only after exact effective-content equality is proven.

**Tech Stack:** Python 3, PyTorch parametrizations, project semantic0031 CUDA resident router, pytest, project Kaggle compound runtime.

## Global Constraints

- Do not modify `engine/source/` or CUDA engine source.
- Router inputs are restricted to public/certain opponent card identities already present in actor-visible semantic observations.
- Exact opponent deck ID, hidden cards, candidate identity, Critic outputs, and Value Meta labels are forbidden router inputs.
- Route table is `001→U165`, `002→U170`, `003→U25`, `007→U125`, `008→U5`, `009→U140`, `011→U160`, `071→U145`.
- Ogerpon-only evidence is provisional U5 and later Hydrapple/Meganium-family evidence overrides it to U145.
- Unknown/conflict default is not finalized until all eight deck-067 Benchmark V2 CUDA-2048 reports pass.
- V14 checkpoints reconstruct only their actually stored effective Option LoRA/decoder/allocation behavior; missing historical StateEncoder LoRA remains zero-delta.
- V19 checkpoints must include StateEncoder attention LoRA, StateEncoder FFN LoRA, OptionEncoder attention/FFN LoRA, final LayerNorm, decoder and allocation head.
- Cross-route sharing requires exact effective tensor equality; differing tensors remain independently routed.

---

### Task 1: V3 Public Routing Rules

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_deck_router_v3_rules.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_deck_router_v3.py`

**Interfaces:**
- Produces: `PublicDeckMemory(job_count, device, default_update)`, `DECK_TO_UPDATE`, `ROUTED_UPDATES`, and `route_manifest(default_update)`.

- [x] Write failing tests for all eight public family routes, Ogerpon→071 override, hidden/unknown default, conflict default, and forbidden input manifest.
- [x] Implement tensorized monotonic public memory with injected default constrained to one of the eight routed updates.
- [x] Run focused rules tests on CPU.

### Task 2: Complete Effective Actor Router

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_deck_router_v3.py`
- Modify test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_deck_router_v3.py`

**Interfaces:**
- Consumes: eight source checkpoints, Policy-0814 actor/value base, exact focal deck, and immutable default update.
- Produces: `materialize_public_deck_router_v3(...) -> PublicDeckRouterV3Runtime` with `decode_compacted(...)`, `allocation_head_for_job(...)`, and a complete identity audit.

- [ ] Test that V19-only State FFN and final norm differences are classified as routed rather than shared.
- [ ] Test that only byte/effective-equal tensors may alias storage across route models.
- [ ] Materialize all candidates on CPU, verify FP16→FP32 audits, deduplicate only equal effective tensors, then move the routed runtime to CUDA.
- [ ] Group decision rows by selected update, execute the complete effective Actor per group, scatter decoded outputs, and hard-fail uncovered rows.
- [ ] Verify per-head logits/greedy actions match independently loaded static candidates for fixed synthetic batches.

### Task 3: CUDA Evaluation and Kaggle Runtime

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_public_deck_router_v3_policy0814.py`
- Create: `train/0045_single_deck_expert_minimal_lora/semantic_runtime/deployment/public_deck_memory_v3.py`
- Create: `train/0045_single_deck_expert_minimal_lora/semantic_runtime/deployment/public_deck_router_v3.py`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`

**Interfaces:**
- Consumes: final default update chosen from V20 Benchmark V2 and the complete V3 router runtime.
- Produces: public-router frozen evidence, package-runtime parity evidence, and synchronized design documentation.

- [ ] Freeze the default update in a versioned route manifest after all V20 static reports pass.
- [ ] Run a CUDA smoke with complete Policy-0814 opponent and verify zero focal/opponent storage aliases.
- [ ] Compare router-selected actions with independent static candidates at each route.
- [ ] Port identical rules and complete routed deltas into the packaged CPU runtime.
- [ ] Update DESIGN.md/HTML with complete routed components, identity hashes, default selection, evidence boundary and current phase.
