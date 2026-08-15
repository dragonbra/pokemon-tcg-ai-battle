# 0045 Public Meta Router V1 CUDA-2048 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the diagnostic exact-deck Meta oracle with a lane-local router that uses only opponent Pokémon identities already present in the focal actor's public semantic observation, then run the immutable Policy-0809 Benchmark V2 CUDA-2048.

**Architecture:** Keep one U282 frozen Actor/Critic backbone and four disjoint U40/U90/U200/U282 Policy Q/V LoRA, Action Decoder, and Allocation Head bundles. A CUDA-resident per-job bitset monotonically accumulates certain public opponent Pokémon identities; an event-driven rule table resolves the earliest safe route and supplies per-row route IDs before Option LoRA and Decoder execution. Original 29-way labels remain reporting metadata, while Dragapult 00/15/16 and all Lopunny-bearing variants are intentionally folded into routing families 00 and 01.

**Tech Stack:** Python 3, PyTorch CUDA, project-local CUDA Engine 2.0, official engine runtime, pytest, static HTML reporting.

## Global Constraints

- Do not modify `engine/source/`.
- Opponent requested and materialized identity must be immutable complete `Policy-0809`.
- Focal routing may consume only current/remembered certain public opponent identities; hidden opponent deck ID, hidden hand, and Critic Meta predictions are forbidden.
- Route changes occur before a complete strategic decision and never inside a compound decision.
- U282 remains the default; ambiguous evidence remains U282.
- `Dreepy/Drakloak/Dragapult` immediately folds to route family 00/U282.
- `Buneary/Mega Lopunny ex` immediately folds to route family 01/U40.
- Starmie-only evidence remains provisional 12/U282; Starmie plus Dusknoir-line evidence resolves 17/U40.
- Every effective source checkpoint is materialized through `kaggle_fp16_storage_fp32_runtime_v1`; raw FP32 is not evaluation evidence.
- V9 Oracle remains immutable diagnostic evidence; Public Router uses a new V10 version and policy identity.

---

### Task 1: Public evidence memory and earliest-confirmation classifier

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_meta_router_v1.py`
- Create: `train/0045_single_deck_expert_minimal_lora/tests/test_public_meta_router_v1.py`

**Interfaces:**
- Produces: `PublicMetaMemory(job_count, device)`, `observe(validated, job_indices) -> Tensor`, `route_update(job_indices) -> Tensor`, and an auditable rule manifest.
- Consumes: semantic `card_cat`, `card_mask`, relative owner `2`, and identity-knowledge values `1/2` only.

- [ ] Write failing unit tests for Dreepy immediate 00, Buneary immediate 01, Staryu provisional 12, Staryu+Duskull 17, ambiguous Ogerpon default, persistence, reset, and candidate-identity exclusion.
- [ ] Run `pytest -q train/0045_single_deck_expert_minimal_lora/tests/test_public_meta_router_v1.py` and verify failure before implementation.
- [ ] Implement GPU-resident bitset/rule lookup without deck IDs or host transfers.
- [ ] Run the focused tests and verify all pass.

### Task 2: Mixed-lane routed trainable heads

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/public_meta_router_v1.py`
- Modify: `engine_cuda_2_0/python/ptcg_cuda_engine/semantic0031_router.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_meta_router_v1.py`

**Interfaces:**
- Produces: a generic optional pre-focal callback in `Semantic0031ResidentRouter` and public routed LoRA/Decoder/Allocation modules whose route tensor aligns with compact focal rows.
- Consumes: the four already-audited materialized head bundles from `materialize_oracle` without reading its oracle route function.

- [ ] Add failing mixed-row tests proving U40/U90/U200/U282 rows use their own tensors in one CUDA/CPU test batch.
- [ ] Add the no-op-by-default pre-focal hook to the CUDA router.
- [ ] Implement grouped row routing for Q/V LoRA and every Decoder primitive, plus per-job Allocation routing.
- [ ] Prove shared frozen tensors remain equal, head storages do not alias, and Critic output is not consumed by routing.
- [ ] Run focused and existing Meta Oracle regression tests.

### Task 3: Official Benchmark V2 runner and hard gates

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_public_meta_router_v1_cuda2048.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/tests/test_public_meta_router_v1.py`

**Interfaces:**
- Produces: V10 report JSON with complete Policy-0809 identity, public-route identity, per-game final/provisional route, lock decision, seen trigger evidence, route counts, and 2,048 terminal-game gates.
- Consumes: immutable Benchmark V2 common-seed schedule and the four exact checkpoint paths used by V9.

- [ ] Write report-gate tests rejecting exact opponent Meta/deck input, wrong Policy-0809 identity, incomplete games, hidden evidence, and route telemetry mismatches.
- [ ] Implement a fresh-output-only runner under `rl_runs/.../V10_.../artifact/public_router_evaluation/`.
- [ ] Run CPU smoke/unit gates, then a short CUDA smoke that exercises at least two different routes in one resident batch.
- [ ] Run the full CUDA-2048 evaluation and validate 2,048 terminal games, zero errors, zero unfinished games, and zero focal/opponent storage aliases.

### Task 4: Formal report and authoritative documentation

**Files:**
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V10_dragapult_007_public_meta_router_v1_cuda2048.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/evaluation/index.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V10_dragapult_007_public_meta_router_v1_cuda2048/artifact/evaluation.json`

**Interfaces:**
- Produces: an auditable HTML comparison against U282 and V9 Oracle, including per-Meta win rates, route coverage, earliest lock timing, unresolved rate, and evidence boundary.

- [ ] Render the formal HTML using the established 0045 evaluation UI and refuse overwrite.
- [ ] Add the report to the evaluation index and immutable reverse link.
- [ ] Synchronize DESIGN Markdown/HTML and decisions with the actual public-memory forward/routing graph and measured identities.
- [ ] Run the focused test suite and validate all report paths and identity hashes.

