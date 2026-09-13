# 0045 Lopunny Package CPU256 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the final assembled and static U75 Deck-003 Kaggle packages on a common 256-game CPU official-engine schedule with GPU-resident FP32 model inference.

**Architecture:** Add a project-local diagnostic adapter under `train/0045_single_deck_expert_minimal_lora/` that preserves the V18/V13 eight-exact-deck distribution at half size, verifies candidate and full Policy-0814 identities before routing, and delegates isolated games to shared `evaluation/` CPU workers. The immutable Kaggle archives remain inputs; reports go only to `.tmp/evaluation/` and are not Promote evidence.

**Tech Stack:** Python, PyTorch CUDA inference, official `cg` CPU runtime, evaluation worker processes, Unix-domain inference sockets, pytest.

## Global Constraints

- Read and obey `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md` before execution.
- Candidate storage is FP16 and runtime tensors are FP32; identity mismatch is fatal before games.
- Opponent is the independent complete `Policy-0814` with effective SHA-256 `476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580`.
- CPU official engine runs every game in an isolated worker; model inference runs on `cuda:0` through independent candidate/opponent sockets.
- Both candidates use the same deck counts, master seed, toss/seat contract, and per-slot schedule.
- Reports are diagnostic and must not be labeled canonical Frozen CPU256 or automatic Promote evidence.
- Do not modify `engine/source/`, either final `.tar.gz`, existing version directories, or existing formal reports.

---

### Task 1: Half-Scale V18 Distribution

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/policy0814_exact_deck_cpu256_schedule.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_package_cpu256.py`

**Interfaces:**
- Consumes: V13/V18 `policy0814_exact_deck_schedule.materialize()` and its immutable 512 realized deck counts/jobs.
- Produces: `materialize(project_root, focal_deployment_identity) -> dict` with 256 deterministic jobs and exact target deck counts.

- [ ] **Step 1: Write a failing schedule test**

Assert 256 jobs, eight expected deck IDs, deterministic largest-remainder halving of the realized 512 counts, common jobs across focal identities, unique preserved source slots, and a new CPU256 contract ID.

- [ ] **Step 2: Run the schedule test and confirm failure**

Run: `pytest -q train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_package_cpu256.py`

- [ ] **Step 3: Implement the schedule adapter**

Materialize the authoritative 512 schedule, halve realized counts using floor plus deterministic deck-ID tie breaking, and select the first required jobs per deck while retaining original engine/search/policy/toss seeds.

- [ ] **Step 4: Run the schedule test**

Expected: schedule assertions pass.

### Task 2: GPU-Resident Compound Package Server

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/compound_package_inference_server.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_package_cpu256.py`

**Interfaces:**
- Consumes: an extracted self-contained 0045 package exposing `POLICY` and the evaluation socket request protocol.
- Produces: a CUDA FP32 inference service with per-session encoder, macro, public-memory, and routed-head state isolation.

- [ ] **Step 1: Write failing lifecycle tests**

Cover module discovery, FP32 CUDA placement, new/close session state, assembled active-head restoration, and fail-closed rejection of FP16 runtime or unrecognized policy shape.

- [ ] **Step 2: Implement serialized state swapping**

Share immutable GPU modules once, serialize calls with a lock, restore only session-local causal/macro/router state around each `POLICY.select`, move encoded tensors to the model device, and implement the existing inference socket commands.

- [ ] **Step 3: Run focused tests and one-game parity smoke**

Require identical legal completion for CPU-package and GPU-server paths on a fixed seeded game before aggregate evaluation.

### Task 3: CPU256 Runner And Reports

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_policy0814_package_cpu256.py`
- Create: `.tmp/evaluation/0045_lopunny_package_cpu256/<candidate>/run-*/report.html`
- Create: `.tmp/evaluation/0045_lopunny_package_cpu256/comparison.json`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy0814_package_cpu256.py`

**Interfaces:**
- Consumes: the two root `.tar.gz` archives, CPU256 schedule, candidate server, full Policy-0814 resolver, and shared evaluation workers.
- Produces: two auditable reports plus one paired comparison summary.

- [ ] **Step 1: Implement fail-closed preflight**

Fresh-extract each archive; validate rootless inventory, exact Deck-003, FP16 storage, strict FP32 runtime, candidate effective identity, complete Policy-0814 identity, zero candidate/opponent storage sharing, matching `cg`, and an idle/usable CUDA device.

- [ ] **Step 2: Materialize the Policy-0814 inference runtime**

Copy the 0045 semantic runtime and immutable registered Policy-0814 model into an isolated temporary package, validate registry hashes, and launch the standard GPU inference server independently from the focal package server.

- [ ] **Step 3: Run one candidate smoke**

Use the CPU official engine, agent-selected first player, turn/repeat guards, one game, and both GPU inference sockets. Require terminal completion and zero error before scaling.

- [ ] **Step 4: Run both common-schedule CPU256 evaluations**

Run assembled first, then static U75, using identical 256 jobs/counts/seeds, `workers=8`, `worker_cpu_threads=1`, and the same Policy-0814 opponent identity.

- [ ] **Step 5: Validate and summarize**

Require 256/256 terminal games, zero error/unfinished, complete per-game seed/seat evidence, expected per-deck counts, candidate/opponent identity PASS, and write aggregate/per-deck/first-second results plus paired outcome flips.

- [ ] **Step 6: Preserve evidence boundaries**

Label the result `0045_v18_distribution_cpu256_gpu_inference_diagnostic_v1`; do not update formal experiment indexes, W&B, opponent catalogs, promotion records, or Kaggle.
