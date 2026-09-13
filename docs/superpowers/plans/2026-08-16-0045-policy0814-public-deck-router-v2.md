# 0045 Policy-0814 Public Deck Router V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a public-information-only composite Deck-007 policy that routes the stopped V14 Policy-0814 checkpoints by observed opponent Pokemon family.

**Architecture:** Keep U70 as the complete default deployment candidate and attach independently materialized U5/U15/U25/U95 Actor deltas. A per-game monotonic public-evidence memory consumes only certain opponent card identities already present in the focal observation, routes at complete strategic decision boundaries, treats Ogerpon as provisional U5 evidence, and lets Hydrapple/Meganium evidence override it to U70. The qualifying eval reuses the immutable V14 Policy-0814 exact-deck CUDA-512 schedule and emits a separate composite identity and route telemetry.

**Tech Stack:** Python 3.11, PyTorch, project-local CUDA Engine 2.0, official engine runtime, pytest, JSON/HTML experiment artifacts.

## Global Constraints

- Do not modify `engine/source/` or shared CUDA engine source.
- Opponent policy is complete immutable `Policy-0814`; focal and opponent storage must remain disjoint.
- Candidate sources are V14 U5/U15/U25/U70/U95 and must materialize under `kaggle_fp16_storage_fp32_runtime_v1`.
- Route inputs are limited to certain/remembered public opponent card identities; exact deck ID, hidden cards, candidate identities, and Critic outputs are forbidden.
- Route manifest is `001/008 -> U5`, `002 -> U15`, `003 -> U25`, `007 -> U95`, `009/011/071/unknown/conflict -> U70`.
- Ogerpon alone is provisional U5; a later public Hydrapple/Meganium-family card changes the route to U70.
- Formal eval is exactly the existing `0045_v13_policy0814_exact_deck_common_seeds_cuda512_v1` schedule: 512 terminal games, zero error, zero unfinished, greedy official engine.
- Historical V9-V12 router files and reports remain immutable.

---

### Task 1: Public Evidence Memory

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_deck_router_v2_rules.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_deck_router_v2.py`

**Interfaces:**
- Produces: `DEFAULT_UPDATE: int`, `ROUTED_UPDATES: tuple[int, ...]`, `ROUTE_MANIFEST: dict`, and `PublicDeckMemory.observe(validated, job_indices) -> torch.Tensor`.
- Consumes: semantic `card_cat`, `card_mask`, and resident job indices.

- [ ] **Step 1: Write failing tests for every eight-deck route, unknown/default behavior, hidden candidate rejection, Ogerpon provisional routing, 071 override, and conflict/default behavior.**
- [ ] **Step 2: Run `pytest -q train/0045_single_deck_expert_minimal_lora/tests/test_public_deck_router_v2.py` and confirm the new module is missing.**
- [ ] **Step 3: Implement the deterministic public-card evidence state machine with per-game telemetry.**
- [ ] **Step 4: Rerun the focused test and require PASS.**

### Task 2: Composite Candidate Identity and CUDA Router

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_deck_router_v2.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/tests/test_public_deck_router_v2.py`

**Interfaces:**
- Consumes: V14 source checkpoints, Policy-0814 portable Actor/Value bases, `PublicDeckMemory`, and the existing candidate materializer.
- Produces: `materialize_public_deck_router_v2(...) -> PublicDeckRouterRuntime` and `PublicDeckResidentRouter`.

- [ ] **Step 1: Add failing identity tests that require five exact source updates, PASS deployment audits, exact-equal shared Actor tensors, disjoint routed modules, and no Critic-to-router edge.**
- [ ] **Step 2: Implement five independent candidate materializations and audit the actual saved V14 boundary: shared Actor backbone plus routed ActionDecoder, Policy Option LoRA, and AllocationHead.**
- [ ] **Step 3: Implement compacted mixed-lane decode and allocation routing at complete decision boundaries.**
- [ ] **Step 4: Run the focused tests and the existing public-router regression suite.**

### Task 3: Comparable Policy-0814 Eval512

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_public_deck_router_v2_policy0814_cuda512.py`
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/render_public_deck_router_v2.py`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V15_public_deck_router_v2_policy0814/artifact/evaluation.json`
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V15_public_deck_router_v2_policy0814_cuda512.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/evaluation/index.html`

**Interfaces:**
- Consumes: the unchanged V14 exact-deck schedule and source checkpoints U5/U15/U25/U70/U95.
- Produces: a validated JSON report with composite/candidate/opponent/CUDA identity audits, per-deck W/L/D, route timing, unresolved/conflict counts, and a formal HTML report.

- [ ] **Step 1: Add report validation gates for schedule hash, source hashes, 512 terminal games, public-only routing, route inventory, candidate/opponent identity, and zero storage alias.**
- [ ] **Step 2: Run a small non-authoritative CUDA smoke in `.tmp/evaluation/0045_public_deck_router_v2/` and confirm route telemetry covers all expected paths.**
- [ ] **Step 3: Run the full official-engine greedy CUDA-512 common-seed evaluation.**
- [ ] **Step 4: Compare composite outcomes against static U70 per game and per exact deck; do not infer composite score by adding historical maxima.**
- [ ] **Step 5: Render the immutable V15 HTML, refresh the evaluation index, and write the reverse-link artifact.**

### Task 4: Canonical Documentation

**Files:**
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Consumes: the final route manifest, source/effective hashes, eval512 report, and the confirmed V14 checkpoint omission audit.
- Produces: synchronized authoritative design and decision records.

- [ ] **Step 1: Record the user-requested V14 stop at complete U105 eval and W&B sync boundary.**
- [ ] **Step 2: Record that historical V14 deployment checkpoints omit 16 trained StateEncoder LoRA tensors and therefore represent the actual reconstructable Decoder/Option-LoRA/Allocation candidates, not the full in-memory behavior model.**
- [ ] **Step 3: Document public input fields, provisional/locked transitions, default behavior, routed tensor boundary, and evidence limitations in both DESIGN formats.**
- [ ] **Step 4: Record the official eval512 result without automatic Promote or Kaggle claims.**
- [ ] **Step 5: Run focused and full 0045 tests, validate JSON/HTML links, and confirm no training process remains.**
