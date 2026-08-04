# CUDA Frozen51 Acceptance And 0031 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove which Frozen Arena decks the CUDA engine can execute with official-engine parity, then use only admitted decks to complete the 0031 latest-checkpoint Frozen evaluation and document the remaining RL integration work.

**Architecture:** Treat support as a fail-closed, per-deck contract: static reachable-rule inventory, official CPU/POD/CUDA mirror canary, and interaction-oriented ordered replay. Keep support evidence separate from policy-strength evaluation. Extend the existing 0031 batch publication only after the CUDA engine can expose the observation/action contract required by the rule-faithful 0031 actor without hidden CPU-engine fallback.

**Tech Stack:** Python 3.11, C++20, CUDA 12.8, RTX 5080, official read-only C++ engine, CUDA POD runtime, PyTorch, HTML/JSON evaluation reports.

## Global Constraints

- Never modify `engine/source/`; mount or include it read-only.
- The unmodified official CPU engine is the semantic oracle.
- A deck is not admitted merely because its card IDs load or one trajectory terminates.
- CUDA errors, unknown effects, state/status mismatch, outcome mismatch, decision-limit overflow, and unsupported observation/action contracts fail closed.
- Preserve the existing four reports under `evaluation/arena/combat_mat/0031_latest_frozen_test/reports/` and refuse conflicting overwrite.
- Frozen identity is `0019_foundation_51_exact_decks_v4`; every exact deck has 60 cards.
- Strength evaluation uses 10 games per opponent with balanced seats and records the actual opponent snapshot.
- Temporary fixtures and binaries stay under `.tmp/cuda_support_audit/` or `engine_cuda/build/`; durable audit output stays under `evaluation/arena/combat_mat/0031_latest_frozen_test/`.
- CUDA parity is engine support evidence, not gameplay-strength evidence.
- No CPU official-engine fallback is allowed in a claimed CUDA rollout hot path.

---

### Task 1: Frozen51 Reachable-Semantics Inventory

**Files:**
- Create: `engine_cuda/tools/audit_frozen51_support.py`
- Test: `engine_cuda/tests/test_audit_frozen51_support.py`
- Generate: `.tmp/cuda_support_audit/official_rules.json`

**Interfaces:**
- Consumes: `evaluation/arena/frozen/*/deck.csv`, private official IR, and CUDA semantic handler sources.
- Produces: `build_support_inventory(...) -> dict[str, object]` with per-deck cards, skills, attacks, effect/target/condition/trigger IDs, prior parity overlap, and static blockers.

- [x] Add a fixture test proving linked skill/attack traversal and exact-deck hash matching.
- [ ] Run `python3 -m unittest -v engine_cuda.tests.test_audit_frozen51_support` and verify the missing module fails.
- [x] Implement recursive rule reachability and handler-case inventory without embedding private card text in durable output.
- [x] Record the 40-deck fixture overlap and list every Frozen-only card ID.
- [x] Run the focused test and generate the local inventory.

### Task 2: Official CPU/POD/CUDA Deck Admission

**Files:**
- Create: `.tmp/cuda_support_audit/frozen51_mirrors.tsv`
- Generate: `.tmp/cuda_support_audit/official_rules.bin`
- Generate: `.tmp/cuda_support_audit/frozen51_mirror_parity.jsonl`
- Generate: `.tmp/cuda_support_audit/frozen51_ordered_parity.jsonl`

**Interfaces:**
- Consumes: exact Frozen decks and the frozen official rule pack.
- Produces: one parity result per deck plus interaction evidence for cross-deck effects.

- [x] Compile the private IR to the official CUDA rule-pack ABI and validate all section counts.
- [x] Compile `official_battle_ordered_matrix_cuda_paired.cu` for the local GPU architecture.
- [x] Run one coverage-first legal mirror seed for every Frozen deck and retain per-deck error/status/state/outcome counters.
- [x] Re-run failed decks with a focused seed window to separate deterministic unsupported semantics from sparse trajectory coverage.
- [x] Run all 38 x 38 admitted ordered interaction cases.
- [x] Mark decks `supported`, `unsupported`, or `not_proven`; never fold `not_proven` into `supported`.

### Task 3: 0031 CUDA Policy Contract

**Files:**
- Modify only if required: `engine_cuda/python/ptcg_cuda_engine/native.py`
- Modify only if required: `engine_cuda/src/torch_binding.cpp`
- Create only if required: `engine_cuda/python/ptcg_cuda_engine/rule_faithful_adapter.py`
- Test only if required: `engine_cuda/tests/test_rule_faithful_adapter.py`

**Interfaces:**
- Consumes: resident `OfficialStatePod`, 0031 causal feature state, and legal official options.
- Produces: device tensors matching the exact 0031 checkpoint schema and normalized official action indices.

- [x] Compare every 0031 feature field with data resident in `OfficialStatePod`; list unavailable chronological or hidden-information fields.
- [x] Compare the 0031 option contract with the CUDA PolicyCodecV1 surface and identify the missing canonical adapter.
- [ ] Run the same initial states through official-engine deployment and the candidate CUDA adapter; require element/action parity.
- [ ] If all inputs are representable, implement the smallest device adapter and focused tests.
- [x] If any required feature is unavailable, stop strength evaluation through CUDA and publish the exact blocking fields instead of substituting zeros or CPU observations.

### Task 4: Complete Supported 0031 Frozen Reports

**Files:**
- Modify: `evaluation/rule_faithful_frozen_batch.py`
- Modify: `tests/test_rule_faithful_frozen_batch.py`
- Generate: `evaluation/arena/combat_mat/0031_latest_frozen_test/reports/<deck_id>.html`
- Modify: `evaluation/arena/combat_mat/0031_latest_frozen_test/manifest.json`
- Modify: `evaluation/arena/combat_mat/0031_latest_frozen_test/index.html`

**Interfaces:**
- Consumes: supported deck IDs, 0031 latest checkpoint, and the admitted CUDA policy contract.
- Produces: immutable 510-game reports for every runnable requested candidate deck.

- [ ] Add resume tests that preserve the four existing reports and select only missing admitted deck IDs.
- [ ] Add manifest fields for engine backend, support-evidence hash, rejected decks, and completion denominator.
- [ ] Run one-deck 51-opponent smoke and compare report identity/counts with the existing official-engine contract.
- [ ] Run each remaining admitted deck for 51 opponents by 10 balanced games and refresh the aggregate after every deck.
- [ ] Validate every published report has 510 completed games, zero errors, the exact checkpoint/deck/pool identity, and a stable run ID.

### Task 5: Support And RL Readiness Report

**Files:**
- Create: `evaluation/arena/combat_mat/0031_latest_frozen_test/cuda_support.json`
- Create: `evaluation/arena/combat_mat/0031_latest_frozen_test/cuda_support.html`
- Create: `evaluation/arena/combat_mat/0031_latest_frozen_test/RL_READINESS.md`

**Interfaces:**
- Consumes: static inventory, parity results, adapter A/B evidence, and completed strength reports.
- Produces: auditable supported/unsupported/not-proven tables and a prioritized RL integration checklist.

- [x] Publish per-deck status, failed seed/matchup, implicated card IDs, and missing effect/continuation IDs.
- [x] Link support evidence from the 0031 combat index while preserving existing strength reports.
- [x] Document resident reset, codec, policy inference, action apply, rollout buffer, reward/value, terminal handling, opponent routing, and monitoring requirements for RL.
- [x] Run focused CUDA/Python tests and verify durable links and hashes.
- [x] State residual coverage limits: finite deterministic policies do not prove every possible card interaction.

### Task 6: Throughput And 0032 Launch Gate

- [x] Preserve and report the measured same-machine CPU vs isolated-CUDA inference A/B and centralized RL baseline.
- [x] Attempt to build the resident PyTorch extension without source changes and document the conda scattered-toolkit / Torch FindCUDA blocker.
- [x] Refuse to substitute a PolicyCodecV1 or engine-only number for 0031 throughput.
- [x] Keep 0032 unlaunched until the strict 0031 adapter or an explicitly new resident policy contract passes feature/action parity.

## Self-Review

- Spec coverage: all 51 decks receive an explicit support state; admitted decks are eligible for resumed evaluation; failures include card/rule causes; RL adaptation is separately tested and documented.
- Placeholder scan: every failure path and publication artifact has an explicit contract.
- Type consistency: the support inventory feeds admission; admission feeds the adapter and batch denominator; the final report consumes immutable hashes from all three stages.
