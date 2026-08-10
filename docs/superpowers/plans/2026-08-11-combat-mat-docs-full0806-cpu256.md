# Combat Mat Docs Migration And Full-0806 CPU256 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the long-lived combat-mat reports under `docs/`, retire the pre-V1 Policy-0806 standard, formalize CPU256/CUDA2048 seed rules, and publish fresh Full-0806 CPU256 reports for decks 002, 003, 007, and 009.

**Architecture:** Keep formal reports in one canonical `docs/evaluation/combat_mat/` tree while temporary run state remains under `.tmp/evaluation/`. The existing Frozen-0806 official CPU runner remains the single execution path; it resolves the requested deck IDs, validates the frozen pool contract, performs a Full-0806 identity preflight, runs one deterministic 256-game unit per candidate, and publishes only complete reports. Seed semantics remain centralized in `evaluation/frozen_0806_contract.py` and are described normatively in the Promote Champion V1 protocol.

**Tech Stack:** Python 3.11, official engine runtime, PyTorch inference service, unittest, Markdown/HTML/JSON report assets.

## Global Constraints

- Do not modify `engine/source/`.
- Policy identity resolution precedes routing and batching; mismatches hard fail.
- Full-0806 means every effective inference component comes from immutable Policy-0806.
- CPU256 is one complete frozen 256-game unit per candidate; CUDA2048 is eight immutable, non-overlapping 256-game units.
- Seeded toss winners must invoke the corresponding Agent on official context 41; the harness must not assign or balance seats.
- Do not retain or relabel pre-V1 Policy-0806 results as current Full-0806 evidence.
- Do not run CUDA2048 before policy identity and CPU validation pass.

---

### Task 1: Canonical report location and retired Policy-0806 evidence

**Files:**
- Move: `evaluation/arena/combat_mat/` to `docs/evaluation/combat_mat/`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `evaluation/README.md`
- Modify: active report writers that still use `evaluation/arena/combat_mat`
- Delete: `docs/evaluation/combat_mat/policy_0806/` pre-V1 standard/results

**Interfaces:**
- Consumes: formal report paths used by CPU and CUDA evaluation entrypoints.
- Produces: canonical `docs/evaluation/combat_mat/` report root and no tracked pre-V1 `policy_0806` standard.

- [ ] Move the tracked report tree with `git mv evaluation/arena/combat_mat docs/evaluation/combat_mat`.
- [ ] Place the migration/validity README at `docs/evaluation/combat_mat/README.md`.
- [ ] Remove the exact tracked `docs/evaluation/combat_mat/policy_0806/` subtree requested for retirement.
- [ ] Update repository guidance and active formal report writers to use the new canonical root.
- [ ] Run `rg -n "evaluation/arena/combat_mat" AGENTS.md CLAUDE.md evaluation engine_cuda docs/rl` and classify any remaining references as historical text or defects.

### Task 2: Normative seed contract

**Files:**
- Modify: `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`
- Modify: `evaluation/frozen_0806_contract.py`
- Test: `tests/test_frozen_0806_contract.py`
- Test: `engine_cuda/tests/test_policy_0806_cuda_evaluation.py`

**Interfaces:**
- Consumes: `FROZEN_0806_EVALUATION_SEED`, deterministic per-game seed functions, frozen schedule hash.
- Produces: explicit CPU replica-0 and CUDA replica-0..7 semantics plus hard assertions for non-overlap and reproducibility.

- [ ] Add failing assertions that CPU256 is exactly replica 0 and CUDA2048 is replicas 0 through 7 of the same contract.
- [ ] Verify every engine/Search seed is deterministic and unique for its candidate, opponent slot, and replica.
- [ ] Document master seed `341512806`, no rerolls/replacements, context-41 seat choice, required report seed evidence, and the requirement for a new contract ID if any seed rule changes.
- [ ] Run the focused seed-contract tests.

### Task 3: Full-0806 identity audit in formal CPU reports

**Files:**
- Modify: `evaluation/frozen_0806_full_evaluation.py`
- Modify: the shared policy identity registry/resolver only if needed to avoid duplicating hashes
- Test: `tests/test_frozen_0806_full_evaluation.py`

**Interfaces:**
- Consumes: requested `Policy-0806`, registered effective identity, immutable checkpoint/component hashes.
- Produces: `policy_identity_audit` with `PASS`, requested/materialized policy IDs, effective hash, and component sources in each published report/manifest.

- [ ] Write a failing report-validation test for missing or mismatched Policy-0806 identity audit.
- [ ] Add one preflight boundary before inference service startup and inject its immutable audit result into formal report metadata.
- [ ] Make report validation hard fail unless the requested and materialized effective identities exactly match Full-0806.
- [ ] Verify training and evaluation resolve `Policy-0806` to the same registered effective hash.
- [ ] Run the focused evaluation and policy-identity tests.

### Task 4: Four-deck official CPU256 execution

**Files:**
- Create: `docs/evaluation/combat_mat/policy_0806/0806_kaggle_top100_plus_v1_cpu_seeded_256_agent_choice_v3/` generated reports and manifest
- Modify: `docs/evaluation/combat_mat/index.html` or the canonical generated index

**Interfaces:**
- Consumes: exact candidate IDs for deck numbers 002, 003, 007, 009; Full-0806 opponent; frozen v3 schedule.
- Produces: four complete 256-game official CPU reports, 1,024 terminal games total.

- [ ] Resolve numbers 002, 003, 007, and 009 to exact frozen candidate IDs and print the mapping before launch.
- [ ] Run a short preflight/smoke proving official engine, inference services, identity audit, and report output paths are valid.
- [ ] Run `python3 -m evaluation.frozen_0806_full_evaluation --opponent-policy 0806 --deck <002-id> --deck <003-id> --deck <007-id> --deck <009-id>` with calibrated worker/thread settings.
- [ ] Validate each report has exactly 256 terminal games, zero errors, zero unfinished games, the frozen schedule/hash, complete per-game seeds, agent-choice toss metadata, and `policy_identity_audit.status == "PASS"`.
- [ ] Regenerate the canonical docs index without presenting unrun decks as completed.

### Task 5: Final verification

**Files:**
- Verify all modified and generated files.

**Interfaces:**
- Consumes: Tasks 1-4 outputs.
- Produces: auditable final status and exact command/result inventory.

- [ ] Run focused unittests for report paths, frozen contract, policy identity, and CUDA schedule construction.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short` and separate this task's files from pre-existing user changes.
- [ ] Report the four W-L-D results, report links, identity hashes, seed contract answer, and any remaining blocker without committing unrelated work.
