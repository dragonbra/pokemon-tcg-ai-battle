# 0020 Source Persona PoC Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how much policy information the 0019 checkpoint stores in the James Cox & Henry Chao persona residual, then correct the interpretation of conditioned BC metrics and neutral RL initialization.

**Architecture:** Hold the epoch-13 checkpoint, exact Raging Bolt deck, official-engine Arena catalog, seat schedule, and games per opponent fixed. Compare the existing neutral `source_id=0` V8 report with one audit-only self-contained candidate using `source_id=98`; preserve the production and RL contract at `source_id=0`. The official runtime does not expose a seeded battle ABI, so this is an unpaired same-protocol comparison rather than a replay of identical random deals.

**Tech Stack:** Python 3.11, PyTorch, official engine evaluation workers, repository HTML evaluation reports.

## Global Constraints

- Never modify `engine/source/`.
- Use checkpoint SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- Use exact deck SHA-256 `f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`.
- The sole policy variable against V8 is `source_id=98`, assigned to exact team identity `James Cox & Henry Chao` in the frozen 0019 catalog.
- This is an audit-only conditioned-imitation PoC, not a deployment candidate, RL starting policy, or proof of unseen-deck generalization.
- Formal 0020 deployment and all deck-specific RL remain fixed at `source_id=0`.
- Preserve the V8 report and create strict version `V10_raging_bolt_source98_persona_poc`.

---

### Task 1: Freeze the single-variable audit contract

**Files:**
- Create: `docs/superpowers/plans/2026-07-30-0020-source-persona-poc-audit.md`
- Inspect: `train/0015_dragapult_conditioned_bc/`
- Inspect: `train/0016_alakazam_multideck_bc/`
- Inspect: `train/0019_universal_winner_bc/`

**Interfaces:**
- Consumes: frozen 0019 source vocabulary and exact team identity mapping.
- Produces: an evidence-backed history of when Persona entered training and which conclusions it affects.

- [x] Verify that source conditioning first entered the numbered BC line in project 0015.
- [x] Verify that 0016 inherited it and 0019 expanded it to 509 non-neutral sources.
- [x] Verify that source 98 is `James Cox & Henry Chao` and that V8 uses the same exact deck with source 0.

### Task 2: Build the audit-only candidate

**Files:**
- Create: `evaluation/arena/candidates/0020_persona98_raging_bolt_poc/`
- Modify: `train/0020_pluggable_deck_rl/package_builder.py`
- Modify: `train/0020_pluggable_deck_rl/tests/test_package_builder.py`

**Interfaces:**
- Consumes: `build_candidate("region_bot")`, frozen checkpoint, ontology, deck, and copied runtime.
- Produces: a self-contained candidate whose manifest explicitly records source 98 and audit-only status.

- [ ] Add a fail-closed persona-PoC build mode without changing the default neutral package output.
- [ ] Assert that only source 98 and the audited team identity are accepted for this PoC.
- [ ] Build the package and run package validation plus focused unit tests.

### Task 3: Run official-engine Arena evaluation

**Files:**
- Create: `experiments/0020_pluggable_deck_rl/evaluation/V10_raging_bolt_source98_persona_poc.html`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V10_raging_bolt_source98_persona_poc/artifact/status.json`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/V10_raging_bolt_source98_persona_poc/artifact/evaluation.json`
- Modify: `experiments/0020_pluggable_deck_rl/evaluation/index.html`

**Interfaces:**
- Consumes: the 30 enabled opponent catalog entries and 10 games per opponent.
- Produces: 300 official-engine games directly comparable with V8's 84-216 neutral result.

- [ ] Run `python3 -m evaluation validate evaluation/arena/candidates/0020_persona98_raging_bolt_poc`.
- [ ] Run the full Arena evaluation with bounded per-worker CPU threads and no trace retention.
- [ ] Publish the immutable HTML, reverse-link JSON, status metrics, and refreshed evaluation index.

### Task 4: Correct model-design interpretation

**Files:**
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.html`

**Interfaces:**
- Consumes: V8 neutral and V10 source-98 official-engine results plus 0019 conditioned/neutral validation curves.
- Produces: a documented boundary for BC quality, neutral deployment, and RL initialization.

- [ ] Explain that conditioned loss measures teacher-conditioned imitation, not the intended source-agnostic policy.
- [ ] Quantify the V8/V10 effect with confidence intervals and avoid attributing all difference causally beyond the fixed Arena contract.
- [ ] State that future universal BC removes Persona from actor input; source remains metadata/grouping only.
- [ ] Explain why neutral PPO is not weight decay: the residual is zero, while shared backbone/decoder knowledge remains and PPO supplies new on-policy gradients.
- [ ] Record the residual risk that shared representations were optimized under a shortcut and require a fresh persona-free BC baseline for a clean answer.

### Task 5: Verify and hand off

**Files:**
- Test: `train/0020_pluggable_deck_rl/tests/test_package_builder.py`
- Test: `tests/test_evaluation_assets.py`

**Interfaces:**
- Consumes: package, report, index, and design updates.
- Produces: reproducible PoC evidence while V9 Dragapult PPO continues independently.

- [ ] Run focused unit tests and syntax checks for touched modules.
- [ ] Verify report totals, completion/error counts, candidate/source/deck hashes, and V8/V10 comparability.
- [ ] Confirm V9 remains healthy and do not alter its optimizer, checkpoint, rollout pool, or W&B run.
