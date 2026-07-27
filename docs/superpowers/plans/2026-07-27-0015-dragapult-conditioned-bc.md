# 0015 Dragapult Conditioned BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the authoritative prospective design for a high-quality, component-transfer Dragapult behavior-cloning project and audit whether 0014 can represent the exact registered deck.

**Architecture:** Preserve the established full-action BC contract while conditioning the policy on the exact own 60-card deck and current actor-visible resource state. Use pure Dragapult to teach the Dragapult module and Starmie-Dusknoir to teach the Dusknoir module, then measure their separate and combined transfer into Dragapult-Dusknoir. Audit storage, batching, model consumption and online inference of the 0014 registered-deck path before accepting it as the 0015 base.

**Tech Stack:** Markdown and self-contained HTML documentation.

## Global Constraints

- This is a prospective BC design: no value head, reward optimization, PPO, or RL run is authorized.
- Never unconditionally mix different deck, team, or implementation-policy labels.
- The complete own deck is an action-time player-known input; expert/source identity is audit metadata unless a deployable persona contract is explicitly designed.
- Strength claims require self-contained packages and comparable official-engine evaluation.

---

### Task 1: Audit the 0014 registered-deck feature path

**Files:**
- Read: `train/0014_faithful_board_causal_features/features/compiler.py`
- Read: `train/0014_faithful_board_causal_features/training/ac_data.py`
- Read: `train/0014_faithful_board_causal_features/ac_model.py`
- Read: `train/0014_faithful_board_causal_features/r1_model.py`
- Read: `train/0014_faithful_board_causal_features/online_runtime.py`
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/001_0014_registered_deck_audit.md`

**Interfaces:**
- Consumes: the frozen 0014 feature cache, V9/V10 model contracts and V9 checkpoint.
- Produces: an evidence-backed verdict separating cache expressiveness, model consumption, runtime wiring and learned multi-build evidence.

- [x] **Step 1: Trace the exact own-deck path**

Record the 60-card validation, `registered_card_ids`, multiplicity, ledger alignment, AC/R1 resource-token readers and candidate `deck.csv` runtime path.

- [x] **Step 2: Measure actual cache diversity and checkpoint sensitivity**

Count exact registered-deck signatures across all train/validation shards and perturb the real one-card multiplicity difference on a fixed V9 validation state. Record the counts and finite-logit delta without changing any checkpoint or dataset.

- [x] **Step 3: State the acceptance verdict**

Distinguish: A0 does not consume deck features; AC/R1 can exactly distinguish builds; V9 is sensitive to the channel; 0014's two highly imbalanced one-card variants do not prove broad cross-archetype conditioning. Require new per-build/source tests and balanced sampling for 0015.

### Task 2: Publish the revised 0015 design

**Files:**
- Create: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Create: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`

**Interfaces:**
- Consumes: official complete Dragapult episodes, exact deck manifests, source provenance, actor-visible observations, and the existing full-action BC contract.
- Produces: a deck-conditioned BC architecture, T0–T3 experiment matrix, milestone engine-evaluation policy, and discussion questions.

- [x] **Step 1: Specify provenance, conditioning, and label-conflict boundaries**

Document one deck manifest per episode, source/deck grouped splits, balanced sampling, full own-deck conditioning, and source-stratified audit.

- [x] **Step 2: Specify the BC-only model and T0-T3 controls**

Document shared board/option/card encoders, own-deck resource memory, optional build adapters, T0 target-only, T1 plus pure Dragapult, T2 plus Starmie-Dusknoir, T3 all three sources, and the deck-input ablation.

- [x] **Step 3: Specify transfer and evaluation evidence**

Document a held-out third-build few-shot fine-tune versus scratch comparison and sparse fixed-checkpoint official-engine evaluation rather than per-epoch battles.

- [x] **Step 4: Verify document consistency**

Run: `python3 - <<'PY'\nfrom pathlib import Path\npaths = (Path('experiments/0015_dragapult_conditioned_bc/DESIGN.md'), Path('experiments/0015_dragapult_conditioned_bc/DESIGN.html'), Path('experiments/0015_dragapult_conditioned_bc/decisions/001_0014_registered_deck_audit.md'))\nfor path in paths:\n    assert path.is_file() and path.stat().st_size > 0, path\n    text = path.read_text(encoding='utf-8')\n    for phrase in ('0014', '0015', 'registered', 'T0', 'T3'):\n        assert phrase in text, (path, phrase)\nPY\ngit diff --check`

Expected: both documents state the same BC-only scope and contain no whitespace errors.

### Task 3: Make target-only engine strength the primary endpoint

**Files:**
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/002_target_only_evaluation_contract.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`

**Interfaces:**
- Consumes: frozen T0–T3 target-deck candidates and target-only validation metrics.
- Produces: a two-axis imitation/strength interpretation and a non-leaky official-engine contract.

- [x] **Step 1: Restrict success evaluation to the target build**

Declare auxiliary decks as training/diagnostic data only; exclude their metrics from checkpoint
selection and headline success.

- [x] **Step 2: Separate imitation diagnostics from the primary strength endpoint**

Permit lower target exact-action agreement with higher confirmed target win rate, while retaining
target validation as a collapse diagnostic and common checkpoint-selection input.

- [x] **Step 3: Prevent engine-test selection leakage**

Freeze one candidate per arm, use matched target-only engine comparisons, and require a fresh-seed
confirmation of any claimed winner without further tuning.

- [x] **Step 4: Verify Markdown/HTML parity**

Parse the HTML, assert both authority documents contain the target-only diagnostic/primary endpoint
contract and run `git diff --check`.

### Task 4: Publish the next-day execution runbook

**Files:**
- Create: `experiments/0015_dragapult_conditioned_bc/RUNBOOK.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/decisions/002_target_only_evaluation_contract.md`

**Interfaces:**
- Consumes: daily official episode bundle, targeted JSON overlay, exact deck/source list and the
  user-delivered 0014 successor model contract.
- Produces: an auditable 12-hour T0–T3 execution sequence with W&B records and optional extensions.

- [x] **Step 1: Define the input handoff and overlay dataset contract**

Reference the base bundle without copying it, store only targeted JSON additions, deduplicate by
episode identity/content and build one shared feature cache with lightweight arm views.

- [x] **Step 2: Freeze the natural-data matrix and logging contract**

Specify all-qualified-data inclusion, natural source proportions, target-only validation, milestone
checkpoints, strict versions and canonical local/TensorBoard/W&B recording. Accept natural increases
in updates/compute and report them rather than downsampling data.

- [x] **Step 3: Define persistent execution and overfitting branches**

Prioritize the complete base matrix, preserve failures, continue for the authorized window, and
change only one declared variable per follow-up version.

- [x] **Step 4: Bound optional extensions**

After the base matrix, allow static win/loss BC weights, at most one controlled related-deck source,
and deeper reward/value work only under its own explicit contract.

### Task 5: Adopt natural data volumes and the Marnie/Munkidori extension

**Files:**
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/RUNBOOK.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/decisions/001_0014_registered_deck_audit.md`

**Interfaces:**
- Consumes: all qualified target/pure-Dragapult/Starmie-Dusknoir episodes and the large
  Marnie's-Grimmsnarl-with-Munkidori pool.
- Produces: a competition-oriented natural-volume base matrix plus one capped T4 transfer arm.

- [x] **Step 1: Remove artificial base-matrix mixture quotas**

Use every qualified trajectory at its natural frequency in T0–T3, accept extra updates/compute and
report the actual counts rather than downsampling for causal isolation.

- [x] **Step 2: Specify T4 Marnie/Munkidori**

Add a post-matrix arm using a deterministic, stratified Marnie/Munkidori episode cap no larger than
the complete T3 training-episode count, with target-only evaluation and damage-counter slices.
