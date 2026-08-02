# Foundation Decoder Information Flow Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable visual report explaining what information reaches the deck-local decoder, where information is compressed or discarded, and how to distinguish input, representation, decoder-capacity, and training-forgetting failures.

**Architecture:** Treat the frozen 0019 Foundation code vendored into 0022 as the executable source of truth. Trace raw observation fields through the codec, causal ledger, board/resource/event/scenario encoders, the final state/option tensor boundary, and the autoregressive pointer decoder; render that trace as a self-contained HTML report with explicit evidence levels and proposed diagnostic experiments.

**Tech Stack:** Static HTML5, CSS, minimal vanilla JavaScript, Python/PyTorch source audit.

## Global Constraints

- Do not modify `engine/source/`.
- Do not change model, feature, action, training, or checkpoint semantics.
- Distinguish official rules/runtime facts from project model-design facts and research hypotheses.
- Preserve user changes in the dirty worktree.

---

### Task 1: Establish The Executable Model Contract

**Files:**
- Read: `train/0022_league_training/foundation/contract.py`
- Read: `train/0022_league_training/foundation/model_source/base_model.py`
- Read: `train/0022_league_training/foundation/model_source/ac_model.py`
- Read: `train/0022_league_training/foundation/model_source/r2_model.py`
- Read: `train/0022_league_training/policy/action_distribution.py`
- Read: `train/0022_league_training/policy/actor_critic.py`

**Interfaces:**
- Consumes: Current 0022/0023 Foundation and league-training implementation.
- Produces: Verified tensor shapes, trainable/frozen boundary, residual paths, and raw-field provenance used in Task 2.

- [x] **Step 1: Verify immutable Foundation identity and configuration**

Record `d_model=320`, four board encoder layers, two scenario layers, 192 entities, 128 options, 64 action steps, four Goal-QKV roles, neutral `source_id=0`, and the exact trainable decoder component list.

- [x] **Step 2: Trace raw field compilation and causal state**

Map global, entity, option, registered-deck, ledger, event-window, opponent-hand, mask, and count fields to their first learned representation.

- [x] **Step 3: Trace every residual and irreversible aggregation**

Separate encoder-internal residuals from a raw-to-decoder bypass, and record mean pooling, CLS pooling, attention, normalization, clipping, count aggregation, and fixed event-window boundaries.

### Task 2: Build The Visual Audit Report

**Files:**
- Create: `docs/reports/foundation-decoder-information-flow-review-2026-08-02.html`

**Interfaces:**
- Consumes: Verified facts from Task 1.
- Produces: One self-contained browser-readable report with model diagram, decoder zoom, loss audit, failure taxonomy, and experimental roadmap.

- [ ] **Step 1: Render the top-level model map**

Show raw observation, deterministic compiler, frozen board/semantic/scenario encoder, final tensor boundary, deck-local decoder/value branches, and ordered option output.

- [ ] **Step 2: Render information-preservation status**

Use explicit labels for exact, structured, aggregated, contextualized, hard constraint, and absent-bypass paths.

- [ ] **Step 3: Render the decoder boundary and equations**

Show state initialization, pointer score, selected-option GRU update, masks, `minCount/maxCount`, and STOP.

- [ ] **Step 4: Render failure attribution and next experiments**

Provide tests that can distinguish missing raw inputs, frozen-encoder collisions, insufficient decoder capacity, and PPO/BC forgetting without claiming a diagnosis before measurement.

### Task 3: Verify The Artifact

**Files:**
- Test: `docs/reports/foundation-decoder-information-flow-review-2026-08-02.html`

**Interfaces:**
- Consumes: Task 2 HTML.
- Produces: A structurally valid, responsive, internally linked report whose claims match current code.

- [ ] **Step 1: Run static contract checks**

Verify required section IDs, no remote dependencies, balanced high-level HTML tags, and the presence of all cited source paths.

- [ ] **Step 2: Inspect desktop and mobile rendering**

Open the local HTML at desktop and mobile widths and verify the pipeline, tables, labels, and navigation do not overlap or overflow.

- [ ] **Step 3: Reconcile every numerical and architectural claim**

Cross-check tensor shapes, limits, trainable modules, residual equations, and information-loss statements against the exact files listed in Task 1.
