# 0015 Outcome-Aware Value Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the authoritative prospective design for 0015, a BC-to-value research project using complete same-source win/loss trajectories without starting RL.

**Architecture:** Place the project design under `experiments/` in both Markdown and HTML, following the repository's project-document contract. The design separates frozen facts from hypotheses, specifies actor-visible outcome labels and policy/value supervision, and leaves explicit questions for the next discussion.

**Tech Stack:** Markdown and self-contained HTML documentation.

## Global Constraints

- `0015` is a prospective design, not a training run or policy-strength claim.
- Keep deck-specific and expert/source-specific data isolated unless an auditable conditioning contract is separately approved.
- Use only action-time actor-visible input; terminal result is a supervision target, never a policy input feature.
- Do not modify `engine/source/`; policy strength remains an official-engine evaluation question.

---

### Task 1: Write the 0015 design and discussion agenda

**Files:**
- Create: `experiments/0015_outcome_aware_value_foundations/DESIGN.md`
- Create: `experiments/0015_outcome_aware_value_foundations/DESIGN.html`

**Interfaces:**
- Consumes: detailed rules research, project BC/RL conventions, and all-outcome episode provenance.
- Produces: a consistent prospective architecture, staged experiment matrix, measurement gates, and a design Q&A agenda.

- [ ] **Step 1: Specify non-negotiable data and visibility boundaries**

Document complete trajectories, actor-perspective terminal outcome, no future/outcome features, grouped splits, and per-deck/per-source isolation.

- [ ] **Step 2: Specify the first-stage multi-task hypothesis**

Document a shared actor-visible encoder with a policy head trained by BC and a value head trained from all outcomes; state that action-level blame requires later Q/rollout evidence.

- [ ] **Step 3: Specify ablations, evidence gates, and Q&A**

Document frozen controls, lambda ablations, calibration and official-engine gates, plus unanswered questions that require data audits or engine experiments.

- [ ] **Step 4: Verify rendered-document consistency**

Run: `python3 - <<'PY'\nfrom pathlib import Path\nfor path in (Path('experiments/0015_outcome_aware_value_foundations/DESIGN.md'), Path('experiments/0015_outcome_aware_value_foundations/DESIGN.html')):\n    assert path.is_file() and path.stat().st_size > 0, path\n    text = path.read_text(encoding='utf-8')\n    for phrase in ('0015', 'actor-visible', 'value', 'official-engine'):\n        assert phrase in text, (path, phrase)\nPY\ngit diff --check`

Expected: both design documents exist, contain the common project vocabulary, and have no whitespace errors.
