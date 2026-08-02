# Team Formation Meeting Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, projector-ready Chinese HTML presentation for the 2026-08-03 Kaggle team formation meeting.

**Architecture:** One standalone HTML file owns the presentation content, responsive styling, keyboard navigation, print layout, and lightweight interaction. All quantitative claims are traced to repository evidence or visibly marked as user-provided/unverified; no engine or training code changes are required.

**Tech Stack:** Semantic HTML5, CSS, vanilla JavaScript, repository Markdown/JSON evidence, remote card artwork with graceful fallbacks.

## Global Constraints

- Do not modify `engine/source/`.
- Distinguish official rules, official-engine measurements, offline diagnostics, Kaggle observations, and project hypotheses.
- Do not present BC exact-action or sampled rollout rates as policy strength.
- Keep the presentation usable as a directly opened local HTML file and printable to PDF.
- Preserve the user's existing worktree changes.

---

### Task 1: Evidence-backed narrative

**Files:**
- Create: `docs/reports/2026-08-03-team-formation-meeting.html`

**Interfaces:**
- Consumes: competition links, submission receipts, experiment designs, official-engine evaluation summaries, and the rules research note.
- Produces: slide copy with inline evidence labels and a final source index.

- [ ] **Step 1: Freeze the evidence ladder**

Use only the following comparable or explicitly separated facts: `983.6` Kaggle public score from 0014; Dragapult BC `41/260` versus RL U15 `68/260`; Alakazam U39 `235/300`; Universal Foundation zero-shot Dragapult `127/300`; 0019 corpus/model scale; and user-provided team/deadline assumptions.

- [ ] **Step 2: Write the meeting sequence**

Cover formal team formation, competition versus solution write-up, iteration history, Transformer mechanics, current architecture and missing pieces, scale-up thesis, four workstreams, operating model, money/NDA principles, merge gate, and immediate actions.

- [ ] **Step 3: Add evidence boundaries**

Label every headline number as Kaggle observation, official-engine evaluation, offline BC diagnostic, repository fact, project hypothesis, or user-provided item needing confirmation.

### Task 2: Presentation UI and visualizations

**Files:**
- Create: `docs/reports/2026-08-03-team-formation-meeting.html`

**Interfaces:**
- Consumes: the narrative sections from Task 1.
- Produces: a responsive 16:9 slide deck with navigation, diagrams, charts, and print output.

- [ ] **Step 1: Build stable slide layout**

Create full-viewport slides with bounded content, readable projector typography, a progress rail, slide counter, overview navigation, and mobile fallback.

- [ ] **Step 2: Build visual explanations**

Use CSS diagrams for the milestone expansion map, evidence ladder, RL before/after bars, Transformer residual block, Q/K/V flow, FFN versus MoE routing, current model data flow, Foundation-to-League flywheel, four-person responsibility matrix, repository ownership map, and submission gate.

- [ ] **Step 3: Add controls**

Support Arrow/PageUp/PageDown/Space/Home/End keyboard navigation, clickable rail navigation, presenter-note toggling, fullscreen, and URL hash restoration.

- [ ] **Step 4: Add print mode**

Render one slide per printed page, hide presentation controls, and preserve evidence notes.

### Task 3: Verification

**Files:**
- Verify: `docs/reports/2026-08-03-team-formation-meeting.html`

**Interfaces:**
- Consumes: completed HTML from Tasks 1 and 2.
- Produces: validated local deliverable and review screenshots.

- [ ] **Step 1: Validate structure and claims**

Check slide count, duplicate IDs, source anchors, core figures, missing placeholders, and that unverified deadline/submission-limit claims are visibly flagged.

- [ ] **Step 2: Exercise interaction**

Open the local file in a browser, navigate by keyboard and controls, toggle notes/overview/fullscreen, and verify the URL hash updates.

- [ ] **Step 3: Review desktop and mobile screenshots**

Capture representative slides at desktop and mobile sizes and fix clipping, overlap, low contrast, or unreadable diagrams.

