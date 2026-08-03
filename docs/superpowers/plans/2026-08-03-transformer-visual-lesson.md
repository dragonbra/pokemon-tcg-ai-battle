# Transformer Visual Lesson Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beginner-friendly, interactive Chinese HTML lesson that teaches Transformer fundamentals from activation functions through the repository's 0028 semantic policy.

**Architecture:** Use one dependency-free HTML lesson containing semantic markup, responsive CSS, and small vanilla-JavaScript simulations. Add a lightweight lessons index and link it from the documentation root; do not alter model code or authoritative design documents because the lesson only explains current behavior.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, Python standard-library validation, Node.js syntax checking.

## Global Constraints

- Keep the lesson usable by a learner who only knows activation functions.
- Clearly separate official game rules, current runtime/card facts, and project hypotheses.
- Reflect 0028's implemented defaults: `d_model=320`, 8 heads, 4 state layers, 3 option layers, FFN width 960, GELU, pre-norm, state cross-attention, and a GRU pointer decoder with STOP.
- Keep the page responsive and keyboard accessible, with reduced-motion support and no build step.
- Do not modify `engine/source/`, training semantics, model code, or DESIGN documents.

---

### Task 1: Build the interactive course

**Files:**
- Create: `docs/lessons/transformer-from-zero/index.html`

**Interfaces:**
- Consumes: 0028 `ModelConfig`, `StateEncoder`, `OptionEncoder`, `ActionDecoder`, and documented Pokemon TCG rule boundaries.
- Produces: a standalone browser lesson with section navigation, progress state, attention simulations, project architecture explorer, and knowledge checks.

- [x] **Step 1: Create semantic lesson structure**

Add a first-viewport course dashboard and ordered chapters for prerequisites, embeddings, similarity, Q/K/V, softmax, multi-head attention, Transformer blocks, masks, 0028 architecture, BC training, and a final challenge.

- [x] **Step 2: Implement the visual system**

Add a restrained light classroom palette, stable responsive diagrams, accessible controls, card imagery with fallbacks, focus states, and reduced-motion handling.

- [x] **Step 3: Implement interactions**

Use vanilla JavaScript for chapter progress, vector sliders, live dot-product and softmax calculations, attention focus changes, head switching, residual-path toggles, mask demonstrations, architecture detail panels, pointer selection, and quiz feedback.

- [x] **Step 4: Verify teaching and project facts**

Cross-check every 0028 width/layer/head statement against model source and label official rules, card/runtime facts, and training hypotheses separately.

### Task 2: Add documentation navigation

**Files:**
- Create: `docs/lessons/index.html`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: the course path from Task 1.
- Produces: discoverable links from the documentation root to the lessons catalog and Transformer course.

- [x] **Step 1: Create the lessons catalog**

Add a compact course listing with audience, duration, chapter count, and a direct start link.

- [x] **Step 2: Link the catalog from docs root**

Add `Interactive lessons` to `docs/README.md` without changing unrelated navigation.

### Task 3: Validate behavior and layout

**Files:**
- Test: `docs/lessons/transformer-from-zero/index.html`
- Test: `docs/lessons/index.html`

**Interfaces:**
- Consumes: both completed HTML pages.
- Produces: syntax, link, required-section, responsive-layout, and interaction evidence.

- [x] **Step 1: Run static HTML contract checks**

Use Python's `html.parser` to assert one title/H1, required section IDs, controls with labels, and valid local links.

- [x] **Step 2: Run JavaScript syntax checking**

Extract the inline script and run `node --check` on a temporary `.js` file.

- [x] **Step 3: Run browser checks when available**

Serve `docs/` locally, inspect desktop and mobile screenshots, exercise the Q/K/V and pointer interactions, and fix overflow or overlap.

- [x] **Step 4: Review the diff**

Confirm only lesson files, the docs navigation, and this plan changed; preserve all unrelated worktree changes.
