# Official Engine Source Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evidence-backed, browsable guide to the official PTCG engine's directory structure, game/state ownership, card prototype representation, and state-transition pipeline.

**Architecture:** Create a self-contained documentation mini-project under `docs/engine-source-guide/` with a detailed Markdown reference and a dependency-free interactive HTML map. Every architectural claim links to read-only `engine/source/ptcgProgram 22/` evidence, and uncertain interpretations are explicitly separated from verified implementation facts.

**Tech Stack:** Markdown, HTML5, CSS, dependency-free JavaScript, shell-based source and link validation.

## Global Constraints

- Never modify, format, or patch `engine/source/`.
- Distinguish official game rules, official engine implementation facts, and architectural interpretation.
- Preserve all unrelated tracked and untracked workspace changes.
- The HTML guide must work by opening the file directly, without a web server.
- Cite exact source files and line numbers for the main ownership and transition claims.

---

### Task 1: Source architecture audit

**Files:**
- Read only: `engine/source/ptcgProgram 22/*.h`
- Read only: `engine/source/ptcgProgram 22/Export.cpp`
- Create: `docs/engine-source-guide/README.md`

**Interfaces:**
- Consumes: Official engine declarations, inline implementations, exported API, and JSON serialization boundary.
- Produces: A source-indexed reference covering modules, ownership, data structures, and call paths.

- [x] **Step 1: Inventory translation units and dependency aggregation**

Trace `Export.cpp`, `All.h`, `Framework.h`, `Core.h`, and `InitializeCard.h`; record which headers provide data types, tables, setup, gameplay, effects, selection, and API serialization.

- [x] **Step 2: Recover the runtime ownership model**

Trace `Game`, `State`, `PlayerState`, `Card`, `CardMaster`, and the master tables. Document which values are immutable prototypes, mutable card instances, per-player zones, per-turn flags, and transient resolution context.

- [x] **Step 3: Recover the card execution IR**

Trace `CardMaster`, `Skill`, `Attack`, `Effect`, `Target`, `TargetCondition`, `Trigger`, `Chain`, and representative `CardImpl.h` definitions. Explain exactly where generic typed effects end and card-specific callbacks begin.

- [x] **Step 4: Recover the state-transition machinery**

Trace `SetupProc`, `GameProc`, `SelectProc`, `EffectProc`, `EffectInstant`, `EffectContinual`, `PullTrigger`, `SatisfyCondition`, `TargetList`, `CardMove`, and `SetProperty`. Document phase transitions, option generation, selection continuations, effect-stack processing, triggers, and win checks.

- [x] **Step 5: Recover the external boundary**

Trace `Export.cpp`, `Api`, `ApiData`, `ApiJson`, `ToJson`, and `Binary`. Explain initialization, action submission, serialization, public observation, and the boundary between engine truth and actor-visible JSON.

- [x] **Step 6: Write the Markdown reference**

Write `README.md` with a directory map, ownership diagrams, end-to-end timelines, representative card walkthroughs, glossary, evidence boundary, and source index. Include explicit answers to whether damage and other card semantics are coefficient-driven, enum-driven, callback-driven, or hard-coded.

### Task 2: Interactive source guide

**Files:**
- Create: `docs/engine-source-guide/index.html`
- Read: `docs/engine-source-guide/README.md`

**Interfaces:**
- Consumes: The audited facts and diagrams from Task 1.
- Produces: A standalone visual map with stable section anchors and source links.

- [x] **Step 1: Build the responsive shell and evidence legend**

Add compact navigation and clearly label official engine facts, official rules context, and architectural interpretation.

- [x] **Step 2: Render the module and ownership maps**

Show the source directory by responsibility and visualize `Game -> State -> PlayerState -> Card` alongside immutable `CardMaster/Attack/Skill` tables.

- [x] **Step 3: Render the card and transition walkthroughs**

Show one attack and one trainer/ability path from builder declaration through selection, effect execution, mutation, trigger processing, and JSON output.

- [x] **Step 4: Add source drill-down interactions**

Provide accessible tabs or filters for architecture layers and make every central claim link to a local source file.

### Task 3: Artifact verification

**Files:**
- Test: `docs/engine-source-guide/README.md`
- Test: `docs/engine-source-guide/index.html`

**Interfaces:**
- Consumes: Completed guide artifacts.
- Produces: Syntax, anchor, source-link, and factual spot-check results.

- [x] **Step 1: Validate text and HTML structure**

Run `git diff --check`, parse the inline JavaScript with Node, and verify required section IDs `overview`, `modules`, `ownership`, `cards`, `transitions`, `effects`, `api`, and `sources`.

- [x] **Step 2: Validate local links and cited line numbers**

Resolve each relative source link, confirm referenced files exist, and spot-check cited line ranges against current source declarations.

- [ ] **Step 3: Inspect desktop and mobile rendering**

Open the standalone page at desktop and mobile widths, verify no overlap or clipped text, and confirm navigation and source drill-down behavior.

Blocked in the current environment: no browser runtime was installed. A temporary Playwright Chromium install reached 100% download but hung before cache installation and was terminated. Responsive CSS, overflow, wrapping, anchors, and JavaScript were checked statically; pixel-level screenshot verification remains outstanding.

- [x] **Step 4: Confirm source immutability**

Use `git status --short engine/source` and a before/after source hash inventory to confirm the official engine remained untouched.
