# Engine-to-Foundation Agent Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rigorous Chinese Markdown report and a responsive visual HTML report explaining the complete path from official-engine dynamics to universal BC, deck-specific RL, and a scalable evolutionary Arena.

**Architecture:** The Markdown document is the maintainable factual narrative. The standalone HTML mirrors the same evidence and adds interactive phase navigation, architecture diagrams, a freeze/train matrix, mathematical objectives, evidence boundaries, and a generational Arena loop. Both artifacts are grounded in the frozen 0019 Epoch 13 and completed 0018 PPO contracts without changing either project.

**Tech Stack:** Markdown, semantic HTML5, standalone CSS, small dependency-free JavaScript, official project JSON/design evidence.

## Global Constraints

- Do not modify `engine/source/`; the official runtime remains the only source of legal transitions and terminal outcomes.
- Distinguish official rules, current runtime/model facts, and future research hypotheses.
- Preserve the exact 0019 Epoch 13 identity and current model dimensions.
- Treat legal action generation as a hard action contract, not an RL reward.
- Do not claim BC exact-action, rollout win rate, or a single zero-shot run proves global policy optimality.
- Keep this task documentation-only and do not touch concurrent 0020/CUDA assets.

---

### Task 1: Evidence-backed narrative

**Files:**
- Create: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.md`

**Interfaces:**
- Consumes: `experiments/0019_universal_winner_bc/DESIGN.md`, `experiments/0018_alakazam_terminal_rl/DESIGN.md`, `experiments/0020_pluggable_deck_rl/evaluation/V2_zero_shot_dragapult.html`, and the rules research document.
- Produces: The canonical conceptual narrative mirrored by the visual report.

- [x] **Step 1: Write the evidence boundary and executive thesis**

Record official rule constraints, runtime facts, current experiment evidence, and unproven research hypotheses as separate categories.

- [x] **Step 2: Document the current 0019 architecture**

Describe state/entity/deck/event memories, option-as-query cross-attention, hidden-as-query pointer decoding, STOP, and the value head boundary with exact shapes and parameter counts.

- [x] **Step 3: Document the staged learning path**

Define dynamics pretraining, optional universal winner BC, value calibration, deck-specific RL, progressive adapters/unfreezing, and generational consolidation.

- [x] **Step 4: Add mathematical objectives and limitations**

Provide explicit transition, BC, PPO/value, KL/distillation, and consolidation objectives. State why dynamics alone does not imply strategy and why self-play does not guarantee a unique global optimum.

### Task 2: Visual standalone report

**Files:**
- Create: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.html`

**Interfaces:**
- Consumes: The Task 1 narrative and the same immutable experiment evidence.
- Produces: A responsive, dependency-free report that opens directly in a browser.

- [x] **Step 1: Build the report shell and navigation**

Create semantic sections, a sticky phase navigator, evidence badges, accessible segmented phase controls, and responsive layout constraints.

- [x] **Step 2: Build the current-model architecture visualization**

Show official observation and legal options flowing through shared memories, option-as-Q attention, pointer decoding, STOP, and value prediction. Explicitly show both Q/K orientations.

- [x] **Step 3: Build the freeze/train and learning-stage visualizations**

Render a module-by-stage matrix and interactive phase details for engine physics, BC, deck RL, and Arena consolidation.

- [x] **Step 4: Build the mathematical and scale-up sections**

Visualize loss composition, stability/plasticity controls, GPU sharing, deck adapters, league snapshots, and the Foundation G0 to G1 loop.

### Task 3: Contract and presentation verification

**Files:**
- Verify: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.md`
- Verify: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.html`

**Interfaces:**
- Consumes: Both finished artifacts.
- Produces: Syntax, factual, responsive-layout, and interaction evidence.

- [x] **Step 1: Run factual token checks**

Check both formats contain `17,756,162`, `81.2871%`, `42.33%`, `source_id=0`, option shape `[B,O,320]`, official-engine evidence boundaries, and all five learning stages.

- [x] **Step 2: Run HTML structural checks**

Parse the HTML, verify unique section IDs, navigation targets, no missing local links, and no horizontal overflow at desktop and mobile widths.

- [x] **Step 3: Render desktop and mobile screenshots**

Open the static HTML in Chromium, exercise the phase controls, and inspect screenshots for clipping, overlap, and readable diagrams.

- [x] **Step 4: Run repository hygiene checks**

Run `git diff --check` and confirm only the plan and two report files belong to this task.
