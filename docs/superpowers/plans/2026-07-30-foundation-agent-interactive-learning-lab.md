# Foundation Agent Interactive Learning Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing evidence-heavy Foundation Agent report into a visual, interactive lesson that a reader who only knows activation functions can use to understand what the model is and how BC, RL, Engine dynamics, and Arena evolution change it.

**Architecture:** Keep the existing standalone HTML and its evidence sections as the canonical visual report. Add dependency-free teaching labs whose JavaScript computes illustrative logits, softmax probabilities, freeze-boundary tradeoffs, temporal credit weights, and Arena generations; label every synthetic quantity as a teaching model rather than measured policy output.

**Tech Stack:** Semantic HTML5, responsive CSS, dependency-free JavaScript, Chrome DevTools Protocol validation.

## Global Constraints

- Do not modify `engine/source/`, training code, checkpoints, or concurrent 0020 assets.
- Preserve all verified 0019/0018/0020 numbers and their evidence boundaries.
- Keep the report Chinese-first and directly openable without a server or external CDN.
- Use stable responsive dimensions and prevent page-level horizontal overflow at 1440px and 390px.
- Every interactive control must have a visible value, accessible label, keyboard support, and deterministic output.
- Illustrative equations and weights must say they are teaching examples, not extracted model parameters.

---

### Task 1: Beginner Mental Model

**Files:**
- Modify: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.html`

**Interfaces:**
- Consumes: existing report hero, verified evidence metrics, and current 0019 architecture section.
- Produces: `#essence` and `#decision-lab` sections that bridge activation functions to a complete policy.

- [x] **Step 1: Add the activation-to-policy visual ladder**

Render five connected modules: raw fact, embedding, attention/memory, decoder logit, and softmax choice. Each module must state its input, output, and trainable quantity.

- [x] **Step 2: Add a representation-depth slider**

Use `#representation-depth` to select raw IDs, entity meaning, board relations, goal-conditioned options, and action distribution. Update the central vector, question, and output text without changing layout dimensions.

- [x] **Step 3: Add the decision microscope**

Add four state-evidence sliders plus one decoder-nudge slider. Implement `updateDecisionLab()` with explicit toy logits, numerically stable softmax, probability bars, and selected-action highlighting.

- [x] **Step 4: Label the teaching boundary**

Place a persistent notice beside the formulas: values explain mechanics and are not inference from `0019-0730-epoch13`.

### Task 2: Training Mechanics Labs

**Files:**
- Modify: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.html`

**Interfaces:**
- Consumes: existing freeze/train matrix and staged Engine → BC → RL narrative.
- Produces: `#freeze-lab` and `#credit-lab` controls with deterministic derived visuals.

- [x] **Step 1: Add the freeze-depth lab**

Use a five-position slider for value-only, current Decoder + Value, adapters, last semantic/scenario layer, and full model. Update trained/frozen layer colors, known parameter counts, plasticity, forgetting risk, and appropriate use.

- [x] **Step 2: Add the terminal-credit lab**

Use a lambda slider and six-decision timeline. Compute each earlier decision's teaching credit as `(0.99 * lambda) ** distance`, render magnitude bars, and explain that real GAE also subtracts a learned value baseline.

- [x] **Step 3: Integrate with existing phase tabs**

Keep current Engine Physics, Universal BC, Value, Deck RL, and Consolidation tabs; ensure labs reinforce rather than contradict their freeze boundaries.

### Task 3: Arena Evolution and Verification

**Files:**
- Modify: `docs/reports/engine_to_foundation_agent_roadmap_2026-07-30.html`

**Interfaces:**
- Consumes: existing Arena branches and G0 → G1 consolidation description.
- Produces: generation slider, structural audit evidence, and rendered screenshots.

- [x] **Step 1: Add an Arena generation slider**

Render G0 foundation, parallel deck branches, evaluation, and G1 consolidation across deterministic generation states. Do not imply that generation count is a measured experiment.

- [x] **Step 2: Add keyboard and ARIA behavior**

Associate every range input with a label/output and keep tab left/right navigation. Make changing a slider announce the derived value through an `aria-live` summary.

- [x] **Step 3: Run structural and interaction audits**

Parse HTML, verify unique IDs/local links/ARIA targets, use Chrome Runtime evaluation to move every slider and assert finite outputs and one selected action.

- [x] **Step 4: Render and inspect desktop/mobile pages**

Capture 1440×1000 and emulated 390×844 views plus full pages. Assert `scrollWidth == clientWidth` for all interaction states and inspect screenshots for clipping or overlap.

- [x] **Step 5: Run repository hygiene checks**

Run `git diff --check` and confirm this task changes only the visual report and this plan; preserve unrelated 0020 worktree changes.
