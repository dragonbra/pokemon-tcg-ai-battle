# PTCG Card Prototype Engine Semantics Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone HTML report that proves the official engine already contains a finite Card/Attack/Skill/Effect prototype system and shows how to compile it into low-noise, actor-visible model features.

**Architecture:** The report is a single dependency-free HTML document in `docs/reports/`. It uses CSS-native UML, effect-chain diagrams, source-evidence tables, and typed token examples; all claims are grounded in read-only official engine sources and the current 0022 feature implementation. No model, checkpoint, training configuration, or official engine source is changed.

**Tech Stack:** HTML5, responsive CSS Grid/Flexbox, dependency-free JavaScript for navigation highlighting and diagram focus.

## Global Constraints

- Do not modify `engine/source/`; cite it read-only.
- Keep the page usable by opening the HTML directly without a server.
- Distinguish official engine facts, current project implementation facts, and proposed model-design hypotheses.
- Show that the schema is finite and variable-length; do not imply one feature column per card.
- Keep hidden engine state out of actor-visible features; player-local knowledge remains a separate causal ledger.
- Preserve unrelated untracked and modified workspace files.

---

### Task 1: Evidence-backed card prototype report

**Files:**
- Create: `docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html`
- Read only: `engine/source/ptcgProgram 22/Card.h`
- Read only: `engine/source/ptcgProgram 22/Skill.h`
- Read only: `engine/source/ptcgProgram 22/Types.h`
- Read only: `engine/source/ptcgProgram 22/CreateCard.h`
- Read only: `engine/source/ptcgProgram 22/CardImpl.h`
- Read only: `engine/source/ptcgProgram 22/Api.h`
- Read only: `train/0022_league_training/foundation/model_source/card_features.py`

**Interfaces:**
- Consumes: Official `CardMaster`, `Attack`, `Skill`, `Effect`, `EffectType`, and public API field definitions.
- Produces: A standalone report with stable anchors `#prototype`, `#proof`, `#compiler`, `#features`, `#boundary`, and `#roadmap`.

- [x] **Step 1: Build the visual shell and executive conclusion**

Create the page header, sticky section navigation, evidence legend, and summary statement: the engine contains a finite execution IR, while the current feature table collapses per-move structure.

- [x] **Step 2: Draw the UML and finite-schema view**

Render `CardMaster -> Skill/Attack -> Effect -> TargetCondition` as connected CSS boxes. Show variable-length lists and explicit `OBSERVED / UNKNOWN / NOT_APPLICABLE / PADDING` states so absent fields are not conflated with numeric zero.

- [x] **Step 3: Add four exact engine proof chains**

Show Teal Mask Ogerpon ex, Alakazam, Mega Froslass ex, and Crispin using their actual IDs, costs, `EffectType`, targets, and numeric values from `CardImpl.h`. Label each chain as an official engine fact.

- [x] **Step 4: Add the model compilation pipeline**

Visualize `official tables -> static semantic sidecar -> actor-visible observation + causal ledger -> dynamic option evaluator -> semantic tokens -> encoder/decoder`. Explicitly prohibit hidden Prize, deck order, or opponent hand leakage.

- [x] **Step 5: Define finite token families and ownership boundaries**

List identity, timing, cost, operation, target, condition, numeric-expression, modifier, and knowledge-state token families. Separate deterministic preprocessing from learned strategy and preserve raw `cardId/attackId/skillId/effectType` embeddings as residual identity.

- [x] **Step 6: Add an incremental implementation roadmap**

Specify the first extractor as an external read-only tool that emits a hashed static sidecar, followed by action binding, dynamic evaluation, paired-state tests, and model ablations. State that engine source modification is out of scope.

- [x] **Step 7: Validate the artifact**

Run:

```bash
git diff --no-index --check /dev/null docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html
rg -n 'id="(prototype|proof|compiler|features|boundary|roadmap)"' docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html
```

Expected: no whitespace errors and all six required section IDs present.

Extract the inline script and parse it with Node:

```bash
sed -n '/^  <script>/,/^  <\/script>/p' docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html \
  | sed '1d;$d' \
  | node -e 'let s=""; process.stdin.on("data", c => s += c); process.stdin.on("end", () => new Function(s));'
```

Expected: exit status 0.

---

### Task 2: Interactive typed projection and attention-mask lab

**Files:**
- Modify: `docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html`

**Interfaces:**
- Consumes: The report's existing token families, Ogerpon/Alakazam engine examples, responsive CSS, and dependency-free navigation observer.
- Produces: A stable `#projection-lab` section whose controls update a fixed-width projection decomposition and a variable-length masked-attention view without external dependencies.

- [x] **Step 1: Add the typed projection explanation and controls**

Add two selectable examples, component-weight sliders, a fixed-width projected-vector bar, and a field-encoding table. Explicitly distinguish categorical embeddings, numeric scalar/MLP projections, Boolean value plus presence, entity references, and field-state embeddings.

- [x] **Step 2: Add the variable-length batching and attention-mask view**

Render Ogerpon and Alakazam as different-length token rows padded only after semantic token construction. Show a real query token's normalized attention over real keys, exact zero weight over PAD keys, the tensor shapes, and the `softmax(-infinity) = 0` rule.

- [x] **Step 3: Add dependency-free interaction and responsive behavior**

Wire example selection and range inputs to deterministic diagram updates. Keep diagram dimensions stable, support narrow screens, add the section to sticky navigation, and label all adjustable values as illustrative rather than learned checkpoint weights.

- [x] **Step 4: Validate the enhanced artifact**

Run whitespace validation, search for `projection-lab`, `attention_mask`, `OBSERVED_ZERO`, and `softmax`, parse inline JavaScript with Node, and verify balanced structural HTML tags.

---

### Task 3: Preserve-and-enrich model input/output contract

**Files:**
- Modify: `docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html`
- Read only: `train/0022_league_training/foundation/model_source/base_model.py`
- Read only: `train/0022_league_training/foundation/model_source/ac_model.py`
- Read only: `train/0022_league_training/foundation/model_source/features/compiler.py`
- Read only: `train/0022_league_training/foundation/model_source/knowledge/state.py`
- Read only: `engine/source/ptcgProgram 22/ApiJson.h`

**Interfaces:**
- Consumes: Current 0022 entity/global/ledger/event/option tensors, official observation option fields, and the proposed typed prototype tokens.
- Produces: A stable `#model-contract` section that distinguishes current facts from the proposed ideal contract and shows how raw observation, causal memory, exact-deck multiset, static prototypes, and action-specific derived values remain jointly available to encoder and decoder.

- [x] **Step 1: Document the current contract and exact information-loss point**

Show that current state encoding preserves entity, global, zone, ledger, event, known-hand, and registered-deck families, while current `option_cat[12]` keeps option type, areas, owner, source/target card and entity references, ordinal, and generic number but drops the official Attack option's `attackId`.

- [x] **Step 2: Draw the additive semantic join rather than replacement**

Visualize four retained memories: actor-visible dynamic observation, causal ledger/events, exact-deck multiset, and static Card/Attack/Skill/Effect prototypes. Join them by stable identity and runtime entity reference; explicitly state that prototype lookup enriches `cardId` and does not replace HP, damage, attachments, zones, turn budgets, or bounded knowledge.

- [x] **Step 3: Define the ideal option and decoder contract**

Specify an enriched option token with raw option identity, `attackId/skillId/effect context`, source/target entity references, ordered selection role, prototype subgraph, and deterministic current-state deltas such as typed energy deficit, resolved damage/counters, KO/prize change, zone transition, budget consumption, and terminal commitment. Show cross-attention from each option to the preserved memories before autoregressive pointer decoding.

- [x] **Step 4: Add migration and non-regression gates**

Require a parallel additive path with zero-gated initialization, old-feature parity, field-presence and ID-binding audits, no hidden-state leakage, fail-closed prototype coverage/truncation, paired-state sensitivity probes, and official-engine evaluations before any old path is retired.

- [x] **Step 5: Validate the extended artifact**

Run whitespace and anchor checks, parse inline JavaScript with Node, verify balanced structural tags, and search for `attackId`, `ledger`, `multiset`, `option_cat[12]`, and `zero-gated` in the new section.

---

### Task 4: Early-compression versus multi-memory retrieval explanation

**Files:**
- Modify: `docs/reports/ptcg-card-prototype-engine-semantics-2026-08-02.html`

**Interfaces:**
- Consumes: The `#model-contract` current/ideal architecture evidence and existing responsive visual components.
- Produces: A final subsection inside `#model-contract` that distinguishes current summary bottlenecks from option-conditioned retrieval across typed memories.

- [x] **Step 1: Visualize current early compression**

Show the current board entity memory as the relatively preserved path, registered deck/ledger compressed into four goal tokens, and zone/event/known-hand information merged into one scenario vector. Label missing `attackId` as loss rather than compression.

- [x] **Step 2: Visualize the proposed multi-memory query**

Show an enriched option independently querying board entity, prototype, deck/ledger, event/known-hand, and turn-budget memories, followed by gated composition with the raw option residual.

- [x] **Step 3: Explain the three retained information scales**

Separate global summaries for overall phase and decoder initialization, typed memories for exact option-conditioned retrieval, and direct deterministic option features for resolved current consequences. State that summaries remain useful and are not removed.

- [x] **Step 4: Validate the subsection**

Run whitespace and structural-tag checks, parse inline JavaScript with Node, and search for `过早压缩`, `Multi-memory`, `Global summaries`, `Typed memories`, and `Direct option features`.
