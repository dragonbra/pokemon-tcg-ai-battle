# 0042 update040 Kaggle Replay Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a root-level HTML report, modeled on the existing Lucario replay audit, that analyzes selected official public Episodes of the 0042 update040 Dragapult agent at deck-specific tactical depth.

**Architecture:** Resolve the exact Kaggle submission from the authenticated team submission history, freeze its current public Episode metadata, and download the relevant official replay JSON into a git-ignored evidence directory. Derive every tactical claim from the learner-visible action-time state and current engine/card semantics, then render a self-contained HTML report with embedded machine-readable provenance and validate its links, counts, hashes, and markup.

**Tech Stack:** Kaggle Python API, repository card catalog, official Episode replay JSON, Python/JSON audit helpers, static HTML/CSS.

## Global Constraints

- Do not submit, retry a submission, upload data, or perform any other externally visible Kaggle mutation.
- Do not modify `engine/source/` or infer strength from static analysis alone.
- Treat the report as a public replay audit, not Frozen evaluation or Promote Champion evidence.
- Use the exact 0042 update040 FP16-storage/FP32-runtime submission identity and keep focal/opponent identities distinct.
- Judge decisions only from information visible to the 0042 agent at that decision; label counterfactual claims that require an official-engine fork.
- Preserve all unrelated worktree changes.

---

### Task 1: Freeze the exact submission and evidence set

**Files:**
- Read: `archive/submission/0042_dragapult_ex_007_policy0809_rl_u040_fp16_storage_fp32_runtime/manifest.json`
- Create: `.tmp/evaluation/0042_u040_kaggle_replay_audit/submission.json`
- Create: `.tmp/evaluation/0042_u040_kaggle_replay_audit/episodes.json`
- Create: `.tmp/evaluation/0042_u040_kaggle_replay_audit/replays/episode-<id>-replay.json`

**Interfaces:**
- Consumes: authenticated Kaggle team submissions and public Episode endpoints.
- Produces: immutable local submission metadata, Episode metadata, replay files, and SHA-256 values.

- [ ] Resolve the unique submission whose filename/message identifies 0042 update040 and confirm its package/deck identity against the archive manifest.
- [ ] List only `PUBLIC + COMPLETED` Episodes containing that exact submission ID and save the frozen metadata snapshot.
- [ ] Select a representative set that covers losses and diagnostically useful wins rather than outcome cherry-picking.
- [ ] Download each selected official replay and record its SHA-256.

### Task 2: Extract Dragapult-specific action timelines

**Files:**
- Read: `.tmp/evaluation/0042_u040_kaggle_replay_audit/replays/*.json`
- Read: `data/official/EN_Card_Data.csv`
- Create: `.tmp/evaluation/0042_u040_kaggle_replay_audit/analysis.json`

**Interfaces:**
- Consumes: selected replay JSON, exact deck, card names/text, and action-time observations.
- Produces: per-Episode timelines, outcome facts, focal seat, opponent archetype, and bounded tactical findings.

- [ ] Decode replay steps into focal choices and visible board/resource transitions.
- [ ] Audit Dreepy→Drakloak→Dragapult evolution timing, energy readiness, Phantom Dive damage-counter allocation, gust/retreat plans, Bench occupancy, Prize map, and next-attacker continuity.
- [ ] Separate deterministic replay facts from plausible alternatives and official-engine counterfactuals.
- [ ] Aggregate repeated symptoms without presenting the selected sample as an overall rate.

### Task 3: Render the root HTML report

**Files:**
- Create: `kaggle_0042_update040_dragapult_replay_audit_2026-08-12.html`

**Interfaces:**
- Consumes: `analysis.json`, the reference report's information architecture, and exact package metadata.
- Produces: a responsive, standalone Chinese HTML audit with embedded `report-data` JSON.

- [ ] Reuse the reference report's visual language, deck overview, snapshot summary, cross-game diagnosis, per-Episode timelines, state judgment, next gates, and evidence boundary.
- [ ] Make every diagnosis specific to Dragapult's Stage-2 setup, mixed Psychic/Fire energy, Phantom Dive active damage plus Bench counters, single-/multi-Prize exposure, and Dusknoir/utility-Pokémon interactions actually present in the replay.
- [ ] Link each audited Episode to its local raw replay evidence and embed complete submission/replay hashes and claim boundaries.

### Task 4: Validate the deliverable

**Files:**
- Test: `kaggle_0042_update040_dragapult_replay_audit_2026-08-12.html`
- Test: `.tmp/evaluation/0042_u040_kaggle_replay_audit/analysis.json`

**Interfaces:**
- Consumes: final HTML and frozen evidence.
- Produces: validation output proving report/evidence consistency.

- [ ] Parse the HTML and embedded JSON; assert report ID, submission ID, update 40, audited Episode count, unique Episode IDs, and replay SHA-256 matches.
- [ ] Assert the exact deck has 60 cards and all local replay links resolve from the root report.
- [ ] Search for unsupported Frozen/Promote claims, hidden-information leakage, placeholder text, and broken section IDs.
- [ ] Inspect the rendered page or a screenshot at desktop width for layout regressions.
