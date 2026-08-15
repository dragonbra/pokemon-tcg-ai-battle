# Storage Retention Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim approximately 65 GiB while preserving current 0045 evidence, immutable policy identities, formal reports, selected checkpoints, and runtime packages.

**Architecture:** Perform three independently audited cleanups: reconstructable evaluation materializations, historical rolling checkpoints, and disposable `.tmp` roots. Build exact path lists with protected-asset gates, record count/bytes before deletion, delete only the validated lists, and verify retained identities/reports afterward.

**Tech Stack:** Python `pathlib` for validated inventory and deletion, repository JSON manifests, SHA-256 identity records, `du`/`df` verification.

## Global Constraints

- Do not modify or delete `engine/source/`.
- Preserve all promoted/frozen policy definitions and effective runtime assets.
- Preserve `archive/submission/`, formal evaluation HTML/JSON, W&B/TensorBoard metrics, and 0045 U40/U90/U95/U100–U200 plus Deck-023 U105 evidence.
- Evaluation cleanup deletes only reconstructable materialized model payloads; lightweight reports and identity hashes remain.
- Checkpoint cleanup protects each version's terminal checkpoint, explicitly named best/latest/final/EMA/Champion files, manifest-referenced checkpoints, and current 0045 evaluated checkpoints.
- Deletion targets must be explicit validated paths beneath the expected repository roots.

---

### Task 1: Evaluation materialization retention

**Files:**
- Delete selectively: `rl_runs/**/materialization/model.bin`
- Preserve: `rl_runs/**/report.json`, `experiments/**`, `archive/**`

**Interfaces:**
- Consumes: evaluation materialization inventory and protected current-evidence paths.
- Produces: at least 25 GiB reclaimed with reports and identity metadata intact.

- [ ] Inventory all materialized model payloads and sort oldest first.
- [ ] Exclude protected 0045 formal U40/U90/U95/U100–U200 materializations.
- [ ] Select the oldest unprotected files until the target is reached.
- [ ] Delete only the validated selection and verify report/HTML survival.

### Task 2: Historical rolling checkpoint retention

**Files:**
- Delete selectively: historical `rl_runs/**/checkpoint/update-*.pt`
- Preserve: current evaluated 0045 checkpoints and manifest-selected checkpoints.

**Interfaces:**
- Consumes: manifest path references, per-version terminal checkpoints, semantic filename protections, and current 0045 evaluation identities.
- Produces: approximately 30 GiB reclaimed without breaking registered policies or current evidence.

- [ ] Build the protected checkpoint set from manifests and version terminal points.
- [ ] Rank unprotected rolling checkpoints from the earliest project/version/update first, prioritizing non-milestone updates.
- [ ] Select up to the requested target and emit an exact count/byte audit.
- [ ] Delete only the validated selection and verify all protected paths remain.

### Task 3: Disposable temporary assets

**Files:**
- Delete selectively: `.tmp/evaluation/<old-purpose>/`, `.tmp/environment_daily/<old-date>/`, and other old top-level temporary roots.
- Preserve: `.tmp/evaluation/jijij_023_cuda2048/` until its current evidence has been fully verified.

**Interfaces:**
- Consumes: top-level temporary-directory size and modification-time inventory.
- Produces: at least 10 GiB reclaimed from non-authoritative assets.

- [ ] Rank temporary roots by age and size while excluding the current 023 audit.
- [ ] Select old completed evaluation/package-parity/export/daily-generation roots to reach the target.
- [ ] Delete only validated temporary roots.
- [ ] Verify formal reports and archives remain available.

### Task 4: Final integrity and decision record

**Files:**
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Consumes: before/after byte counts and protected-path verification.
- Produces: durable audit of what was removed, retained, and reclaimed.

- [ ] Recompute repository and category sizes.
- [ ] Verify current 0045 checkpoints, Kaggle archives, formal HTML, report JSON, and policy registry assets.
- [ ] Record exact deleted counts/bytes and evidence boundaries in the decision log.
- [ ] Run `git diff --check` and report final free-space change.
