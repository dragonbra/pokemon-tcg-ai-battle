# 0042 Operations Manual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained runbook that takes a newly cloned 0042 workspace from environment validation through smoke, formal PPO, Frozen evaluation, monitoring, and safe shutdown.

**Architecture:** Keep the canonical policy and project contracts normative; the new document translates them into ordered operator actions without duplicating authority. Explain CPU/CUDA roles, immutable Policy-0809 opponent materialization, per-lane exact-deck routing, candidate deployment conversion, and PPO Protocol V2 beside the exact commands that exercise those boundaries.

**Tech Stack:** Markdown, Python 3.11, PyTorch/CUDA, CMake/Ninja, official seeded engine runtime, W&B, unittest.

## Global Constraints

- Do not modify `engine/source/`; all official-engine builds go below `engine/build/`.
- 0042 is Policy-0809-only and must obey both mandatory Frozen-policy contracts.
- Formal PPO requires explicit human authorization and the `--launch-formal` token.
- Formal candidate evidence uses FP16 storage followed by strict FP32 runtime materialization.
- Model-only checkpoints retain every update; versions and formal reports are immutable.

---

### Task 1: Write and link the operator runbook

**Files:**
- Create: `docs/rl/0042_full_model_design_operations_manual.md`
- Modify: `docs/rl/0042_full_model_design.md`

**Interfaces:**
- Consumes: canonical Frozen protocol, 0042 contract, current runner CLIs, model registry, league catalog, CUDA-engine guide, and experiment manifest.
- Produces: one ordered clone-to-run procedure and a discoverable link from the existing 0042 onboarding page.

- [x] **Step 1: Document authority, status, architecture, and evidence boundaries**

State which files are normative, identify the current preflight/smoke status, and separate official CPU semantics from CUDA acceleration.

- [x] **Step 2: Document environment and immutable-asset verification**

Provide commands for Python/CUDA/W&B checks and SHA-256 verification of the actor, Value, allocation sidecar, candidate base, rule pack, and extension.

- [x] **Step 3: Document staged execution**

Provide exact commands for focused tests, identity checks, Protocol V2 preflight, isolated smoke, formal launch, watchdog operation, safe stop, candidate evaluation, and export.

- [x] **Step 4: Explain Frozen, opponent weighting, and special PPO behavior**

Tie each semantic claim to the actual registry, 55-deck/256-slot catalog, lane routing, candidate materialization, evaluation cadence, checkpoint retention, and W&B ordering.

- [x] **Step 5: Add troubleshooting and anti-footgun guidance**

List fail-closed symptoms and the prohibited legacy evaluators, version reuse, raw-FP32 strength claims, focal/opponent sharing, and first-lane deck-feature reuse.

### Task 2: Validate the documentation against the repository

**Files:**
- Test: `docs/rl/0042_full_model_design_operations_manual.md`

**Interfaces:**
- Consumes: the runbook from Task 1 and current repository files.
- Produces: path, command, hash, and coverage checks with no source-code behavior changes.

- [x] **Step 1: Check every referenced repository path exists**

Run a scripted extraction/check over the manual's repository-relative code spans and manually inspect intentional runtime-created paths.

- [x] **Step 2: Check every documented Python entry point exposes help**

Run `python3 -m <module> --help` for non-destructive CLIs referenced by the runbook.

- [x] **Step 3: Run focused 0042 contract tests**

Run the policy identity, Frozen contract, candidate deployment, Protocol V2, smoke diagnostic, and project identity tests.

- [x] **Step 4: Scan for stale identities and placeholders**

Confirm the operational path names only `Policy-0809`/Frozen-0809, labels legacy 0806 references as history, and contains no unfinished placeholder language.

- [x] **Step 5: Review the final diff**

Confirm only documentation changed and that the authoritative DESIGN files need no synchronization because model/training semantics did not change.
