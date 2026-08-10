# Policy Batch Divergence Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify the eight existing production action divergences as numerical near-ties or batch-dependent policy implementation errors.

**Architecture:** Re-run only the existing fixed 256-seed cohort to capture the exact routed mixed policy batches for the eight known non-exact games. Remove the environment, replay each captured row as batch-1, homogeneous-256, and its original mixed batch, then compare stage outputs and legal logits.

**Tech Stack:** Python 3.11, PyTorch CUDA FP32, existing update-270 and Policy-0806 models, JSON/PT temporary artifacts.

## Global Constraints

- Do not add new games or seeds; only replay the existing 256-game cohort for missing tensors.
- Do not modify engine, model, routing, precision, or policy checkpoints.
- Do not print or persist full network activations; retain only input batches and compact numeric summaries.
- Diagnose all eight first divergences before classifying the failure mode.

---

### Task 1: Capture Existing Divergence Inputs

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/policy_batch_diagnostic/run.py`
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/policy_batch_diagnostic/captures.pt`

**Interfaces:**
- Consumes: prior fixed seed manifest, eight failed game indices, canonical batch-1 references, and production router callbacks.
- Produces: exact target row plus original mixed router batch for every first mismatch.

- [ ] Rebuild canonical references for only the eight existing failed seeds.
- [ ] Replay the same fixed 256 CUDA cohort once.
- [ ] Capture the first routed action mismatch for each failed job without changing actions.
- [ ] Verify captured observation/legal hashes and actions against prior evidence.

### Task 2: Offline Batch and Layer Comparison

**Files:**
- Create: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/policy_batch_diagnostic/report.json`

**Interfaces:**
- Consumes: Task 1 captures.
- Produces: per-stage diffs, raw/masked legal logits, margins, argmax, and per-seed classification.

- [ ] Run target row alone as batch-1.
- [ ] Repeat the row 256 times and verify homogeneous lane equality.
- [ ] Run the original mixed routed batch and select the original target row.
- [ ] Compare state trunk, option encoder/LoRA, decoder hidden/raw logits, legal mask, masked logits, and argmax.
- [ ] Record FP32/TF32/autocast state and the actual mixed model batch size.

### Task 3: Classification

**Files:**
- Modify: `.tmp/evaluation/0040_cpu_cuda_full0806_parity/policy_batch_diagnostic/report.json`

**Interfaces:**
- Consumes: eight per-seed comparisons.
- Produces: one of `NUMERICAL NEAR-TIE`, `BATCH IMPLEMENTATION BUG`, or `MIXED / NEEDS FURTHER ISOLATION`.

- [ ] Classify each divergence using margin and structural-difference evidence.
- [ ] Report whether all homogeneous copies are identical and whether cross-lane coupling exists.
- [ ] State the smallest next action without applying a fix.
