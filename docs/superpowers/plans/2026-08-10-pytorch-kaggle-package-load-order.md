# PyTorch Kaggle Package Load-Order Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Torch-before-`cg/libcg` loading a documented, machine-assisted requirement for future Kaggle packages and their official-engine smoke tests.

**Architecture:** PyTorch package manifests explicitly declare `runtime_framework: pytorch`, while generated `main.py` files configure thread limits and import Torch before any `cg`, `ctypes`, or native engine load. The evaluation worker reads that declaration, with a backward-compatible `strategy/model.bin` fallback for historical packages, and preloads Torch before loading the official `cg` runtime. Repository instructions and the submission archive README describe the same fail-closed extraction and smoke workflow.

**Tech Stack:** Python 3.11, PyTorch, JSON package manifests, official `cg` runtime, `unittest`.

## Global Constraints

- Do not modify `engine/source/`.
- Do not preload Torch in engine workers using only shared remote inference.
- Preserve compatibility with historical PyTorch packages that predate `runtime_framework`.
- A failed or crashed smoke is never a successful packaging result.

---

### Task 1: Specify the package contract

**Files:**
- Modify: `AGENTS.md`
- Modify: `archive/submission/README.md`

**Interfaces:**
- Consumes: the observed Torch-after-`libcg` crash and existing final-package gates.
- Produces: one discoverable normative contract and one operator-facing checklist.

- [x] Require explicit PyTorch runtime metadata and Torch-before-native-runtime import order.
- [x] Require extracted-archive tests to prove the order under the official-engine smoke path.
- [x] Document fail-closed diagnosis and prohibit bypassing the smoke by validating only source directories.

### Task 2: Make the evaluation gate honor the contract

**Files:**
- Modify: `evaluation/cli.py`
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/runner/worker.py`
- Modify: `tests/test_evaluation_batch.py`
- Modify: `tests/test_evaluation_cli.py`
- Modify: `tests/test_evaluation_worker.py`

**Interfaces:**
- Consumes: `SubmissionPackage.package_manifest` and package runtime layout.
- Produces: package runtime metadata serialized into worker requests and `_preload_local_agent_runtime_dependencies(request: GameRequest) -> None` called before `_load_game_api_with_runtime`.

- [x] Add tests proving local PyTorch packages preload Torch before `cg`, historical model packages use the compatibility fallback, and shared-only workers skip Torch.
- [x] Implement the minimal dependency detection and preload behavior.
- [x] Expose the existing per-game worker timeout through the CLI so slow local package smoke tests can distinguish a timeout from a native load-order crash.
- [x] Run the focused worker tests.

### Task 3: Update the active 0031 exporter contract

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/export_candidate.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_deployment.py`

**Interfaces:**
- Consumes: the standard 0031 zero-shot exporter.
- Produces: generated `main.py` with an explicit eager Torch import and manifests containing `runtime_framework: pytorch` plus `native_runtime_load_order: torch_before_cg`.

- [x] Add failing exporter assertions for manifest metadata and import ordering.
- [x] Update the generated entrypoint and manifest.
- [x] Run the focused deployment tests.

### Task 4: Verify the integrated gate

**Files:**
- Test: the files changed in Tasks 1–3.

**Interfaces:**
- Consumes: the documented and executable load-order contract.
- Produces: regression evidence and an updated implementation checklist.

- [x] Run focused unit tests and compile checks.
- [x] Run an extracted 0809 package official-engine smoke through the corrected worker path.
- [x] Record the result without claiming policy strength from the smoke.
