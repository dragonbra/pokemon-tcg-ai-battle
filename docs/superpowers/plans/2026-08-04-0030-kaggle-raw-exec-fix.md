# 0030 Kaggle Raw-Exec Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce U50 and U81 archives whose entrypoints load under Kaggle's `exec(code_object, env)` contract without `__file__`, then submit each fixed archive exactly once in U50-first order.

**Architecture:** Add a generic isolated raw-exec validation gate without changing the existing official-engine package loader. Make the 0030 exporter normalize copied entrypoints and invoke the new gate before publishing an output directory. Preserve the failed archives and create new `kaggle_exec_fix` directories and tarballs with separate hashes and receipts.

**Tech Stack:** Python 3.11, `subprocess`, Kaggle CLI, repository evaluation package APIs, deterministic GNU tar, Git LFS.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve the previously submitted U50 and U81 package directories, tarballs, hashes, and failure receipt.
- Submit U50 before U81, with exactly one Kaggle submit command per fixed archive and no automatic retry.
- Keep all unrelated 0031 worktree changes unstaged and unmodified.
- Final tar roots must directly contain `main.py`, `deck.csv`, `cg/`, `strategy/`, and `manifest.json`.

---

### Task 1: Kaggle Raw-Exec Validation Gate

**Files:**
- Modify: `evaluation/packages/loader.py`
- Modify: `tests/test_evaluation_packages.py`

**Interfaces:**
- Consumes: package `main.py`, package root, and exact deck list.
- Produces: `validate_kaggle_raw_exec(entrypoint: Path, root: Path, deck: list[int]) -> None`.

- [x] **Step 1: Write a failing test**

Add one package whose entrypoint uses `Path(__file__)` and assert raw-exec validation rejects it with `name '__file__' is not defined`. Add a compatible package using `Path(globals().get('__file__', Path.cwd()))` and assert it passes and returns the exact deck.

- [x] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_packages.EvaluationPackageTests.test_kaggle_raw_exec_rejects_unconditional_dunder_file tests.test_evaluation_packages.EvaluationPackageTests.test_kaggle_raw_exec_accepts_cwd_fallback`

Expected: FAIL because `validate_kaggle_raw_exec` does not exist.

- [x] **Step 3: Implement isolated raw-exec validation**

Run `main.py` in a subprocess with the package as CWD, compile and execute it into an empty globals dictionary, require a callable `agent`, call `agent({'select': None})`, compare the returned exact deck, and use the existing inherited success FD so `SystemExit` or `os._exit(0)` cannot spoof success.

- [x] **Step 4: Run the focused and complete evaluation package tests**

Run: `python3 -m unittest -v tests.test_evaluation_packages`

Expected: PASS.

### Task 2: 0030 Exporter Compatibility

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/export_candidate.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_contract.py`

**Interfaces:**
- Consumes: copied 0028 source `main.py` and the Task 1 validator.
- Produces: every future 0030 export with a CWD fallback and a pre-publication raw-exec gate.

- [x] **Step 1: Write a failing exporter contract test**

Assert the exporter source contains `globals().get("__file__", Path.cwd())`, the fixed Kaggle root fallback, and a call to `validate_kaggle_raw_exec` before `staging.replace(output)`.

- [x] **Step 2: Run the focused contract test and verify failure**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_contract.ProjectContractTest.test_exporter_enforces_kaggle_raw_exec_contract`

Expected: FAIL against the current copy-only exporter.

- [x] **Step 3: Normalize and validate the exported entrypoint**

Replace the unsafe `ROOT = Path(__file__).resolve().parent` line in staging with the established `globals().get('__file__', Path.cwd())` plus `/kaggle_simulations/agent` fallback. Fail closed if the expected unsafe or already-safe source pattern cannot be identified, and invoke raw-exec validation after writing model and manifest but before publishing the directory.

- [x] **Step 4: Run all 0030 tests**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_contract train.0030_dragapult_shared_encoder_decoder_rl.tests.test_monitor_best_checkpoints train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_diagnostics train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_imports`

Expected: PASS.

### Task 3: Fixed U50 and U81 Submission Assets

**Files:**
- Create: `archive/submission/0030_dragapult_ex_001_rl_update50_kaggle_exec_fix/`
- Create: `archive/submission/0030_dragapult_ex_001_rl_update81_kaggle_exec_fix/`
- Create: `archive/submission/dist/0030_dragapult_ex_001_rl_update50_kaggle_exec_fix.tar.gz`
- Create: `archive/submission/dist/0030_dragapult_ex_001_rl_update81_kaggle_exec_fix.tar.gz`

**Interfaces:**
- Consumes: immutable failed package directories and the corrected entrypoint contract.
- Produces: two distinct, auditable, flat-root Kaggle archives.

- [x] **Step 1: Copy immutable payloads to new fixed identities**

Physically copy each failed package, update only its entrypoint and manifest identity/provenance, and leave the original package and tarball untouched.

- [x] **Step 2: Build deterministic flat-root archives**

Use sorted members, epoch mtime, numeric owner/group 0, and no outer directory. Exclude bytecode, cache, optimizer, rollout, trace, and symlink content.

- [x] **Step 3: Validate directories and extracted tar payloads**

Run standard `evaluation validate`, raw-exec validation, exact initialization deck comparison, model/manifest hashes, member traversal checks, and extracted-package validation for both archives.

- [x] **Step 4: Record final SHA-256 values**

Capture the two archive hashes before any external submission and use those exact values in the resubmission receipt.

### Task 4: One-Shot Ordered Kaggle Resubmission

**Files:**
- Create: `experiments/0030_dragapult_shared_encoder_decoder_rl/submission_receipt_u50_u81_kaggle_exec_fix.json`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DECISIONS.md`

**Interfaces:**
- Consumes: Task 3 validated archives and the two user-provided Japanese messages.
- Produces: exactly two Kaggle submission refs and an immutable receipt.

- [x] **Step 1: Run read-only credential, quota, and duplicate preflight**

Require at least two remaining daily slots and confirm neither fixed filename already appears in submission history.

- [x] **Step 2: Submit U50 exactly once**

Issue one `kaggle competitions submit` command for the fixed U50 archive with the U50 message, then only poll its returned ref.

- [x] **Step 3: Submit U81 exactly once**

After U50 has a confirmed ref, issue one submit command for fixed U81 with the U81 message, then only poll its returned ref.

- [x] **Step 4: Persist terminal receipts and commit scoped assets**

Record refs, timestamps, statuses, scores/errors, archive hashes, command counts, and `automatic_retry_issued: false`. Run tests and `git diff --check`, stage only 0030/shared-validator/fixed-package assets, and commit without touching 0031.
