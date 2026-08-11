# 0042 Clean-Clone Runtime Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every non-0809, redistributable 0042 startup dependency available or reproducibly buildable from a clean Git clone.

**Architecture:** Replace the cross-project ignored 0038 checkpoint dependency with a project-local allocation-head-only sidecar whose file and tensor provenance are strictly verified. Keep ABI- and machine-specific CUDA binaries local, but provide a 0042-owned build/preflight entry point that selects the local GPU architecture and emits an auditable manifest. Document remaining intentionally local assets separately from Git omissions.

**Tech Stack:** Python 3.11, PyTorch, CMake, Ninja, CUDA, unittest, Git LFS attributes for binary artifacts where applicable.

## Global Constraints

- Do not add or modify Policy-0809 actor, Value, or portable package weights.
- Do not modify `engine/source/`.
- Do not commit `_ptcg_cuda.so`, the private official rule pack, generated official card data, run checkpoints, or W&B data.
- 0042 must not depend at runtime on executable code or mutable run artifacts from another numbered training project.
- Allocation-head loading must fail closed on file hash, schema, source provenance, tensor inventory, shape, or dtype mismatch.
- Existing user work in the dirty worktree must be preserved and reviewed separately before staging.

---

### Task 1: Freeze the allocation-only initialization sidecar

**Files:**
- Create: `train/0042_full_model_design/assets/allocation_head_bc_v1.pt`
- Create: `train/0042_full_model_design/assets/allocation_head_bc_v1.manifest.json`
- Modify: `.gitignore`
- Modify: `train/0042_full_model_design/initialization.py`
- Modify: `train/0042_full_model_design/tests/test_0040_initialization.py`

**Interfaces:**
- Consumes: the immutable allocation tensors from 0038 update-0 checkpoint SHA-256 `d572c8673fcf16c39fb17d8d261bec6578885a69c227e579bc4e6f616bb22820`.
- Produces: `ALLOCATION_HEAD_CHECKPOINT`, `ALLOCATION_HEAD_SHA256`, and strict project-local allocation-head materialization.

- [x] **Step 1: Add a failing clean-clone boundary test**

Assert that the allocation asset is under `train/0042_full_model_design/assets`, is tracked, contains only `allocation_head_state_dict` inference tensors plus primitive provenance metadata, and that `initialization.py` contains no `rl_runs/0038_action_boundary_rl` runtime path.

- [x] **Step 2: Run the focused initialization tests and confirm the new boundary fails**

Run: `python3 -m unittest -v train.0042_full_model_design.tests.test_0040_initialization`

Expected: failure because the project-local sidecar does not yet exist.

- [x] **Step 3: Materialize and register the minimal sidecar**

Extract exactly the ten `allocation_head.*` tensors, strip the prefix inside the sidecar, preserve source checkpoint/dataset hashes and `ppo_updates=0`, compute the sidecar SHA-256, and add a human-readable manifest. Add a narrow `.gitignore` exception for this one file.

- [x] **Step 4: Implement strict sidecar loading**

Verify the sidecar file hash before `torch.load(..., weights_only=True)`, validate schema/provenance/zero-PPO metadata and exact tensor inventory, then strict-load only `model.allocation_head`. Keep Policy-0809 actor and Value loading unchanged.

- [x] **Step 5: Run initialization and policy-identity tests**

Run: `python3 -m unittest -v train.0042_full_model_design.tests.test_0040_initialization train.0042_full_model_design.tests.test_initialization_contract train.0042_full_model_design.tests.test_policy_identity`

Expected: all pass.

### Task 2: Make the CUDA extension reproducibly local

**Files:**
- Create: `train/0042_full_model_design/tools/build_cuda_extension.py`
- Create: `train/0042_full_model_design/tools/__init__.py`
- Create: `train/0042_full_model_design/tests/test_build_cuda_extension.py`
- Modify: `docs/rl/0042_full_model_design_operations_manual.md`

**Interfaces:**
- Consumes: tracked `engine_cuda/CMakeLists.txt`, the active Python/PyTorch environment, CMake/Ninja, and the detected CUDA compute capability.
- Produces: local `.tmp/engine_cuda_benchmark/build_sm120_staged/_ptcg_cuda.so` plus a JSON build manifest recording the actual target architecture; no binary is added to Git.

- [x] **Step 1: Add failing command/planning tests**

Test compute-capability normalization, build-directory selection, Torch CMake discovery, and rejection of a missing CUDA toolchain without invoking a real build.

- [x] **Step 2: Implement the 0042-owned builder**

Provide `--architecture`, `--build-dir`, and `--jobs`; auto-detect architecture from `torch.cuda.get_device_capability(0)` when omitted; configure/build `_ptcg_cuda`; import it from the exact output directory; and atomically write provenance containing tool versions, PyTorch/CUDA versions, architecture, source commit, file path, and SHA-256.

- [x] **Step 3: Remove the misleading clean-clone implication from the operations manual**

State that `.so` is intentionally not in Git, point to the builder, and keep private rule-pack preparation a separate required step.

- [x] **Step 4: Run builder unit tests without compiling CUDA**

Run: `python3 -m unittest -v train.0042_full_model_design.tests.test_build_cuda_extension`

Expected: all pass.

### Task 3: Audit, verify, commit, and push

**Files:**
- Modify only if tests establish necessity: existing dirty 0042 diagnostic files.

**Interfaces:**
- Consumes: Tasks 1–2 and the pre-existing dirty worktree.
- Produces: one reviewed commit on `dev/cyd_main`, pushed to `origin/dev/cyd_main`.

- [x] **Step 1: Run clean-clone dependency audit**

Confirm tracked availability of source code, decks, fixtures, and the allocation sidecar; report Policy-0809 weights, private rule pack, generated official data, Gate-C local manifest, CPU/CUDA build outputs, credentials, and run directories as intentionally local prerequisites.

- [x] **Step 2: Run focused and full 0042 test suites**

Run the touched focused tests, then `python3 -m unittest discover -s train/0042_full_model_design/tests -p 'test_*.py' -v`. Record skips caused solely by explicitly excluded local weights/data; fix regressions in tracked code.

- [x] **Step 3: Review existing dirty changes**

Verify the raw-FP32 CPU diagnostic remains explicitly non-candidate/non-Promote, preserves official-engine semantics, and its rollout trajectory suppression and chance-boundary naming tests pass.

- [x] **Step 4: Stage only reviewed files and inspect the staged diff**

Use explicit paths. Confirm no 0809 checkpoint, private rule pack, `.so`, official data, run checkpoint, W&B file, or unrelated user file is staged.

- [x] **Step 5: Commit and push**

Commit with a scoped 0042 message, fetch/rebase only if needed without discarding local work, and push `HEAD` to `origin/dev/cyd_main`. Verify the remote branch contains the new commit.
