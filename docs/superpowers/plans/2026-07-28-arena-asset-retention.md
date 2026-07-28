# Arena Asset Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain formal Arena opponent packages with LFS-backed binary assets while removing all Git history for mutable Candidates packages.

**Architecture:** Rewrite only the local branch history before pushing. Keep source, deck, catalog entries, and formal Combat Matrix reports in ordinary Git; put model weights and package-runtime native binaries in Git LFS. Remove `evaluation/arena/candidates/` from every reachable commit and ignore it for future local staging.

**Tech Stack:** Git, Git LFS, git-filter-repo, Python unittest

## Global Constraints

- Do not modify `engine/source/`.
- Do not rerun the Combat Matrix.
- Formal opponent packages remain self-contained.
- Preserve unrelated working-tree changes.
- Push the completed local commit to `origin/dev/cyd_main`.

---

### Task 1: Audit Stored Assets

**Files:**
- Modify: `.gitattributes`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: formal packages at `evaluation/arena/opponents/`
- Produces: path-specific LFS and ignore rules

- [ ] **Step 1: Enumerate candidate history and formal binary paths**

Run: `git log --all -- evaluation/arena/candidates && git ls-tree -r --long HEAD evaluation/arena/opponents`

Expected: Candidate history is identified separately from formal opponent assets.

- [ ] **Step 2: Classify retention**

Keep formal model weights and native runtime binaries in LFS. Remove all Candidates paths and their local tracking rule from Git history.

### Task 2: Rewrite Local History

**Files:**
- Modify: `.gitattributes`
- Modify: `.gitignore`
- Delete: all historical `evaluation/arena/candidates/**`

**Interfaces:**
- Consumes: audited path lists
- Produces: a branch containing no Candidates tree entries and LFS pointers for retained binary assets

- [ ] **Step 1: Create a local backup ref**

Run: `git branch backup/pre-arena-asset-retention dev/cyd_main`

Expected: original local commits remain recoverable before rewriting.

- [ ] **Step 2: Migrate retained formal package binaries to LFS**

Run: `git lfs migrate import --include='evaluation/arena/opponents/**/policy*.pt,evaluation/arena/opponents/**/strategy/model.bin,evaluation/arena/opponents/**/cg/*.dll,evaluation/arena/opponents/**/cg/*.so,evaluation/arena/opponents/**/cg/*.dylib' --include-ref=refs/heads/dev/cyd_main`

Expected: formal package binary files are Git LFS pointers.

- [ ] **Step 3: Remove Candidates from all reachable branch history**

Run: `git filter-repo --path evaluation/arena/candidates --invert-paths --force`

Expected: `git log --all -- evaluation/arena/candidates` has no output.

- [ ] **Step 4: Add the future Candidates ignore rule**

Add `evaluation/arena/candidates/` to `.gitignore` and commit it with the rewritten branch state.

### Task 3: Verify the Result

**Files:**
- Test: `tests/test_evaluation_assets.py`
- Test: `tests/test_evaluation_combat_matrix.py`

**Interfaces:**
- Consumes: rewritten branch and LFS object store
- Produces: verified formal opponent catalog and valid LFS pointer coverage

- [ ] **Step 1: Verify absence and LFS tracking**

Run: `git ls-files evaluation/arena/candidates && git lfs ls-files`

Expected: no Candidates paths; all retained formal binaries appear in LFS output.

- [ ] **Step 2: Validate assets**

Run: `python3 -m evaluation validate evaluation/arena/opponents/alakazam_dudunsparce_04_sota && python3 -m unittest -v tests.test_evaluation_assets tests.test_evaluation_combat_matrix`

Expected: validation and targeted tests pass.

### Task 4: Publish

**Files:**
- Modify: remote branch `origin/dev/cyd_main`

**Interfaces:**
- Consumes: verified rewritten local branch
- Produces: matching remote ref and uploaded LFS objects

- [ ] **Step 1: Push branch and LFS objects**

Run: `git push --force-with-lease origin dev/cyd_main`

Expected: Git LFS uploads the formal binary objects and remote accepts the rewritten history.

- [ ] **Step 2: Confirm remote HEAD**

Run: `git ls-remote --heads origin dev/cyd_main`

Expected: remote SHA equals local `git rev-parse HEAD`.
