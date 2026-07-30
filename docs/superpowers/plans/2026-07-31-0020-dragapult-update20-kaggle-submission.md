# 0020 Dragapult Update 20 Kaggle Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the formally evaluated 0020 Dragapult PPO update-20 policy as a self-contained Kaggle archive, pass every repository submission gate on a fresh extraction, and use the user's one authorized Kaggle submission attempt.

**Architecture:** Resolve the policy identity from the immutable evaluation/status/checkpoint chain, then use the 0020 package builder to copy the deck, inference implementation, official runtime, ontology, and model-only checkpoint into `archive/submission/<name>/`. Create a root-layout tarball under `archive/submission/dist/`, validate only a fresh extraction, run an official-engine 10-game smoke, and submit that exact tarball after credential and quota checks.

**Tech Stack:** Python 3.11+, PyTorch model-only checkpoints, repository evaluation CLI, POSIX tar, SHA-256, Kaggle CLI/API.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve all pre-existing dirty-worktree changes and do not overwrite an existing submission directory, archive, or evaluation report.
- The archive must extract with `main.py`, `deck.csv`, `cg/`, strategy code, and model weights at its root, with no symlinks, caches, optimizer state, or adjacent-project dependency.
- Validate the fresh extraction with package structure checks, exact 60-card initialization, dynamic `exec` without `__file__`, `python3 -m evaluation validate`, and an official-engine opponent × 10-game smoke with 10/10 finished and zero errors.
- Perform at most the single Kaggle submission explicitly authorized by the user; do not retry a successful or uncertain submission call without new authorization.

---

### Task 1: Resolve the immutable evaluated policy

**Files:**
- Read: `experiments/0020_pluggable_deck_rl/evaluation/V15_dragapult_ppo_update20_probe.html`
- Read: `rl_runs/0020_pluggable_deck_rl/versions/V15_dragapult_ppo_update20_probe/artifact/status.json`
- Read: `rl_runs/0020_pluggable_deck_rl/versions/V9_dragapult_ppo_initial/checkpoint/update-000020.pt`

**Interfaces:**
- Consumes: formal evaluation metadata and model-only checkpoint metadata.
- Produces: exact source checkpoint path, hash, deck ID, model contract, and expected 49.00% arena result.

- [ ] Verify the formal report contains 300 completed games, 147 wins, zero errors, and 49.00% win rate.
- [ ] Cross-check the status file's checkpoint path/hash and package manifest hash against the checkpoint payload and evaluated candidate.
- [ ] Refuse packaging if any identity edge is ambiguous or mismatched.

### Task 2: Build the self-contained package and archive

**Files:**
- Read: `train/0020_pluggable_deck_rl/package_builder.py`
- Create: `archive/submission/0020_dragapult_ppo_update20/`
- Create: `archive/submission/dist/0020_dragapult_ppo_update20.tar.gz`

**Interfaces:**
- Consumes: the verified update-20 checkpoint and the existing 0020 package builder contract.
- Produces: one complete source directory and one root-layout tarball containing the identical payload.

- [ ] Run the 0020 builder with the Dragapult deck and verified update-20 checkpoint.
- [ ] Inspect the resulting manifest, exact 60-line deck, weights, runtime files, and absence of symlinks/caches/forbidden state.
- [ ] Create the archive from inside the package directory so no extra top-level directory is introduced.
- [ ] Record the archive SHA-256 and verify every archived member is relative and safe.

### Task 3: Gate a fresh extraction

**Files:**
- Create: `.tmp/evaluation/0020_dragapult_ppo_update20_kaggle_gate/run-*/report.html`

**Interfaces:**
- Consumes: only a fresh extraction of the final tarball.
- Produces: validation output and a 10-game official-engine smoke report with 100% completion and zero errors.

- [ ] Extract the final tarball into a new temporary directory and compare its file hashes with the source package.
- [ ] Run `python3 -m evaluation validate <extracted-root>`.
- [ ] Dynamically execute `main.py` without `__file__` or repository `PYTHONPATH`, call `read_deck_csv()` and `agent({"select": null})`, and require the same exact 60-card deck.
- [ ] Run one fixed catalog opponent for 10 games through the official engine with eight workers and one CPU thread per worker, writing the report under `.tmp/evaluation/0020_dragapult_ppo_update20_kaggle_gate/`.
- [ ] Parse the report payload and require 10 completed games, 100% completion, zero unfinished games, and zero errors.

### Task 4: Submit once and record the receipt

**Files:**
- Create: `archive/submission/0020_dragapult_ppo_update20/submission_receipt.json` only if a repository-local receipt is appropriate and does not alter the submitted tarball.

**Interfaces:**
- Consumes: the exact gated archive, Kaggle competition `pokemon-tcg-ai-battle`, authenticated account context, and remaining daily quota.
- Produces: one Kaggle submission reference plus terminal or latest visible submission status/score.

- [ ] Verify credentials without printing secrets, inspect the authenticated account, list current submissions, and check daily quota.
- [ ] Submit `archive/submission/dist/0020_dragapult_ppo_update20.tar.gz` once with an update-20-identifying message.
- [ ] Capture the returned submission reference and poll the existing submission until a processed score/status appears or a concrete external blocker is reached.
- [ ] Do not issue a second submit command if the first call succeeded, timed out ambiguously, or created a visible submission.

### Task 5: Self-review and handoff

**Files:**
- Read: all produced artifacts and command evidence from Tasks 1–4.

**Interfaces:**
- Consumes: checkpoint evidence, package/archive hashes, gate report, and Kaggle receipt.
- Produces: concise user handoff with clickable local paths and explicit caveats.

- [ ] Confirm the selected checkpoint is update 20 and its formal local result is 49.00%, without treating rollout diagnostics as frozen greedy strength.
- [ ] Report package directory, archive path, SHA-256, all gate outcomes, Kaggle competition, submission ID/status/public score, and any remaining evaluation risk.
