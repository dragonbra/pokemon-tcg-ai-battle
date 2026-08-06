# 0031 Friend 0806 Pretrain And Dragapult Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `large-model-0806.tar.gz` as an auditable 0031 pretrained asset and evaluate that exact checkpoint with the 0033 Third PTCG Club Dragapult deck against the frozen 0019 pool.

**Architecture:** Preserve the supplied model-only payload byte-for-byte in a new immutable `archive/pretrained/` directory with self-contained 0031 model source, contracts, assets, loader, verifier, and provenance sidecars. Export one temporary zero-shot candidate using the exact 0033 deck, run 51 opponents x 10 games through the unmodified official engine, and publish the completed report as a new 0031 evaluation version with a matching `rl_runs/0031.../versions/` record.

**Tech Stack:** Python 3.11, PyTorch, repository evaluation CLI, official engine runtime, HTML evaluation reports, JSON manifests.

## Global Constraints

- Do not modify `engine/source/`.
- Do not overwrite any existing pretrained archive, run version, or formal evaluation HTML.
- Preserve the supplied checkpoint as model-only; reject optimizer, scheduler, scaler, RNG, rollout, and replay state.
- Bind evaluation to `0019_foundation_51_exact_decks_v4`, 10 games per opponent, 510 total games, and require 510/510 finished with zero errors.
- Use `train/0033_dragapult_third_ptcg_club_rl/league/decks/dragapult_third_ptcg_club/deck.csv` unchanged as the exact 60-card candidate deck.
- Store the authoritative report under `experiments/0031_rule_faithful_semantic_foundation_pretraining/evaluation/` and add a reverse link under the same V9 run version.

---

### Task 1: Publish The Pretrained Asset

**Files:**
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/`
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/manifest.json`
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/checkpoint_metadata.json`
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/README.md`
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/loader.py`
- Create: `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/verify_archive.py`

**Interfaces:**
- Consumes: `large-model-0806.tar.gz` containing `best_validation_loss_0806.pt`.
- Produces: `model.pt` with SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8` and a strict `load_model(device)` API.

- [x] **Step 1: Validate source payload identity**

Run a weights-only load and require schema `0031_model_only_checkpoint_v1`, project `0031_rule_faithful_semantic_foundation_pretraining`, arm `rule_faithful_semantic`, version `V2_full_winners_bs1024_20260616_20260803`, epoch 11, step 141878, and `SemanticPolicy` parameter contract 56,352,322.

- [x] **Step 2: Materialize the release without changing weights**

Copy the checkpoint to `model.pt`, copy the self-contained 0031 inference/model source and prototype assets, and write release metadata containing archive/checkpoint hashes, selection metrics, source tar hash, and the model-only boundary.

- [x] **Step 3: Verify the archive**

Run `python3 archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/verify_archive.py` and require strict state loading, manifest hash agreement, no symlinks/caches, and a finite minimal forward pass.

### Task 2: Evaluate The 0033 Dragapult Deck

**Files:**
- Create: `.tmp/model_loading/0031_friend_0806/candidates/dragapult_third_ptcg_club/`
- Create: `.tmp/evaluation/0031_friend_0806_dragapult_frozen0019/`
- Create: `experiments/0031_rule_faithful_semantic_foundation_pretraining/evaluation/V9_friend0806_epoch11_dragapult_third_frozen0019.html`

**Interfaces:**
- Consumes: the published `model.pt`, 0033 deck SHA-256 `5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`, and frozen pool `0019_foundation_51_exact_decks_v4`.
- Produces: one embedded, authoritative official-engine report with 510 ordered games.

- [x] **Step 1: Export and validate the zero-shot candidate**

Use `train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate.export_candidate`, then run `python3 -m evaluation validate <candidate>`.

- [x] **Step 2: Execute the formal evaluation**

Run `python3 -m evaluation --pool frozen run --candidate <candidate> --opponents all --games 10 --workers 8 --worker-cpu-threads 1 --candidate-device cuda:0 --opponent-device cuda:0 --metric-profile league_deck_quality --output .tmp/evaluation/0031_friend_0806_dragapult_frozen0019`.

- [x] **Step 3: Fail closed and publish**

Parse embedded report data and require the exact checkpoint hash, deck hash, pool identity, 51 opponents, 510 games, 510 completed, zero errors, and zero unfinished games before copying it to the V9 authoritative path.

### Task 3: Record The Formal Version

**Files:**
- Create: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V9_friend0806_epoch11_dragapult_third_frozen0019/artifact/evaluation.json`
- Create: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V9_friend0806_epoch11_dragapult_third_frozen0019/artifact/status.json`
- Create: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V9_friend0806_epoch11_dragapult_third_frozen0019/artifact/training_config.json`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/evaluation/index.html`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/manifest.json`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`

**Interfaces:**
- Consumes: validated V9 report summary and immutable identities from Tasks 1-2.
- Produces: auditable version metadata and clickable 0031 evaluation index entry.

- [x] **Step 1: Write immutable V9 provenance**

Record checkpoint/archive/deck/pool hashes, run ID, game counts, W-L-D, completion rate, metric profile, worker settings, and the authoritative report path. Mark this as evaluation-only zero-shot evidence, not a training run or automatic promotion claim.

- [x] **Step 2: Refresh project records**

Append the 0806 release/evaluation decision, update the project manifest without changing the current training semantics, and regenerate the evaluation index from all `V*.html` reports.

- [x] **Step 3: Run final checks**

Run the pretrained verifier, evaluation report parser checks, relevant experiment/evaluation tests, `git diff --check`, and confirm `git status` contains no changes under `engine/source/`.
