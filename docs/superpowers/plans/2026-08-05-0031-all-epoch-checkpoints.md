# 0031 All-Epoch Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve one immutable model-only checkpoint for every completed 0031 epoch while retaining the existing latest and best-selection aliases.

**Architecture:** Extend the existing atomic model-only checkpoint publisher with an `epoch_XXXX` retention name emitted at every completed validation boundary. Keep `latest` and the three best aliases for consumers, and record immutable epoch entries separately so selection metadata never discards history. The already-running epoch 3 artifact is recovered from its current `latest.pt` before epoch 4 can overwrite it.

**Tech Stack:** Python 3.11, PyTorch checkpoint serialization, `unittest`, JSON experiment metadata.

## Global Constraints

- Do not modify `engine/source/`.
- Checkpoints remain model-only and atomically published; optimizer, scheduler, RNG and rollout state are excluded.
- `checkpoint_retention` is `all`; no epoch checkpoint may be rotated or deleted.
- Preserve the active epoch 4 training process and W&B run.

---

### Task 1: Immutable Epoch Retention

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/training/trainer.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/run_bc.py`
- Test: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_checkpoints.py`

**Interfaces:**
- Consumes: `save_checkpoint(root, name, model, metadata)` and epoch metadata already built after validation.
- Produces: `epoch_XXXX.pt` for every completed epoch plus an `all_epochs` selection map keyed by the decimal epoch string.

- [x] **Step 1: Write the failing test**

Add a trainer-level assertion that two completed epochs retain both `epoch_0001.pt` and `epoch_0002.pt`, that each payload has the matching `metadata["epoch"]`, and that neither contains optimizer state.

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_checkpoints`

Expected: FAIL because the trainer currently publishes only replaceable aliases.

- [x] **Step 3: Write minimal implementation**

For every completed arm/epoch, call:

```python
epoch_name = f"epoch_{epoch:04d}"
epoch_record = save_checkpoint(
    arm_root,
    name=epoch_name,
    model=model,
    metadata=metadata,
)
checkpoint_records[arm].setdefault("all_epochs", {})[str(epoch)] = epoch_record
```

Keep alias publication unchanged. Update the run config to declare `checkpoint_retention: "all"`, `model_only_checkpoint_retention_slots: "all_completed_epochs_plus_four_aliases"`, and the completed-epoch boundary.

- [x] **Step 4: Run focused and repository tests**

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests.test_checkpoints`

Run: `python3 -m unittest -v train.0031_rule_faithful_semantic_foundation_pretraining.tests`

Expected: PASS, with every serialized epoch payload remaining model-only.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-05-0031-all-epoch-checkpoints.md \
  train/0031_rule_faithful_semantic_foundation_pretraining/training/trainer.py \
  train/0031_rule_faithful_semantic_foundation_pretraining/run_bc.py \
  train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_checkpoints.py
git commit -m "retain every 0031 epoch checkpoint"
```

### Task 2: Recover The Active Run Boundary

**Files:**
- Create: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/checkpoint/rule_faithful_semantic/epoch_0003.pt`
- Modify: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/V4_lr5e4_no_early_stop_b512/artifact/checkpoint_selection.json`

**Interfaces:**
- Consumes: epoch 3 `latest.pt`, whose payload metadata reports `epoch == 3`.
- Produces: immutable epoch 3 model-only checkpoint and auditable selection metadata.

- [x] **Step 1: Verify source identity**

Load `latest.pt` read-only and assert `schema_version`, `metadata["epoch"] == 3`, and absence of optimizer/scheduler/RNG keys.

- [x] **Step 2: Recover without recomputation**

Create `epoch_0003.pt` from the verified bytes using a no-clobber reflink/copy, then verify identical SHA-256.

- [ ] **Step 3: Audit the running boundary**

After epoch 4 completes, verify `latest.pt` reports epoch 4 and preserve it as `epoch_0004.pt` if the already-loaded trainer has not emitted that file. Continue this boundary audit until a process restart loads the fixed implementation.

- [x] **Step 4: Verify training continuity**

Confirm the watchdog reports a live child, W&B `active/active`, no alert, and forward epoch progress after recovery.
