# 0025 James Cox Raging Bolt BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train matched legacy and semantic BC policies on the same 1,073 winning Episodes from the James Cox Raging Bolt expert and measure whether 0025 semantics improve imitation.

**Architecture:** A strict catalog slicer combines two audited TeamNames only at the corpus layer while preserving source provenance and one exact 60-card deck hash. One semantic materialization is shared by both arms; the legacy arm reads only the frozen legacy payload, while the semantic arm reads all typed memories. Both arms use the same Episode split, action targets, optimization loop, validation decoder, seeds, and checkpoint contract.

**Tech Stack:** Python 3.11, PyTorch, gzip JSONL, official Kaggle Episode archives, TensorBoard, W&B online, official engine evaluation.

## Global Constraints

- Do not modify `engine/source/`.
- Keep `train/0025_semantic_foundation_pretraining/` independent of all other numbered training projects.
- Team/source identity is provenance only and must never enter actor forward.
- The corpus must contain only exact TeamNames `James Cox` and `James Cox & Henry Chao`, winning views, and deck SHA-256 `f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`.
- Preserve whole-Episode train/validation assignments from the audited universal catalog.
- Formal training uses a new immutable `V2_james_cox_raging_bolt_ablation` version, W&B online, per-epoch full validation, and model-only checkpoints.
- Strength conclusions require official engine runtime games; offline exact-action metrics alone are imitation evidence.

---

### Task 1: Exact Expert Slice

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/data/expert_slice.py`
- Create: `train/0025_semantic_foundation_pretraining/tests/test_expert_slice.py`
- Create: `experiments/0025_semantic_foundation_pretraining/data_audit/james_cox_raging_bolt_2026-07-20_2026-08-01.json`

**Interfaces:**
- Consumes: `slice_catalog(catalog, team_names, deck_sha256, start_date, end_date)` arguments and the strict replay catalog schema.
- Produces: a raw-builder-compatible catalog plus an audit containing per-source, per-date, seat, split, deck, and Episode commitments.

- [ ] **Step 1: Write failing source/deck boundary tests**

```python
def test_slice_rejects_wrong_deck_and_preserves_sources():
    sliced, audit = slice_catalog(fixture_catalog(), ("James Cox", "James Cox & Henry Chao"), EXPECTED_DECK, "2026-07-20", "2026-08-01")
    assert audit["episodes"] == 2
    assert [row["team_name"] for row in sliced["sources"]] == ["James Cox", "James Cox & Henry Chao"]
    assert all(row["deck_sha256"] == EXPECTED_DECK for row in sliced["episodes"])
```

- [ ] **Step 2: Run the focused test and observe the missing module failure**

Run: `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests.test_expert_slice`

- [ ] **Step 3: Implement fail-closed slicing and canonical commitments**

```python
def slice_catalog(catalog, team_names, deck_sha256, start_date, end_date):
    accepted = [row for row in catalog["episodes"] if row["team_name"] in team_names and row["deck_sha256"] == deck_sha256 and start_date <= row["episode_date"] <= end_date]
    validate_unique_winning_exact_decks(accepted)
    return rebuild_catalog(accepted), build_audit(accepted)
```

- [ ] **Step 4: Run tests and generate the committed audit from the 13-day catalog**

Run: `python3 -m train.0025_semantic_foundation_pretraining.data.expert_slice --catalog .tmp/0025_james_cox_audit/winner_catalog_2026-07-20_2026-08-01.json --team-name 'James Cox' --team-name 'James Cox & Henry Chao' --deck-sha256 f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a --start-date 2026-07-20 --end-date 2026-08-01 --output-catalog .tmp/0025_james_cox_audit/expert_catalog.json --output-audit experiments/0025_semantic_foundation_pretraining/data_audit/james_cox_raging_bolt_2026-07-20_2026-08-01.json`

### Task 2: Shared Raw And Semantic Dataset

**Files:**
- Modify: `train/0025_semantic_foundation_pretraining/data/materialize.py`
- Create: `train/0025_semantic_foundation_pretraining/training/dataset.py`
- Create: `train/0025_semantic_foundation_pretraining/tests/test_training_dataset.py`

**Interfaces:**
- Consumes: the expert catalog and official Episode locators.
- Produces: immutable raw decisions and semantic shards with split counts, hashes, and target tensors shared by both experiment arms.

- [ ] **Step 1: Add tests for split preservation, target STOP construction, shard hashes, and source exclusion**

```python
def test_collate_builds_shared_targets_without_source_input():
    batch = collate_training_records(records)
    assert batch["targets"][0, -1] == batch["option_mask"].shape[1]
    assert "source_id" not in batch
```

- [ ] **Step 2: Implement streaming shard validation and deterministic epoch shuffling**

```python
def iter_batches(root: Path, split: str, batch_size: int, seed: int):
    records = load_verified_split(root, split)
    for indices in deterministic_batches(len(records), batch_size, seed):
        yield collate_training_records([records[index] for index in indices])
```

- [ ] **Step 3: Build the raw expert corpus and semantic dataset**

Run: `python3 -m train.0025_semantic_foundation_pretraining.data.raw_dataset --catalog .tmp/0025_james_cox_audit/expert_catalog.json --output rl_runs/0025_semantic_foundation_pretraining/dataset/V1_james_cox_raging_bolt_raw --workers 8`

Run: `python3 -m train.0025_semantic_foundation_pretraining.data.materialize --raw-root rl_runs/0025_semantic_foundation_pretraining/dataset/V1_james_cox_raging_bolt_raw --output rl_runs/0025_semantic_foundation_pretraining/dataset/V1_james_cox_raging_bolt_semantic --prototypes train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json --start-date 2026-07-20 --end-date 2026-08-01`

- [ ] **Step 4: Verify counts, hashes, split disjointness, exact deck, and actor provenance exclusion**

Run: `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests.test_training_dataset`

### Task 3: Matched Legacy And Semantic Policies

**Files:**
- Modify: `train/0025_semantic_foundation_pretraining/model/multi_memory.py`
- Create: `train/0025_semantic_foundation_pretraining/training/objective.py`
- Create: `train/0025_semantic_foundation_pretraining/tests/test_bc_objective.py`

**Interfaces:**
- Consumes: one collated batch containing shared targets and actor tensors.
- Produces: teacher logits `[B,S,O+1]`, legal deterministic sequences, token loss, token accuracy, teacher exact, greedy exact, and action-length accuracy for either arm.

- [ ] **Step 1: Add tests for teacher prefix consumption and legal greedy decoding**

```python
def test_semantic_teacher_logits_follow_target_prefix():
    logits = semantic.teacher_logits(batch)
    assert logits.shape[:2] == batch["targets"].shape
    assert torch.isfinite(logits[batch["targets"] != -100]).all()
```

- [ ] **Step 2: Implement semantic teacher logits and batched deterministic decode**

```python
def teacher_logits(self, batch):
    outputs = []
    for step in range(batch["targets"].shape[1]):
        outputs.append(self(batch, batch["targets"][:, :step]))
    return torch.stack(outputs, dim=1)
```

- [ ] **Step 3: Implement one shared objective and evaluator for both policies**

```python
def teacher_loss(model, batch):
    logits = model.teacher_logits(batch)
    mask = batch["targets"] != -100
    return F.cross_entropy(logits[mask], batch["targets"][mask])
```

- [ ] **Step 4: Run focused model and objective tests**

Run: `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests.test_bc_objective`

### Task 4: Formal Paired Trainer

**Files:**
- Create: `train/0025_semantic_foundation_pretraining/training/checkpoints.py`
- Create: `train/0025_semantic_foundation_pretraining/training/trainer.py`
- Create: `train/0025_semantic_foundation_pretraining/run_bc_ablation.py`
- Create: `train/0025_semantic_foundation_pretraining/tests/test_checkpoint_contract.py`

**Interfaces:**
- Consumes: shared dataset, two models, optimizer config, immutable version paths.
- Produces: arm-specific metrics, model-only checkpoints, combined comparison summary, TensorBoard events, and W&B online metrics.

- [ ] **Step 1: Test that checkpoints reject optimizer/RNG/replay state**

```python
def test_checkpoint_is_model_only():
    payload = load_checkpoint(save_checkpoint(model, metadata))
    assert set(payload) == {"schema_version", "state_dict", "metadata"}
```

- [ ] **Step 2: Implement per-epoch single-pass training and full validation**

```python
for epoch in range(1, epochs + 1):
    train_metrics = optimize_once(train_batches(epoch))
    validation_metrics = evaluate(validation_batches(epoch))
    logger.log(epoch, {**train_metrics, **validation_metrics})
    retain_latest_and_best_model_only_checkpoints()
```

- [ ] **Step 3: Implement immutable V2 allocation and W&B environment**

```python
paths = initialize_version(PROJECT_ID, "V2_james_cox_raging_bolt_ablation")
os.environ.update({"WANDB_MODE": "online", "WANDB_ENTITY": "dragon_bra", "WANDB_PROJECT": "pokemon-tcg-policy-learning"})
```

- [ ] **Step 4: Run checkpoint and runner preflight tests**

Run: `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests.test_checkpoint_contract`

### Task 5: End-To-End Smoke

**Files:**
- Modify: `train/0025_semantic_foundation_pretraining/tools/smoke_and_benchmark.py`
- Create: `.tmp/0025_james_cox_bc_smoke/<run-id>/`

**Interfaces:**
- Consumes: real expert semantic shards.
- Produces: two-arm one-epoch smoke metrics and reloadable checkpoints outside formal version paths.

- [ ] **Step 1: Run two train batches and two validation batches per arm with W&B disabled**

Run: `python3 -m train.0025_semantic_foundation_pretraining.run_bc_ablation --smoke --maximum-train-batches 2 --maximum-validation-batches 2`

- [ ] **Step 2: Verify both losses are finite, both checkpoints reload, and semantic gradients reach all four memory queries**

Run: `python3 -m unittest -v train.0025_semantic_foundation_pretraining.tests`

### Task 6: Formal V2 Training And Comparison

**Files:**
- Create: `rl_runs/0025_semantic_foundation_pretraining/versions/V2_james_cox_raging_bolt_ablation/`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0025_semantic_foundation_pretraining/DESIGN.html`

**Interfaces:**
- Consumes: verified shared dataset and smoke-proven runner.
- Produces: monitored paired BC run, best checkpoints, and an explicit legacy-versus-semantic comparison.

- [ ] **Step 1: Launch the formal foreground training session with watchdog-style resource checks**

Run: `python3 -m train.0025_semantic_foundation_pretraining.run_bc_ablation --version V2_james_cox_raging_bolt_ablation --epochs 40 --early-stopping-patience 6`

- [ ] **Step 2: Monitor metrics, checkpoint writes, W&B status, GPU/RAM/SSD, and failures at intervals below 60 seconds**

- [ ] **Step 3: Select each arm by validation loss and report paired offline deltas without claiming gameplay strength**

- [ ] **Step 4: Synchronize both authoritative DESIGN documents with corpus commitments, tensor shapes, parameter counts, objectives, metrics, and current phase**

### Task 7: Official Engine Evaluation Boundary

**Files:**
- Create: `evaluation/arena/candidates/0025_james_cox_raging_bolt_<arm>/`
- Create: `experiments/0025_semantic_foundation_pretraining/evaluation/V2_james_cox_raging_bolt_ablation.html`

**Interfaces:**
- Consumes: selected model-only checkpoints and exact 60-card deck.
- Produces: self-contained candidate packages and official-engine matchup evidence under the fixed evaluation catalog.

- [ ] **Step 1: Export self-contained legacy and semantic candidates with identical deck and legal action contract**

- [ ] **Step 2: Validate both packages independently**

Run: `python3 -m evaluation validate evaluation/arena/candidates/<candidate>`

- [ ] **Step 3: Run balanced official-engine evaluation under the same opponents, seeds, and worker contract**

Run: `python3 -m evaluation run --candidate evaluation/arena/candidates/<candidate> --opponents all --workers 8 --worker-cpu-threads 1 --output <formal-report-path>`

- [ ] **Step 4: Record the authoritative report backlink and separate offline imitation gains from official-engine strength gains**
