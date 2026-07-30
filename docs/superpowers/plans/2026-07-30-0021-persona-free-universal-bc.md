# 0021 Corrected Persona-Free Universal BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained universal Pokemon TCG pretraining project whose Actor learns
winner-side actions from state, exact own deck, causal history, and an unordered legal-option set,
without teacher identity or engine option position entering policy inference.

**Architecture:** V1 is a corrected rematerialization of the frozen 0019 winner Episode membership,
not a source-free-only ablation. It fixes relative seat encoding, removes option-position features,
supervises STOP only for optional termination, and gives every decision equal sequence-loss weight.
Source/team remains provenance metadata outside the Actor. A later, separately versioned experiment
may add self-supervised state/transition objectives from both players' visible states; loser actions
must never become BC targets.

**Tech Stack:** Python 3.11, PyTorch, BF16 CUDA training, standard-library `unittest`, TensorBoard,
W&B, and official-engine Evaluation.

## Global Constraints

- Never modify `engine/source/`.
- Use project ID `0021_persona_free_universal_bc` under `train/`, `experiments/`, and `rl_runs/`.
- `train/0021_persona_free_universal_bc/` must not import executable code from another numbered
  project; copy and freeze the required 0019 implementation with provenance.
- Winner-side actions are the only BC labels. Do not challenge or silently broaden that dataset rule.
- Source, team, player name, submission, and Persona may be retained only as provenance/audit
  metadata. They must never enter Actor tensors, `forward`, `encode`, decoder state, loss weighting,
  sampling weighting, or deployment.
- Actor input is `actor-relative causal state + exact own deck + causal visible history + legal
  option contents`. Do not encode absolute player identity or engine option-list position.
- The policy over legal options must be permutation equivariant: permuting legal options may only
  permute option logits and decoded indices after inverse remapping.
- Use `action_termination` to distinguish `optional_stop` and `forced_max`. Append a STOP target only
  for `optional_stop`; `forced_max` ends without a learned STOP decision.
- Optimize mean per-decision sequence NLL. Do not let actions with more selected tokens receive more
  total weight merely because they are longer.
- Keep IID Episode validation and deck-OOD validation separate. Neither offline split is policy
  strength evidence.
- Every formal epoch performs exactly one train update pass and one complete validation pass for
  each declared validation split.
- Formal V1 must accumulate at least 43,200 seconds of complete CUDA epoch work before early
  stopping is allowed. Persist and restore this counter in every recovery checkpoint.
- Formal training uses W&B online project `dragon_bra/pokemon-tcg-policy-learning`; canonical facts
  remain in `training_metrics.jsonl`.
- 0021 V1 is an explicit exception to the repository's model-only default: save bounded, atomic,
  exact epoch-boundary recovery checkpoints containing model, AdamW optimizer, trainer counters,
  Python/NumPy/Torch CPU/Torch CUDA RNG state, and scheduler/GradScaler state when those components
  exist. Raw replay, materialized batches, and DataLoader workers are still forbidden.
- Resume only into the same repository version and stable W&B run. The checkpoint must commit the
  dataset, model, feature compiler, training config, completed epoch, and global step; a mismatch
  fails closed. Because V1 saves only after a complete epoch and derives shard/batch shuffling from
  `(seed, epoch)`, no mid-epoch DataLoader cursor is claimed or required.
- Policy strength claims require official-engine Arena evaluation.
- Do not start formal 0021 training until the corrected rematerialization contract and deck-OOD
  split audit both pass.

---

### Task 1: Allocate the Self-Contained Project and Freeze Provenance

**Files:**
- Create: `train/0021_persona_free_universal_bc/`
- Create: `train/0021_persona_free_universal_bc/tests/test_project_contract.py`
- Create: `experiments/0021_persona_free_universal_bc/manifest.json`
- Create: `experiments/0021_persona_free_universal_bc/decisions/001_corrected_pretraining_contract.md`
- Create: `rl_runs/0021_persona_free_universal_bc/dataset/`

**Interfaces:**
- Consumes: frozen 0019 winner Episode membership and raw replay commitments.
- Produces: `PROJECT_ID`, project-local Python implementation, and immutable provenance hashes.

- [ ] **Step 1: Write failing identity and import-boundary tests**

```python
def test_project_identity(self) -> None:
    self.assertEqual(PROJECT.PROJECT_ID, "0021_persona_free_universal_bc")

def test_no_cross_numbered_runtime_imports(self) -> None:
    self.assertEqual(find_forbidden_numbered_imports(PROJECT_ROOT), [])
```

- [ ] **Step 2: Verify the tests fail before allocation**

Run: `python3 -m unittest -v train.0021_persona_free_universal_bc.tests.test_project_contract`

Expected: import failure because the 0021 package does not exist.

- [ ] **Step 3: Copy the required 0019 Python sources into 0021**

Copy source files, rename project identity, and remove `source_model.py` and
`source_r15_model.py`. Preserve a file-by-file source SHA-256 map in the experiment manifest.

- [ ] **Step 4: Freeze Episode membership without reusing old feature shards**

The manifest must record the exact winner Episode/player keys and raw payload hashes accepted by
0019. Do not hard-link the old 0019 model-ready shards because their absolute-seat,
option-position, STOP, and loss contracts are invalid for 0021.

- [ ] **Step 5: Run the focused contract test**

Run: `python3 -m unittest -v train.0021_persona_free_universal_bc.tests.test_project_contract`

Expected: PASS, including deletion-safety and no numbered-project runtime import.

### Task 2: Implement Actor-Relative and Option-Order-Free Features

**Files:**
- Modify: `train/0021_persona_free_universal_bc/base_model.py`
- Modify: `train/0021_persona_free_universal_bc/ac_model.py`
- Modify: `train/0021_persona_free_universal_bc/features/compiler.py`
- Create: `train/0021_persona_free_universal_bc/tests/test_feature_contract.py`

**Interfaces:**
- Produces: codec schema `0021_actor_relative_option_set_v1` and a permutation-equivariant option
  encoder.

- [ ] **Step 1: Write a failing relative-seat test**

```python
def test_seat_feature_is_actor_relative(self) -> None:
    actor0 = encode_global(make_observation(your_index=0, first_player=0))
    actor1 = encode_global(make_observation(your_index=1, first_player=1))
    self.assertEqual(actor0.self_went_first, actor1.self_went_first)
    self.assertEqual(actor0.opponent_went_first, actor1.opponent_went_first)
```

- [ ] **Step 2: Write failing option-permutation tests**

```python
def test_option_features_exclude_list_position(self) -> None:
    encoded = codec.encode(observation_with_distinct_options(), action=None)
    self.assertNotIn("option_position", codec.schema_fields)
    self.assertEqual(encoded["option_cat"].shape[-1], 11)

def test_logits_are_equivariant_to_option_permutation(self) -> None:
    original = model.option_logits(batch)
    permuted = model.option_logits(permute_options(batch, PERMUTATION))
    torch.testing.assert_close(original, inverse_permute(permuted, PERMUTATION))
```

- [ ] **Step 3: Replace absolute `firstPlayer` with relative seat state**

Encode one categorical field with values `unknown=0`, `self_first=1`, `opponent_first=2`, derived
jointly from `current.yourIndex` and `current.firstPlayer`. Do not expose either absolute index.

- [ ] **Step 4: Remove option position from codec and model**

Delete `option_index + 1`, `option_position`, and all position-dependent option paths in both the
base and AC encoder. Option features may describe only action type, card/entity, actor-relative
source/target ownership, zones, slots within semantic zones, and card-defined values.

- [ ] **Step 5: Add randomized permutation property coverage**

For 128 deterministic seeds, permute 2-128 legal options, inverse-remap logits and decoded actions,
and require exact greedy-action identity plus numerical closeness of logits.

- [ ] **Step 6: Run feature tests**

Run: `python3 -m unittest -v train.0021_persona_free_universal_bc.tests.test_feature_contract`

Expected: PASS.

### Task 3: Correct Termination Targets and Decision-Balanced Loss

**Files:**
- Modify: `train/0021_persona_free_universal_bc/training/materialized.py`
- Modify: `train/0021_persona_free_universal_bc/training/ac_data.py`
- Modify: `train/0021_persona_free_universal_bc/training/a0_trainer.py`
- Create: `train/0021_persona_free_universal_bc/tests/test_training_contract.py`

**Interfaces:**
- Produces: `targets`, `target_mask`, `decision_loss`, and metadata-only provenance rows.

- [ ] **Step 1: Write failing termination-label tests**

```python
def test_optional_stop_has_stop_target(self) -> None:
    batch = select_ac_batch(cache_row("optional_stop", action=[2]), INDICES)
    self.assertEqual(batch["targets"][0, 1].item(), batch["option_mask"].size(1))
    self.assertTrue(batch["target_mask"][0, 1])

def test_forced_max_has_no_stop_target(self) -> None:
    batch = select_ac_batch(cache_row("forced_max", action=[2]), INDICES)
    self.assertEqual(batch["target_mask"][0].sum().item(), 1)
```

- [ ] **Step 2: Write a failing decision-balance test**

```python
def test_each_decision_has_equal_outer_weight(self) -> None:
    token_nll = torch.tensor([[2.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    losses = per_decision_sequence_nll(token_nll, mask)
    torch.testing.assert_close(losses, torch.tensor([2.0, 2.0]))
```

- [ ] **Step 3: Preserve termination metadata while constructing targets**

Map `action_termination` to an integer cache tensor. Append option selections in teacher order;
append STOP only when termination is `optional_stop`. Reject unknown termination values.

- [ ] **Step 4: Keep source identity outside the Actor batch**

`select_ac_batch` must not return `source_id`, team name, player ID, or submission ID. A separate
audit iterator may yield those values with row identity, but the trainer and model signatures must
not accept them.

- [ ] **Step 5: Implement per-decision sequence NLL**

```python
def per_decision_sequence_nll(token_nll: Tensor, mask: Tensor) -> Tensor:
    counts = mask.sum(dim=1).clamp_min(1)
    return (token_nll * mask).sum(dim=1) / counts
```

Optimize `per_decision_sequence_nll(...).mean()`. Continue reporting token accuracy, teacher exact,
greedy exact, legal action, and action-length accuracy as diagnostics.

- [ ] **Step 6: Run training-contract tests**

Run: `python3 -m unittest -v train.0021_persona_free_universal_bc.tests.test_training_contract`

Expected: PASS.

### Task 4: Rematerialize IID and Deck-OOD Datasets

**Files:**
- Modify: `train/0021_persona_free_universal_bc/data/replay_catalog.py`
- Modify: `train/0021_persona_free_universal_bc/training/materialized.py`
- Create: `train/0021_persona_free_universal_bc/tests/test_dataset_splits.py`
- Create: `experiments/0021_persona_free_universal_bc/data_audit.json`

**Interfaces:**
- Produces: train, IID validation, and deck-OOD validation shards under a new 0021 schema hash.

- [ ] **Step 1: Write split leakage tests**

```python
def test_episode_player_groups_do_not_cross_splits(self) -> None:
    self.assertFalse(train_episode_players & iid_episode_players)
    self.assertFalse(train_episode_players & ood_episode_players)

def test_ood_decks_are_absent_from_training(self) -> None:
    self.assertFalse(train_deck_hashes & ood_deck_hashes)
```

- [ ] **Step 2: Define deterministic split manifests**

Keep the frozen 0019 winner Episode membership. Create an Episode-disjoint IID validation split and
a deck-hash-disjoint OOD split with enough Episodes per held-out archetype to report uncertainty.
Fail closed if the corpus cannot satisfy the minimum declared in `data_audit.json`; do not call six
single-Episode decks a meaningful OOD benchmark.

- [ ] **Step 3: Build new feature shards from raw committed rows**

Run:

```bash
python3 -m train.0021_persona_free_universal_bc.training.materialized build \
  --raw rl_runs/0021_persona_free_universal_bc/dataset/V1_universal_winner_raw \
  --output rl_runs/0021_persona_free_universal_bc/dataset/V1_corrected_model_ready \
  --workers 8 --shard-records 1024
```

- [ ] **Step 4: Audit invariants and publish commitments**

Verify 60-card decks, unique row identities, causal chronology, winner-side action ownership,
termination counts, option-count distributions, zero Actor source fields, and split intersections.
Record all shard sizes and SHA-256 commitments in `data_audit.json`.

- [ ] **Step 5: Validate without loading 0019 executable code**

Run: `python3 -m train.0021_persona_free_universal_bc.training.materialized validate rl_runs/0021_persona_free_universal_bc/dataset/V1_corrected_model_ready`

Expected: PASS with schema `0021_actor_relative_option_set_v1`.

### Task 5: Freeze the V1 Model, Logging, and Exact-Resume Checkpoint Contract

**Files:**
- Modify: `train/0021_persona_free_universal_bc/run_r15.py`
- Modify: `train/0021_persona_free_universal_bc/training/checkpoints.py`
- Create: `experiments/0021_persona_free_universal_bc/DESIGN.md`
- Create: `experiments/0021_persona_free_universal_bc/DESIGN.html`
- Create: `train/0021_persona_free_universal_bc/tests/test_checkpoints.py`

**Interfaces:**
- Produces: version `V1_corrected_persona_free_r15`, canonical metrics, and bounded exact
  epoch-boundary recovery checkpoints.

- [ ] **Step 1: Write model-boundary and checkpoint tests**

```python
def test_actor_signature_has_no_source_identity(self) -> None:
    self.assertFalse(forbidden_actor_fields(model, {"source", "persona", "team", "player"}))

def test_checkpoint_round_trip_restores_exact_training_state(self) -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    self.assertEqual(
        set(payload),
        {"model", "optimizer", "scheduler", "grad_scaler", "trainer", "rng", "metadata"},
    )
    restored = load_checkpoint(
        CHECKPOINT, model=new_model, optimizer=new_optimizer,
        scheduler=None, grad_scaler=None, expected_metadata=EXPECTED,
    )
    self.assertEqual(restored.completed_epoch, 3)
    self.assertEqual(restored.global_step, 123)
```

- [ ] **Step 2: Set explicit V1 defaults**

Use batch 256, validation batch 512, learning rate `3e-4`, weight decay `0.02`, BF16, seed
`20260723`, patience 5, min delta `0.001`, and selection metric `bc/validation_iid/loss`. W&B tags:
`0021,persona_free,corrected_contract,universal_winner,r15,gradual_option`.

- [ ] **Step 3: Emit separate validation namespaces**

Log `bc/validation_iid/*` and `bc/validation_deck_ood/*`. Do not emit conditioned/neutral Persona
metrics. Each epoch must fully scan both fixed validation splits.

- [ ] **Step 4: Implement fail-closed exact epoch-boundary recovery**

`save_checkpoint(...)` writes schema `0021_resumable_bc_checkpoint_v1` with top-level keys `model`,
`optimizer`, `scheduler`, `grad_scaler`, `trainer`, `rng`, and `metadata`. `trainer` contains
`completed_epoch`, `global_step`, `best`, `no_loss_improvement`, `progress_iteration`, and canonical
history through that epoch. `rng` contains Python state, NumPy state converted to safe primitives
plus a Torch tensor, Torch CPU state, and all CUDA generator states.

`load_checkpoint(path, ..., expected_metadata)` verifies schema, project/version, dataset content
hash, feature compiler hash, model-config hash, and training-config hash before mutating any state.
It restores all components and returns a frozen `ResumeState`. A missing key, wrong hash, wrong
version, non-epoch-boundary marker, or optimizer parameter-group mismatch raises `ValueError`.

- [ ] **Step 5: Add same-version resume CLI**

Add `--resume-checkpoint PATH`. Without it, `initialize_version` still requires unused output
directories. With it, resolve `project_version_paths(PROJECT_ID, --version)`, require the existing
version and checkpoint to agree, reuse the stable W&B run ID, and call `train` with restored epoch,
global step, best metrics, patience counter, progress counter, and history. Never resume into a
different `V<n>_<tag>` and never append when the training config or dataset changed.

- [ ] **Step 6: Bound checkpoint retention**

Retain latest, best IID loss, best IID greedy exact, and best deck-OOD greedy exact, with at most
eight full recovery files. Prune only files unreferenced by every criterion manifest and report the
retained byte total in status/summary.

- [ ] **Step 7: Synchronize DESIGN Markdown and HTML**

Document exact tensor shapes, relative-seat encoding, option-set equivariance, termination targets,
decision-balanced NLL, source metadata boundary, split limitations, model parameter count measured
from current code, exact-resume payload and epoch-boundary limitation, retention/disk policy, and
official-engine acceptance criteria. Do not reuse the old 17,387,842 count after removing
embeddings; compute and test the new count.

- [ ] **Step 8: Run contract tests**

Run: `python3 -m unittest -v train.0021_persona_free_universal_bc.tests.test_project_contract train.0021_persona_free_universal_bc.tests.test_checkpoints`

Expected: PASS, including an interrupted-vs-uninterrupted deterministic CPU fixture whose final
model, optimizer, counters, and next RNG draws are identical.

### Task 6: Run Smoke and Formal Corrected BC Training

**Files:**
- Create: `.tmp/0021_persona_free_universal_bc/V0_corrected_contract_smoke/`
- Create: `rl_runs/0021_persona_free_universal_bc/versions/V1_corrected_persona_free_r15/`

**Interfaces:**
- Produces: finite smoke evidence, then a W&B-backed formal training version.

- [ ] **Step 1: Run one-batch forward/backward smoke**

Run: `python3 -m train.0021_persona_free_universal_bc.run_r15 --smoke --smoke-tag V0_corrected_contract_smoke`

Expected: finite decision-balanced loss and gradients; permutation and source-boundary assertions
pass; no formal W&B run. Resume that smoke checkpoint in the same smoke version and verify the next
epoch completes without duplicate metric epochs.

- [ ] **Step 2: Validate smoke artifacts**

Confirm dataset commitment, input keys, parameter count, termination counts, exact-resume
checkpoint, finite IID/OOD validation metrics, and bounded retained checkpoint bytes.

- [ ] **Step 3: Calibrate batch throughput on the target RTX 5080**

Run the same fixed 20-train-batch smoke at batch sizes 256 and 512. Record decisions/second,
iterations/second, peak allocated/reserved CUDA memory, and finite loss under identical seed and
BF16 settings. Select 512 only if it completes without OOM and leaves at least 10% of 16,303 MiB
unallocated; otherwise retain 256. Store the chosen batch and benchmark evidence in the V1 training
config rather than inferring speed from model size.

- [ ] **Step 4: Allocate and start the formal version**

Run: `python3 -m train.0021_persona_free_universal_bc.run_r15 --version V1_corrected_persona_free_r15`

Expected: unused artifact/checkpoint/tensorboard/wandb directories and one stable online W&B ID.

- [ ] **Step 5: Monitor every epoch and enforce the 12-hour floor**

Audit train optimization diagnostics, complete IID/OOD validation, throughput, GPU memory, W&B
sync, checkpoint retention, and cumulative CUDA epoch seconds. A project-local watchdog may restart
the same version at most three times from `criteria/latest.json`; it must fail closed without a
valid epoch-boundary checkpoint. Preserve failed versions and allocate V2 for any semantic change.

### Task 7: Evaluate Selected Checkpoints in the Official Engine

**Files:**
- Create: `evaluation/arena/candidates/0021_persona_free_<deck>/`
- Create: `experiments/0021_persona_free_universal_bc/evaluation/V<n>_<tag>.html`
- Create: `experiments/0021_persona_free_universal_bc/evaluation/index.html`
- Create: `rl_runs/0021_persona_free_universal_bc/versions/V<n>_<tag>/artifact/evaluation.json`

**Interfaces:**
- Produces: self-contained packages and official-engine Arena evidence comparable to 0019 neutral.

- [ ] **Step 1: Export self-contained candidates**

Each package contains a 60-card deck, copied runtime, model-only checkpoint, ontology, manifest, and
physical `cg/`. Package manifests must state feature schema and contain no Persona deployment field.

- [ ] **Step 2: Validate every package**

Run: `python3 -m evaluation validate evaluation/arena/candidates/0021_persona_free_<deck>`

Expected: exact 60-card, self-contained, model hash committed.

- [ ] **Step 3: Run the frozen Arena contract**

Use all enabled opponents, at least 10 games per opponent, balanced seats, the same metric profile,
`--workers 2`, and `--worker-cpu-threads 1` while GPU training is active.

- [ ] **Step 4: Publish immutable reports and comparisons**

Keep offline imitation and Arena strength separate. Compare to 0019 neutral only with identical
deck, opponent catalog hash, games, seats, runtime, and metric profile; report uncertainty for
unpaired differences.

### Task 8: Reserve Auxiliary State/Transition Pretraining for a New Version

**Files:**
- Create later: `experiments/0021_persona_free_universal_bc/decisions/002_auxiliary_state_transition_contract.md`

**Interfaces:**
- Consumes: V1 evidence and a separately audited dataset of both players' visible states.
- Produces: a new V2+ experiment only after its targets and leakage boundaries are approved.

- [ ] **Step 1: Keep V1 free of unvalidated auxiliary objectives**

Do not add transition, value, masked-card, or loser-action terms to V1 merely to improve a scalar.

- [ ] **Step 2: Specify a separate auxiliary contract before implementation**

The contract may use winner and loser visible states for representation or transition targets, but
must exclude loser actions from BC, exclude future/private information from current-state inputs,
define stop-gradient/weighting precisely, and allocate a new repository version and W&B run.

### Task 9: Final Verification

- [ ] **Step 1: Run the complete focused suite**

Run: `python3 -m unittest discover -s train/0021_persona_free_universal_bc -t . -p 'test_*.py' -v`

- [ ] **Step 2: Run repository checks**

Run: `python3 -m unittest -v tests.test_evaluation_assets`

Run: `python3 -m compileall -q train/0021_persona_free_universal_bc evaluation`

Run: `git diff --check`

- [ ] **Step 3: Cross-check authoritative evidence**

Verify DESIGN Markdown/HTML against code, data audit, checkpoint metadata, training summary, W&B
identity, and official-engine reports before declaring 0021 V1 complete.
