# 0015 R15 Rule-Contract Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline because the repository does not
> expose `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Track every
> step with the checkboxes below and do not delegate or commit.

**Goal:** Keep the selected 0014 R15 architecture intact while adding a gradual option-level rule
reader that can model Dragapult/Dusknoir turn budgets, prize races, library pressure and post-blast
relay decisions.

**Architecture:** V7 subclasses the existing source-conditioned R15 policy. Four actor-visible
numeric rule tokens attend into each legal full-action option, then a residual with a configurable
initial scale of `0.10` updates only option embeddings. The default `r15` model path remains byte-
compatible with V1–V6; `r15_rule_contract` is an explicit new model family and version.

**Tech Stack:** Python 3.11, PyTorch 2.11 BF16/CUDA, unittest, TensorBoard, W&B online, official
engine evaluation.

## Global Constraints

- R15 at `0014/V25_r15_deterministic_gradual_option` remains the only selected 0014 baseline.
- Use the frozen T1 dataset: 54,764 train decisions and 1,000 target-only validation decisions.
- Preserve exact registered-deck conditioning, source persona, actor-visible causality and legal
  full-action decoding.
- V7 changes only the model family; seed `20260723`, optimizer, batch sizes, validation and
  checkpoint selection remain identical to V2.
- The new rule residual initializes at `0.10`, not the old R16 neutral scale `1.0`, because 0015 has
  much less target data and must avoid an immediate shortcut.
- Do not modify `engine/source/`, frozen V1–V6 artifacts or the existing candidate packages.
- Every formal run uses a new monotonic version and W&B run. No commit, push, Kaggle submission or
  opponent admission is authorized.

---

### Task 1: Source-conditioned R15 rule reader

**Files:**
- Create: `train/0015_dragapult_conditioned_bc/rule_contract_model.py`
- Create: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: the existing R15 batch dictionary, including `global_num`, `global_cat`,
  `zone_inventory_num` and `source_id`.
- Produces: `SourceConditionedR15RuleConfig` and `SourceConditionedR15RulePolicy` with the same
  `forward`, `encode` and greedy decode contract as `SourceConditionedR15Policy`.

- [ ] Write tests that reject invalid rule depth/width/scale and accept `0 < scale < 2`.
- [ ] Write a model test that loads one frozen cache batch and checks finite logits, unchanged
  `[B, T, C]` output shape and source-conditioned forward behavior.
- [ ] Write an initialization test that proves every rule ScaleGate channel starts at `0.10`.
- [ ] Implement four rule tokens using only actor-visible fields: `turn_budget` width 10,
  `prize_race` width 16, `library_pressure` width 12 and `board_relay` width 19.
- [ ] Add one typed Transformer layer, legal-option cross-attention and an option-only residual.
- [ ] Run the focused unittest and compile the new module.

### Task 2: Explicit model-family training dispatch

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/training/outcome_weighted_trainer.py`
- Test: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: `--model-family {r15,r15_rule_contract}` and `--rule-initial-scale 0.10`.
- Produces: immutable config/model-contract/checkpoint metadata identifying the concrete family and
  full nested configuration.

- [ ] Add a model factory whose default produces the exact existing R15 class/config.
- [ ] Add CLI validation that rule-only options cannot silently change an `r15` run.
- [ ] Generalize the weighted optimization callback type to `nn.Module` without changing its math.
- [ ] Record `model_family`, parameter count, rule token widths and initial scale in formal
  artifacts and W&B tags.
- [ ] Prove with a regression test that default arguments still instantiate
  `SourceConditionedR15Policy` with the V1–V6 schema and parameter count `17,416,642`.
- [ ] Run all 0015 tests and `git diff --check`.

### Task 3: Portable candidate support

**Files:**
- Create: `train/0015_dragapult_conditioned_bc/portable/rule_contract_model.py`
- Modify: `train/0015_dragapult_conditioned_bc/portable/source_inference.py`
- Modify: `train/0015_dragapult_conditioned_bc/export_candidate.py`
- Test: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: checkpoint metadata `model_family` and nested model config.
- Produces: portable CPU policy with strict family dispatch and a self-contained candidate package.

- [ ] Make inference reject unknown families rather than falling back to R15.
- [ ] Copy the rule model only for rule-family exports and record the family in candidate manifest.
- [ ] Export a smoke checkpoint to `.tmp/evaluation/0015_v7_export_smoke/` and validate its exact
  60-card deck, copied `cg/`, checkpoint load and one legal selection.
- [ ] Verify old V2 export/load behavior remains supported.

### Task 4: CUDA gate and formal V7 training

**Files:**
- Create: `.tmp/0015_dragapult_conditioned_bc/smoke_v7_rule_contract/`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V7_r15_rule_contract_reader/`

**Interfaces:**
- Consumes: T1 data, seed `20260723`, rule initial scale `0.10`.
- Produces: canonical metrics, TensorBoard, checkpoints, W&B online run and completion status.

- [ ] Wait for V6 to release enough VRAM, then run a five-batch CUDA BF16 optimizer smoke.
- [ ] Verify finite loss/gradients, 1,000-decision validation, legal greedy rate `1.0`, exact initial
  rule scale and reloadable checkpoint.
- [ ] Preflight all V7 artifact paths and authoritative report path for nonexistence.
- [ ] Start V7 immediately after the smoke; do not leave the GPU idle between the two commands.
- [ ] Monitor per-epoch loss/exact/legality, W&B sync and early stopping without changing the run.

### Task 4B: Predeclared gradual-rule prior comparison

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V8_r15_rule_scale_003/`

**Interfaces:**
- Consumes: the exact V7 model/data/optimizer/seed contract with `rule_initial_scale=0.03`.
- Produces: a one-variable comparison against V7 that tests slower rule-evidence introduction.

- [ ] Freeze V8 before observing any V7 official-engine result; do not tune from engine feedback.
- [ ] Use `0.03` because the measured 0014 R2 state-scale mean was `0.0327`; change no other value.
- [ ] Preflight all V8 paths while V7 trains.
- [ ] Start V8 immediately when V7 releases the GPU, after checking V7 completion and W&B sync.
- [ ] Apply the same target-only checkpoint selector and official-engine evaluation contract.

### Task 4C: Strong-policy rule-adapter initialization

**Files:**
- Create: `train/0015_dragapult_conditioned_bc/initialization.py`
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V9_v2_rule_adapter/`

**Interfaces:**
- Consumes: V2's frozen best-target-exact R15 checkpoint and a new rule-contract model.
- Produces: exact inherited R15/source weights, randomly initialized rule-only parameters and an
  auditable initialization provenance block.

- [ ] Write a failing test that loads V2 into a rule model and proves all inherited tensors match
  exactly while every missing tensor belongs only to the new rule reader.
- [ ] Reject checkpoints from unknown schema/model families and any unexpected/missing base key.
- [ ] Add `--initial-checkpoint`; allow it only for `r15_rule_contract` formal runs.
- [ ] Record source path, SHA-256, source version, loaded-key count and new-key list in config,
  model contract, checkpoint metadata and W&B tags.
- [ ] Predeclare V9 with rule scale `0.03`, LR `1e-4`, seed `20260723` and all other T1 settings
  unchanged; it is a practical adapter experiment, not a fresh-init causal comparison.
- [ ] Run focused tests and a CUDA smoke while V8 trains if VRAM permits; otherwise run it between
  V8 completion and V9 launch.
- [ ] Start V9 immediately after V8 and apply the same checkpoint/evaluation contract.

### Task 4D: Frozen-base rule adapter

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/initialization.py`
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V10_v2_frozen_rule_adapter/`

**Interfaces:**
- Consumes: the V9 initialization contract and its audited `new_model_keys`.
- Produces: an optimizer over only new rule-reader tensors, with every inherited V2 tensor frozen.

- [ ] Write a failing test proving every inherited parameter has `requires_grad=False`, every new
  rule parameter remains trainable and the frozen/trainable counts sum to the total.
- [ ] Add `--train-rule-only`, valid only with `--initial-checkpoint` and
  `r15_rule_contract`.
- [ ] Build AdamW from trainable tensors only and record trainable/frozen tensor and parameter
  counts in config, model contract, checkpoint metadata and W&B tags.
- [ ] Predeclare V10 with V2 warm-start, rule scale `0.03`, LR `3e-4`, seed `20260723` and no other
  changes. This is derived from V9's immediate full-model degradation at LR `1e-4`.
- [ ] CUDA-smoke exact inherited-weight preservation after one optimizer step, then launch V10 as
  soon as V9 releases the GPU.

### Task 4E: Frozen-adapter learning-rate control

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V11_v2_frozen_rule_lr1e4/`

**Interfaces:**
- Consumes: the exact V10 warm-start/freeze/model/data/seed contract with LR `1e-4`.
- Produces: a one-variable test of whether slower adapter optimization reduces V10 overfitting.

- [ ] Freeze V11 before V10 official-engine results are available.
- [ ] Change only rule-only LR from `3e-4` to `1e-4`; retain scale `0.03` and patience 5.
- [ ] Preflight V11 paths and start immediately after V10 W&B sync.
- [ ] Use the same target-exact checkpoint selector and official-engine evaluation contract.

### Task 4F: Frozen-adapter rule-prior control

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V12_v2_frozen_rule_scale_010/`

**Interfaces:**
- Consumes: the exact V10 warm-start/freeze/LR/data/seed contract with scale `0.10`.
- Produces: the final bounded comparison of `0.03` versus `0.10` under a frozen V2 base.

- [ ] Freeze V12 before V10/V11 official-engine results are available.
- [ ] Change only rule initial scale from `0.03` to `0.10`; retain LR `3e-4`.
- [ ] Start immediately after V11 W&B sync.
- [ ] Stop rule scale/LR exploration after V12 and complete the official-engine queue.

### Task 4G: Frozen-adapter independent-seed replication

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V13_v2_frozen_rule_seed_20260727/`

**Interfaces:**
- Consumes: the exact V12 warm-start/freeze/scale/LR/data contract with seed `20260727`.
- Produces: one independent-seed stability check without reopening the completed scale/LR sweep.

- [ ] Freeze V13 before any V10–V12 official-engine result is used to tune its contract.
- [ ] Change only seed `20260723` to `20260727`; retain rule scale `0.10`, LR `3e-4`, V2
  warm-start and rule-only optimization.
- [ ] Start immediately after V12 W&B sync so the GPU does not remain idle.
- [ ] Export the same target-exact checkpoint and apply the same official-engine evaluation gate.
- [ ] Stop seed replication after V13; use V10–V13 official-engine evidence to decide the next
  architectural experiment instead of launching another blind seed or hyperparameter run.

### Task 5: Frozen-package comparison and design synchronization

**Files:**
- Create: `evaluation/arena/candidates/0015_v6_t1_seed_20260727_exact_best/`
- Create: `evaluation/arena/candidates/0015_v7_r15_rule_contract_exact_best/`
- Create: `evaluation/arena/candidates/0015_v8_r15_rule_scale_003_exact_best/`
- Create: `experiments/0015_dragapult_conditioned_bc/evaluation/V6_t1_seed_20260727.html`
- Create: `experiments/0015_dragapult_conditioned_bc/evaluation/V7_r15_rule_contract_reader.html`
- Create: `experiments/0015_dragapult_conditioned_bc/evaluation/V8_r15_rule_scale_003.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/manifest.json`
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/007_r15_seed_and_rule_reader.md`

**Interfaces:**
- Consumes: each version's target greedy-exact checkpoint selected without engine feedback.
- Produces: two 200-game official-engine reports under the frozen catalog and an evidence-backed
  keep/reject decision relative to V2.

- [ ] Export and validate V6 and V7 without admitting either to the formal opponent pool.
- [ ] Run 20 opponents × 10 games with `--workers 8 --worker-cpu-threads 1`.
- [ ] Record W/L/error, completion, seat split, per-opponent results and uncertainty against V2.
- [ ] Keep the strongest supported candidate; do not rank by validation label match alone.
- [ ] Synchronize model tensors, parameter count, current phase and V6/V7 results in both DESIGN
  formats and refresh the evaluation index.
- [ ] Verify report SHA backlinks, W&B status, candidate packages, tests, compileall and diff check.
