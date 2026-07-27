# 0015 Low-Rank Rule Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline because the repository does not
> expose `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Track the
> steps below without delegating or committing.

**Goal:** Prepare a lower-capacity typed rule adapter that preserves the frozen V2 policy while
reducing the 3,311,360-parameter rule-only learner that rapidly overfits the 54,764-decision T1
training set.

**Architecture:** Keep all four actor-visible rule token inputs and the R15 base unchanged. Project
R15 option/state/categorical evidence from width 320 into a configurable rule width, perform typed
rule self-attention and option cross-attention in that bottleneck, then project one gated residual
back to width 320. Existing width-320 checkpoints remain loadable through explicit metadata
defaults; the first proposed formal run uses width 80 and four heads.

**Tech Stack:** Python 3.11, PyTorch 2.11 BF16/CUDA, unittest, portable CPU inference, W&B online,
official-engine evaluation.

## Global Constraints

- Do not modify `engine/source/`, frozen datasets, V1–V13 artifacts or candidate packages.
- Preserve the exact registered-deck, source-persona, actor-visible and full-action contracts.
- Existing `r15_rule_contract` checkpoints without bottleneck fields must reconstruct width 320 and
  eight heads exactly.
- A conditional next version may start only after focused tests, portable export smoke, CUDA BF16
  smoke and an official-engine gate establish that another bounded rule-reader experiment is
  justified. The gate failed, so no version was allocated; `V14` was subsequently used by the
  independent label-smoothing experiment.
- No commit, push, Kaggle submission or opponent admission is authorized.

---

### Task 1: Configurable rule bottleneck

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/rule_contract_model.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: `rule_model_width: int` and `rule_heads: int` in
  `SourceConditionedR15RuleConfig`.
- Produces: the unchanged `[B, O, 320]` option representation and checkpoint key contract.

- [x] Add failing tests for non-positive width/heads and non-divisible width/head combinations.
- [x] Add a width-80 test that checks finite logits, unchanged decoder shape, exact initial gate and
  fewer trainable rule parameters than the width-320 reader.
- [x] Add rule-prefixed option/state/categorical projections and run encoder/attention/delta/gate in
  the configured bottleneck width.
- [x] Keep defaults at width 320 and eight heads and prove the default parameter count is still
  `20,728,002`.

### Task 2: Training and portable metadata

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/portable/rule_contract_model.py`
- Modify: `train/0015_dragapult_conditioned_bc/portable/source_inference.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: CLI `--rule-model-width` and `--rule-heads`.
- Produces: immutable training/checkpoint metadata and strict portable reconstruction.

- [x] Reject bottleneck CLI changes for the plain `r15` family.
- [x] Record width, heads, total parameters and frozen/trainable counts in config/model contract.
- [x] Make portable loading default missing legacy fields to 320/eight and strictly honor new fields.
- [x] Export a width-80 smoke checkpoint and validate a legal CPU selection from its 60-card package.

### Task 3: CUDA gate and bounded formal-version decision

**Files:**
- Create: `.tmp/0015_dragapult_conditioned_bc/smoke_v14_low_rank_rule/`
- Conditionally create: the next unused formal version directory (not allocated because the gate
  failed)

**Interfaces:**
- Consumes: V2 exact-best warm start, frozen inherited base, rule width 80, four rule heads, scale
  `0.10`, LR `3e-4`, seed `20260723`.
- Produces: one bounded low-capacity comparison, only if the official-engine gate permits it.

- [x] Run five CUDA BF16 optimizer batches and prove every inherited V2 tensor remains bitwise
  unchanged.
- [x] Compare exact trainable parameter count with 3,311,360 and record the reduction.
- [x] V11 and V12 did not beat V2's 39/200, so keep the implementation as a prepared rejected
  direction and do not consume a formal version. `V14` now names the separate label-smoothing run.
- [x] The launch branch was not applicable because the gate failed; no formal run or evaluation was
  created for the low-rank reader.
