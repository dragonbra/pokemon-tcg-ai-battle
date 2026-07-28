# 0015 R2 Strong Scenario T1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and officially evaluate the 0014 V12 R2 strong-scenario architecture on the strongest-supported 0015 T1 dataset.

**Architecture:** Preserve the exact `R2StrongScenarioPolicy` trunk and its neutral 1.0 state/option Scenario ScaleGates. Add the same audited source-persona residual used by 0015 R15 so multi-expert labels remain conditioned; keep T1 rows, target validation, seed, optimizer, loss and checkpoint selector unchanged.

**Tech Stack:** Python 3.11, PyTorch CUDA BF16, unittest, official-engine evaluation, TensorBoard, W&B online.

## Global Constraints

- Version is `V18_r2_strong_scenario_t1`; never reuse or overwrite an existing version.
- Dataset is `V1_core_r15_features`, arm `t1`: 54,764 train decisions and 1,000 target validation decisions.
- Win/loss/draw weights are all `1.0`; no label smoothing and no warm start.
- Do not modify `engine/source/`, admit an opponent, submit to Kaggle, commit, or push.
- Formal checkpoint selector is best validation greedy exact; official strength is 200 games at 10 per fixed opponent with 8 workers and one CPU thread per worker.

---

### Task 1: Source-conditioned R2 training family

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/model.py`
- Modify: `train/0015_dragapult_conditioned_bc/run.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: `R2ModelConfig`, `R2StrongScenarioPolicy`, frozen source vocabulary.
- Produces: `SourceConditionedR2Config`, `SourceConditionedR2Policy`, `build_policy(model_family="r2")`.

- [ ] Add a failing factory test asserting the R2 type, 17,416,642 parameters, R2 ScaleGate neutral scale 1.0 and source scale 0.10.
- [ ] Run the focused test and confirm `model_family="r2"` is rejected before implementation.
- [ ] Implement the R2 wrapper by applying the existing source embedding residual after the unchanged R2 `encode()` output.
- [ ] Extend CLI validation, metadata schema and model contract so R2 is explicit and R15 defaults remain unchanged.
- [ ] Run all 0015 focused tests and confirm the new factory test passes.

### Task 2: Portable R2 candidate contract

**Files:**
- Modify: `train/0015_dragapult_conditioned_bc/portable/source_model.py`
- Modify: `train/0015_dragapult_conditioned_bc/portable/source_inference.py`
- Modify: `train/0015_dragapult_conditioned_bc/export_candidate.py`
- Modify: `train/0015_dragapult_conditioned_bc/tests/test_rule_contract_model.py`

**Interfaces:**
- Consumes: checkpoint metadata `model_family="r2"` and `model.r2`.
- Produces: strict CPU reconstruction of source-conditioned R2 and a standard self-contained candidate.

- [ ] Add a failing test that exports a minimal R2 checkpoint and verifies candidate manifest family/schema.
- [ ] Extend portable config reconstruction to parse either `model.r2` or `model.r15` without weakening strict state loading.
- [ ] Extend export schema/family allowlists and use the generic portable policy entry point.
- [ ] Run candidate export tests, scoped compile and candidate validation.

### Task 3: V18 smoke, formal training and evaluation

**Files:**
- Create: `.tmp/0015_dragapult_conditioned_bc/smoke_v18_r2_strong_scenario_t1/`
- Create: `rl_runs/0015_dragapult_conditioned_bc/versions/V18_r2_strong_scenario_t1/`
- Create: `evaluation/arena/candidates/0015_v18_r2_strong_scenario_t1_exact_best/`
- Create: `experiments/0015_dragapult_conditioned_bc/evaluation/V18_r2_strong_scenario_t1.html`

**Interfaces:**
- Consumes: unchanged T1 cache and source-conditioned R2 family.
- Produces: W&B-synced per-epoch evidence, exact-best candidate and 200-game official report.

- [ ] Run five CUDA BF16 train batches plus full 1,000-decision validation; require finite loss and 100% legal actions.
- [ ] Launch V18 immediately after smoke with batch 256, validation batch 512, LR `3e-4`, weight decay `0.02`, seed `20260723`, patience 5 and min delta `0.001`.
- [ ] Monitor every epoch, verify W&B sync and export the earliest maximum greedy-exact checkpoint.
- [ ] Validate the candidate and run 200 official-engine games with `--workers 8 --worker-cpu-threads 1`.
- [ ] Create immutable evaluation backlink and compare V18 with V2 under the same catalog.

### Task 4: Authoritative documentation and audit

**Files:**
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.md`
- Modify: `experiments/0015_dragapult_conditioned_bc/DESIGN.html`
- Modify: `experiments/0015_dragapult_conditioned_bc/manifest.json`
- Create: `experiments/0015_dragapult_conditioned_bc/decisions/010_r2_strong_scenario_t1.md`

**Interfaces:**
- Consumes: V18 config, status, W&B state, exact metrics and official report.
- Produces: synchronized model shapes/data flow/stage conclusion and immutable provenance.

- [ ] Record that R2 uses the same 22 actor-visible input groups, a two-layer scenario bank, Goal-QKV and `(0, 2)` dynamic ScaleGates initialized at 1.0.
- [ ] Record the source-persona residual, parameter count, T1 row counts, exact-best epoch and official W/L/error.
- [ ] Parse all JSON, verify report hashes/backlinks, run 0015 tests, evaluation asset tests, scoped compile and `git diff --check`.
