# 0030 Best Checkpoint Audit Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuously detect strict 0019 frozen-evaluation improvements in the active 0030 RL run, preserve the checkpoint, export it, and publish an independent 510-game official-engine Frozen 0019 report without interrupting training.

**Architecture:** A foreground polling process reads the canonical `training_metrics.jsonl` and persists an atomic audit state under `.tmp/training_monitor/`. Pure selection helpers decide whether a record is a strict new best and allocate the next repository version. The orchestrator records an attempt before expensive work, copies the model-only checkpoint into the new immutable version, exports a candidate, invokes the existing evaluation CLI, and lets the evaluation framework publish the report, index, and provenance backlink.

**Tech Stack:** Python 3.11, standard library JSON/subprocess/pathlib, PyTorch model-only checkpoints, existing `rl_environment.runs` version APIs, existing `evaluation` CLI, unittest.

## Global Constraints

- Do not modify `engine/source/`; all strength evidence must use the official engine runtime.
- Poll `eval/foundation_0019/win_rate`; never merge it with the 0028 evaluation namespace.
- The initial best is update 5 at `0.6078431372549019`; later triggers must be strictly greater.
- Each formal audit runs all 51 Frozen 0019 decks with 10 games each and both policies resident on `cuda:0`.
- Audit failures must be recorded but must never stop or signal the active RL training process.
- Record an attempted update atomically before export or evaluation so restart cannot duplicate it.
- Formal reports go to `experiments/0030_dragapult_shared_encoder_decoder_rl/evaluation/V<n>_update<n>_frozen0019_audit.html` with a matching runtime backlink.
- Candidate export directories and watcher state remain ignored runtime assets; reports and evaluation index are Git-trackable.
- Never submit to Kaggle or perform any externally visible submission action.
- Do not modify or commit unrelated 0031 worktree changes.

---

### Task 1: Strict-Best Selection And Durable State

**Files:**
- Create: `train/0030_dragapult_shared_encoder_decoder_rl/monitor_best_checkpoints.py`
- Create: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_monitor_best_checkpoints.py`

**Interfaces:**
- Consumes: JSON objects from `artifact/training_metrics.jsonl` containing `trainer/update`, `eval/checkpoint_update`, and `eval/foundation_0019/win_rate`.
- Produces: `evaluation_record(record) -> EvaluationPoint | None`, `select_new_bests(records, state) -> list[EvaluationPoint]`, and atomic `AuditState` JSON with attempted/completed/failed updates.

- [ ] **Step 1: Write failing tests for strict improvement, equal/worse suppression, and restart idempotence**

```python
def test_selects_only_strict_new_best(self):
    state = AuditState.initial(5, 62 / 102)
    points = select_new_bests([
        metric(10, 57 / 102), metric(15, 62 / 102), metric(20, 63 / 102)
    ], state)
    self.assertEqual([point.update for point in points], [20])

def test_attempted_update_is_not_selected_after_restart(self):
    state = AuditState.initial(5, 62 / 102).mark_attempted(EvaluationPoint(20, 63 / 102))
    self.assertEqual(select_new_bests([metric(20, 63 / 102)], state), [])
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_monitor_best_checkpoints`

Expected: FAIL because `monitor_best_checkpoints` does not exist.

- [ ] **Step 3: Implement immutable evaluation points and atomic state transitions**

```python
@dataclass(frozen=True)
class EvaluationPoint:
    update: int
    win_rate: float

def evaluation_record(record: dict[str, Any]) -> EvaluationPoint | None:
    if "eval/foundation_0019/win_rate" not in record:
        return None
    update = int(record.get("eval/checkpoint_update", record["trainer/update"]))
    return EvaluationPoint(update, float(record["eval/foundation_0019/win_rate"]))
```

State writes use a sibling temporary file followed by `Path.replace`; `mark_attempted` advances the observed best before any subprocess starts.

- [ ] **Step 4: Run the focused selection tests**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_monitor_best_checkpoints`

Expected: strict-best and idempotence tests PASS.

### Task 2: Immutable Formal Audit Orchestration

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/monitor_best_checkpoints.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_monitor_best_checkpoints.py`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/manifest.json`

**Interfaces:**
- Consumes: `rl_environment.runs.initialize_version(PROJECT_ID, version_name)`, `export_candidate.export_candidate`, and `python3 -m evaluation --pool frozen run`.
- Produces: `allocate_audit_version(update) -> VersionPaths`, `audit_checkpoint(point, config, state)`, a preserved checkpoint copy, formal HTML/index/backlink, and structured heartbeat/error events.

- [ ] **Step 1: Write failing tests for missing checkpoint recording and monotonic version naming**

```python
def test_missing_checkpoint_records_failure_without_subprocess(self):
    result = audit_checkpoint(EvaluationPoint(20, 63 / 102), config_with_missing_checkpoint())
    self.assertEqual(result.status, "failed")
    self.assertEqual(result.reason, "checkpoint_missing")

def test_next_version_is_monotonic(self):
    self.assertEqual(next_audit_version(["V1_a", "V3_c"], 20), "V4_update20_frozen0019_audit")
```

- [ ] **Step 2: Run the orchestration tests and verify they fail**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_monitor_best_checkpoints`

Expected: FAIL because orchestration helpers are not implemented.

- [ ] **Step 3: Repair the formal experiment manifest**

Set `status` to `initialized`; add non-empty `deck`, `expert_source`, `dataset_contract`, and `engine_revision`; describe the equal-weight immutable 0019/0028 training foundations and preserve exact source identities in the existing fields.

- [ ] **Step 4: Implement checkpoint preservation, export, and formal evaluation**

```python
command = [
    sys.executable, "-m", "evaluation", "--pool", "frozen", "run",
    "--candidate", str(candidate), "--opponents", "all", "--games", "10",
    "--output", str(paths.evaluation), "--workers", "8",
    "--worker-cpu-threads", "1", "--candidate-device", "cuda:0",
    "--opponent-device", "cuda:0",
]
```

Call `initialize_version` before copying the source checkpoint to `paths.checkpoints`; write an evaluation-only config and status. On failure, retain the allocated version and write `state=failed`, but return to polling.

- [ ] **Step 5: Run focused and project tests**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_monitor_best_checkpoints train.0030_dragapult_shared_encoder_decoder_rl.tests.test_contract train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_diagnostics train.0030_dragapult_shared_encoder_decoder_rl.tests.test_worker_imports`

Expected: all 0030 tests PASS.

### Task 3: Publish Baseline Audit And Start Foreground Watcher

**Files:**
- Create: `experiments/0030_dragapult_shared_encoder_decoder_rl/evaluation/V4_update5_frozen0019_audit.html`
- Create: `experiments/0030_dragapult_shared_encoder_decoder_rl/evaluation/index.html`
- Create: `rl_runs/0030_dragapult_shared_encoder_decoder_rl/versions/V4_update5_frozen0019_audit/artifact/evaluation.json`

**Interfaces:**
- Consumes: V3 `update-0005.pt`, the self-contained exporter, and the formal evaluation framework.
- Produces: a Git-trackable 510-game baseline report and a long-lived watcher terminal session for later strict bests.

- [ ] **Step 1: Seed the watcher state and formally audit update 5**

Run:

```bash
python3 -m train.0030_dragapult_shared_encoder_decoder_rl.monitor_best_checkpoints \
  --training-version V3_lightweight_engine_workers \
  --audit-update 5 --audit-win-rate 0.6078431372549019 --once
```

Expected: V4 is allocated, 510/510 games complete with zero errors, and report/index/backlink exist.

- [ ] **Step 2: Validate formal provenance and report summary**

Run:

```bash
python3 -m evaluation validate evaluation/arena/candidates/0030_dragapult_ex_001_v3_update5
python3 -m unittest -v tests.test_experiment_projects
```

Expected: exact 60-card package validation passes and formal project tests pass.

- [ ] **Step 3: Start the continuous foreground watcher**

Run:

```bash
python3 -m train.0030_dragapult_shared_encoder_decoder_rl.monitor_best_checkpoints \
  --training-version V3_lightweight_engine_workers --interval-seconds 30
```

Expected: a foreground exec session emits `BEST_CHECKPOINT_MONITOR_HEARTBEAT`; later strict improvements emit attempt, completion, or alert records without terminating training.

- [ ] **Step 4: Commit only the 0030 implementation, plan, manifest, and formal report assets**

```bash
git add docs/superpowers/plans/2026-08-04-0030-best-checkpoint-audit-watcher.md \
  train/0030_dragapult_shared_encoder_decoder_rl/monitor_best_checkpoints.py \
  train/0030_dragapult_shared_encoder_decoder_rl/tests/test_monitor_best_checkpoints.py \
  experiments/0030_dragapult_shared_encoder_decoder_rl/manifest.json \
  experiments/0030_dragapult_shared_encoder_decoder_rl/evaluation
git commit -m "feat: audit 0030 best checkpoints"
```

Expected: the commit contains no `0031` paths, ignored candidate packages, checkpoints, W&B staging, or `.tmp` state.
