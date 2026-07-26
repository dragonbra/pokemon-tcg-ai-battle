# Historical W&B Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror every recoverable historical TensorBoard run into the existing private W&B project without changing metric values, steps, or source-of-truth artifacts.

**Architecture:** Add a standalone `rl_environment.wandb_backfill` CLI that discovers TensorBoard event files below `rl_runs/tensorboard`, uses matching canonical JSONL records when available, and otherwise reads scalar events directly. It writes one deterministic W&B run per experiment/version, records source hashes and provenance in W&B config/summary, and emits a local JSON audit report so reruns are inspectable and idempotent.

**Tech Stack:** Python 3.11+, TensorBoard event accumulator, existing `wandb` SDK, standard-library `unittest`.

## Global Constraints

- Upload only finite scalar metrics from committed historical JSONL or TensorBoard event files.
- Prefer `training_metrics.jsonl` for matching historical versions; TensorBoard is the fallback and fills runs lacking JSONL.
- Preserve the original scalar step; do not manufacture timestamps, checkpoints, traces, or evaluation results.
- Mark every mirrored run with `historical_backfill=true`, metric source, and SHA-256 source hashes.
- Target private W&B project `dragon_bra/pokemon-tcg-policy-learning` and never upload checkpoints, datasets, replays, traces, or source code.
- Skip event files with no scalar records and report them locally instead of creating empty W&B runs.
- Keep local backfill staging/audit outputs under `.tmp/wandb/`, which remains Git-ignored.

---

### Task 1: Parse and normalize historical metric sources

**Files:**
- Create: `rl_environment/wandb_backfill.py`
- Test: `tests/test_rl_wandb_backfill.py`

**Interfaces:**
- Produces: `HistoricalRun`, `MetricRecord`, `discover_historical_runs(tensorboard_root, artifact_root) -> list[HistoricalRun]`.
- Produces: `load_metric_records(run) -> list[MetricRecord]`, where a record contains `step: int` and `metrics: dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prefers_matching_jsonl_over_tensorboard(tmp_path):
    run = make_historical_run(tmp_path, jsonl_records=[{"step": 1, "train/loss": 0.5}])
    assert load_metric_records(run) == [MetricRecord(step=1, metrics={"train/loss": 0.5})]


def test_reads_scalar_events_when_jsonl_is_missing(tmp_path):
    run = make_historical_run(tmp_path, tensorboard_records=[("train/loss", 1, 0.5)])
    assert load_metric_records(run) == [MetricRecord(step=1, metrics={"train/loss": 0.5})]


def test_skips_event_file_without_scalars(tmp_path):
    run = make_historical_run(tmp_path)
    assert load_metric_records(run) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: FAIL because `rl_environment.wandb_backfill` does not exist.

- [ ] **Step 3: Implement minimal discovery and parsing**

```python
@dataclass(frozen=True)
class MetricRecord:
    step: int
    metrics: dict[str, float]


def load_metric_records(run: HistoricalRun) -> list[MetricRecord]:
    if run.jsonl_path is not None:
        return _load_jsonl_records(run.jsonl_path)
    return _load_tensorboard_records(run.event_path)
```

Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` with scalar-size guidance `0` to retain every event. Filter non-finite values. Merge all scalar tags at equal steps into one `MetricRecord`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: PASS.

### Task 2: Mirror normalized history with provenance

**Files:**
- Modify: `rl_environment/wandb_backfill.py`
- Modify: `tests/test_rl_wandb_backfill.py`

**Interfaces:**
- Consumes: `HistoricalRun` and `list[MetricRecord]` from Task 1.
- Produces: `mirror_run(run, records, *, sdk) -> BackfillResult`.
- Produces: W&B run ID `historical-<stable hash>` and config containing original experiment/version, source kind, paths relative to repository root, and SHA-256 hashes.

- [ ] **Step 1: Write the failing tests**

```python
def test_mirror_logs_original_steps_and_backfill_provenance(tmp_path):
    run = make_historical_run(tmp_path, jsonl_records=[{"step": 3, "train/loss": 0.25}])
    sdk = FakeWandb()
    result = mirror_run(run, load_metric_records(run), sdk=sdk)
    assert sdk.init_calls[0]["project"] == "pokemon-tcg-policy-learning"
    assert sdk.init_calls[0]["config"]["historical_backfill"] is True
    assert sdk.run.logged == [{"trainer/epoch": 3, "bc/train/loss": 0.25}]
    assert result.status == "uploaded"


def test_mirror_does_not_create_run_for_empty_history(tmp_path):
    sdk = FakeWandb()
    result = mirror_run(make_historical_run(tmp_path), [], sdk=sdk)
    assert result.status == "skipped_empty"
    assert sdk.init_calls == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: FAIL because `mirror_run` does not exist.

- [ ] **Step 3: Implement the W&B mirror**

```python
def mirror_run(run: HistoricalRun, records: list[MetricRecord], *, sdk: Any) -> BackfillResult:
    if not records:
        return BackfillResult(run=run, status="skipped_empty", record_count=0)
    settings = _historical_settings(run)
    wandb_run = sdk.init(**settings)
    for record in records:
        wandb_run.log(_wandb_payload(run.job_type, record))
    wandb_run.summary.update(_summary(run, records))
    wandb_run.finish(exit_code=0)
    return BackfillResult(run=run, status="uploaded", record_count=len(records))
```

Use the existing `_wandb_metric_name` and job-type inference rules from `rl_environment.wandb_logging`. Use W&B `resume="allow"` only for online mode, and set `save_code=False` and `disable_code=True`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: PASS.

### Task 3: Add audited CLI and execute the backfill

**Files:**
- Modify: `rl_environment/wandb_backfill.py`
- Modify: `tests/test_rl_wandb_backfill.py`
- Create at runtime: `.tmp/wandb/historical-backfill-2026-07-26.json`

**Interfaces:**
- Produces: `python3 -m rl_environment.wandb_backfill --mode online --audit <path>`.
- Produces: an audit document listing every discovered run, source paths/hashes, metric count, W&B run ID, and `uploaded` or `skipped_empty` status.

- [ ] **Step 1: Write the failing CLI audit test**

```python
def test_cli_dry_run_writes_audit_for_uploaded_and_empty_runs(tmp_path):
    exit_code = main([
        "--tensorboard-root", str(tmp_path / "tensorboard"),
        "--artifact-root", str(tmp_path / "artifact"),
        "--mode", "dry-run",
        "--audit", str(tmp_path / "audit.json"),
    ])
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert exit_code == 0
    assert {item["status"] for item in audit["runs"]} == {"dry_run", "skipped_empty"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: FAIL because `main` does not create the audit file.

- [ ] **Step 3: Implement CLI, then run a dry-run audit**

```bash
python3 -m rl_environment.wandb_backfill \
  --mode dry-run \
  --audit .tmp/wandb/historical-backfill-2026-07-26.json
```

Require explicit `--mode online` to invoke the W&B SDK. Default roots are `rl_runs/tensorboard` and `rl_runs/artifact`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest -v tests.test_rl_wandb_backfill`

Expected: PASS.

- [ ] **Step 5: Authenticate, upload once, and verify audit results**

Run after the user performs interactive login:

```bash
python3 -m wandb login
python3 -m rl_environment.wandb_backfill \
  --mode online \
  --audit .tmp/wandb/historical-backfill-2026-07-26.json
```

Expected: Every non-empty run is `uploaded`; runs with zero scalar records are `skipped_empty`; the audit’s aggregate counts match the console summary.

- [ ] **Step 6: Run focused regression tests**

Run:

```bash
python3 -m unittest -v tests.test_rl_wandb_backfill tests.test_rl_wandb_logging tests.test_rl_wandb_identity
```

Expected: PASS.
