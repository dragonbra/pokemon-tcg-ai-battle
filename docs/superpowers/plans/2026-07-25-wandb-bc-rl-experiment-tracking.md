# W&B BC/RL Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, free-tier-conscious W&B mirror that makes BC, value calibration, PPO, and frozen official-engine evaluation runs comparable without replacing canonical JSONL, TensorBoard, checkpoints, or evaluation HTML.

**Architecture:** `TrainingLogger` remains the single write path and writes canonical JSONL first. Optional TensorBoard and W&B sinks receive normalized scalar records after the JSONL flush; W&B failures never invalidate or truncate the local record. One repository `V<n>_<tag>` maps to one stable W&B run ID, while custom axes and namespaced metrics keep BC and RL histories distinct and frozen evaluation summaries provide the only cross-stage policy-quality comparison.

**Tech Stack:** Python 3.11+, standard-library `unittest`, optional `wandb` Python SDK, existing PyTorch TensorBoard writer, canonical JSONL, official-engine evaluation HTML.

## Global Constraints

- Never modify `engine/source/`; all policy-quality evidence comes from real official-engine evaluation.
- Keep `training_metrics.jsonl` authoritative, TensorBoard available, and W&B optional and failure-isolated.
- Preserve strict experiment and version paths under `rl_runs/artifact`, `rl_runs/tensorboard`, `rl_runs/checkpoint`, and `rl_runs/evaluation`.
- Never upload API keys, raw observations, complete traces, replay payloads, checkpoints, datasets, or source patches by default.
- Use one W&B project for comparable BC/RL lifecycle runs; use `group=experiment_id` and a stable repository-derived run ID per version.
- A resumed process may reuse a W&B run only when it resumes the same repository version; a new policy update requires the next `V<n>_<tag>`.
- Update `train/alakazam_bc_rl/DESIGN.html` in the implementation change because the training observation and run-record interface changes.

---

### Task 1: Optional W&B dependency and immutable run identity

**Files:**
- Modify: `pyproject.toml`
- Modify: `rl_environment/runs.py`
- Create: `tests/test_rl_wandb_identity.py`

**Interfaces:**
- Consumes: `TrainingPaths` and `training_paths(output: Path)`.
- Produces: `wandb_run_id(experiment_id: str, version_name: str) -> str` and manifest fields `tracking.wandb_run_id`, `tracking.wandb_project`, and `tracking.metric_schema`.

- [ ] **Step 1: Write identity tests**

Create tests asserting that `wandb_run_id("0012-alakazam_sota_feature_engineering", "V10_online_ppo")` is deterministic, at most 64 characters, contains only ASCII word characters or hyphens, and changes when either input changes. Assert that an initialized experiment records tracking as disabled without an API key.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python3 -m unittest -v tests.test_rl_wandb_identity`

Expected: failure because `wandb_run_id` and tracking manifest fields do not exist.

- [ ] **Step 3: Add the optional dependency and identity helper**

Add `wandb>=0.28,<1` to a new `tracking` optional-dependency extra rather than the mandatory or `rl` dependency sets. Implement a readable prefix plus SHA-256 suffix so IDs are stable and fit W&B's 64-character limit. Add only non-secret W&B defaults to the experiment manifest.

- [ ] **Step 4: Run the focused tests**

Run: `python3 -m unittest -v tests.test_rl_wandb_identity`

Expected: all tests pass.

### Task 2: Failure-isolated W&B scalar sink

**Files:**
- Create: `rl_environment/wandb_logging.py`
- Modify: `rl_environment/logging.py`
- Create: `tests/test_rl_wandb_logging.py`

**Interfaces:**
- Consumes: normalized `dict[str, Any]` records returned by `TrainingLogger.log`.
- Produces: `WandbSettings`, `WandbSink.log(record: dict[str, Any])`, `WandbSink.set_summary(values: dict[str, Any])`, and `WandbSink.close(exit_code: int = 0)`.

- [ ] **Step 1: Write fake-client unit tests**

Cover disabled mode, missing SDK, offline mode, stable `id` with `resume="allow"`, config/group/job-type/tag propagation, custom axes, numeric filtering, and an injected upload exception. Assert in every case that JSONL receives exactly one complete record before the W&B fake is called.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python3 -m unittest -v tests.test_rl_wandb_logging`

Expected: failure because the W&B sink is undefined.

- [ ] **Step 3: Implement settings and the optional sink**

Support `disabled`, `offline`, and `online` modes. Initialize a run with the repository version ID, experiment group, stage-specific job type, immutable config, and tags. Define `trainer/epoch`, `trainer/update`, `env/decisions`, and `env/episodes` as custom axes. Catch SDK import, initialization, logging, and finish errors; emit a local warning and continue without changing the canonical JSONL.

- [ ] **Step 4: Extend `TrainingLogger` without breaking callers**

Add a keyword-only optional sink parameter. Keep the current constructor calls valid. Flush JSONL, then TensorBoard, then W&B. Do not send nested objects, NaN, infinity, strings from per-step metric payloads, or high-cardinality case data through the scalar sink.

- [ ] **Step 5: Run logging tests**

Run: `python3 -m unittest -v tests.test_rl_wandb_logging`

Expected: all tests pass, including the injected network failure case.

### Task 3: Canonical BC/RL metric contract

**Files:**
- Create: `rl_environment/metric_contract.py`
- Modify: `train/alakazam_bc_rl/training/train_behavior_cloning.py`
- Modify: `train/alakazam_bc_rl/training/train_full_action_bc.py`
- Modify: `train/alakazam_bc_rl/training/calibrate_value.py`
- Modify: `train/alakazam_bc_rl/training/train_ppo.py`
- Create: `tests/test_rl_metric_contract.py`

**Interfaces:**
- Consumes: existing trainer metric dictionaries.
- Produces: `normalize_training_metrics(stage: str, axes: dict[str, int], metrics: dict[str, Any]) -> dict[str, Any]` using the schema documented in `docs/reports/rl/wandb-bc-rl-experiment-tracking-design-20260725.html`.

- [ ] **Step 1: Write schema tests**

Assert required identity axes, allowed namespaces, finite numeric values, and stage-specific required keys. Assert that BC loss and PPO loss remain separate and that no training-time rollout win rate is renamed to `eval/outcome/win_rate`.

- [ ] **Step 2: Run schema tests and confirm failure**

Run: `python3 -m unittest -v tests.test_rl_metric_contract`

Expected: failure because normalization is undefined.

- [ ] **Step 3: Implement normalization and wire all four trainers**

Use `bc/*`, `value/*`, `ppo/*`, `rollout/*`, `system/*`, and `trainer/*` namespaces. Preserve existing JSONL keys for backward compatibility during the first migration version, adding normalized aliases rather than silently renaming historical fields.

- [ ] **Step 4: Run trainer and schema tests**

Run: `python3 -m unittest -v tests.test_rl_metric_contract tests.test_full_action_bc tests.test_rl_framework`

Expected: all tests pass and every epoch still emits aligned train and validation metrics.

### Task 4: Frozen evaluation summaries and compact Tables

**Files:**
- Create: `evaluation/reporting/wandb_export.py`
- Modify: `evaluation/cli.py`
- Create: `tests/test_evaluation_wandb_export.py`

**Interfaces:**
- Consumes: finalized evaluation report manifest, aggregate metrics, catalog-ordered opponent results, and the source W&B run ID recorded in the version manifest.
- Produces: summary keys under `eval/outcome/*`, `eval/seat/*`, and `eval/quality/*`, plus a compact per-opponent table with one row per opponent and seat aggregate.

- [ ] **Step 1: Write exporter tests with a fake W&B run**

Assert official-engine provenance is required; verify win rate, completion rate, error rate, game count, average complete turns, first/second-player splits, opponent package, and report path. Assert no trace, replay frame, observation, or checkpoint bytes are uploaded.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `python3 -m unittest -v tests.test_evaluation_wandb_export`

Expected: failure because the exporter does not exist.

- [ ] **Step 3: Implement explicit post-report export**

Export only after the HTML report is finalized. Reopen the matching lifecycle run with `resume="must"`, update summary fields, log one compact Table, record the local report path and hash, and close. Make export opt-in and non-fatal; the local report remains complete when W&B is unavailable.

- [ ] **Step 4: Run evaluation tests**

Run: `python3 -m unittest -v tests.test_evaluation_wandb_export tests.test_evaluation_reporting`

Expected: all tests pass with catalog order preserved.

### Task 5: Documentation, smoke validation, and rollout gate

**Files:**
- Modify: `train/alakazam_bc_rl/DESIGN.html`
- Modify: `rl_environment/README.md`
- Modify: `rl_runs/README.md`
- Modify: `docs/reports/README.md`

**Interfaces:**
- Consumes: completed logger, metric schema, and evaluation exporter.
- Produces: operator instructions for disabled/offline/online modes and a documented go/no-go gate for broader adoption.

- [ ] **Step 1: Document credentials and modes**

Document `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`, and repository-local `WANDB_DIR`. State that secrets never enter Git and that `wandb sync` is a deliberate external upload action.

- [ ] **Step 2: Document the comparison contract**

Explain that only frozen official-engine `eval/*` summaries compare policy quality across BC and RL; trainer loss, rollout win rate, and wall-clock throughput are diagnostic within compatible cohorts.

- [ ] **Step 3: Run an offline smoke**

Run a minimal BC smoke with W&B offline, confirm canonical JSONL and TensorBoard output, inspect the offline run directory, and do not sync it externally without explicit authorization.

- [ ] **Step 4: Run repository validation**

Run: `python3 -m unittest discover -s tests -p 'test_*.py'`

Run: `python3 -m compileall -q evaluation visualization rl_environment train`

Expected: both commands succeed.

- [ ] **Step 5: Adopt only after the smoke gate**

Require zero canonical logging regressions, zero secret or large-file uploads, a W&B-disabled run identical in local metrics to the baseline, and an offline run that can be inspected locally. Online synchronization remains a separate user-authorized action.
