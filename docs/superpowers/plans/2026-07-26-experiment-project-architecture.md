# 实验项目目录架构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `0013` 起启用以 `train/<project_id>`、`experiments/<project_id>` 和 `rl_runs/<project_id>` 为核心的独立实验项目架构，并保留 `0001`–`0012` 历史资产不移动。

**Architecture:** 将新项目的可运行业务实现、长期可读档案和运行时物理资产分离。`rl_environment.runs` 变为唯一的项目/版本路径分配器，并为每个版本返回同目录下的 artifact、checkpoint、TensorBoard 与 W&B staging 路径；正式 evaluation 归档到 `experiments/<project_id>/evaluation/`，由 artifact 保存双向追溯元数据。

**Tech Stack:** Python 3.11+、标准库 `unittest`、JSON、Markdown、HTML、现有 `rl_environment` CLI。

## Global Constraints

- `0013` 起的项目 ID 格式为 `<四位全局递增编号>_<ascii_snake_case_slug>`，例如 `0013_alakazam_rollout_value_calibration`。
- 项目内版本格式固定为 `V<n>_<ascii_snake_case_tag>`，`n` 从 1 起严格递增且不可复用。
- `rl_environment/` 不得包含 deck、expert/source、feature schema、模型、reward、loss 或具体训练算法。
- 新项目必须同时拥有 `train/<project_id>/`、`experiments/<project_id>/`、`rl_runs/<project_id>/`；项目档案为唯一人工与 AI 导航入口。
- 正式官方 engine evaluation 的权威 HTML 归档在 `experiments/<project_id>/evaluation/`；候选 package 仍只导出到 `evaluation/arena/candidates/<candidate_name>/`。
- `0001`–`0012` 的现有路径、报告和历史链接不得移动或覆盖。
- 任何官方能力结论仍必须来自官方 engine runtime 的真实对局。
- 不修改 `engine/source/`。
- 不创建 Git commit，除非用户另行明确要求。

---

## File structure

### New files

- `experiments/README.md` — 新项目架构的总入口与使用规则。
- `experiments/INDEX.md` — 新项目人工可读索引，初始时包含表头与 `0013` 起的条目格式。
- `experiments/INDEX.html` — 新项目 HTML 总览，链接每个项目档案。
- `experiments/legacy/INDEX.md` — `0001`–`0012` 兼容索引，保留旧 `rl_runs`、训练实现与评测入口链接。
- `tests/test_experiment_projects.py` — 项目初始化、manifest 合同、版本分配、拒绝覆盖与评测追溯测试。

### Modified files

- `rl_environment/runs.py` — 新项目 ID 校验、三目录原子项目初始化、嵌套版本运行路径、版本递增检查、运行 artifact 初始化和评测归档引用。
- `rl_environment/README.md` — 共享层边界和新 CLI/目录契约。
- `rl_runs/README.md` — 说明新结构与历史只读结构的边界，删除将新项目写入顶层 asset-group 的指引。
- `CLAUDE.md` — 将 `rl_runs/artifact|checkpoint|tensorboard|evaluation` 平铺规则替换为新项目结构，并声明 `experiments/` 为项目档案与 DESIGN 权威位置。
- `AGENTS.md` — 与 `CLAUDE.md` 保持一致，移除 `work/` 等过时候选 package 约定。
- `.gitignore` — 针对嵌套 `rl_runs/<project>/dataset|checkpoint|wandb|tensorboard` 路径更新 ignore 规则，同时让 tracked artifact 与 experiments 内容保持可见。
- `pyproject.toml` — 如现有 package-data 或 test 配置需要，更新新测试/资源约定；否则不修改。

### Files deliberately unchanged

- `rl_runs/artifact/0008-*` 到 `rl_runs/artifact/0012-*`、`rl_runs/evaluation/0001-*` 到 `rl_runs/evaluation/0012-*`、`rl_runs/tensorboard/0001-*` 到 `rl_runs/tensorboard/0012-*`。
- 所有现有 `train/alakazam_*` 与 `train/kaggle_bc_top20` 训练实现。
- `evaluation/arena/candidates/`、`evaluation/arena/opponents/` 和 `engine/source/`。

## Task 1: Define project identity and immutable path contracts

**Files:**
- Modify: `rl_environment/runs.py:14-35`
- Create: `tests/test_experiment_projects.py`

**Interfaces:**
- Produces `PROJECT_ID = re.compile(r"^(?P<number>\d{4})_(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*)$")`.
- Produces `VersionPaths` dataclass with `project_id: str`, `version_name: str`, `project_archive: Path`, `run_root: Path`, `artifact: Path`, `checkpoints: Path`, `tensorboard: Path`, `wandb: Path`, `evaluation: Path`, `config: Path`, `metrics: Path`, `summary: Path`, and `status: Path`.
- Consumes only `Path` and project/version strings; no deck/model-specific fields.

- [ ] **Step 1: Write failing project-ID and version-path tests**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rl_environment.runs import project_version_paths


class ExperimentProjectPathTests(unittest.TestCase):
    def test_project_version_paths_nest_all_runtime_assets_under_one_project(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Path(directory)
            with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
                paths = project_version_paths(
                    "0013_alakazam_rollout_value_calibration",
                    "V1_initial_contract",
                )

            self.assertEqual(
                paths.artifact,
                repository / "rl_runs/0013_alakazam_rollout_value_calibration/versions/V1_initial_contract/artifact",
            )
            self.assertEqual(
                paths.checkpoints,
                repository / "rl_runs/0013_alakazam_rollout_value_calibration/versions/V1_initial_contract/checkpoint",
            )
            self.assertEqual(
                paths.evaluation,
                repository / "experiments/0013_alakazam_rollout_value_calibration/evaluation/V1_initial_contract.html",
            )

    def test_project_version_paths_reject_hyphenated_project_slug(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid project ID"):
            project_version_paths("0013-alakazam", "V1_initial_contract")
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectPathTests
```

Expected: FAIL because `project_version_paths` is not defined.

- [ ] **Step 3: Add project path constants, `VersionPaths`, and pure path resolver**

In `rl_environment/runs.py`, replace the old new-project constants with repository-relative roots and add:

```python
PROJECT_ID = re.compile(r"^(?P<number>\d{4})_(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*)$")


@dataclass(frozen=True)
class VersionPaths:
    project_id: str
    version_name: str
    project_archive: Path
    run_root: Path
    artifact: Path
    checkpoints: Path
    tensorboard: Path
    wandb: Path
    evaluation: Path
    config: Path
    metrics: Path
    summary: Path
    status: Path


def project_version_paths(project_id: str, version_name: str) -> VersionPaths:
    if PROJECT_ID.fullmatch(project_id) is None:
        raise ValueError(f"invalid project ID: {project_id}")
    if VERSIONED_ATTEMPT.fullmatch(version_name) is None:
        raise ValueError(f"invalid version name: {version_name}")
    project_archive = REPOSITORY_ROOT / "experiments" / project_id
    run_root = REPOSITORY_ROOT / "rl_runs" / project_id / "versions" / version_name
    artifact = run_root / "artifact"
    return VersionPaths(
        project_id=project_id,
        version_name=version_name,
        project_archive=project_archive,
        run_root=run_root,
        artifact=artifact,
        checkpoints=run_root / "checkpoint",
        tensorboard=run_root / "tensorboard",
        wandb=run_root / "wandb",
        evaluation=project_archive / "evaluation" / f"{version_name}.html",
        config=artifact / "training_config.json",
        metrics=artifact / "training_metrics.jsonl",
        summary=artifact / "training_summary.json",
        status=artifact / "status.json",
    )
```

Do not delete legacy `TrainingPaths`, `training_paths`, or `numbered_artifact_path` in this task; existing 0001–0012 callers remain supported until their consumers are explicitly migrated.

- [ ] **Step 4: Run targeted tests to verify they pass**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectPathTests tests.test_run_versioning
```

Expected: PASS.

## Task 2: Create an atomic project scaffold and validated manifest

**Files:**
- Modify: `rl_environment/runs.py:131-202`
- Modify: `tests/test_experiment_projects.py`

**Interfaces:**
- Consumes `initialize_project(slug: str, *, objective: str, deck: str, expert_source: str, dataset_contract: str, engine_revision: str, opponent_pool_snapshot: str) -> ProjectPaths`.
- Produces project directories and `experiments/<project_id>/manifest.json` with `schema_version: "ptcg_experiment_project_v1"`.
- Produces `ProjectPaths` dataclass containing `project_id`, `train`, `archive`, `runs`, and `manifest`.

- [ ] **Step 1: Write failing scaffold tests**

```python
from rl_environment.runs import initialize_project


def test_initialize_project_creates_three_project_roots_and_manifest(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository), patch(
            "rl_environment.runs._git_value", return_value="abc123"
        ):
            paths = initialize_project(
                "alakazam_rollout_value_calibration",
                objective="Calibrate value targets from official-engine rollouts",
                deck="alakazam_dudunsparce",
                expert_source="team_policy_2026_07",
                dataset_contract="single_team_episode_v1",
                engine_revision="official-engine-2026-07-26",
                opponent_pool_snapshot="catalog-sha256:abc",
            )

        self.assertEqual(paths.project_id, "0001_alakazam_rollout_value_calibration")
        self.assertTrue(paths.train.is_dir())
        self.assertTrue(paths.archive.is_dir())
        self.assertTrue(paths.runs.is_dir())
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["deck"], "alakazam_dudunsparce")
        self.assertEqual(manifest["paths"]["training"], "train/0001_alakazam_rollout_value_calibration")


def test_initialize_project_rejects_missing_contract_field(self) -> None:
    with TemporaryDirectory() as directory:
        with patch("rl_environment.runs.REPOSITORY_ROOT", Path(directory)):
            with self.assertRaisesRegex(ValueError, "expert_source must not be empty"):
                initialize_project(
                    "alakazam_rollout",
                    objective="value calibration",
                    deck="alakazam",
                    expert_source="",
                    dataset_contract="dataset_v1",
                    engine_revision="official",
                    opponent_pool_snapshot="snapshot_v1",
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectScaffoldTests
```

Expected: FAIL because `initialize_project` is not defined.

- [ ] **Step 3: Implement `ProjectPaths`, next-ID discovery and atomic initialization**

Add these constructs in `rl_environment/runs.py`:

```python
@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    train: Path
    archive: Path
    runs: Path
    manifest: Path


def _next_project_id() -> int:
    roots = (REPOSITORY_ROOT / "train", REPOSITORY_ROOT / "experiments", REPOSITORY_ROOT / "rl_runs")
    ids: set[int] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            match = PROJECT_ID.fullmatch(child.name)
            if child.is_dir() and match is not None:
                ids.add(int(match.group("number")))
    return max(ids, default=12) + 1


def _required_value(name: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value
```

`initialize_project` must normalize its `slug` to ASCII snake_case, allocate the next number with `_next_project_id`, check all three target directories are absent before writing, then create:

```text
train/<id>/
experiments/<id>/data_audit/
experiments/<id>/decisions/
experiments/<id>/versions/
experiments/<id>/evaluation/
rl_runs/<id>/dataset/
rl_runs/<id>/versions/
```

Write `manifest.json` atomically by first writing `manifest.json.tmp`, then replacing it. Include `schema_version`, `project_id`, `objective`, all five required contract fields, `status: "initialized"`, UTC `created_at`, Git commit/status, and relative paths for training/archive/runs. If any directory creation or manifest write fails, remove only directories created by this invocation and re-raise the exception.

- [ ] **Step 4: Run targeted tests to verify scaffold behavior**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectScaffoldTests tests.test_run_versioning
```

Expected: PASS.

- [ ] **Step 5: Add CLI command without breaking legacy `create`**

Extend `_main()` so this command prints the archive directory after creating a new project:

```bash
python3 -m rl_environment.runs create-project alakazam_rollout_value_calibration \
  --objective "Calibrate value targets from official-engine rollouts" \
  --deck alakazam_dudunsparce \
  --expert-source team_policy_2026_07 \
  --dataset-contract single_team_episode_v1 \
  --engine-revision official-engine-2026-07-26 \
  --opponent-pool-snapshot catalog-sha256:abc
```

Keep the existing `create` subcommand intact for historical tooling. Do not make it create new-format projects.

- [ ] **Step 6: Verify CLI help and all targeted tests**

Run:

```bash
python3 -m rl_environment.runs --help
python3 -m unittest -v tests.test_experiment_projects tests.test_run_versioning
```

Expected: help lists `create-project`; all tests PASS.

## Task 3: Allocate immutable version directories and record evaluation provenance

**Files:**
- Modify: `rl_environment/runs.py`
- Modify: `tests/test_experiment_projects.py`

**Interfaces:**
- Consumes `initialize_version(project_id: str, version_name: str) -> VersionPaths`.
- Consumes `record_evaluation(paths: VersionPaths, *, run_id: str, report_sha256: str) -> None` after the report exists.
- Produces version directories, initial `status.json`, and `artifact/evaluation.json` with report path, hash and run ID.

- [ ] **Step 1: Write failing version-allocation and provenance tests**

```python
from rl_environment.runs import initialize_version, record_evaluation


def test_initialize_version_creates_all_runtime_paths_once(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        project_id = "0013_alakazam_rollout"
        (repository / "experiments" / project_id).mkdir(parents=True)
        (repository / "rl_runs" / project_id / "versions").mkdir(parents=True)
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
            paths = initialize_version(project_id, "V1_initial_contract")
            self.assertTrue(paths.artifact.is_dir())
            self.assertTrue(paths.checkpoints.is_dir())
            self.assertTrue(paths.tensorboard.is_dir())
            self.assertTrue(paths.wandb.is_dir())
            self.assertEqual(json.loads(paths.status.read_text())["state"], "allocated")
            with self.assertRaises(FileExistsError):
                initialize_version(project_id, "V1_initial_contract")


def test_record_evaluation_requires_existing_report_and_writes_backlink(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        project_id = "0013_alakazam_rollout"
        archive = repository / "experiments" / project_id / "evaluation"
        archive.mkdir(parents=True)
        (repository / "rl_runs" / project_id / "versions").mkdir(parents=True)
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
            paths = initialize_version(project_id, "V1_initial_contract")
            with self.assertRaises(FileNotFoundError):
                record_evaluation(paths, run_id="run-a", report_sha256="abc")
            paths.evaluation.write_text("<html></html>\n", encoding="utf-8")
            record_evaluation(paths, run_id="run-a", report_sha256="abc")

        record = json.loads((paths.artifact / "evaluation.json").read_text())
        self.assertEqual(record["run_id"], "run-a")
        self.assertEqual(record["report"], "experiments/0013_alakazam_rollout/evaluation/V1_initial_contract.html")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectVersionTests
```

Expected: FAIL because `initialize_version` and `record_evaluation` are not defined.

- [ ] **Step 3: Implement strict version initialization**

Implement `initialize_version` with these exact guards:

1. `experiments/<project_id>/manifest.json` must exist; otherwise raise `FileNotFoundError`.
2. `V<n>_<tag>` must match `VERSIONED_ATTEMPT`.
3. Read existing directories under `rl_runs/<project_id>/versions/`; parse all valid `Vn` names. If any version exists, the requested number must equal the greatest existing number plus one. Otherwise it must be `V1`.
4. Reject if `rl_runs/<project_id>/versions/<version_name>` exists, or if archive evaluation `<version_name>.html` already exists.
5. Create `artifact/`, `checkpoint/`, `tensorboard/`, and `wandb/` together; write `artifact/status.json` as:

```json
{
  "state": "allocated",
  "version": "V1_initial_contract"
}
```

Use temporary files and `Path.replace()` for JSON writes. On failure, remove only the new version root and do not delete existing project assets.

- [ ] **Step 4: Implement report provenance recording**

`record_evaluation` must require `paths.evaluation.is_file()`, then write `artifact/evaluation.json` with:

```json
{
  "report": "experiments/<project_id>/evaluation/<version_name>.html",
  "report_sha256": "<provided digest>",
  "run_id": "<provided run id>",
  "version": "<version name>"
}
```

Reject a pre-existing `artifact/evaluation.json` to preserve immutable version provenance.

- [ ] **Step 5: Run tests to verify success and legacy compatibility**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects tests.test_run_versioning
```

Expected: PASS.

## Task 4: Add the project archive templates and top-level discovery indexes

**Files:**
- Create: `experiments/README.md`
- Create: `experiments/INDEX.md`
- Create: `experiments/INDEX.html`
- Create: `experiments/legacy/INDEX.md`
- Modify: `rl_environment/runs.py`
- Modify: `tests/test_experiment_projects.py`

**Interfaces:**
- Consumes `initialize_project` from Task 2.
- Produces archive templates `README.md`, `DESIGN.md`, `DESIGN.html`, `versions/README.md`, and `evaluation/index.html` in each newly initialized project.
- Produces `refresh_experiment_index() -> None` to render stable Markdown and HTML indexes from `experiments/*/manifest.json`.

- [ ] **Step 1: Write failing archive-template and index tests**

```python
from rl_environment.runs import initialize_project, refresh_experiment_index


def test_initialized_project_has_navigation_and_design_templates(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
            paths = initialize_project(
                "alakazam_value",
                objective="value calibration",
                deck="alakazam",
                expert_source="team_policy",
                dataset_contract="dataset_v1",
                engine_revision="official",
                opponent_pool_snapshot="catalog_v1",
            )

        self.assertTrue((paths.archive / "README.md").is_file())
        self.assertTrue((paths.archive / "DESIGN.md").is_file())
        self.assertTrue((paths.archive / "DESIGN.html").is_file())
        self.assertTrue((paths.archive / "evaluation/index.html").is_file())


def test_refresh_experiment_index_links_project_design_and_evaluation(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        archive = repository / "experiments" / "0013_alakazam_value"
        archive.mkdir(parents=True)
        (archive / "manifest.json").write_text(
            json.dumps({"project_id": "0013_alakazam_value", "objective": "value calibration", "status": "initialized"}),
            encoding="utf-8",
        )
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
            refresh_experiment_index()

        index = (repository / "experiments" / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("0013_alakazam_value/DESIGN.html", index)
        self.assertIn("0013_alakazam_value/evaluation/index.html", index)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentArchiveTests
```

Expected: FAIL because templates and `refresh_experiment_index` do not exist.

- [ ] **Step 3: Add new-project archive templates**

Have `initialize_project` create these minimal, intentional templates:

- `README.md`: project hypothesis, status, links to `DESIGN.md`, `manifest.json`, `data_audit/`, `decisions/`, `versions/`, `evaluation/index.html`, training implementation and `rl_runs` root.
- `DESIGN.md`: title, project ID, and required sections `Current Stage`, `Input and Action Contract`, `Model and Objective`, `Data Contract`, `Evaluation Contract`, `Next Decision`.
- `DESIGN.html`: a small static HTML page linking to `DESIGN.md`, `README.md`, `manifest.json`, and `evaluation/index.html`; do not copy training claims into a generated placeholder.
- `versions/README.md`: convention for one Markdown decision record per V.
- `evaluation/index.html`: empty-project table with columns version, candidate, run ID, games, W/L/error, win rate, completion rate, report.

The templates must state that they are initialized outlines requiring completion before a training version can start, not fabricated experiment facts.

- [ ] **Step 4: Implement deterministic top-level index refresh**

`refresh_experiment_index` scans direct child directories of `experiments/`, skips `legacy`, requires a readable `manifest.json`, sorts by `project_id`, and writes:

- `INDEX.md`: a Markdown table containing project ID, objective, status, and relative links to `README.md`, `DESIGN.html`, `evaluation/index.html`.
- `INDEX.html`: a matching static HTML table with the same relative links and no external dependencies.

Call it at the end of successful `initialize_project`. Invalid/unreadable manifests must raise `ValueError` rather than silently omit a project.

- [ ] **Step 5: Create a legacy compatibility index**

Write `experiments/legacy/INDEX.md` with entries `0001` through `0012`, each linking only to existing repository paths. For each entry include available legacy evaluation index under `rl_runs/evaluation/<old-id>/index.html`; where a matching old artifact directory exists, link it; where the project’s training source is known from current tree, link it. Do not create synthetic manifest, DESIGN, V records or report copies for legacy projects.

- [ ] **Step 6: Run tests and manually validate links**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects
python3 - <<'PY'
from pathlib import Path
for line in Path("experiments/legacy/INDEX.md").read_text(encoding="utf-8").splitlines():
    if "](" not in line:
        continue
    for target in [part.split(")", 1)[0] for part in line.split("](")[1:]]:
        if target.startswith("../"):
            assert (Path("experiments/legacy") / target).resolve().exists(), target
PY
```

Expected: all tests PASS and every legacy Markdown link target exists.

## Task 5: Update documentation, project instructions, and ignore rules

**Files:**
- Modify: `rl_environment/README.md`
- Modify: `rl_runs/README.md`
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `.gitignore`
- Modify: `pyproject.toml` only if test/package configuration requires it

**Interfaces:**
- Consumes implemented CLI and directory contracts from Tasks 1–4.
- Produces a single consistent written authority for new project paths while retaining historical compatibility guidance.

- [ ] **Step 1: Write a failing documentation-consistency test**

Add to `tests/test_experiment_projects.py`:

```python
def test_project_documentation_uses_new_authoritative_paths(self) -> None:
    expected = "experiments/<project_id>/"
    for path in (Path("CLAUDE.md"), Path("AGENTS.md"), Path("rl_environment/README.md"), Path("rl_runs/README.md")):
        self.assertIn(expected, path.read_text(encoding="utf-8"), path)
```

- [ ] **Step 2: Run it to verify it fails before documentation changes**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentDocumentationTests
```

Expected: FAIL because current repository instructions describe the old `rl_runs/artifact|checkpoint|tensorboard|evaluation` roots as authoritative for new projects.

- [ ] **Step 3: Update `rl_environment/README.md`**

Replace the old allocation example with `create-project` and all required contract flags. Document this exact separation:

```text
rl_environment/       shared infrastructure
train/<project_id>/   project-specific implementation
experiments/<project_id>/ project archive and authoritative DESIGN/evaluation
rl_runs/<project_id>/ runtime assets
```

State explicitly that the legacy `create` command exists only for historical tooling and must not be used for `0013` onward.

- [ ] **Step 4: Update `rl_runs/README.md`**

Replace old asset-group tree examples with nested `<project_id>/versions/<V>/...` examples. State that formal evaluation reports are not stored in `rl_runs`; `artifact/evaluation.json` points to the authoritative archive report. Preserve a brief note that `rl_runs/artifact|checkpoint|tensorboard|evaluation` are read-only legacy paths for `0001`–`0012`.

- [ ] **Step 5: Update `CLAUDE.md` and `AGENTS.md` coherently**

Update every new-project instruction that currently mandates:

```text
rl_runs/artifact/<experiment>/V<n>_<tag>/
rl_runs/tensorboard/<experiment>/V<n>_<tag>/
rl_runs/checkpoint/<experiment>/V<n>_<tag>/
rl_runs/evaluation/<experiment>/V<n>_<tag>.html
```

to instead mandate:

```text
train/<project_id>/
experiments/<project_id>/
rl_runs/<project_id>/versions/<V<n>_<tag>>/
experiments/<project_id>/evaluation/<V<n>_<tag>.html
```

Retain the requirements for strict V numbering, full per-epoch train/validation metrics, engine evaluation, candidate admission and non-overwrite. Declare `experiments/<project_id>/DESIGN.html` and `DESIGN.md` the authoritative design documents. Remove stale references to `work/<name>/` as a candidate destination; candidates remain under `evaluation/arena/candidates/`.

- [ ] **Step 6: Update `.gitignore` for nested run assets**

Add rules that ignore only nested large/local assets:

```gitignore
/rl_runs/*/dataset/
/rl_runs/*/versions/*/checkpoint/
/rl_runs/*/versions/*/wandb/
/rl_runs/*/versions/*/tensorboard/
```

Do not ignore `rl_runs/*/versions/*/artifact/` or any `experiments/` file. Preserve existing legacy ignore patterns until historical assets are no longer relevant.

- [ ] **Step 7: Run documentation test, syntax and full tests**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects
python3 -m compileall -q rl_environment
python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: all commands exit 0.

## Task 6: Validate a disposable `0013`-format project lifecycle

**Files:**
- Modify: `tests/test_experiment_projects.py`
- No persistent source files beyond the prior tasks.

**Interfaces:**
- Consumes `initialize_project`, `initialize_version`, `record_evaluation`, and `refresh_experiment_index`.
- Produces an end-to-end regression test using a `TemporaryDirectory`, with no artifact written to the repository.

- [ ] **Step 1: Write the lifecycle regression test**

```python
def test_new_project_lifecycle_is_navigable_and_immutable(self) -> None:
    with TemporaryDirectory() as directory:
        repository = Path(directory)
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository), patch(
            "rl_environment.runs._git_value", return_value="abc123"
        ):
            project = initialize_project(
                "alakazam_rollout",
                objective="value calibration",
                deck="alakazam",
                expert_source="team_policy",
                dataset_contract="dataset_v1",
                engine_revision="official",
                opponent_pool_snapshot="catalog_v1",
            )
            version = initialize_version(project.project_id, "V1_initial_contract")
            version.evaluation.write_text("<html>report</html>\n", encoding="utf-8")
            record_evaluation(version, run_id="engine-run-123", report_sha256="deadbeef")
            refresh_experiment_index()

        self.assertTrue((project.archive / "README.md").is_file())
        self.assertTrue((project.archive / "DESIGN.html").is_file())
        self.assertTrue((project.archive / "evaluation" / "index.html").is_file())
        self.assertTrue((version.artifact / "evaluation.json").is_file())
        self.assertIn(project.project_id, (repository / "experiments" / "INDEX.md").read_text())
        with patch("rl_environment.runs.REPOSITORY_ROOT", repository):
            with self.assertRaises(FileExistsError):
                initialize_version(project.project_id, "V1_initial_contract")
```

- [ ] **Step 2: Run the lifecycle test to verify it fails if any integration is absent**

Run:

```bash
python3 -m unittest -v tests.test_experiment_projects.ExperimentProjectLifecycleTests
```

Expected: PASS after Tasks 1–5; if it fails, correct only the missing contract integration before proceeding.

- [ ] **Step 3: Run repository validation**

Run:

```bash
python3 -m compileall -q evaluation visualization rl_environment train
python3 -m unittest discover -s tests -p 'test_*.py'
git diff --check
git status --short
```

Expected: compilation and tests exit 0; `git diff --check` emits no whitespace errors; status contains only intentional architecture changes plus the user’s pre-existing working-tree changes.

- [ ] **Step 4: Report verification evidence without committing**

State the exact test command and result, the new `create-project` CLI, the authoritative entrypoint `experiments/<project_id>/README.md`, and confirm that no `0001`–`0012` file moved. Do not create a commit unless the user explicitly asks.

## Spec coverage self-review

- Project identity and strict V naming: Tasks 1–3.
- Independent `train/`, centralized `experiments/`, nested `rl_runs/`, and shared-only environment boundary: Tasks 1–2 and Task 5.
- Concentrated DESIGN documents and project navigation: Task 4 and Task 5.
- Formal evaluation as archive authority and artifact backlink: Task 3.
- Candidate/opponent boundary: preserved in Task 5 instructions; no package moves planned.
- Legacy 0001–0012 preservation and discoverability: Task 4 and Task 5.
- Fail-closed allocation, immutable versions, contract validation, indexes, and verification: Tasks 2–6.

Placeholder scan: no `TBD`, `TODO`, deferred implementation, or unspecified test steps remain. Names used by later tasks are defined in Tasks 1–4.
