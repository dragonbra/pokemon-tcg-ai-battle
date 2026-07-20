# Visualization Directory Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 replay 可视化核心和 CLI 从 `scripts/` 完整迁移到 `visualization/replay/`，并把所有使用说明切换到新的模块入口。

**Architecture:** `visualization/replay/core.py` 负责 replay 格式识别、帧校验、动作元数据补齐和外部 viewer launcher；`visualization/replay/cli.py` 负责命令行参数和退出码；`visualization/replay/__init__.py` 暴露 core API。现有 `scripts/replay_visualizer.py` 与 `scripts/visualize_replay.py` 删除，不保留兼容入口。

**Tech Stack:** Python 3.11+、标准库 `argparse`/`json`/`html`/`tempfile`/`webbrowser`、`unittest`、现有官方 simulator 和 shell 校验脚本。

## Global Constraints

- Python 使用 4 个空格、类型注解和清晰的小函数，遵守 100 字符行宽。
- 不新增第三方依赖，不修改 agent 策略、卡组或官方引擎。
- 默认本地 battle 仍只保存轻量 trace；完整 visualize replay 写入 `/tmp` 等临时路径。
- 不保留 `scripts/` 的 replay 可视化兼容入口；文档和代码全部使用 `python3 -m visualization.replay.cli`。
- 不执行 `git commit`，保留当前工作区中用户已有的改动。

## File Map

- Create: `visualization/__init__.py`，将 visualization 标记为 Python 包。
- Create: `visualization/replay/__init__.py`，导出 core 的公共 API。
- Create: `visualization/replay/core.py`，迁移 `scripts/replay_visualizer.py` 的实现。
- Create: `visualization/replay/cli.py`，迁移 `scripts/visualize_replay.py` 的 CLI。
- Modify: `visualization/README.md`，替换占位内容为完整使用说明。
- Modify: `scripts/run_local_battle.py`，改为导入 `visualization.replay.core`。
- Modify: `tests/test_replay_visualization.py`，切换 import 和 subprocess 入口。
- Modify: `README.md`、`AGENTS.md`、`docs/replay-visualization.md`，切换命令和链接，避免旧路径残留。
- Delete: `scripts/replay_visualizer.py`、`scripts/visualize_replay.py`。

### Task 1: 建立 visualization replay 包并迁移核心逻辑

**Files:**
- Create: `visualization/__init__.py`
- Create: `visualization/replay/__init__.py`
- Create: `visualization/replay/core.py`
- Modify: `tests/test_replay_visualization.py`
- Delete: `scripts/replay_visualizer.py`

**Interfaces:**
- Produces `ReplayFormatError`, `NormalizedReplay`, `ReplayLaunch`, `load_replay`, `extract_visualize_frames`, `attach_trace_metadata`, `create_viewer_launcher`, `show_replay` from `visualization.replay.core`.
- `visualization.replay` re-exports the same names for callers that prefer the package boundary.

- [ ] **Step 1: Add package markers and update test imports.**

Create empty `visualization/__init__.py`, create `visualization/replay/__init__.py` with:

```python
from .core import (
    DEFAULT_VIEWER_URL,
    NormalizedReplay,
    ReplayFormatError,
    ReplayLaunch,
    attach_trace_metadata,
    create_viewer_launcher,
    extract_visualize_frames,
    load_replay,
    show_replay,
)

__all__ = [
    "DEFAULT_VIEWER_URL",
    "NormalizedReplay",
    "ReplayFormatError",
    "ReplayLaunch",
    "attach_trace_metadata",
    "create_viewer_launcher",
    "extract_visualize_frames",
    "load_replay",
    "show_replay",
]
```

In `tests/test_replay_visualization.py`, replace imports from `scripts.replay_visualizer` with `visualization.replay`, and replace the local import used by the launcher test the same way.

- [ ] **Step 2: Run the focused tests and verify the expected import failure.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization -v
```

Expected: FAIL because `visualization.replay.core` does not exist yet.

- [ ] **Step 3: Move the core implementation.**

Create `visualization/replay/core.py` by moving the complete implementation from `scripts/replay_visualizer.py`, preserving the public signatures and behavior. Keep the Chinese format errors, default endpoint, `html.escape` launcher encoding, and `setdefault()` behavior in `attach_trace_metadata()` unchanged. Then delete `scripts/replay_visualizer.py`.

- [ ] **Step 4: Run focused tests and verify the core migration.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization.ReplayAdapterTests -v
```

Expected: all adapter, metadata, validation, and launcher tests pass.

### Task 2: Migrate the CLI and runner dependency

**Files:**
- Create: `visualization/replay/cli.py`
- Modify: `scripts/run_local_battle.py`
- Modify: `tests/test_replay_visualization.py`
- Delete: `scripts/visualize_replay.py`

**Interfaces:**
- CLI command: `python3 -m visualization.replay.cli REPLAY [--no-open] [--viewer-url URL]`.
- `scripts/run_local_battle.py` consumes `attach_trace_metadata` and `extract_visualize_frames` from `visualization.replay.core`.

- [ ] **Step 1: Change the CLI test command before implementation.**

In `tests/test_replay_visualization.py`, replace `scripts/visualize_replay.py` in both subprocess invocations with `-m visualization.replay.cli` immediately after `sys.executable`.

- [ ] **Step 2: Run CLI tests to verify the old path is no longer used.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization.ReplayAdapterTests.test_cli_no_open_prints_launcher_path -v
```

Expected: FAIL because `visualization.replay.cli` is not implemented yet.

- [ ] **Step 3: Move the CLI implementation and update runner imports.**

Create `visualization/replay/cli.py` from the existing CLI. Use absolute imports:

```python
from .core import DEFAULT_VIEWER_URL, ReplayFormatError, show_replay
```

Keep `main(argv: list[str] | None = None) -> int`, the `--no-open` and `--viewer-url` options, Chinese help/error text, and `SystemExit(main())`. In `scripts/run_local_battle.py`, import `attach_trace_metadata` and `extract_visualize_frames` from `visualization.replay.core` in both its package and script execution paths. Ensure the script path adds the repository root before importing the package if needed. Then delete `scripts/visualize_replay.py`.

- [ ] **Step 4: Run all replay tests and the CLI help.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization -v
python3 -m visualization.replay.cli --help
```

Expected: all replay tests pass; help shows the replay positional argument, `--no-open`, and `--viewer-url`.

### Task 3: Consolidate documentation and remove stale paths

**Files:**
- Modify: `visualization/README.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/replay-visualization.md`

**Interfaces:**
- All user-facing examples use `python3 -m visualization.replay.cli`.
- The canonical detailed guide is `visualization/README.md`; `docs/replay-visualization.md` becomes a short link or is removed after all references are updated.

- [ ] **Step 1: Replace the placeholder visualization README.**

Write a Chinese guide covering: the module command; default browser launch and `--no-open`; `--viewer-url`; supported top-level and Kaggle `steps` frame locations; local `run_local_battle.sh --visualize-output` generation; old trace limitation; external viewer POST protocol; black-screen troubleshooting; and the Python import example from `visualization.replay`.

- [ ] **Step 2: Update repository navigation and operational instructions.**

In `README.md` replace the scripts command with the module command and make `visualization/` the directory description for replay viewing. In `AGENTS.md` replace every old CLI/import documentation path with `python3 -m visualization.replay.cli`, update the detailed guide link to `visualization/README.md`, and state that implementation ownership is under `visualization/replay/`.

- [ ] **Step 3: Remove the duplicate old guide or turn it into a redirect note.**

Prefer deleting `docs/replay-visualization.md` after `rg` confirms no references remain, because the user requested the documentation location to move. If repository navigation requires a stable `docs/` link, retain only a short Chinese note pointing to `../visualization/README.md`; do not duplicate the full guide.

- [ ] **Step 4: Search for stale implementation paths.**

Run:

```bash
rg -n 'scripts/(replay_visualizer|visualize_replay)|from scripts\.replay_visualizer|docs/replay-visualization' AGENTS.md README.md docs scripts tests visualization
```

Expected: no stale runtime imports or user-facing commands; historical superpowers plan/spec files may mention the old path and should be left unchanged unless they are part of this migration documentation.

### Task 4: Validate the migration

**Files:**
- No new source files; inspect all files changed by Tasks 1-3.

- [ ] **Step 1: Run focused replay tests.**

```bash
python3 -m unittest tests.test_replay_visualization -v
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax checks.**

```bash
python3 -m py_compile visualization/__init__.py visualization/replay/__init__.py visualization/replay/core.py visualization/replay/cli.py scripts/run_local_battle.py tests/test_replay_visualization.py
python3 -m compileall -q scripts visualization tests
```

Expected: both commands exit 0.

- [ ] **Step 3: Run repository asset validation.**

```bash
python3 scripts/check_assets.py
```

Expected: existing asset, deck, and simulator checks pass. This does not run a local battle or write replay data into the repository.

- [ ] **Step 4: Exercise the new CLI with a temporary replay.**

```bash
python3 -c 'import json, tempfile; from pathlib import Path; p=Path(tempfile.gettempdir()) / "ptcg-visualization-smoke.json"; p.write_text(json.dumps({"visualize": [{"state": 1}]}), encoding="utf-8"); print(p)'
python3 -m visualization.replay.cli /tmp/ptcg-visualization-smoke.json --no-open
```

Expected: exit 0, print a temporary `launcher:` path and the default viewer endpoint.

- [ ] **Step 5: Check the final diff.**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only this migration, the existing user changes, and the newly created design/plan records appear in the worktree. Do not commit.
