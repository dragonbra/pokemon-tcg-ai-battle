# Replay 可视化工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立统一的 replay 可视化入口，兼容 Kaggle replay、本地带可视化帧的 replay，并让本地 runner 可选保存 viewer 所需帧。

**Architecture:** `scripts/replay_visualizer.py` 负责纯数据识别、校验、标准化和 launcher 生成；`scripts/visualize_replay.py` 只负责 CLI；`scripts/run_local_battle.py` 在显式传入输出路径时调用 `visualize_data()` 并复用标准化辅助函数。外部 viewer 通过临时 HTML form 使用 `POST` 接收帧数组，原始 JSON 不被改写。

**Tech Stack:** Python 3.11 标准库、现有 `unittest` 测试发现、`webbrowser`、官方 cg `visualize_data()`。

## Global Constraints

- 项目要求 Python 3.11+，Python 使用 4 个空格和类型注解。
- 单行长度遵守 `pyproject.toml` 中的 100 字符限制。
- 不增加第三方依赖，不修改 agent 策略、卡组或官方引擎。
- 普通本地 smoke test 默认不保存可视化数据；可视化文件写到用户传入路径，示例使用 `/tmp`。
- 不自动执行 `git commit` 或 `git push`。

---

### Task 1: Replay 数据适配与校验

**Files:**
- Create: `scripts/replay_visualizer.py`
- Create: `tests/test_replay_visualization.py`

**Interfaces:**
- Produces `ReplayFormatError`, `NormalizedReplay`, `ReplayLaunch`, `extract_visualize_frames(record)`, `load_replay(path)`, and `attach_trace_metadata(frames, trace)` for later tasks.
- `NormalizedReplay.frames` is a non-empty `list[dict[str, Any]]`; `NormalizedReplay.source` is one of `"top-level"`, `"visualize_frames"`, or `"kaggle-steps"`.

- [ ] **Step 1: Write the failing tests for supported formats.**

```python
from scripts.replay_visualizer import extract_visualize_frames, load_replay


def test_extracts_kaggle_visualize_from_first_step():
    frames = [{"obs": "initial"}, {"obs": "after-action"}]
    record = {"steps": [[{"visualize": frames}]]}

    assert extract_visualize_frames(record) == frames


def test_extracts_local_top_level_visualize_before_legacy_alias():
    frames = [{"state": 1}]
    record = {"visualize": frames, "visualize_frames": [{"state": 2}]}

    assert extract_visualize_frames(record) == frames


def test_extracts_local_visualize_frames_alias():
    frames = [{"state": 1}]

    assert extract_visualize_frames({"visualize_frames": frames}) == frames


def test_load_replay_records_source_and_original_path(tmp_path):
    path = tmp_path / "kaggle.json"
    path.write_text('{"steps": [[{"visualize": [{"state": 1}]}]]}', encoding="utf-8")

    replay = load_replay(path)

    assert replay.path == path
    assert replay.source == "kaggle-steps"
    assert replay.frames == [{"state": 1}]
```

- [ ] **Step 2: Run the focused tests and verify the intended failure.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: FAIL because `scripts.replay_visualizer` and its extraction functions do not exist yet.

- [ ] **Step 3: Write failing tests for invalid input and legacy traces.**

```python
import json

import pytest

from scripts.replay_visualizer import ReplayFormatError, extract_visualize_frames, load_replay


@pytest.mark.parametrize("record", [{}, {"trace": [{"step": 1}]}, {"steps": []}])
def test_rejects_replay_without_visualize_frames(record):
    with pytest.raises(ReplayFormatError, match="visualize"):
        extract_visualize_frames(record)


def test_rejects_non_object_frame():
    with pytest.raises(ReplayFormatError, match="帧"):
        extract_visualize_frames({"visualize": ["not-an-object"]})


def test_load_replay_reports_invalid_json_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ReplayFormatError, match=str(path)):
        load_replay(path)
```

- [ ] **Step 4: Run the new invalid-input tests and verify they fail for the missing implementation.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: FAIL because the adapter module is not implemented.

- [ ] **Step 5: Implement the minimal adapter.**

```python
def _validate_frames(value: object, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ReplayFormatError(f"回放没有可用的 visualize 帧，来源：{source}")
    if not all(isinstance(frame, dict) for frame in value):
        raise ReplayFormatError(f"visualize 帧必须是对象，来源：{source}")
    return value


def extract_visualize_frames(record: object) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        raise ReplayFormatError("回放 JSON 顶层必须是对象")
    if "visualize" in record:
        return _validate_frames(record["visualize"], "顶层 visualize")
    if "visualize_frames" in record:
        return _validate_frames(record["visualize_frames"], "顶层 visualize_frames")
    for step in record.get("steps", []):
        if not isinstance(step, list):
            continue
        for item in step:
            if isinstance(item, dict) and item.get("visualize"):
                return _validate_frames(item["visualize"], "Kaggle steps")
    raise ReplayFormatError("回放没有 visualize 帧；旧 trace 需要用 --visualize-output 重新生成")
```

`load_replay()` 使用 `json.loads(path.read_text())`，将文件、JSON 解码和格式错误统一包装为带路径的 `ReplayFormatError`，并创建 `NormalizedReplay`。

- [ ] **Step 6: Run the focused tests and verify they pass.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: PASS for all adapter tests.

- [ ] **Step 7: Add and test local trace metadata alignment.**

```python
def test_attach_trace_metadata_does_not_overwrite_existing_fields():
    frames = [{"obs": "existing", "action": [[9], [9]]}, {}]
    trace = [
        {"observation": {"turn": 1}, "action": [2]},
    ]

    attach_trace_metadata(frames, trace)

    assert frames[0]["obs"] == "existing"
    assert frames[0]["action"] == [[9], [9]]
    assert frames[1]["obs"] == {"turn": 1}
    assert frames[1]["action"] == [[2], [2]]
```

Run the test before implementation and confirm the missing helper failure. Then implement `attach_trace_metadata()` by selecting only trace entries containing `action`, reserving frame zero for initial state, filling frame `index + 1`, and using `setdefault()` for `obs` and `action`. Run `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v` again and confirm PASS.

---

### Task 2: Viewer launcher 与 CLI

**Files:**
- Modify: `scripts/replay_visualizer.py`
- Create: `scripts/visualize_replay.py`
- Modify: `tests/test_replay_visualization.py`

**Interfaces:**
- `create_viewer_launcher(frames, output, viewer_url) -> Path` writes a self-submitting HTML form with `method="POST"`, `name="json"`, and the configured endpoint.
- `show_replay(path, open_browser=True, viewer_url=DEFAULT_VIEWER_URL) -> ReplayLaunch` creates the launcher and calls `webbrowser.open()` only when requested.
- `scripts/visualize_replay.py` returns exit code 0 on valid input and 1 on `ReplayFormatError`/I/O errors.

- [ ] **Step 1: Write the failing launcher and no-open CLI tests.**

```python
import json
import subprocess
import sys

from scripts.replay_visualizer import create_viewer_launcher


def test_launcher_posts_json_to_viewer(tmp_path):
    frames = [{"state": "x", "text": "a'b"}]
    launcher = create_viewer_launcher(
        frames,
        tmp_path / "launcher.html",
        "https://viewer.test/replay",
    )
    html = launcher.read_text(encoding="utf-8")

    assert 'action="https://viewer.test/replay"' in html
    assert 'method="POST"' in html
    assert 'name="json"' in html
    assert json.dumps(frames, ensure_ascii=False) in html.replace("&#x27;", "'")


def test_cli_no_open_prints_launcher_path(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text('{"visualize": [{"state": 1}]}', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/visualize_replay.py", str(replay), "--no-open"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "launcher" in result.stdout
    assert "ptcgvis.heroz.jp/Visualizer/Replay/0" in result.stdout
```

- [ ] **Step 2: Run the launcher tests and verify they fail.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: FAIL because launcher and CLI behavior do not exist yet.

- [ ] **Step 3: Implement launcher generation and browser control.**

Use `html.escape(json.dumps(frames, ensure_ascii=False), quote=True)` as the hidden input value. Write a UTF-8 document with a form, a `window.onload` submit handler, and `target="_blank"`. `show_replay()` creates a temporary file with `tempfile.NamedTemporaryFile(prefix="ptcg-replay-", suffix=".html", delete=False)`, calls `webbrowser.open(path.as_uri())` when enabled, and returns `ReplayLaunch`.

- [ ] **Step 4: Run the focused launcher tests and verify they pass.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: PASS.

- [ ] **Step 5: Add CLI error tests, implement parser, and verify.**

```python
def test_cli_reports_old_trace_with_nonzero_status(tmp_path):
    replay = tmp_path / "trace.json"
    replay.write_text('{"trace": [{"step": 1}]}', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/visualize_replay.py", str(replay), "--no-open"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--visualize-output" in result.stderr
```

Implement `argparse` options for `replay`, `--no-open`, and `--viewer-url`; print the generated launcher and endpoint to stdout, and print `ReplayFormatError` to stderr. Run:

`python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: PASS.

---

### Task 3: Local runner 可选保存可视化 replay

**Files:**
- Modify: `scripts/run_local_battle.py`
- Modify: `tests/test_replay_visualization.py`

**Interfaces:**
- Add `visualize_output: Path | None = None` to `run()` without changing existing positional behavior.
- Add `--visualize-output PATH` to the CLI.
- Keep the normal output JSON unchanged when the option is omitted.

- [ ] **Step 1: Write the failing unit test for local replay assembly.**

```python
from scripts.run_local_battle import _build_visual_replay


def test_build_visual_replay_keeps_trace_and_attaches_frames():
    result = {"agent0": "a", "agent1": "b", "trace": [{"action": [0]}]}
    frames = [{"state": "initial"}, {"state": "after"}]

    replay = _build_visual_replay(result, frames)

    assert replay["replay_format"] == "ptcg-local-v1"
    assert replay["trace"] == result["trace"]
    assert replay["visualize"][1]["action"] == [[0], [0]]
```

- [ ] **Step 2: Run the test and verify the expected missing-helper failure.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v`

Expected: FAIL because `_build_visual_replay()` and the runner option do not exist.

- [ ] **Step 3: Implement local replay assembly and optional engine capture.**

Import `json`, `attach_trace_metadata`, and `visualize_data` inside the existing simulator import block. Add `_build_visual_replay(result, frames)` that copies the result dictionary, sets `replay_format`, copies the frame list, and calls `attach_trace_metadata()`.

In `run()`, initialize `visualize_frames = None`; in `finally`, before `battle_finish()`, if `visualize_output` is set, call `json.loads(visualize_data())`. Preserve the original exception in `error`; if capture fails and no earlier error exists, set `error` to `VisualizeError: ...`. After writing the normal output, write `_build_visual_replay(result, visualize_frames)` to the requested path only when frames were captured. Ensure `battle_finish()` remains in a `finally` path even when capture fails.

- [ ] **Step 4: Run focused tests and syntax checks.**

Run: `python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v && python3 -m py_compile scripts/run_local_battle.py`

Expected: PASS and exit code 0.

- [ ] **Step 5: Add the CLI argument and verify help output.**

```python
parser.add_argument(
    "--visualize-output",
    type=Path,
    default=None,
    help="可选：保存官方 viewer 所需的完整 visualize replay",
)
```

Pass `args.visualize_output` to `run()` and run:

`python3 scripts/run_local_battle.py --help`

Expected: help output contains `--visualize-output` and the command exits 0 without loading the simulator.

---

### Task 4: 使用文档与集成验证

**Files:**
- Create: `docs/replay-visualization.md`
- Modify: `README.md`

- [ ] **Step 1: Write the user-facing command examples.**

Document the Kaggle command, the local runner command with `/tmp` output, the `--no-open` mode, the old-trace diagnostic, and the fact that the external viewer needs card assets to render non-black card faces.

- [ ] **Step 2: Run the focused and repository checks.**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_replay_visualization.py' -v
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
git diff --check
```

Expected: each command exits 0. If the dynamic library prevents a local battle, report that environment limitation separately while still validating the adapter against a repository Kaggle replay.

- [ ] **Step 3: Run a real Kaggle replay through `--no-open`.**

Run:

`python3 scripts/visualize_replay.py replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json --no-open`

Expected: exit 0, print a `/tmp/ptcg-replay-*.html` launcher path and the viewer endpoint. Inspect the generated HTML for a non-empty JSON payload, then remove only that temporary launcher if it is no longer needed.

- [ ] **Step 4: Review the final diff and report uncommitted status.**

Run: `git status --short && git diff --stat`

Report the files changed, verification results, and explicitly state that no commit or push was performed.
