# Replay Launcher Same-Tab Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 replay launcher 在当前 Chrome 标签页 POST 到外部 viewer，避免自动打开新窗口导致的弹窗拦截。

**Architecture:** 保持现有临时 HTML launcher 和外部 viewer POST 协议，只把 form 的目标从新窗口改为当前窗口。测试验证 HTML target，文档说明新的浏览器行为。

**Tech Stack:** Python 3.11+ 标准库、HTML form POST、`unittest`。

## Global Constraints

- 不新增第三方依赖或本地代理服务。
- 不改变 `show_replay()`、CLI 参数、viewer endpoint 和 replay payload 格式。
- 默认提交目标为当前标签页 `target="_self"`，不得生成 `target="_blank"`。
- 不执行 git commit；保留工作区已有用户改动。

## Task 1: 修改 launcher、测试和文档

**Files:**
- Modify: `visualization/replay/core.py:119-127`
- Modify: `tests/test_replay_visualization.py:92-107`
- Modify: `visualization/README.md:12-20,71`
- Modify: `README.md:62-64`
- Modify: `AGENTS.md:57-58`

**Interfaces:**
- `create_viewer_launcher(frames, output, viewer_url)` 继续返回 launcher `Path`，生成 `POST` form，并把提交结果导航到当前标签页。

- [ ] **Step 1: Add the failing launcher assertion.**

Extend `test_launcher_posts_json_to_viewer` with:

```python
self.assertIn('target="_self"', html)
self.assertNotIn('target="_blank"', html)
```

- [ ] **Step 2: Run the focused test and verify it fails.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization.ReplayAdapterTests.test_launcher_posts_json_to_viewer -v
```

Expected: FAIL because the existing launcher still contains `target="_blank"`.

- [ ] **Step 3: Change the form target.**

In `visualization/replay/core.py`, change only the generated form line from:

```html
<form method="POST" action="{action}" target="_blank">
```

to:

```html
<form method="POST" action="{action}" target="_self">
```

Keep the automatic submit script and hidden `json` input unchanged.

- [ ] **Step 4: Update user-facing browser behavior documentation.**

In `visualization/README.md`, explain that the launcher submits in the current tab and does not create a popup. Update the concise command descriptions in `README.md` and `AGENTS.md` to refer to current-tab navigation.

- [ ] **Step 5: Run focused tests and verify the behavior.**

Run:

```bash
python3 -m unittest tests.test_replay_visualization -v
python3 -m visualization.replay.cli replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json --no-open
```

Expected: all replay tests pass; the generated launcher output is successful and the HTML contains `target="_self"` without `target="_blank"`.

## Task 2: Final validation

**Files:**
- Inspect all files changed in Task 1.

- [ ] **Step 1: Run the complete test suite.**

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: 0 failures.

- [ ] **Step 2: Run syntax and repository checks.**

```bash
python3 -m py_compile visualization/replay/core.py visualization/replay/cli.py tests/test_replay_visualization.py
python3 -m compileall -q scripts visualization tests
python3 scripts/check_assets.py
git diff --check
```

Expected: all commands exit 0.
