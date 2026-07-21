# V8 History HTML Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 V8 semantic recovery 的每个 Target/Start/iteration 生成独立 HTML 报告，并提供跨轮次总览页。

**Architecture:** `scripts/render_v8_history.py` 只读取轻量 `summary.json` 与对应 Markdown，生成 history 根目录 `index.html` 以及每条 record 的 `<record-id>/index.html`。页面使用内嵌 CSS/JSON，不依赖网络或前端构建；完整 trace 仍保留在 `/tmp`。

**Tech Stack:** Python 3.11+ 标准库、静态 HTML/CSS、标准库 `unittest`。

## Global Constraints

- 评测来源只使用隔壁 Auto-Iteration Sample，不使用当前仓库 `evaluation/`。
- Full Sample 固定 17 个 opponent × 10 局 = 170 局，先后手按实际 trace 统计。
- focused record 只用于机制诊断，页面必须标注 focused，不能伪装为 full Sample。
- 中文页面、中文记录优先；不复制完整 trace 到 history。
- 不执行 git commit/push。

### Task 1: 定义 HTML 数据读取与显示测试

**Files:**
- Create: `tests/test_v8_history_html.py`
- Read: `work/auto-iteration/history_iterations/v8-semantic-recovery/summary.json`

**Interfaces:**
- Test will import `build_history_pages(summary, history_dir)` from `scripts.render_v8_history`.
- The function returns `dict[str, str]` mapping relative output paths to HTML strings; it must not read trace files.

- [ ] **Step 1: Write the failing test**

```python
def test_build_history_pages_renders_full_and_focused_records(self):
    pages = build_history_pages(self.summary, self.history_dir)
    self.assertIn("index.html", pages)
    self.assertIn("target-baseline/index.html", pages)
    self.assertIn("start-baseline/index.html", pages)
    self.assertIn("iteration-004/index.html", pages)
    self.assertIn("118/50/2", pages["target-baseline/index.html"])
    self.assertIn("8/128/34", pages["start-baseline/index.html"])
    self.assertIn("17×10", pages["target-baseline/index.html"])
    self.assertIn("focused", pages["iteration-004/index.html"])
    self.assertIn("attackId=1072", pages["index.html"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest -v tests.test_v8_history_html`

Expected: FAIL because `scripts.render_v8_history` does not yet exist.

### Task 2: Implement deterministic page generation

**Files:**
- Create: `scripts/render_v8_history.py`
- Modify: `tests/test_v8_history_html.py`

**Interfaces:**
- `load_summary(history_dir: Path) -> dict[str, Any]` loads and validates JSON object input.
- `build_history_pages(summary: Mapping[str, Any], history_dir: Path) -> dict[str, str]` returns the root and per-record pages.
- `render_history(history_dir: Path, output_root: Path | None = None) -> list[Path]` writes pages and creates one directory per record.
- `main(argv: Sequence[str] | None = None) -> int` supports `--history-dir` and optional `--output-root`.

- [ ] **Step 1: Add the minimal renderer**

The renderer must:

1. Format W/L/D, rates, errors, Powerful Hand and post-KO fields without inventing missing denominators.
2. Render full and focused labels visibly.
3. Link each record page to `../index.html` and its sibling Markdown when it exists.
4. Escape all summary/Markdown text with `html.escape`.
5. Embed only summary values and a short “trace remains in /tmp” notice.

- [ ] **Step 2: Run the focused renderer test**

Run: `python3 -m unittest -v tests.test_v8_history_html`

Expected: PASS, with pages for Target, Start and all existing iteration records.

### Task 3: Add the approved history layout and usage documentation

**Files:**
- Modify: `work/auto-iteration/history_iterations/v8-semantic-recovery/README.md`
- Modify: `work/auto-iteration/history_iterations/v8-semantic-recovery/summary.json`
- Modify: `docs/superpowers/specs/2026-07-21-v8-history-html-dashboard-design.md`

- [ ] **Step 1: Record iteration-005 focused result**

Add `iteration-005` to `summary.json` with `sample_type: "focused"`, `games: 12`, `wins: 5`,
`losses: 7`, `draws: 0`, `errors: 0`, `second_turn_powerful_hand: 0/12`,
`post_ko: 15`, `post_ko_zero_ready: 13/15`, `game_level_breaks: 5/12`, and decision `observe`.

- [ ] **Step 2: Document the renderer command and directory layout**

Add:

```bash
python3 scripts/render_v8_history.py \
  --history-dir work/auto-iteration/history_iterations/v8-semantic-recovery
```

and explain that `target-baseline/`, `start-baseline/`, and each `iteration-NNN/` contain the
primary HTML report for that record.

### Task 4: Generate and verify the real pages

**Files:**
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/target-baseline/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/start-baseline/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/iteration-001/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/iteration-002/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/iteration-003/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/iteration-004/index.html`
- Create: `work/auto-iteration/history_iterations/v8-semantic-recovery/iteration-005/index.html`

- [ ] **Step 1: Render the real history**

Run: `python3 scripts/render_v8_history.py --history-dir work/auto-iteration/history_iterations/v8-semantic-recovery`

Expected: exit 0 and one root page plus one page per summary record.

- [ ] **Step 2: Verify generated content and safety**

Run:

```bash
python3 -m unittest -v tests.test_v8_history_html
python3 -m json.tool work/auto-iteration/history_iterations/v8-semantic-recovery/summary.json >/dev/null
rg -n '118/50/2|8/128/34|iteration-005|focused|attackId=1072' work/auto-iteration/history_iterations/v8-semantic-recovery -g '*.html'
git diff --check
```

Expected: all commands exit 0; generated pages contain the required values and no full trace JSON.

## Self-review checklist

- Target and Start use identical full-sample sections, with their different metrics preserved.
- Focused rows/pages visibly carry the focused label and are not used as promotion evidence.
- Missing metrics render as `—`; no denominator is inferred from another record.
- The root page links to every generated record page.
- The renderer is deterministic and the test covers the user-visible distinctions.
