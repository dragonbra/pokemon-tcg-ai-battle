# v5_auto_iter 策略执行流程图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `v5_auto_iter` 提交目录中新增一个能展示真实策略执行顺序、评分优先级和关键 gate 的中文 HTML 流程图。

**Architecture:** 页面是单文件、自包含的静态文档：内嵌 CSS 绘制响应式卡片与流程连线，内嵌 SVG 绘制主流程，内嵌 JavaScript 管理节点点击后的详情面板。页面内容直接对应 `main.py` 的 `agent`、`_main_action`、`_select_effect` 及其辅助判断，不引入运行时依赖，也不修改策略行为。

**Tech Stack:** HTML5、CSS3、原生 SVG、原生 JavaScript；验证使用 Python 标准库和仓库现有 shell 工具。

## Global Constraints

- 只新增 `submission/alakazam_v5_auto_iter/strategy-flow.html` 以及本计划文档，不修改 `main.py`、`deck.csv`、`cg/` 或现有报告。
- 页面文案使用中文；说明“score 越低越优先”，并区分主行动与效果选择。
- 页面不依赖 CDN、字体、图片或网络资源，打开本地文件即可阅读。
- 节点内容必须覆盖击倒、进攻路线、接力、贴能量、检索、恢复、抽牌安全、干扰、撤退和 `END`。
- 页面交互仅用于展开说明，不改变任何卡组或 agent 逻辑。

---

### Task 1: 创建单文件流程图页面

**Files:**
- Create: `submission/alakazam_v5_auto_iter/strategy-flow.html`
- Modify: none

**Interfaces:**
- Consumes: `v5_auto_iter/main.py` 中 `agent(obs)` 的三路入口、`_main_action` 的 tuple score 排序、`_select_effect` 的 context/select type 分支。
- Produces: 一个可直接用浏览器打开的 HTML 文档；点击流程节点后，右侧详情区显示 `函数 / 优先级 / 实际规则 / 风险`。

- [ ] **Step 1: 写入页面骨架、流程节点和优先级图例**

  页面包含标题、策略摘要、颜色图例，以及 `agent → 初始化 / 主行动 / 效果选择` 的 SVG/HTML 流程；主行动按 `0`、`1–8`、`9–30`、`50+`、`99` 分组。

- [ ] **Step 2: 写入关键 gate 与节点详情数据**

  用原生 JavaScript 的静态数据对象承载节点详情，至少包含首回合进化、攻击线数量、Rare Candy 直通、Dunsparce 接力、单线单能量、Telepath Energy、抽牌保护、Mist Energy、Boss、Fezandipiti、Retreat、Enriching Energy 和牌库 Dudunsparce 风险点。

- [ ] **Step 3: 添加点击展开、键盘焦点和移动端布局**

  节点使用按钮语义或 `tabindex`，点击/回车更新详情面板；CSS 在窄屏下将左右布局改为上下布局，保证流程文字和评分仍可阅读。

### Task 2: 页面与工作区验证

**Files:**
- Test: `submission/alakazam_v5_auto_iter/strategy-flow.html`
- Inspect: `submission/alakazam_v5_auto_iter/main.py`

- [ ] **Step 1: 验证文件存在、非空且不含外部依赖**

  Run: `test -s submission/alakazam_v5_auto_iter/strategy-flow.html && ! rg -n 'https?://|<script[^>]+src=|<link[^>]+href=' submission/alakazam_v5_auto_iter/strategy-flow.html`

  Expected: exit code `0`，没有输出。

- [ ] **Step 2: 用 Python 标准库解析 HTML 结构**

  Run: `python3 - <<'PY'\nfrom html.parser import HTMLParser\nfrom pathlib import Path\n\nclass Parser(HTMLParser):\n    pass\n\npath = Path('submission/alakazam_v5_auto_iter/strategy-flow.html')\nparser = Parser()\nparser.feed(path.read_text())\nparser.close()\ntext = path.read_text()\nfor marker in ('agent(obs)', '_main_action', '_select_effect', 'score 越低越优先', 'DOMContentLoaded'):\n    assert marker in text, marker\nprint('HTML parse and markers: OK')\nPY`

  Expected: 输出 `HTML parse and markers: OK`。

- [ ] **Step 3: 检查差异和策略文件未被改动**

  Run: `git diff --check && git diff -- submission/alakazam_v5_auto_iter/main.py submission/alakazam_v5_auto_iter/deck.csv`

  Expected: `git diff --check` 成功，策略与卡组没有差异输出。
