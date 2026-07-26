# Kaggle Top 100 对战环境日报

主要入口：打开 [`index.html`](index.html)。

## 目录合同

- 本目录保存 Kaggle Pokémon TCG AI Battle **Top 100 对战环境分析日报**。
- 它不是项目开发进展日报，也不负责规定固定的数据来源。
- 当用户要求制作日报时，由当次任务明确提供或指定需要分析的源数据。
- 最终交付物是可直接阅读的 HTML，保存为 `daily/YYYY-MM-DD.html`。
- 每新增一份日报，必须同步更新 `index.html`，并按日期倒序排列。
- 报告正文应说明自己的分析日期、输入范围和证据口径；不需要在本目录额外保存一套 snapshot。

## UI 继承合同

- 整体日报固定以 [`daily/2026-07-26.html`](daily/2026-07-26.html) 为基准：保留导航、摘要、
  个人筛选、构筑分布、跨日变化、Match-up/节奏证据入口、静态排名和逐人 exact deck 展开。
- “Top 100 全局构筑卡池”固定以 [`daily/2026-07-25.html`](daily/2026-07-25.html) 为基准：
  使用卡图网格展示 Card ID、类别、覆盖套数和覆盖率，并提供按牌型的典型构筑卡池。
- 某日缺少逐局 opponent、firstPlayer、turn 或墙钟字段时，保留组件并如实说明不可重建；
  禁止沿用前日报数值或将无证据格子写成 0% 胜率。
- 禁止将正式日报退化为单个搜索框加排行榜表格。发布前须通过
  `tests.test_environment_daily_contract` 的结构回归检查。

当前报告：

- [2026-07-27 — Top 100 实时环境快照 0727（最终 submission + 官方 completed/public Meta）](daily/2026-07-27.html)
- [2026-07-26 — Top 100 今日环境快照 0726（最终 submission + 官方 completed/public Meta）](daily/2026-07-26.html)
- [2026-07-25 — Top 100 Deck 环境与 Match-up 分析](daily/2026-07-25.html)
- [2026-07-23 — Kaggle Top 100 BC 候选调研](daily/2026-07-23.html)
