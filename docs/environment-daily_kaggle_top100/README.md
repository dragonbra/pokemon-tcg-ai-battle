# Kaggle Top 100 对战环境日报

主要入口：打开 [`index.html`](index.html)。

## 目录合同

- 本目录保存 Kaggle Pokémon TCG AI Battle **Top 100 对战环境分析日报**。
- 它不是项目开发进展日报，也不负责规定固定的数据来源。
- 当用户说“根据现在的快照构建日报”时，固定含义是：立即冻结调用时可见的官方 Top 100
  leaderboard；按每行 `submissionDate` 唯一绑定该行代表的 submission；对每个 submission
  获取同一冻结时刻以前最新的 `PUBLIC + COMPLETED` Episode，并从该 Episode 中 submission
  自身唯一的 player index 读取 exact 60-card deck。禁止复用前一天、同日旧 run 或任意缓存
  replay 冒充当前快照。
- 胜率、W-L-D 和 match-up 只使用同一批冻结 submission 在冻结时间以前的官方 Episode Meta；
  opponent 必须来自 Episode 对侧的真实 submission。缺少证据时显示不可重建或 `n=0`，不得复制
  前日报数值、把缺失值写成 0% 或从代表 replay 外推全量先后攻/回合统计。
- Kaggle Episode 端点具有最终一致性且约 1,000 条滚动窗口会抛出较老记录。生成器必须对多轮
  查询到的冻结点前 Episode 按 `episode_id` 做单调并集，并在连续两轮全量查询零新增后才允许
  完成；胜率从该冻结并集重算。不得把单次端点返回列表当成完整、稳定的历史样本。
- 最终交付物是可直接阅读的 HTML，保存为 `daily/YYYY-MM-DD.html`。
- 每新增一份日报，必须同步更新 `index.html`，并按日期倒序排列。
- 原始 snapshot、Episode Meta 与代表 replay 写入独立的 `.tmp/environment_daily/<date>/run-*`
  目录；已标记 complete 的 run 不可继续采集或改写。正式 HTML 内嵌 100 条身份链审计摘要。

## 固定构建入口

在仓库根目录运行：

```bash
python3 -m data.processed.environment_daily.generate_live_snapshot --date YYYY-MM-DD
```

该命令使用全新的 run 目录完成 leaderboard 冻结、submission 绑定、Meta 抓取、代表 replay
下载、exact deck 提取、胜率和卡池聚合、100/100 身份审计、HTML 渲染及索引更新。目标日报已存在时
默认拒绝覆盖；只有明确重建同一天时才加 `--overwrite`。审计已有 run 使用：

```bash
python3 -m data.processed.environment_daily.generate_live_snapshot \
  --date YYYY-MM-DD --work .tmp/environment_daily/YYYY-MM-DD/run-* --validate-only
```

## UI 继承合同

- 整体日报固定以 [`daily/2026-07-26.html`](daily/2026-07-26.html) 为基准：保留导航、摘要、
  个人筛选、构筑分布、跨日变化、Match-up/节奏证据入口、静态排名和逐人 exact deck 展开。
- “Top 100 全局构筑卡池”固定以 [`daily/2026-07-25.html`](daily/2026-07-25.html) 为基准：
  使用带卡图的 9 列审计表展示卡牌资料、使用构筑数、覆盖率、合计投入、使用时均值、中位数、
  范围和投入分布；按牌型的典型投入表与个人特例同样必须带卡图。
- 某日缺少逐局 opponent、firstPlayer、turn 或墙钟字段时，保留组件并如实说明不可重建；
  禁止沿用前日报数值或将无证据格子写成 0% 胜率。
- 禁止将正式日报退化为单个搜索框加排行榜表格。发布前须通过
  `tests.test_environment_daily_contract` 的结构回归检查。
- 渲染是 fail-closed：100 行中任一 leaderboard 字段、submissionDate 绑定、截止时间、最新
  Episode、player index、replay 文件、60 张 deck、deck hash 或 W-L-D 对不上，就不得生成正式日报。

当前报告：

- [0726–0728 — Top 100 跨日环境变迁分析（构筑、排名、卡池与证据边界）](environment-transition.html)
- [2026-07-28 — Top 100 实时环境快照 0728（最终 submission + 官方 completed/public Meta）](daily/2026-07-28.html)
- [2026-07-27 — Top 100 实时环境快照 0727（最终 submission + 官方 completed/public Meta）](daily/2026-07-27.html)
- [2026-07-26 — Top 100 今日环境快照 0726（最终 submission + 官方 completed/public Meta）](daily/2026-07-26.html)
- [2026-07-25 — Top 100 Deck 环境与 Match-up 分析](daily/2026-07-25.html)
- [2026-07-23 — Kaggle Top 100 BC 候选调研](daily/2026-07-23.html)
