# 研究与实现资料索引

这个目录按“先确认事实，再选择卡组，再实现与评估”的顺序组织。

## 1. 比赛与 simulator 事实

- [`../notes/competition-facts.md`](../notes/competition-facts.md)：官方事实、提交约束和已知规则差异。
- [`research/community-research-zh.md`](research/community-research-zh.md)：Discussion、公开 Notebook 和社区路线的研究摘要。
- [`implementation/interactive-battle-brief-zh.md`](implementation/interactive-battle-brief-zh.md)：交互式 battle 调试需求，暂不作为当前第一阶段阻塞项。

## 2. 卡组与策略研究

- [`decks/deck-meta-alakazam-zh.md`](decks/deck-meta-alakazam-zh.md)：排行榜 archetype 与 Alakazam 的 meta 研究。
- [`decks/alakazam-baseline-notebook-analysis-zh.md`](decks/alakazam-baseline-notebook-analysis-zh.md)：Alakazam 规则策略的拆解。

## 3. 当前实现

- [`implementation/implementation-brief.md`](implementation/implementation-brief.md)：第一版官方 simulator baseline 的目标、验收标准和运行记录。
- [`implementation/baseline-run-zh.md`](implementation/baseline-run-zh.md)：已完成的资源整理、smoke test、提交包和本地 battle 结果。
- 实际提交在 [`../submission/`](../submission/) 下，每个子目录都是一套可独立打包的 agent；官方参考资源在 [`../data/official/`](../data/official/)。
- [`implementation/alakazam-v1-run-zh.md`](implementation/alakazam-v1-run-zh.md)：胡地 V1 卡表映射、目录结构和本地对局验收。
- [`kaggle/alakazam-v2-episodes-2026-07-18.md`](kaggle/alakazam-v2-episodes-2026-07-18.md)：V2 submission 的 Kaggle 官方 Episode 初步复盘。
- [`kaggle/alakazam-v3-episodes-2026-07-19.md`](kaggle/alakazam-v3-episodes-2026-07-19.md)：V3 submission 的 Kaggle 官方 Episode 复盘。
- [`kaggle/alakazam-public-reference-2026-07-19.md`](kaggle/alakazam-public-reference-2026-07-19.md)：Kaggle 公开 Alakazam agent 调研报告及参考 notebook。
- [`kaggle/alakazam-public-code-survey-2026-07-19.md`](kaggle/alakazam-public-code-survey-2026-07-19.md)：多份公开 Alakazam code、卡组和实验结果的整合调研。
- [`kaggle/alakazam-v5-fixed-deck-strategy-research-and-v5-mixed-brief-2026-07-19.md`](kaggle/alakazam-v5-fixed-deck-strategy-research-and-v5-mixed-brief-2026-07-19.md)：固定 V5 卡组、面向 Codex 的 `v5_mixed` 策略研究与实现任务书。当前阶段的优先级是：下载 Kaggle 官方 Episode → 提取 observation/action/replay 指标 → 复盘真实对手与失败原因 → 再扩展策略。
