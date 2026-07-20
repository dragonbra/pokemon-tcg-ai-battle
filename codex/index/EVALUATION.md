# 评测索引

官方 Kaggle 对局和 replay 是主要评测依据，历史报告集中在 [reports/kaggle](../../reports/kaggle/)。

当前记录原则：明确对手、样本量、胜负分母、步数和指标口径；本地对局只用于合法性、崩溃和回归检查。历史提交入口见 [submission](../../submission/)。

## 历史结果摘要

| 版本 / 记录 | 数据范围 | 结果 | 关键指标 | 来源与限制 |
| --- | --- | --- | --- | --- |
| V2 | 3 场 public | 1 胜 / 2 负；public score 611.8 | 一场未完成 Alakazam 线，一场牌库耗尽 | [V2 报告](../../reports/kaggle/alakazam-v2-episodes-2026-07-18.md)；样本很小 |
| V3 | 5 场 public | 3 胜 / 2 负；public score 652.4 | 5 场均形成 Alakazam；首只平均 turn 6.0 | [V3 报告](../../reports/kaggle/alakazam-v3-episodes-2026-07-19.md)；不含 validation self-play |
| V5 Auto Iteration | 各 10 场最新 public replay | 5 胜 / 5 负；样本胜率 50%；public score 647.7 | 首次 Alakazam 平均 turn 5.56；牌库耗尽败局 2 场 | [V5 Auto 报告](../../reports/kaggle/alakazam-v5-auto-iter-episodes-2026-07-19/analysis.md)；非配对 A/B 样本 |
| V6 | 10 场 public | 6 胜 / 4 负；public score 668.3 | 首只 Alakazam 9/10，平均 turn 4.44；终局平均 3.5 Prize | [V6 报告](../../reports/kaggle/alakazam-v6-episodes-2026-07-19/analysis.md)；胜局多为对手提前终止，不能直接当 Prize race 胜率 |
| V7 Auto Iteration iter-44 | 170 场 trace | 127 胜 / 43 负 / 0 平；74.7% | 第二回合 Powerful Hand 40/170；错误 0 | [iter-44 报告](../../reports/kaggle/alakazam-v7-auto-iter/iter-44/analysis.md)；这是迭代 trace，不是 Kaggle public score |

V8 当前候选尚未产生新的官方评测结果；后续结果应补充样本范围、control/candidate、分母、对手范围和异常终局说明。

新增结果应链接原始 Episode 或对应报告，不在索引中复制逐局 trace。
