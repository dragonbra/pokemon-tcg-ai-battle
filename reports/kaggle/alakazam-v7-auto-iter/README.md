# V7 AutoIter 报告归档说明

这里保存每一轮 AutoIter 的轻量决策记录和必要的指标摘要。完整逐局 trace 不放在本仓库，
而是由评测仓库维护：

`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/`

本 repo 可以正常保留 `decision.md`、`advisor.md`、`metrics.json`、`cases.jsonl` 和
`analysis.md`；完整 trace 只在外部评测目录保留最新一轮，旧轮次在确认关键 case 已经提取
后再清理。

## 关于 iter-09 至 iter-17

这些编号的 V7 目录曾经不在当前工作区中保留。当前检查没有找到对应的原始 trace、指标
摘要或 Git 历史副本，因此不能从别的版本（例如 V5 AutoIter）推断或补写具体结果。
对应目录中的 `decision.md` 是透明的缺档记录，而不是虚构的实验报告。

从 `iter-18` 开始，正式的 AutoIter V9 及后续记录已经恢复为目录级摘要；完整过程也在
[`submission/alakazam_v7_auto_iter/ITER_PROCESS.md`](../../../submission/alakazam_v7_auto_iter/ITER_PROCESS.md)
中维护。

## 保留约定

- 每轮至少保留 `decision.md`，记录改动、control/candidate 结果、采纳状态和证据来源。
- 有 advisor 时保留 `advisor.md`；有可复核统计时保留 `metrics.json`。
- 大体积逐局 JSON 优先放到外部评测仓库；正常情况下只保留最新完整批次，旧批次清理前
  先确认关键 case 已保存到 `cases.jsonl`/`decision.md`。
- 如果某轮需要长期复盘，应只保留该轮的少量 case trace，不保留整批 170 局 JSON。
- 没有可靠指标的轮次不得写成“提升”或“回归”，也不得更新 `BEST_STRATEGY.json`。
