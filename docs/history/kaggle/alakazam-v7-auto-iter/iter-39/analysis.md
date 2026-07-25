# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：114 / 55 / 1
- 我方错误：0
- 胜率：67.1%
- Meta 加权胜率：68.5%
- 第二回合 Powerful Hand：51/170 (30.0%)
- 我方被击倒事件：164
- 击倒后无 ready attacker：106/164 (64.6%)
- 出现过打手断档的对局：57/170 (33.5%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 242 |
| post_ko_no_ready_attacker | 106 |
| second_turn_powerful_hand_missing | 119 |

## 评测配置

```json
{
  "agent_label": "iter-39-fresh-bench-second-turn-20260720",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-39-fresh-bench-second-turn-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
