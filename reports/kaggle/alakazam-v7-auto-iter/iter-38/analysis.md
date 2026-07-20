# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：101 / 66 / 3
- 我方错误：0
- 胜率：59.4%
- Meta 加权胜率：61.2%
- 第二回合 Powerful Hand：39/170 (22.9%)
- 我方被击倒事件：188
- 击倒后无 ready attacker：113/188 (60.1%)
- 出现过打手断档的对局：65/170 (38.2%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 284 |
| post_ko_no_ready_attacker | 113 |
| second_turn_powerful_hand_missing | 131 |

## 评测配置

```json
{
  "agent_label": "iter-38-second-turn-bench-insurance-20260720",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-38-second-turn-bench-insurance-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
