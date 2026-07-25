# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：20
- 对局：20；胜 / 负 / 平：13 / 7 / 0
- 我方错误：0
- 胜率：65.0%
- Meta 加权胜率：62.7%
- 第二回合 Powerful Hand：4/20 (20.0%)
- 我方被击倒事件：25
- 击倒后无 ready attacker：21/25 (84.0%)
- 出现过打手断档的对局：6/20 (30.0%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 29 |
| post_ko_no_ready_attacker | 21 |
| second_turn_powerful_hand_unavailable | 16 |

## 评测配置

```json
{
  "agent_label": "iter-47-enriching-focus-20260720",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-47-enriching-focus-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
