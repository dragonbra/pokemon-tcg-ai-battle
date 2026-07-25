# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：106 / 63 / 1
- 我方错误：0
- 胜率：62.4%
- Meta 加权胜率：61.8%
- 第二回合 Powerful Hand：36/170 (21.2%)
- 我方被击倒事件：174
- 击倒后无 ready attacker：117/174 (67.2%)
- 出现过打手断档的对局：68/170 (40.0%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 13 |
| post_ko_no_ready_attacker | 117 |
| second_turn_powerful_hand_missing | 134 |

## 评测配置

```json
{
  "agent_label": null,
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-04-repeat/candidate"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
