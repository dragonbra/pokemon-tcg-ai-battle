# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：0
- 对局：170；胜 / 负 / 平：93 / 74 / 3
- 我方错误：3
- 胜率：54.7%
- Meta 加权胜率：54.7%
- 第二回合 Powerful Hand：0/170 (0.0%)
- 我方被击倒事件：0
- 击倒后无 ready attacker：0/0 (0.0%)
- 出现过打手断档的对局：0/170 (0.0%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| second_turn_powerful_hand_missing | 170 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter_candidate",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter-candidate-summary"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
