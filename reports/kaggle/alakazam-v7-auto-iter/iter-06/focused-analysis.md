# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：30
- 对局：30；胜 / 负 / 平：15 / 15 / 0
- 我方错误：0
- 胜率：50.0%
- Meta 加权胜率：47.6%
- 第二回合 Powerful Hand：0/30 (0.0%)
- 我方被击倒事件：50
- 击倒后无 ready attacker：43/50 (86.0%)
- 出现过打手断档的对局：16/30 (53.3%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| post_ko_no_ready_attacker | 43 |
| second_turn_powerful_hand_missing | 30 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-06-focused"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
