# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：105 / 65 / 0
- 我方错误：0
- 胜率：61.8%
- Meta 加权胜率：61.3%
- 第二回合 Powerful Hand：34/170 (20.0%)
- 我方被击倒事件：170
- 击倒后无 ready attacker：117/170 (68.8%)
- 出现过打手断档的对局：63/170 (37.1%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| post_ko_no_ready_attacker | 117 |
| second_turn_powerful_hand_missing | 136 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_candidate_iter01",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-01/candidate"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
