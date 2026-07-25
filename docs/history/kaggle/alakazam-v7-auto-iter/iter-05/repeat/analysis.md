# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：97 / 72 / 1
- 我方错误：0
- 胜率：57.1%
- Meta 加权胜率：57.8%
- 第二回合 Powerful Hand：49/170 (28.8%)
- 我方被击倒事件：183
- 击倒后无 ready attacker：137/183 (74.9%)
- 出现过打手断档的对局：63/170 (37.1%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 6 |
| post_ko_no_ready_attacker | 137 |
| second_turn_powerful_hand_missing | 121 |

## 评测配置

```json
{
  "agent_label": null,
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-05-repeat"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
