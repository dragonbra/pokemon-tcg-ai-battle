# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：105 / 61 / 4
- 我方错误：0
- 胜率：61.8%
- Meta 加权胜率：62.2%
- 第二回合 Powerful Hand：40/170 (23.5%)
- 我方被击倒事件：168
- 击倒后无 ready attacker：102/168 (60.7%)
- 出现过打手断档的对局：50/170 (29.4%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 365 |
| post_ko_no_ready_attacker | 102 |
| second_turn_powerful_hand_missing | 130 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter_iter31_telepath_active_anchor_full",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter-31-full.yUD8RP"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
