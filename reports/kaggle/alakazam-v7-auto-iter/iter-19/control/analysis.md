# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：116 / 52 / 2
- 我方错误：0
- 胜率：68.2%
- Meta 加权胜率：70.4%
- 第二回合 Powerful Hand：46/170 (27.1%)
- 我方被击倒事件：181
- 击倒后无 ready attacker：124/181 (68.5%)
- 出现过打手断档的对局：62/170 (36.5%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 257 |
| post_ko_no_ready_attacker | 124 |
| second_turn_powerful_hand_missing | 124 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter_iter15_patch_priority",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter-15-patch-priority.lrZV28"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
