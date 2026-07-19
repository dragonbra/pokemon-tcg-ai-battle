# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：101 / 67 / 2
- 我方错误：2
- 胜率：59.4%
- Meta 加权胜率：59.1%
- 第二回合 Powerful Hand：40/170 (23.5%)
- 我方被击倒事件：180
- 击倒后无 ready attacker：119/180 (66.1%)
- 出现过打手断档的对局：59/170 (34.7%)
- 空 Bench Run Away Draw：4

## Case 摘要

| failure_class | 数量 |
|---|---:|
| empty_bench_run_away_draw | 4 |
| post_ko_no_ready_attacker | 119 |
| second_turn_powerful_hand_missing | 130 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_control_iter01",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-01/control"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
