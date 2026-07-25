# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：20
- 对局：20；胜 / 负 / 平：10 / 10 / 0
- 我方错误：0
- 胜率：50.0%
- Meta 加权胜率：48.2%
- 第二回合 Powerful Hand：4/20 (20.0%)
- 我方被击倒事件：51
- 击倒后无 ready attacker：34/51 (66.7%)
- 出现过打手断档的对局：12/20 (60.0%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 45 |
| post_ko_no_ready_attacker | 34 |
| second_turn_powerful_hand_missing | 16 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter_iter31_telepath_active_anchor",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter-31-focus.J3Nmxh"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
