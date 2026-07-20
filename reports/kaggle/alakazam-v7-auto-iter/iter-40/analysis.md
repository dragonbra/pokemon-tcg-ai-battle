# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：102 / 65 / 3
- 我方错误：0
- 胜率：60.0%
- Meta 加权胜率：61.0%
- 第二回合 Powerful Hand：40/170 (23.5%)
- 我方被击倒事件：205
- 击倒后无 ready attacker：128/205 (62.4%)
- 出现过打手断档的对局：65/170 (38.2%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 230 |
| post_ko_no_ready_attacker | 128 |
| second_turn_powerful_hand_missing | 130 |

## 评测配置

```json
{
  "agent_label": null,
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-40-night-stretcher-lana-gate-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
