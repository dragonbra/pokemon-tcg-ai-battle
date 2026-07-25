# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：97 / 71 / 2
- 我方错误：0
- 胜率：57.1%
- Meta 加权胜率：57.2%
- 第二回合 Powerful Hand：34/170 (20.0%)
- 我方被击倒事件：185
- 击倒后无 ready attacker：146/185 (78.9%)
- 出现过打手断档的对局：72/170 (42.4%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 228 |
| post_ko_no_ready_attacker | 146 |
| second_turn_powerful_hand_missing | 136 |

## 评测配置

```json
{
  "agent_label": "alakazam_v7_auto_iter",
  "command": "analyze",
  "report_dir": "/private/tmp/alakazam-v7-auto-iter/iter-07-full"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
