# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：109 / 59 / 2
- 我方错误：0
- 胜率：64.1%
- Meta 加权胜率：64.9%
- 第二回合 Powerful Hand：42/170 (24.7%)
- 我方被击倒事件：187
- 击倒后无 ready attacker：110/187 (58.8%)
- 出现过打手断档的对局：64/170 (37.6%)
- 空 Bench Run Away Draw：0

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed（全部 pass） | 257 |
| post_ko_no_ready_attacker | 110 |
| second_turn_powerful_hand_unavailable | 128 |

## 诊断口径说明

- `second_turn_powerful_hand_unavailable` 的 128 局在目标回合没有合法的
  `attackId=1072` option，不能记为策略漏攻。
- `bench_insurance_missed` 的 257 个 case 都有同回合或后续同回合的合法铺场/接力进展，
  没有 `fail` case。Wondrous Patch 只有在目标为 Kadabra/Alakazam，或 Abra 已有可见
  下一阶段进化来源时，才计为有效接力准备。
- `post_ko_no_ready_attacker` 是 KO 后状态指标；要判定策略错误，必须回溯 KO 前一个己方
  决策点，并确认当时确实存在合法且未被选择的替代动作。

## 评测配置

```json
{
  "agent_label": "iter-46-discovery-20260720",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-46-discovery-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
